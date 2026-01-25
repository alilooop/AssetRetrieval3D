"""
Generate SigLip embeddings for text and images using distributed multi-GPU inference.

This script uses torchrun for distributed processing. Each rank processes a shard
of the data and saves to temporary files. Rank 0 then merges all temporary files
directly into HDF5 format incrementally, without loading all embeddings into memory.
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import numpy as np
import h5py

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.data_loader import DataLoader
from utils.image_utils import get_asset_image_paths, load_images

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed process group and return rank, world_size, local_rank."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class SigLipEmbedder:
    """Generate SigLip embeddings for text and images with distributed support."""
    
    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.processor = None
        self.embedding_dim = 1152  # Default for SigLip2-so400m
    
    def load_model(self):
        """Load SigLip model and processor."""
        if dist.get_rank() == 0:
            logger.info(f"Loading SigLip model: {config.SIGLIP_MODEL} on device {self.device}")
        
        # Use specific device for this rank
        self.model = AutoModel.from_pretrained(
            config.SIGLIP_MODEL
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
        
        # Determine embedding dimension if not fixed
        with torch.no_grad():
            dummy_text = self.processor(text="dummy", return_tensors="pt").to(self.device)
            self.embedding_dim = self.model.get_text_features(**dummy_text).shape[-1]
            
        if dist.get_rank() == 0:
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """Embed texts and return embeddings."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = [t for t in texts[i : i + batch_size]]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            text_embeddings = self.model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.extend(text_embeddings.cpu().numpy())
            
            if i % (batch_size * 10) == 0 and dist.get_rank() == 0:
                logger.info(f"Embedded {i}/{len(texts)} texts on Rank 0")
        
        return embeddings

    @torch.no_grad()
    def embed_images(
        self, 
        asset_ids: List[str], 
        batch_size: int = 16
    ) -> Dict[str, List[np.ndarray]]:
        """Embed images for assets and return embeddings."""
        embeddings_dict = {}
        
        for asset_id in tqdm(asset_ids, desc=f"Rank {dist.get_rank()} processing images"):
            image_paths = get_asset_image_paths(asset_id)
            if not image_paths:
                continue
                
            images = load_images(image_paths)
            if not images:
                continue
            
            asset_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(self.device)
                
                image_embeddings = self.model.get_image_features(**inputs)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                
                asset_embeddings.extend(image_embeddings.cpu().numpy())
            
            embeddings_dict[asset_id] = asset_embeddings
        
        return embeddings_dict


def main():
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    
    data_loader = DataLoader()
    embedder = SigLipEmbedder(device)
    
    # 1. Load and Shard Text Data
    captions_en = data_loader.load_english_captions()
    if config.MAX_ASSETS:
        all_asset_ids = sorted(list(captions_en.keys()))[:config.MAX_ASSETS]
        captions_en = {aid: captions_en[aid] for aid in all_asset_ids}
    else:
        all_asset_ids = sorted(list(captions_en.keys()))
        
    total_texts = len(all_asset_ids)
    
    # Check disk space on Rank 0
    if rank == 0:
        required_size = total_texts * embedder.embedding_dim * 4  # float32 = 4 bytes
        _, _, free_space = shutil.disk_usage(config.EMBEDDINGS_DIR)
        if free_space < required_size * 1.2:  # 20% buffer
            raise RuntimeError(f"Insufficient disk space for text embeddings. Required: {required_size/1e9:.2f}GB, Free: {free_space/1e9:.2f}GB")
        
        logger.info(f"Processing {total_texts} text embeddings")
    
    dist.barrier()
    
    # Shard text assets
    rank_asset_ids = all_asset_ids[rank::world_size]
    rank_texts = [captions_en[aid] for aid in rank_asset_ids]
    
    embedder.load_model()
     
    # Text Embedding Generation
    logger.info(f"Rank {rank} embedding {len(rank_asset_ids)} texts...")
    
    # Generate embeddings for this rank
    rank_text_embeddings = embedder.embed_texts(rank_texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    
    # Save rank embeddings to temporary file
    temp_dir = config.EMBEDDINGS_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    temp_text_file = temp_dir / f"text_rank_{rank}.npz"
    np.savez_compressed(
        temp_text_file,
        asset_ids=rank_asset_ids,
        embeddings=np.array(rank_text_embeddings)
    )
    logger.info(f"Rank {rank} saved {len(rank_asset_ids)} text embeddings to temp file")
    
    dist.barrier()
    
    # 2. Image Data Pre-scan and Sharding
    all_image_asset_ids = data_loader.get_asset_ids_with_images(max_assets=config.MAX_ASSETS)
    
    # Shard image assets
    rank_image_asset_ids = all_image_asset_ids[rank::world_size]
    
    logger.info(f"Rank {rank} embedding images for {len(rank_image_asset_ids)} assets...")
    rank_image_embeddings = embedder.embed_images(rank_image_asset_ids, batch_size=16)
    
    # Save rank image embeddings to temporary file
    temp_image_file = temp_dir / f"image_rank_{rank}.npz"
    
    # Flatten the structure for npz saving
    image_data = {
        'asset_ids': list(rank_image_embeddings.keys()),
        'counts': [len(embs) for embs in rank_image_embeddings.values()]
    }
    
    # Flatten embeddings
    all_embeddings = []
    for embs in rank_image_embeddings.values():
        all_embeddings.extend(embs)
    image_data['embeddings'] = np.array(all_embeddings)
    
    np.savez_compressed(temp_image_file, **image_data)
    logger.info(f"Rank {rank} saved image embeddings for {len(rank_image_embeddings)} assets to temp file")
    
    dist.barrier()
    
    # 3. Merge on Rank 0 and save to HDF5 (incrementally, without loading all in memory)
    if rank == 0:
        logger.info("\n=== Merging embeddings from all ranks ===")
        
        # First pass: collect metadata without loading embeddings
        logger.info("Collecting metadata from temporary files...")
        text_metadata = []
        image_metadata = []
        total_viewpoints = 0
        
        for r in range(world_size):
            # Text metadata
            temp_file = temp_dir / f"text_rank_{r}.npz"
            data = np.load(temp_file)
            text_metadata.append({
                'rank': r,
                'asset_ids': data['asset_ids'],
                'count': len(data['asset_ids'])
            })
            data.close()
            
            # Image metadata
            temp_file = temp_dir / f"image_rank_{r}.npz"
            data = np.load(temp_file)
            counts = data['counts']
            total_viewpoints += np.sum(counts)
            image_metadata.append({
                'rank': r,
                'asset_ids': data['asset_ids'],
                'counts': counts,
                'total_viewpoints': np.sum(counts)
            })
            data.close()
        
        total_text_items = sum(m['count'] for m in text_metadata)
        logger.info(f"Total text embeddings: {total_text_items}")
        logger.info(f"Total image viewpoints: {total_viewpoints}")
        
        # Create HDF5 files and write incrementally
        # Text embeddings
        logger.info(f"Creating and writing text embeddings to {config.SIGLIP_TEXT_EMBEDDINGS_FILE}")
        with h5py.File(config.SIGLIP_TEXT_EMBEDDINGS_FILE, 'w') as f:
            # Create datasets
            embeddings_ds = f.create_dataset(
                'embeddings',
                shape=(total_text_items, embedder.embedding_dim),
                dtype='float32',
                chunks=(min(1000, total_text_items), embedder.embedding_dim),
                compression='gzip',
                compression_opts=4
            )
            
            asset_ids_ds = f.create_dataset(
                'asset_ids',
                shape=(total_text_items,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(min(1000, total_text_items),),
                compression='gzip',
                compression_opts=4
            )
            
            # Write data rank by rank
            write_offset = 0
            for meta in text_metadata:
                temp_file = temp_dir / f"text_rank_{meta['rank']}.npz"
                data = np.load(temp_file)
                
                count = meta['count']
                embeddings_ds[write_offset:write_offset + count] = data['embeddings']
                asset_ids_ds[write_offset:write_offset + count] = data['asset_ids']
                
                write_offset += count
                data.close()
                logger.info(f"  Wrote {count} embeddings from rank {meta['rank']}")
            
            # Add metadata
            f.attrs['total_items'] = total_text_items
            f.attrs['embedding_dim'] = embedder.embedding_dim
        
        logger.info(f"✓ Text embeddings saved: {total_text_items} items")
        
        # Image embeddings
        logger.info(f"Creating and writing image embeddings to {config.SIGLIP_IMAGE_EMBEDDINGS_FILE}")
        with h5py.File(config.SIGLIP_IMAGE_EMBEDDINGS_FILE, 'w') as f:
            # Create datasets
            embeddings_ds = f.create_dataset(
                'embeddings',
                shape=(total_viewpoints, embedder.embedding_dim),
                dtype='float32',
                chunks=(min(1000, total_viewpoints), embedder.embedding_dim),
                compression='gzip',
                compression_opts=4
            )
            
            # Create metadata dataset
            metadata_dtype = np.dtype([
                ('asset_id', h5py.string_dtype(encoding='utf-8')),
                ('start_idx', np.int64),
                ('count', np.int32)
            ])
            
            total_assets = sum(len(m['asset_ids']) for m in image_metadata)
            metadata_ds = f.create_dataset(
                'metadata',
                shape=(total_assets,),
                dtype=metadata_dtype,
                compression='gzip',
                compression_opts=4
            )
            
            # Write data rank by rank
            viewpoint_offset = 0
            asset_offset = 0
            
            for meta in image_metadata:
                temp_file = temp_dir / f"image_rank_{meta['rank']}.npz"
                data = np.load(temp_file)
                
                asset_ids = data['asset_ids']
                counts = data['counts']
                embeddings = data['embeddings']
                
                # Write embeddings
                total_rank_viewpoints = len(embeddings)
                embeddings_ds[viewpoint_offset:viewpoint_offset + total_rank_viewpoints] = embeddings
                
                # Write metadata for each asset
                emb_offset = 0
                for i, (aid, count) in enumerate(zip(asset_ids, counts)):
                    aid = str(aid)  # Ensure string type
                    metadata_ds[asset_offset + i] = (aid, viewpoint_offset + emb_offset, count)
                    emb_offset += count
                
                viewpoint_offset += total_rank_viewpoints
                asset_offset += len(asset_ids)
                
                data.close()
                logger.info(f"  Wrote {total_rank_viewpoints} viewpoints from rank {meta['rank']}")
            
            # Add file-level attributes
            f.attrs['total_viewpoints'] = total_viewpoints
            f.attrs['total_assets'] = total_assets
            f.attrs['embedding_dim'] = embedder.embedding_dim
        
        logger.info(f"✓ Image embeddings saved: {total_viewpoints} viewpoints from {total_assets} assets")
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        logger.info("\n✓ SigLip embedding generation complete!")
        logger.info(f"  Text embeddings: {total_text_items} assets")
        logger.info(f"  Image embeddings: {total_viewpoints} viewpoints across {total_assets} assets")
        logger.info(f"  Files saved to {config.EMBEDDINGS_DIR}")
    
    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Distributed SigLip embedding failed: {e}", exc_info=True)
        # Ensure we cleanup even on failure if initialized
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
