"""
Generate SigLip embeddings for text and images using distributed multi-GPU inference.

This script uses torchrun for distributed processing and numpy.memmap for 
memory-efficient aggregation of large-scale embeddings.
"""
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import numpy as np
import numpy.lib.format as npy_format

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
    def embed_texts_to_memmap(
        self, 
        texts: List[str], 
        memmap: np.ndarray, 
        start_idx: int,
        batch_size: int = 32
    ):
        """Embed texts and write directly to memmap."""
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            text_embeddings = self.model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Write to memmap
            curr_idx = start_idx + i
            memmap[curr_idx : curr_idx + len(batch_texts)] = text_embeddings.cpu().numpy()
            
            if i % (batch_size * 10) == 0 and dist.get_rank() == 0:
                logger.info(f"Embedded {i}/{len(texts)} texts on Rank 0")

    @torch.no_grad()
    def embed_images_to_memmap(
        self, 
        asset_ids: List[str], 
        memmap: np.ndarray, 
        asset_offsets: Dict[str, int],
        batch_size: int = 16
    ):
        """Embed images for assets and write to memmap using pre-calculated offsets."""
        for asset_id in tqdm(asset_ids, desc=f"Rank {dist.get_rank()} processing images"):
            image_paths = get_asset_image_paths(asset_id)
            if not image_paths:
                continue
                
            images = load_images(image_paths)
            if not images:
                continue
                
            asset_start_idx = asset_offsets[asset_id]
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(self.device)
                
                image_embeddings = self.model.get_image_features(**inputs)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                
                curr_idx = asset_start_idx + i
                memmap[curr_idx : curr_idx + len(batch_images)] = image_embeddings.cpu().numpy()


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
    
    # Pre-allocate Text Memmap on Rank 0
    if rank == 0:
        # Check disk space
        required_size = total_texts * embedder.embedding_dim * 4  # float32 = 4 bytes
        _, _, free_space = shutil.disk_usage(config.EMBEDDINGS_DIR)
        if free_space < required_size * 1.2:  # 20% buffer
            raise RuntimeError(f"Insufficient disk space for text embeddings. Required: {required_size/1e9:.2f}GB, Free: {free_space/1e9:.2f}GB")

        logger.info(f"Pre-allocating text memmap for {total_texts} texts at {config.SIGLIP_TEXT_EMBEDDINGS_MEMMAP}")
        shape = (total_texts, embedder.embedding_dim)
        # open_memmap handles the NPY header and pre-allocation
        text_memmap = npy_format.open_memmap(
            config.SIGLIP_TEXT_EMBEDDINGS_MEMMAP, 
            mode='w+', 
            dtype='float32', 
            shape=shape
        )
        del text_memmap # Flush and close header
        
        # Save metadata
        with open(config.SIGLIP_TEXT_METADATA_FILE, 'w') as f:
            json.dump(all_asset_ids, f)

    dist.barrier()
    
    # Map text memmap in each rank for writing
    text_memmap = npy_format.open_memmap(
        config.SIGLIP_TEXT_EMBEDDINGS_MEMMAP, 
        mode='r+', 
        dtype='float32', 
        shape=(total_texts, embedder.embedding_dim)
    )
    
    # Shard text assets
    rank_asset_ids = all_asset_ids[rank::world_size]
    rank_texts = [captions_en[aid] for aid in rank_asset_ids]
    # Each rank needs its global offset. For simplicity since we use rank::world_size, 
    # we need to calculate the actual indices.
    
    embedder.load_model()
     
    # Text Embedding Generation
    logger.info(f"Rank {rank} embedding {len(rank_asset_ids)} texts...")
    
    # Use a dictionary for fast global index lookup
    aid_to_global_idx = {aid: i for i, aid in enumerate(all_asset_ids)}
    
    def embed_rank_texts_optimized():
        batch_size = config.EMBEDDING_BATCH_SIZE
        for i in range(0, len(rank_asset_ids), batch_size):
            batch_aids = rank_asset_ids[i : i + batch_size]
            batch_texts = rank_texts[i : i + batch_size]
            
            inputs = embedder.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                text_embeddings = embedder.model.get_text_features(**inputs)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            emb_np = text_embeddings.cpu().numpy()
            for j, aid in enumerate(batch_aids):
                text_memmap[aid_to_global_idx[aid]] = emb_np[j]

    embed_rank_texts_optimized()
    text_memmap.flush()
    dist.barrier()
    
    # 2. Image Data Pre-scan and Sharding
    all_image_asset_ids = data_loader.get_asset_ids_with_images(max_assets=config.MAX_ASSETS)
    
    # Pre-scan viewpoint counts (All ranks can do this to save time, then sync, or Rank 0 does it)
    # Let's have Rank 0 do it and broadcast or save to file
    viewpoint_counts = {}
    total_viewpoints = 0
    asset_offsets = {}

    if rank == 0:
        logger.info("Pre-scanning viewpoints for all assets...")
        for aid in tqdm(all_image_asset_ids, desc="Scanning viewpoints"):
            count = len(get_asset_image_paths(aid))
            viewpoint_counts[aid] = count
            asset_offsets[aid] = total_viewpoints
            total_viewpoints += count
            
        logger.info(f"Total viewpoints to embed: {total_viewpoints}")
        
        # Check disk space for images
        required_size = total_viewpoints * embedder.embedding_dim * 4
        _, _, free_space = shutil.disk_usage(config.EMBEDDINGS_DIR)
        if free_space < required_size * 1.2:
            raise RuntimeError(f"Insufficient disk space for image embeddings. Required: {required_size/1e9:.2f}GB, Free: {free_space/1e9:.2f}GB")

        # Allocate Image Memmap
        logger.info(f"Pre-allocating image memmap for {total_viewpoints} viewpoints at {config.SIGLIP_IMAGE_EMBEDDINGS_MEMMAP}")
        shape = (total_viewpoints, embedder.embedding_dim)
        image_memmap = npy_format.open_memmap(
            config.SIGLIP_IMAGE_EMBEDDINGS_MEMMAP, 
            mode='w+', 
            dtype='float32', 
            shape=shape
        )
        del image_memmap
        
        # Save image metadata (asset_id -> {start_idx, count})
        image_metadata = {
            aid: {"start_idx": asset_offsets[aid], "count": viewpoint_counts[aid]}
            for aid in all_image_asset_ids
        }
        with open(config.SIGLIP_IMAGE_METADATA_FILE, 'w') as f:
            json.dump(image_metadata, f)
            
        # Broadcast total_viewpoints and metadata info
        meta_bundle = (total_viewpoints, asset_offsets, viewpoint_counts)
    else:
        meta_bundle = None

    # Broadcast metadata
    # dist.broadcast_object_list requires a list
    bundle_list = [meta_bundle]
    dist.broadcast_object_list(bundle_list, src=0)
    total_viewpoints, asset_offsets, viewpoint_counts = bundle_list[0]
    
    dist.barrier()
    
    # Map image memmap in 'r+' mode for distributed writing
    image_memmap = npy_format.open_memmap(
        config.SIGLIP_IMAGE_EMBEDDINGS_MEMMAP, 
        mode='r+', 
        dtype='float32', 
        shape=(total_viewpoints, embedder.embedding_dim)
    )
    
    # Shard image assets
    rank_image_asset_ids = all_image_asset_ids[rank::world_size]
    
    logger.info(f"Rank {rank} embedding images for {len(rank_image_asset_ids)} assets...")
    embedder.embed_images_to_memmap(rank_image_asset_ids, image_memmap, asset_offsets)
    
    image_memmap.flush()
    dist.barrier()
    
    if rank == 0:
        logger.info("\n✓ SigLip embedding generation complete!")
        logger.info(f"  Text embeddings: {total_texts}")
        logger.info(f"  Image embeddings: {total_viewpoints} viewpoints across {len(all_image_asset_ids)} assets")

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
