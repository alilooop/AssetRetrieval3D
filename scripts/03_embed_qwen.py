"""
Generate Qwen embeddings for text (EN and CN) and multi-image.

This script:
1. Generates English text embeddings using batch API
2. Generates Chinese text embeddings using batch API
3. Generates multi-image embeddings (up to 8 images per asset) using API
4. Saves embeddings to disk
"""
import sys
import logging
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List
import dashscope
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.data_loader import DataLoader
from utils.image_utils import sample_viewpoint_images

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class QwenEmbedder:
    """Generate Qwen embeddings for text and images."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL
        )
    
    def create_text_embedding_batch_jsonl(
        self,
        texts: Dict[str, str],
        output_file: Path,
        start_idx: int = 0
    ) -> int:
        """
        Create JSONL file for text embedding batch API.
        
        Args:
            texts: Dictionary of id -> text
            output_file: Output JSONL file path
            start_idx: Starting custom_id index
        
        Returns:
            Number of entries created
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (text_id, text) in enumerate(texts.items()):
                entry = {
                    "custom_id": f"{start_idx + i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": config.QWEN_EMBEDDING_MODEL,
                        "input": text
                    },
                    "metadata": {"text_id": text_id}
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return len(texts)
    
    def split_texts_into_batches(
        self,
        texts: Dict[str, str],
        batch_size: int,
        prefix: str
    ) -> List[Path]:
        """Split texts into batch JSONL files."""
        batch_dir = config.BATCH_JSONL_DIR / prefix
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        text_items = list(texts.items())
        jsonl_files = []
        
        for batch_idx in range(0, len(text_items), batch_size):
            batch_texts = dict(text_items[batch_idx:batch_idx + batch_size])
            
            output_file = batch_dir / f"batch_{batch_idx // batch_size:04d}.jsonl"
            
            logger.info(f"Creating {prefix} batch {batch_idx // batch_size}: {len(batch_texts)} texts")
            self.create_text_embedding_batch_jsonl(
                batch_texts,
                output_file,
                start_idx=batch_idx
            )
            
            jsonl_files.append(output_file)
        
        return jsonl_files
    
    def save_batch_ids(self, batch_info: Dict, filepath: Path):
        """Save batch IDs and metadata to disk."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch IDs saved to {filepath}")
    
    def load_batch_ids(self, filepath: Path) -> Dict:
        """Load batch IDs and metadata from disk."""
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def submit_all_batches(self, jsonl_files: List[Path], batch_ids_file: Path) -> Dict[str, Dict]:
        """
        Submit all batch jobs at once and save batch IDs.
        
        Args:
            jsonl_files: List of JSONL files to process
            batch_ids_file: Path to save batch IDs
        
        Returns:
            Dictionary mapping batch_id -> {jsonl_file, result_file, status}
        """
        # Load existing batch info if available
        batch_info = self.load_batch_ids(batch_ids_file)
        
        # Check which batches are already completed
        completed_batches = set()
        for batch_id, info in list(batch_info.items()):
            result_file = Path(info['result_file'])
            if result_file.exists():
                completed_batches.add(info['jsonl_file'])
                logger.info(f"Batch already completed: {result_file.name}")
        
        # Submit new batch jobs
        for jsonl_file in jsonl_files:
            jsonl_file_str = str(jsonl_file)
            
            # Skip if already processed
            if jsonl_file_str in completed_batches:
                continue
            
            # Check if batch job already exists
            existing_batch_id = None
            for batch_id, info in batch_info.items():
                if info['jsonl_file'] == jsonl_file_str:
                    existing_batch_id = batch_id
                    break
            
            if existing_batch_id:
                logger.info(f"Batch job already submitted for {jsonl_file.name}: {existing_batch_id}")
                continue
            
            # Upload and create batch job
            result_file = jsonl_file.parent / f"result_{jsonl_file.name}"
            
            try:
                logger.info(f"Uploading {jsonl_file.name}...")
                file_object = self.client.files.create(file=jsonl_file, purpose="batch")
                file_id = file_object.id
                
                logger.info("Creating batch job...")
                batch = self.client.batches.create(
                    input_file_id=file_id,
                    endpoint="/v1/embeddings",
                    completion_window="24h"
                )
                batch_id = batch.id
                logger.info(f"Batch job created: {batch_id}")
                
                # Store batch info
                batch_info[batch_id] = {
                    'jsonl_file': jsonl_file_str,
                    'result_file': str(result_file),
                    'status': 'submitted',
                    'file_id': file_id
                }
                
                # Save immediately after each submission
                self.save_batch_ids(batch_info, batch_ids_file)
                
            except Exception as e:
                logger.error(f"Failed to submit batch for {jsonl_file.name}: {e}")
                continue
        
        return batch_info
    
    def monitor_all_batches(self, batch_info: Dict[str, Dict], batch_ids_file: Path, poll_interval: int = 30):
        """
        Monitor all batch jobs until completion.
        
        Args:
            batch_info: Dictionary of batch_id -> metadata
            batch_ids_file: Path to batch IDs file
            poll_interval: Seconds between status checks
        """
        pending_batches = set(batch_info.keys())
        
        logger.info(f"Monitoring {len(pending_batches)} batch jobs...")
        
        while pending_batches:
            for batch_id in list(pending_batches):
                try:
                    batch = self.client.batches.retrieve(batch_id=batch_id)
                    status = batch.status
                    batch_info[batch_id]['status'] = status
                    
                    if status == "completed":
                        logger.info(f"✓ Batch {batch_id} completed")
                        
                        # Download results
                        result_file = Path(batch_info[batch_id]['result_file'])
                        if not result_file.exists() and batch.output_file_id:
                            logger.info("Downloading results...")
                            content = self.client.files.content(batch.output_file_id)
                            content.write_to_file(result_file)
                            logger.info(f"Results saved to {result_file}")
                        
                        pending_batches.remove(batch_id)
                        
                    elif status in ["failed", "expired", "cancelled"]:
                        logger.error(f"✗ Batch {batch_id} {status}")
                        pending_batches.remove(batch_id)
                        
                        if status == "failed":
                            logger.error(f"  Error details: {batch.errors}")
                    
                    else:
                        logger.info(f"Batch {batch_id}: {status}")
                    
                    # Save updated status
                    self.save_batch_ids(batch_info, batch_ids_file)
                    
                except Exception as e:
                    logger.error(f"Error checking batch {batch_id}: {e}")
            
            if pending_batches:
                logger.info(f"Waiting {poll_interval}s... ({len(pending_batches)} batches pending)")
                time.sleep(poll_interval)
    
    def parse_text_embedding_results(
        self,
        result_files: List[Path],
        id_mapping: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """
        Parse text embedding results from batch API.
        
        Args:
            result_files: List of result JSONL files
            id_mapping: Mapping from custom_id to text_id
        
        Returns:
            Dictionary of text_id -> embedding
        """
        embeddings = {}
        
        for result_file in result_files:
            if not result_file.exists():
                logger.warning(f"Result file not found: {result_file}")
                continue
            
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    result = json.loads(line)
                    custom_id = result.get('custom_id')
                    
                    response = result.get('response', {})
                    body = response.get('body', {})
                    data = body.get('data', [])
                    
                    if data:
                        embedding = data[0].get('embedding', [])
                        if custom_id in id_mapping:
                            text_id = id_mapping[custom_id]
                            embeddings[text_id] = np.array(embedding, dtype=np.float32)
        
        return embeddings
    
    def generate_text_embeddings_batch(
        self,
        texts: Dict[str, str],
        prefix: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate text embeddings using batch API.
        
        Args:
            texts: Dictionary of text_id -> text
            prefix: Prefix for batch files (e.g., 'en_text', 'cn_text')
        
        Returns:
            Dictionary of text_id -> embedding
        """
        logger.info(f"Generating {prefix} embeddings for {len(texts)} texts")
        
        # Split into batches
        jsonl_files = self.split_texts_into_batches(
            texts,
            config.TEXT_EMBEDDING_API_BATCH_SIZE,
            prefix
        )
        
        # Batch IDs file path
        batch_dir = config.BATCH_JSONL_DIR / prefix
        batch_ids_file = batch_dir / "batch_ids.json"
        
        # Submit all batch jobs at once
        batch_info = self.submit_all_batches(jsonl_files, batch_ids_file)
        
        # Monitor all batches until completion
        self.monitor_all_batches(batch_info, batch_ids_file)
        
        # Collect result files
        result_files = [Path(info['result_file']) for info in batch_info.values()]
        
        # Create custom_id to text_id mapping
        text_ids = list(texts.keys())
        id_mapping = {str(i): text_ids[i] for i in range(len(text_ids))}
        
        # Parse results
        embeddings = self.parse_text_embedding_results(result_files, id_mapping)
        
        logger.info(f"Generated {len(embeddings)} {prefix} embeddings")
        return embeddings
    
    def embed_single_image(self, image_path: Path) -> np.ndarray:
        """
        Embed a single image using Qwen API.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Embedding as numpy array
        """
        input_data = [{'image': str(image_path)}]
        
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data
        )
        
        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get('embeddings', [])
            if embeddings:
                return np.array(embeddings[0]['embedding'], dtype=np.float32)
        
        raise RuntimeError(f"Failed to embed image: {image_path}")
    
    def embed_multi_images(self, image_paths: List[Path]) -> np.ndarray:
        """
        Embed multiple images as a single embedding using Qwen API.
        
        Args:
            image_paths: List of image paths (up to 8)
        
        Returns:
            Single embedding representing all images
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        # Qwen supports up to 8 images
        image_paths = image_paths[:8]
        
        # Prepare input
        input_data = [{'image': str(path)} for path in image_paths]
        
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data
        )
        
        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get('embeddings', [])
            if embeddings:
                # For multi-image, we get one embedding
                return np.array(embeddings[0]['embedding'], dtype=np.float32)
        
        raise RuntimeError("Failed to embed multi-images")
    
    def generate_image_embeddings(self, asset_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate multi-image embeddings for all assets.
        
        Args:
            asset_ids: List of asset IDs
        
        Returns:
            Dictionary of asset_id -> multi-image embedding
        """
        logger.info(f"Generating multi-image embeddings for {len(asset_ids)} assets")
        
        image_embeddings = {}
        skipped = 0
        
        for asset_id in tqdm(asset_ids, desc="Processing assets"):
            try:
                # Sample 8 viewpoint images uniformly
                image_paths = sample_viewpoint_images(
                    asset_id,
                    num_samples=config.QWEN_NUM_IMAGES
                )
                
                if not image_paths:
                    logger.warning(f"No images found for asset {asset_id}")
                    skipped += 1
                    continue
                
                # Generate multi-image embedding
                embedding = self.embed_multi_images(image_paths)
                image_embeddings[asset_id] = embedding
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to embed images for asset {asset_id}: {e}")
                skipped += 1
                continue
        
        logger.info(f"Generated image embeddings for {len(image_embeddings)} assets")
        logger.info(f"Skipped {skipped} assets")
        
        return image_embeddings
    
    def save_embeddings(
        self,
        text_en_embeddings: Dict[str, np.ndarray],
        text_cn_embeddings: Dict[str, np.ndarray],
        image_embeddings: Dict[str, np.ndarray]
    ):
        """Save embeddings to disk."""
        logger.info("Saving embeddings to disk...")
        
        with open(config.QWEN_TEXT_EN_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(text_en_embeddings, f)
        logger.info(f"Saved English text embeddings to {config.QWEN_TEXT_EN_EMBEDDINGS_FILE}")
        
        with open(config.QWEN_TEXT_CN_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(text_cn_embeddings, f)
        logger.info(f"Saved Chinese text embeddings to {config.QWEN_TEXT_CN_EMBEDDINGS_FILE}")
        
        with open(config.QWEN_IMAGE_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(image_embeddings, f)
        logger.info(f"Saved image embeddings to {config.QWEN_IMAGE_EMBEDDINGS_FILE}")
    
    def load_existing_embeddings(self) -> tuple:
        """Load existing embeddings if available."""
        text_en = None
        text_cn = None
        images = None
        
        if config.QWEN_TEXT_EN_EMBEDDINGS_FILE.exists():
            with open(config.QWEN_TEXT_EN_EMBEDDINGS_FILE, 'rb') as f:
                text_en = pickle.load(f)
            logger.info(f"Loaded {len(text_en)} English text embeddings")
        
        if config.QWEN_TEXT_CN_EMBEDDINGS_FILE.exists():
            with open(config.QWEN_TEXT_CN_EMBEDDINGS_FILE, 'rb') as f:
                text_cn = pickle.load(f)
            logger.info(f"Loaded {len(text_cn)} Chinese text embeddings")
        
        if config.QWEN_IMAGE_EMBEDDINGS_FILE.exists():
            with open(config.QWEN_IMAGE_EMBEDDINGS_FILE, 'rb') as f:
                images = pickle.load(f)
            logger.info(f"Loaded {len(images)} image embeddings")
        
        return text_en, text_cn, images
    
    def generate_all(self, force_regenerate: bool = False):
        """
        Main method to generate all Qwen embeddings.
        
        Args:
            force_regenerate: If True, regenerate even if files exist
        """
        # Check if embeddings already exist
        if not force_regenerate:
            text_en, text_cn, images = self.load_existing_embeddings()
            if text_en is not None and text_cn is not None and images is not None:
                logger.info("✓ Qwen embeddings already exist. Use force_regenerate=True to regenerate.")
                return
        
        # Load captions for ALL assets (text embeddings work without images)
        captions_en = self.data_loader.load_english_captions()
        captions_cn = self.data_loader.load_chinese_captions()
        
        # Limit if MAX_ASSETS is set
        if config.MAX_ASSETS:
            asset_ids = list(captions_en.keys())[:config.MAX_ASSETS]
            captions_en = {aid: captions_en[aid] for aid in asset_ids if aid in captions_en}
            captions_cn = {aid: captions_cn[aid] for aid in asset_ids if aid in captions_cn}
            logger.info(f"Limited to {len(asset_ids)} assets for processing")
        
        # Generate English text embeddings for ALL assets with captions
        text_en_embeddings = self.generate_text_embeddings_batch(captions_en, "en_text")
        
        # Generate Chinese text embeddings for ALL assets with captions
        if captions_cn:
            text_cn_embeddings = self.generate_text_embeddings_batch(captions_cn, "cn_text")
        else:
            logger.warning("No Chinese captions available, skipping Chinese text embeddings")
            text_cn_embeddings = {}
        
        # Generate multi-image embeddings ONLY for assets with gobjaverse images
        asset_ids_with_images = self.data_loader.get_asset_ids_with_images(max_assets=config.MAX_ASSETS)
        logger.info(f"Generating image embeddings for {len(asset_ids_with_images)} assets with images")
        image_embeddings = self.generate_image_embeddings(asset_ids_with_images)
        
        # Save to disk
        self.save_embeddings(text_en_embeddings, text_cn_embeddings, image_embeddings)
        
        logger.info("\n✓ Qwen embedding generation complete!")
        logger.info(f"  English text embeddings: {len(text_en_embeddings)}")
        logger.info(f"  Chinese text embeddings: {len(text_cn_embeddings)}")
        logger.info(f"  Multi-image embeddings: {len(image_embeddings)}")


def main():
    """Main entry point."""
    try:
        config.validate_config()
        
        embedder = QwenEmbedder()
        embedder.generate_all(force_regenerate=False)
        
    except Exception as e:
        logger.error(f"Qwen embedding generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

