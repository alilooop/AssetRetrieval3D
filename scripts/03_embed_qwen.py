"""
Generate Qwen embeddings for text (EN and CN) and multi-image.

This script:
1. Generates English text embeddings using online API with multithreading
2. Generates Chinese text embeddings using online API with multithreading
3. Generates multi-image embeddings (up to 8 images per asset) using online API with multithreading
4. Saves embeddings to disk
"""
import sys
import logging
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional
import dashscope
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.data_loader import DataLoader
from utils.image_utils import sample_viewpoint_images
from utils.h5_utils import (
    save_text_embeddings_h5,
    save_multimodal_text_embeddings_h5,
    load_multimodal_text_embeddings_h5,
    load_text_embeddings_h5
)

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class QwenEmbedder:
    """Generate Qwen embeddings for text and images using online API."""
    
    def __init__(self):
        self.embed_dim = 768 
        self.data_loader = DataLoader()
        # DashScope API key is set via config which sets env var or used directly
        # dashscope.api_key = config.DASHSCOPE_API_KEY # If needed globally
        
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embed a single text string using Qwen API.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding as numpy array or None if failed
        """
        # Add small delay to avoid rate limiting
        time.sleep(np.random.rand())

        try:
            input_data = [{'text': text}]
            resp = dashscope.MultiModalEmbedding.call(
                api_key=config.DASHSCOPE_API_KEY,
                model=config.QWEN_EMBEDDING_MODEL,
                input=input_data
            )
            
            if resp.status_code == 200 and resp.output:
                embeddings = resp.output.get('embeddings', [])
                if embeddings:
                    return np.array(embeddings[0]['embedding'], dtype=np.float32)
            else:
                logger.warning(f"Failed to embed text (status {resp.status_code}): {resp.message}")
                return np.zeros(self.embed_dim, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Exception embedding text: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    def embed_multi_images(self, image_paths: List[Path]) -> Optional[np.ndarray]:
        """
        Embed multiple images as a single embedding using Qwen API.
        
        Args:
            image_paths: List of image paths (up to 8)
        
        Returns:
            Single embedding representing all images or None if failed
        """
        # Add small delay to avoid rate limiting
        time.sleep(np.random.rand())

        if not image_paths:
            return None
        
        try:
            # Qwen supports up to 8 images
            image_paths = image_paths[:8]
            
            # Prepare input with base64 encoded images
            input_data = []
            for path in image_paths:
                try:
                    with open(path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    # Assuming PNG format as per utils/image_utils.py
                    image_data = f"data:image/png;base64,{base64_image}"
                    input_data.append({'image': image_data})
                except Exception as e:
                    logger.error(f"Failed to encode image {path}: {e}")
            
            if not input_data:
                return None

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
            else:
                logger.warning(f"Failed to embed images (status {resp.status_code}): {resp.message}")
                return np.zeros(self.embed_dim, dtype=np.float32)
        except Exception as e:
            logger.error(f"Exception embedding images: {e}")
            
        return np.zeros(self.embed_dim, dtype=np.float32)

    def generate_text_embeddings_parallel(
        self,
        texts: Dict[str, str],
        prefix: str,
        max_workers: int = 4        # to avoid rate limit
    ) -> Dict[str, np.ndarray]:
        """
        Generate text embeddings using online API with multithreading.
        
        Args:
            texts: Dictionary of text_id -> text
            prefix: Prefix for logging (e.g., 'en_text', 'cn_text')
            max_workers: Number of threads
        
        Returns:
            Dictionary of text_id -> embedding
        """
        logger.info(f"Generating {prefix} embeddings for {len(texts)} texts using {max_workers} threads")
        
        embeddings = {}
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each text
            future_to_id = {
                executor.submit(self.embed_text, text): text_id 
                for text_id, text in texts.items()
            }
            
            for future in tqdm(as_completed(future_to_id), total=len(texts), desc=f"Processing {prefix}"):
                text_id = future_to_id[future]
                try:
                    embedding = future.result()
                    if embedding is not None:
                        embeddings[text_id] = embedding
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing text {text_id}: {e}")
                    failed_count += 1
        
        logger.info(f"Generated {len(embeddings)} {prefix} embeddings. Failed: {failed_count}")
        return embeddings
    
    def generate_image_embeddings_parallel(
        self, 
        asset_ids: List[str],
        max_workers: int = 8
    ) -> Dict[str, np.ndarray]:
        """
        Generate multi-image embeddings for all assets using multithreading.
        
        Args:
            asset_ids: List of asset IDs
            max_workers: Number of threads (lower than text due to image upload size)
        
        Returns:
            Dictionary of asset_id -> multi-image embedding
        """
        logger.info(f"Generating multi-image embeddings for {len(asset_ids)} assets using {max_workers} threads")
        
        image_embeddings = {}
        failed_count = 0
        skipped_count = 0

        def process_asset(asset_id):
            # Sample 8 viewpoint images uniformly
            image_paths = sample_viewpoint_images(
                asset_id,
                num_samples=config.QWEN_NUM_IMAGES
            )
            
            if not image_paths:
                return asset_id, None, "no_images"
            
            # Generate multi-image embedding
            embedding = self.embed_multi_images(image_paths)
            if embedding is not None:
                return asset_id, embedding, "success"
            else:
                return asset_id, None, "failed"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_asset = {
                executor.submit(process_asset, asset_id): asset_id 
                for asset_id in asset_ids
            }
            
            for future in tqdm(as_completed(future_to_asset), total=len(asset_ids), desc="Processing images"):
                try:
                    asset_id, embedding, status = future.result()
                    if status == "success":
                        image_embeddings[asset_id] = embedding
                    elif status == "no_images":
                        skipped_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing asset images: {e}")
                    failed_count += 1
        
        logger.info(f"Generated image embeddings for {len(image_embeddings)} assets")
        logger.info(f"Skipped {skipped_count} (no images), Failed {failed_count}")
        
        return image_embeddings
    
    def save_embeddings(
        self,
        text_en_embeddings: Optional[Dict[str, np.ndarray]] = None,
        text_cn_embeddings: Optional[Dict[str, np.ndarray]] = None,
        image_embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        """Save embeddings to disk in HDF5 format."""
        logger.info("Saving embeddings to HDF5 files...")
        
        if text_cn_embeddings is not None and text_en_embeddings is not None:
            # Save text embeddings (English and Chinese in one file)
            save_multimodal_text_embeddings_h5(
                config.QWEN_TEXT_EMBEDDINGS_FILE,
                text_en_embeddings,
                text_cn_embeddings
            )
            logger.info(f"Saved text embeddings to {config.QWEN_TEXT_EMBEDDINGS_FILE}")
        
        if image_embeddings is not None:
            # Save image embeddings
            save_text_embeddings_h5(
                config.QWEN_IMAGE_EMBEDDINGS_FILE,
                image_embeddings
            )
            logger.info(f"Saved image embeddings to {config.QWEN_IMAGE_EMBEDDINGS_FILE}")
    
    def load_existing_embeddings(self) -> tuple:
        """Load existing embeddings if available."""
        text_en = None
        text_cn = None
        images = None
        
        if config.QWEN_TEXT_EMBEDDINGS_FILE.exists():
            text_en, text_cn = load_multimodal_text_embeddings_h5(config.QWEN_TEXT_EMBEDDINGS_FILE)
            logger.info(f"Loaded {len(text_en)} English and {len(text_cn)} Chinese text embeddings")
        
        if config.QWEN_IMAGE_EMBEDDINGS_FILE.exists():
            images = load_text_embeddings_h5(config.QWEN_IMAGE_EMBEDDINGS_FILE)
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
        
        text_en_embeddings = text_cn_embeddings = None
        if False:
            # Limit if MAX_ASSETS is set
            if config.MAX_ASSETS:
                asset_ids = list(captions_en.keys())[:config.MAX_ASSETS]
                captions_en = {aid: captions_en[aid] for aid in asset_ids if aid in captions_en}
                captions_cn = {aid: captions_cn[aid] for aid in asset_ids if aid in captions_cn}
                logger.info(f"Limited to {len(asset_ids)} assets for processing")
            
            # Generate English text embeddings
            text_en_embeddings = self.generate_text_embeddings_parallel(captions_en, "en_text")
            
            # Generate Chinese text embeddings
            if captions_cn:
                text_cn_embeddings = self.generate_text_embeddings_parallel(captions_cn, "cn_text")
            else:
                logger.warning("No Chinese captions available, skipping Chinese text embeddings")
                text_cn_embeddings = {}
        
        # Generate multi-image embeddings ONLY for assets with gobjaverse images
        asset_ids_with_images = self.data_loader.get_asset_ids_with_images(max_assets=config.MAX_ASSETS)
        logger.info(f"Generating image embeddings for {len(asset_ids_with_images)} assets with images")
        
        image_embeddings = self.generate_image_embeddings_parallel(asset_ids_with_images)
        # image_embeddings = None
        
        # Save to disk
        self.save_embeddings(text_en_embeddings, text_cn_embeddings, image_embeddings)
        
        logger.info("\n✓ Qwen embedding generation complete!")
        # logger.info(f"  English text embeddings: {len(text_en_embeddings)}")
        # logger.info(f"  Chinese text embeddings: {len(text_cn_embeddings)}")
        logger.info(f"  Multi-image embeddings: {len(image_embeddings)}")


def main():
    """Main entry point."""
    try:
        config.validate_config()
        
        embedder = QwenEmbedder()
        embedder.generate_all(force_regenerate=True)
        
    except Exception as e:
        logger.error(f"Qwen embedding generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

