"""
Image loading and preprocessing utilities.
"""
import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np

import config

logger = logging.getLogger(__name__)


def get_asset_image_paths(gobjaverse_id: str, max_images: Optional[int] = None) -> List[Path]:
    """
    Get all image paths for a given asset.
    
    Args:
        gobjaverse_id: Asset ID in format like "0/10002"
        max_images: Maximum number of images to return
    
    Returns:
        List of paths to PNG images
    """
    asset_dir = config.get_asset_image_dir(gobjaverse_id)
    
    if not asset_dir.exists():
        logger.warning(f"Asset directory not found: {asset_dir}")
        return []
    
    # Collect all PNG images from viewpoint subdirectories
    image_paths = []
    
    # Each asset has subdirectories like 00000, 00005, 00010, etc.
    # Each subdirectory contains viewpoint images
    viewpoint_dirs = sorted([d for d in asset_dir.iterdir() if d.is_dir()])
    
    for viewpoint_dir in viewpoint_dirs:
        png_files = list(viewpoint_dir.glob("*.png"))
        if png_files:
            # Take the first PNG in each viewpoint directory
            image_paths.append(png_files[0])
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    return image_paths


def sample_viewpoint_images(gobjaverse_id: str, num_samples: int = 8) -> List[Path]:
    """
    Uniformly sample viewpoint images for an asset.
    
    Args:
        gobjaverse_id: Asset ID in format like "0/10002"
        num_samples: Number of images to sample
    
    Returns:
        List of uniformly sampled image paths
    """
    all_images = get_asset_image_paths(gobjaverse_id)
    
    if not all_images:
        return []
    
    # Uniformly sample indices
    if len(all_images) <= num_samples:
        return all_images
    
    indices = np.linspace(0, len(all_images) - 1, num_samples, dtype=int)
    return [all_images[i] for i in indices]


def load_image(image_path: Path) -> Optional[Image.Image]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image or None if failed
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def load_images(image_paths: List[Path]) -> List[Image.Image]:
    """
    Load multiple images from disk.
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        List of PIL Images (skips failed loads)
    """
    images = []
    for path in image_paths:
        img = load_image(path)
        if img is not None:
            images.append(img)
    return images


def get_image_url_from_path(image_path: Path) -> str:
    """
    Convert a local image path to a URL (for API calls that need URLs).
    For local files, this returns the file:// URL.
    
    Args:
        image_path: Path to image file
    
    Returns:
        File URL string
    """
    return image_path.as_uri()


def count_available_images(gobjaverse_id: str) -> int:
    """
    Count number of available images for an asset.
    
    Args:
        gobjaverse_id: Asset ID in format like "0/10002"
    
    Returns:
        Number of available images
    """
    return len(get_asset_image_paths(gobjaverse_id))


def validate_asset_images(gobjaverse_id: str, min_images: int = 1) -> bool:
    """
    Check if an asset has sufficient images.
    
    Args:
        gobjaverse_id: Asset ID in format like "0/10002"
        min_images: Minimum number of required images
    
    Returns:
        True if asset has enough images
    """
    return count_available_images(gobjaverse_id) >= min_images


if __name__ == "__main__":
    # Test image utilities
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample asset
    test_asset_id = "0/10005"
    
    print(f"Testing with asset: {test_asset_id}")
    
    # Get all images
    all_images = get_asset_image_paths(test_asset_id)
    print(f"Total images: {len(all_images)}")
    if all_images:
        print(f"Sample paths: {all_images[:3]}")
    
    # Sample images
    sampled = sample_viewpoint_images(test_asset_id, num_samples=8)
    print(f"\nSampled {len(sampled)} images:")
    for i, path in enumerate(sampled):
        print(f"  {i}: {path}")
    
    # Test loading
    if sampled:
        img = load_image(sampled[0])
        if img:
            print(f"\nLoaded image: {img.size}, mode: {img.mode}")
    
    # Test validation
    is_valid = validate_asset_images(test_asset_id, min_images=8)
    print(f"\nAsset has >= 8 images: {is_valid}")

