"""
Data loading utilities for captions and ID mappings.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and accessing caption and mapping data."""
    
    def __init__(self):
        self.captions_en: Optional[Dict[str, str]] = None
        self.captions_cn: Optional[Dict[str, str]] = None
        self.gobjaverse_to_objaverse: Optional[Dict[str, str]] = None
        self.asset_ids: Optional[List[str]] = None
    
    def load_english_captions(self) -> Dict[str, str]:
        """Load English captions from JSON file."""
        if self.captions_en is None:
            logger.info(f"Loading English captions from {config.CAPTIONS_FILE}")
            with open(config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                self.captions_en = json.load(f)
            logger.info(f"Loaded {len(self.captions_en)} English captions")
        return self.captions_en
    
    def load_chinese_captions(self) -> Dict[str, str]:
        """Load Chinese captions from JSON file."""
        if self.captions_cn is None:
            if not config.CAPTIONS_CN_FILE.exists():
                logger.warning(f"Chinese captions file not found: {config.CAPTIONS_CN_FILE}")
                return {}
            
            logger.info(f"Loading Chinese captions from {config.CAPTIONS_CN_FILE}")
            with open(config.CAPTIONS_CN_FILE, 'r', encoding='utf-8') as f:
                self.captions_cn = json.load(f)
            logger.info(f"Loaded {len(self.captions_cn)} Chinese captions")
        return self.captions_cn
    
    def load_index_mapping(self) -> Dict[str, str]:
        """Load gobjaverse ID to objaverse ID mapping."""
        if self.gobjaverse_to_objaverse is None:
            logger.info(f"Loading index mapping from {config.INDEX_MAPPING_FILE}")
            with open(config.INDEX_MAPPING_FILE, 'r', encoding='utf-8') as f:
                self.gobjaverse_to_objaverse = json.load(f)
            logger.info(f"Loaded {len(self.gobjaverse_to_objaverse)} ID mappings")
        return self.gobjaverse_to_objaverse
    
    def get_asset_ids(self, max_assets: Optional[int] = None) -> List[str]:
        """
        Get list of asset IDs to process (all assets with captions).
        
        Args:
            max_assets: Maximum number of assets to return (for debugging)
        
        Returns:
            List of asset IDs (all assets with captions)
        """
        if self.asset_ids is None:
            captions = self.load_english_captions()
            self.asset_ids = sorted(captions.keys())
        
        if max_assets is not None:
            return self.asset_ids[:max_assets]
        return self.asset_ids
    
    def get_asset_ids_with_images(self, max_assets: Optional[int] = None) -> List[str]:
        """
        Get list of asset IDs that have gobjaverse images.
        
        Args:
            max_assets: Maximum number of assets to return (for debugging)
        
        Returns:
            List of asset IDs that exist in gobjaverse (have images)
        """
        captions = self.load_english_captions()
        gobjaverse_mapping = self.load_index_mapping()
        
        # Filter to only assets that exist in gobjaverse
        asset_ids_with_images = sorted([
            asset_id for asset_id in captions.keys()
            if asset_id in gobjaverse_mapping
        ])
        
        logger.info(f"Found {len(asset_ids_with_images)} assets with gobjaverse images "
                   f"(out of {len(captions)} total captions)")
        
        if max_assets is not None:
            return asset_ids_with_images[:max_assets]
        return asset_ids_with_images
    
    def get_objaverse_id(self, gobjaverse_id: str) -> Optional[str]:
        """
        Convert gobjaverse ID to objaverse ID.
        
        Args:
            gobjaverse_id: Asset ID in format like "0/10002"
        
        Returns:
            Objaverse ID string or None if not found
        """
        mapping = self.load_index_mapping()
        objaverse_path = mapping.get(gobjaverse_id)
        if objaverse_path:
            # Extract objaverse ID from path like "000-001/bc2fe4e89ff44b8a9b54614346f7ad29.glb"
            return Path(objaverse_path).stem
        return None
    
    def save_chinese_captions(self, captions_cn: Dict[str, str]):
        """
        Save Chinese captions to JSON file.
        
        Args:
            captions_cn: Dictionary mapping asset IDs to Chinese captions
        """
        logger.info(f"Saving {len(captions_cn)} Chinese captions to {config.CAPTIONS_CN_FILE}")
        with open(config.CAPTIONS_CN_FILE, 'w', encoding='utf-8') as f:
            json.dump(captions_cn, f, ensure_ascii=False, indent=2)
        self.captions_cn = captions_cn
        logger.info("Chinese captions saved successfully")


def load_captions(language: str = 'english') -> Dict[str, str]:
    """
    Convenience function to load captions.
    
    Args:
        language: 'english' or 'chinese'
    
    Returns:
        Dictionary mapping asset IDs to captions
    """
    loader = DataLoader()
    if language.lower() == 'english':
        return loader.load_english_captions()
    elif language.lower() == 'chinese':
        return loader.load_chinese_captions()
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_asset_list(max_assets: Optional[int] = None) -> List[str]:
    """
    Get list of asset IDs to process (all assets with captions).
    
    Args:
        max_assets: Maximum number of assets (for debugging)
    
    Returns:
        List of asset IDs
    """
    loader = DataLoader()
    return loader.get_asset_ids(max_assets=max_assets)


def get_asset_list_with_images(max_assets: Optional[int] = None) -> List[str]:
    """
    Get list of asset IDs that have gobjaverse images.
    
    Args:
        max_assets: Maximum number of assets (for debugging)
    
    Returns:
        List of asset IDs with images
    """
    loader = DataLoader()
    return loader.get_asset_ids_with_images(max_assets=max_assets)


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
    # Test loading English captions
    captions_en = loader.load_english_captions()
    print(f"English captions: {len(captions_en)}")
    print(f"Sample: {list(captions_en.items())[:3]}")
    
    # Test loading index mapping
    mapping = loader.load_index_mapping()
    print(f"\nIndex mappings: {len(mapping)}")
    print(f"Sample: {list(mapping.items())[:3]}")
    
    # Test getting asset IDs
    asset_ids = loader.get_asset_ids(max_assets=10)
    print(f"\nFirst 10 asset IDs: {asset_ids}")
    
    # Test objaverse ID conversion
    for asset_id in asset_ids[:3]:
        objaverse_id = loader.get_objaverse_id(asset_id)
        print(f"  {asset_id} -> {objaverse_id}")

