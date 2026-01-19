"""
Utility functions for HDF5-based embedding storage.

This module provides helper functions for reading and writing embeddings
in HDF5 format, enabling efficient random access by objaverse ID.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class HDF5EmbeddingWriter:
    """Context manager for writing embeddings to HDF5 in distributed settings."""
    
    def __init__(
        self,
        filepath: Path,
        embedding_dim: int,
        total_items: int,
        compression: str = "gzip",
        compression_level: int = 4,
        chunk_size: int = 1000
    ):
        """
        Initialize HDF5 writer.
        
        Args:
            filepath: Path to HDF5 file
            embedding_dim: Dimension of embeddings
            total_items: Total number of items to store
            compression: Compression algorithm (gzip, lzf, None)
            compression_level: Compression level (1-9 for gzip)
            chunk_size: Chunk size for HDF5 dataset
        """
        self.filepath = filepath
        self.embedding_dim = embedding_dim
        self.total_items = total_items
        self.compression = compression
        self.compression_level = compression_level
        self.chunk_size = chunk_size
        self.file = None
    
    def __enter__(self):
        """Open HDF5 file for writing."""
        self.file = h5py.File(self.filepath, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file:
            self.file.close()
    
    def create_embedding_dataset(
        self,
        name: str = "embeddings"
    ) -> h5py.Dataset:
        """
        Create embedding dataset.
        
        Args:
            name: Name of the dataset
        
        Returns:
            HDF5 dataset
        """
        return self.file.create_dataset(
            name,
            shape=(self.total_items, self.embedding_dim),
            dtype='float32',
            chunks=(min(self.chunk_size, self.total_items), self.embedding_dim),
            compression=self.compression,
            compression_opts=self.compression_level if self.compression == "gzip" else None
        )
    
    def create_asset_id_dataset(
        self,
        name: str = "asset_ids"
    ) -> h5py.Dataset:
        """
        Create asset ID dataset.
        
        Args:
            name: Name of the dataset
        
        Returns:
            HDF5 dataset
        """
        return self.file.create_dataset(
            name,
            shape=(self.total_items,),
            dtype=h5py.string_dtype(encoding='utf-8'),
            chunks=(min(self.chunk_size, self.total_items),),
            compression=self.compression,
            compression_opts=self.compression_level if self.compression == "gzip" else None
        )
    
    def create_metadata_dataset(
        self,
        name: str,
        dtype: np.dtype,
        shape: Optional[Tuple] = None
    ) -> h5py.Dataset:
        """
        Create custom metadata dataset.
        
        Args:
            name: Name of the dataset
            dtype: NumPy dtype for the dataset
            shape: Shape of the dataset (if None, uses (total_items,))
        
        Returns:
            HDF5 dataset
        """
        if shape is None:
            shape = (self.total_items,)
        
        return self.file.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=True,
            compression=self.compression,
            compression_opts=self.compression_level if self.compression == "gzip" else None
        )


def save_text_embeddings_h5(
    filepath: Path,
    embeddings_dict: Dict[str, np.ndarray],
    compression: str = "gzip",
    compression_level: int = 4
):
    """
    Save text embeddings to HDF5 file.
    
    Args:
        filepath: Path to output HDF5 file
        embeddings_dict: Dictionary mapping asset_id -> embedding
        compression: Compression algorithm
        compression_level: Compression level
    """
    if not embeddings_dict:
        raise ValueError("No embeddings to save")
    
    # Get dimensions
    asset_ids = sorted(embeddings_dict.keys())
    first_embedding = embeddings_dict[asset_ids[0]]
    embedding_dim = first_embedding.shape[0]
    total_items = len(asset_ids)
    
    logger.info(f"Saving {total_items} text embeddings to {filepath}")
    
    with HDF5EmbeddingWriter(
        filepath,
        embedding_dim,
        total_items,
        compression,
        compression_level
    ) as writer:
        # Create datasets
        embeddings_ds = writer.create_embedding_dataset()
        asset_ids_ds = writer.create_asset_id_dataset()
        
        # Write data
        for i, asset_id in enumerate(asset_ids):
            embeddings_ds[i] = embeddings_dict[asset_id]
            asset_ids_ds[i] = asset_id
        
        # Add metadata attributes
        writer.file.attrs['total_items'] = total_items
        writer.file.attrs['embedding_dim'] = embedding_dim
    
    logger.info(f"Successfully saved {total_items} embeddings")


def save_image_embeddings_h5(
    filepath: Path,
    embeddings_dict: Dict[str, List[np.ndarray]],
    compression: str = "gzip",
    compression_level: int = 4
):
    """
    Save image embeddings to HDF5 file (multiple viewpoints per asset).
    
    Args:
        filepath: Path to output HDF5 file
        embeddings_dict: Dictionary mapping asset_id -> list of embeddings
        compression: Compression algorithm
        compression_level: Compression level
    """
    if not embeddings_dict:
        raise ValueError("No embeddings to save")
    
    # Calculate total viewpoints and gather metadata
    asset_ids = sorted(embeddings_dict.keys())
    metadata_list = []
    total_viewpoints = 0
    
    for asset_id in asset_ids:
        embeddings_list = embeddings_dict[asset_id]
        count = len(embeddings_list)
        metadata_list.append((asset_id, total_viewpoints, count))
        total_viewpoints += count
    
    # Get embedding dimension
    first_embedding = embeddings_dict[asset_ids[0]][0]
    embedding_dim = first_embedding.shape[0]
    
    logger.info(f"Saving {total_viewpoints} image embeddings from {len(asset_ids)} assets to {filepath}")
    
    with HDF5EmbeddingWriter(
        filepath,
        embedding_dim,
        total_viewpoints,
        compression,
        compression_level
    ) as writer:
        # Create embedding dataset
        embeddings_ds = writer.create_embedding_dataset()
        
        # Create metadata dataset with compound dtype
        metadata_dtype = np.dtype([
            ('asset_id', h5py.string_dtype(encoding='utf-8')),
            ('start_idx', np.int64),
            ('count', np.int32)
        ])
        
        metadata_ds = writer.file.create_dataset(
            'metadata',
            shape=(len(asset_ids),),
            dtype=metadata_dtype,
            compression=compression,
            compression_opts=compression_level if compression == "gzip" else None
        )
        
        # Write embeddings and metadata
        for i, (asset_id, start_idx, count) in enumerate(metadata_list):
            embeddings_list = embeddings_dict[asset_id]
            for j, embedding in enumerate(embeddings_list):
                embeddings_ds[start_idx + j] = embedding
            
            metadata_ds[i] = (asset_id, start_idx, count)
        
        # Add file-level attributes
        writer.file.attrs['total_viewpoints'] = total_viewpoints
        writer.file.attrs['total_assets'] = len(asset_ids)
        writer.file.attrs['embedding_dim'] = embedding_dim
    
    logger.info(f"Successfully saved {total_viewpoints} viewpoint embeddings from {len(asset_ids)} assets")


def save_multimodal_text_embeddings_h5(
    filepath: Path,
    en_embeddings_dict: Dict[str, np.ndarray],
    cn_embeddings_dict: Dict[str, np.ndarray],
    compression: str = "gzip",
    compression_level: int = 4
):
    """
    Save multimodal (English + Chinese) text embeddings to HDF5 file.
    
    Args:
        filepath: Path to output HDF5 file
        en_embeddings_dict: Dictionary mapping asset_id -> English embedding
        cn_embeddings_dict: Dictionary mapping asset_id -> Chinese embedding
        compression: Compression algorithm
        compression_level: Compression level
    """
    # Get all unique asset IDs
    all_asset_ids = sorted(set(en_embeddings_dict.keys()) | set(cn_embeddings_dict.keys()))
    
    if not all_asset_ids:
        raise ValueError("No embeddings to save")
    
    # Get embedding dimension from first available embedding
    sample_embedding = (
        en_embeddings_dict[next(iter(en_embeddings_dict))] if en_embeddings_dict
        else cn_embeddings_dict[next(iter(cn_embeddings_dict))]
    )
    embedding_dim = sample_embedding.shape[0]
    total_items = len(all_asset_ids)
    
    logger.info(f"Saving {total_items} multimodal text embeddings to {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Create groups for English and Chinese
        en_group = f.create_group('english')
        cn_group = f.create_group('chinese')
        
        # Helper function to save to a group
        def save_to_group(group, embeddings_dict, asset_ids):
            embeddings_ds = group.create_dataset(
                'embeddings',
                shape=(len(asset_ids), embedding_dim),
                dtype='float32',
                chunks=(min(1000, len(asset_ids)), embedding_dim),
                compression=compression,
                compression_opts=compression_level if compression == "gzip" else None
            )
            
            asset_ids_ds = group.create_dataset(
                'asset_ids',
                shape=(len(asset_ids),),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(min(1000, len(asset_ids)),),
                compression=compression,
                compression_opts=compression_level if compression == "gzip" else None
            )
            
            # Write data
            for i, asset_id in enumerate(asset_ids):
                if asset_id in embeddings_dict:
                    embeddings_ds[i] = embeddings_dict[asset_id]
                    asset_ids_ds[i] = asset_id
            
            group.attrs['total_items'] = len(asset_ids)
            group.attrs['embedding_dim'] = embedding_dim
        
        # Save English embeddings
        en_asset_ids = sorted(en_embeddings_dict.keys())
        save_to_group(en_group, en_embeddings_dict, en_asset_ids)
        
        # Save Chinese embeddings
        cn_asset_ids = sorted(cn_embeddings_dict.keys())
        save_to_group(cn_group, cn_embeddings_dict, cn_asset_ids)
        
        # Add file-level attributes
        f.attrs['total_assets'] = total_items
        f.attrs['embedding_dim'] = embedding_dim
    
    logger.info(f"Successfully saved EN: {len(en_asset_ids)}, CN: {len(cn_asset_ids)} embeddings")


def load_text_embeddings_h5(
    filepath: Path,
    asset_ids: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load text embeddings from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        asset_ids: Optional list of specific asset IDs to load (None = load all)
    
    Returns:
        Dictionary mapping asset_id -> embedding
    """
    embeddings_dict = {}
    
    with h5py.File(filepath, 'r') as f:
        all_asset_ids = f['asset_ids'][:]
        
        # Decode byte strings to regular strings if needed
        if isinstance(all_asset_ids[0], bytes):
            all_asset_ids = [aid.decode('utf-8') for aid in all_asset_ids]
        
        if asset_ids is None:
            # Load all embeddings
            all_embeddings = f['embeddings'][:]
            embeddings_dict = dict(zip(all_asset_ids, all_embeddings))
        else:
            # Load specific embeddings
            aid_to_idx = {aid: i for i, aid in enumerate(all_asset_ids)}
            target_indices = []
            valid_aids = []
            
            for aid in asset_ids:
                if aid in aid_to_idx:
                    target_indices.append(aid_to_idx[aid])
                    valid_aids.append(aid)
            
            if target_indices:
                # Sort indices for better HDF5 access performance
                sorted_idx_and_aids = sorted(zip(target_indices, valid_aids))
                sorted_indices = [idx for idx, aid in sorted_idx_and_aids]
                sorted_aids = [aid for idx, aid in sorted_idx_and_aids]
                
                # Batch read (h5py supports fancy indexing)
                batch_embeddings = f['embeddings'][sorted_indices]
                embeddings_dict = dict(zip(sorted_aids, batch_embeddings))
    
    return embeddings_dict


def load_image_embeddings_h5(
    filepath: Path,
    asset_ids: Optional[List[str]] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Load image embeddings from HDF5 file (multiple viewpoints per asset).
    
    Args:
        filepath: Path to HDF5 file
        asset_ids: Optional list of specific asset IDs to load (None = load all)
    
    Returns:
        Dictionary mapping asset_id -> list of embeddings
    """
    embeddings_dict = {}
    
    with h5py.File(filepath, 'r') as f:
        metadata = f['metadata'][:]
        embeddings_ds = f['embeddings']
        
        # Build lookup if we need specific asset IDs
        if asset_ids is not None:
            asset_ids_set = set(asset_ids)
        
        for record in metadata:
            asset_id = record['asset_id']
            if isinstance(asset_id, bytes):
                asset_id = asset_id.decode('utf-8')
            
            # Skip if not in requested list
            if asset_ids is not None and asset_id not in asset_ids_set:
                continue
            
            start_idx = int(record['start_idx'])
            count = int(record['count'])
            
            # Load all viewpoint embeddings for this asset using slicing for performance
            embeddings_dict[asset_id] = embeddings_ds[start_idx : start_idx + count]
    
    return embeddings_dict


def yield_text_embeddings_h5(
    filepath: Path,
    batch_size: int = 1000
):
    """
    Yield text embeddings from HDF5 file in batches to save memory.
    
    Args:
        filepath: Path to HDF5 file
        batch_size: Number of items per yield
    
    Yields:
        Tuple of (list of asset_ids, numpy array of embeddings)
    """
    with h5py.File(filepath, 'r') as f:
        all_asset_ids = f['asset_ids']
        all_embeddings = f['embeddings']
        total_items = len(all_asset_ids)
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            
            batch_ids = all_asset_ids[i:end_idx]
            if isinstance(batch_ids[0], bytes):
                batch_ids = [aid.decode('utf-8') for aid in batch_ids]
            
            batch_embeddings = all_embeddings[i:end_idx]
            yield batch_ids, batch_embeddings


def yield_image_embeddings_h5(
    filepath: Path,
    batch_size: int = 100
):
    """
    Yield image embeddings from HDF5 file asset by asset.
    
    Args:
        filepath: Path to HDF5 file
        batch_size: Number of assets per yield
    
    Yields:
        Dictionary mapping asset_id -> list of embeddings for a batch of assets
    """
    with h5py.File(filepath, 'r') as f:
        metadata = f['metadata']
        embeddings_ds = f['embeddings']
        total_assets = len(metadata)
        
        for i in range(0, total_assets, batch_size):
            end_idx = min(i + batch_size, total_assets)
            batch_dict = {}
            
            for j in range(i, end_idx):
                record = metadata[j]
                asset_id = record['asset_id']
                if isinstance(asset_id, bytes):
                    asset_id = asset_id.decode('utf-8')
                
                start_idx = int(record['start_idx'])
                count = int(record['count'])
                
                batch_dict[asset_id] = embeddings_ds[start_idx : start_idx + count]
            
            yield batch_dict


def load_multimodal_text_embeddings_h5(
    filepath: Path,
    asset_ids: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load multimodal (English + Chinese) text embeddings from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        asset_ids: Optional list of specific asset IDs to load (None = load all)
    
    Returns:
        Tuple of (en_embeddings_dict, cn_embeddings_dict)
    """
    en_embeddings_dict = {}
    cn_embeddings_dict = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load English embeddings
        if 'english' in f:
            en_group = f['english']
            en_asset_ids = en_group['asset_ids'][:]
            if isinstance(en_asset_ids[0], bytes):
                en_asset_ids = [aid.decode('utf-8') for aid in en_asset_ids]
            
            if asset_ids is None:
                en_embeddings = en_group['embeddings'][:]
                en_embeddings_dict = dict(zip(en_asset_ids, en_embeddings))
            else:
                aid_to_idx = {aid: i for i, aid in enumerate(en_asset_ids)}
                for asset_id in asset_ids:
                    if asset_id in aid_to_idx:
                        idx = aid_to_idx[asset_id]
                        en_embeddings_dict[asset_id] = en_group['embeddings'][idx]
        
        # Load Chinese embeddings
        if 'chinese' in f:
            cn_group = f['chinese']
            cn_asset_ids = cn_group['asset_ids'][:]
            if isinstance(cn_asset_ids[0], bytes):
                cn_asset_ids = [aid.decode('utf-8') for aid in cn_asset_ids]
            
            if asset_ids is None:
                cn_embeddings = cn_group['embeddings'][:]
                cn_embeddings_dict = dict(zip(cn_asset_ids, cn_embeddings))
            else:
                aid_to_idx = {aid: i for i, aid in enumerate(cn_asset_ids)}
                for asset_id in asset_ids:
                    if asset_id in aid_to_idx:
                        idx = aid_to_idx[asset_id]
                        cn_embeddings_dict[asset_id] = cn_group['embeddings'][idx]
    
    return en_embeddings_dict, cn_embeddings_dict


def get_embedding_by_id(
    filepath: Path,
    asset_id: str,
    group_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Get a single embedding by asset ID.
    
    Args:
        filepath: Path to HDF5 file
        asset_id: Asset ID to query
        group_name: Optional group name for multimodal files ('english' or 'chinese')
    
    Returns:
        Embedding array or None if not found
    """
    with h5py.File(filepath, 'r') as f:
        # Navigate to the appropriate location
        if group_name:
            if group_name not in f:
                return None
            group = f[group_name]
            asset_ids = group['asset_ids'][:]
            embeddings = group['embeddings']
        else:
            asset_ids = f['asset_ids'][:]
            embeddings = f['embeddings']
        
        # Decode if needed
        if isinstance(asset_ids[0], bytes):
            asset_ids = [aid.decode('utf-8') for aid in asset_ids]
        
        # Find the index
        try:
            idx = list(asset_ids).index(asset_id)
            return embeddings[idx]
        except ValueError:
            return None


def list_asset_ids_h5(
    filepath: Path,
    group_name: Optional[str] = None
) -> List[str]:
    """
    List all asset IDs in an HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        group_name: Optional group name for multimodal files
    
    Returns:
        List of asset IDs
    """
    with h5py.File(filepath, 'r') as f:
        if group_name:
            if group_name not in f:
                return []
            asset_ids = f[group_name]['asset_ids'][:]
        elif 'metadata' in f:
            # Image embeddings file with metadata
            metadata = f['metadata'][:]
            asset_ids = metadata['asset_id']
        else:
            # Simple text embeddings file
            asset_ids = f['asset_ids'][:]
        
        # Decode if needed
        if isinstance(asset_ids[0], bytes):
            asset_ids = [aid.decode('utf-8') for aid in asset_ids]
        else:
            asset_ids = list(asset_ids)
        
        return asset_ids


def yield_multimodal_text_embeddings_h5(
    filepath: Path,
    batch_size: int = 1000
):
    """
    Yield multimodal text embeddings from HDF5 file in batches.
    
    Args:
        filepath: Path to HDF5 file
        batch_size: Number of items per yield
    
    Yields:
        Tuple of (batch_ids, en_embeddings, cn_embeddings)
    """
    with h5py.File(filepath, 'r') as f:
        en_group = f.get('english')
        cn_group = f.get('chinese')
        
        # Get all unique asset IDs and map them
        en_ids = []
        if en_group:
            en_ids = en_group['asset_ids'][:]
            if isinstance(en_ids[0], bytes):
                en_ids = [aid.decode('utf-8') for aid in en_ids]
        
        cn_ids = []
        if cn_group:
            cn_ids = cn_group['asset_ids'][:]
            if isinstance(cn_ids[0], bytes):
                cn_ids = [aid.decode('utf-8') for aid in cn_ids]
        
        all_asset_ids = sorted(list(set(en_ids) | set(cn_ids)))
        total_items = len(all_asset_ids)
        
        en_id_to_idx = {aid: i for i, aid in enumerate(en_ids)}
        cn_id_to_idx = {aid: i for i, aid in enumerate(cn_ids)}
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            batch_ids = all_asset_ids[i:end_idx]
            
            en_batch = None
            if en_group:
                en_indices = [en_id_to_idx[aid] for aid in batch_ids if aid in en_id_to_idx]
                if en_indices:
                    # Sort for performance, then map back to batch_ids order
                    idx_map = {idx: pos for pos, idx in enumerate(en_indices)}
                    sorted_en_indices = sorted(en_indices)
                    sorted_en_embs = en_group['embeddings'][sorted_en_indices]
                    
                    en_batch = np.zeros((len(batch_ids), en_group['embeddings'].shape[1]), dtype='float32')
                    # This part is still a bit slow but better than total load
                    for j, aid in enumerate(batch_ids):
                        if aid in en_id_to_idx:
                            orig_idx = en_id_to_idx[aid]
                            # Find where this orig_idx went in the sorted array
                            # Actually, a simpler way:
                            pass
                    
                    # Simpler way to map back:
                    temp_map = {idx: emb for idx, emb in zip(sorted_en_indices, sorted_en_embs)}
                    for j, aid in enumerate(batch_ids):
                        if aid in en_id_to_idx:
                            en_batch[j] = temp_map[en_id_to_idx[aid]]
                        else:
                            en_batch[j] = 0 # Or some indicator
            
            cn_batch = None
            if cn_group:
                cn_indices = [cn_id_to_idx[aid] for aid in batch_ids if aid in cn_id_to_idx]
                if cn_indices:
                    sorted_cn_indices = sorted(cn_indices)
                    sorted_cn_embs = cn_group['embeddings'][sorted_cn_indices]
                    temp_map = {idx: emb for idx, emb in zip(sorted_cn_indices, sorted_cn_embs)}
                    
                    cn_batch = np.zeros((len(batch_ids), cn_group['embeddings'].shape[1]), dtype='float32')
                    for j, aid in enumerate(batch_ids):
                        if aid in cn_id_to_idx:
                            cn_batch[j] = temp_map[cn_id_to_idx[aid]]
                        else:
                            cn_batch[j] = 0
            
            yield batch_ids, en_batch, cn_batch
    info = {}
    
    with h5py.File(filepath, 'r') as f:
        info['file_attrs'] = dict(f.attrs)
        info['datasets'] = {}
        info['groups'] = {}
        
        def visit_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                info['datasets'][name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_mb': obj.size * obj.dtype.itemsize / (1024 * 1024)
                }
            elif isinstance(obj, h5py.Group):
                info['groups'][name] = dict(obj.attrs)
        
        f.visititems(visit_item)
    
    return info

