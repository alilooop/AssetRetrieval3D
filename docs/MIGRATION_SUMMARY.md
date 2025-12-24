# HDF5 Storage Migration Summary

## Overview

Successfully migrated all embedding storage from `.npy` memmap and pickle formats to HDF5 (`.h5`) format for efficient random access by objaverse ID and better scalability.

## Changes Made

### 1. Configuration Updates (`config.py`)

**Removed:**
- `SIGLIP_TEXT_EMBEDDINGS_FILE` (`.pkl`)
- `SIGLIP_IMAGE_EMBEDDINGS_FILE` (`.pkl`)
- `SIGLIP_TEXT_EMBEDDINGS_MEMMAP` (`.npy`)
- `SIGLIP_IMAGE_EMBEDDINGS_MEMMAP` (`.npy`)
- `SIGLIP_TEXT_METADATA_FILE` (`.json`)
- `SIGLIP_IMAGE_METADATA_FILE` (`.json`)
- `QWEN_TEXT_EN_EMBEDDINGS_FILE` (`.pkl`)
- `QWEN_TEXT_CN_EMBEDDINGS_FILE` (`.pkl`)
- `QWEN_IMAGE_EMBEDDINGS_FILE` (`.pkl`)

**Added:**
- `SIGLIP_TEXT_EMBEDDINGS_FILE` → `siglip_text_embeddings.h5`
- `SIGLIP_IMAGE_EMBEDDINGS_FILE` → `siglip_image_embeddings.h5`
- `QWEN_TEXT_EMBEDDINGS_FILE` → `qwen_text_embeddings.h5`
- `QWEN_IMAGE_EMBEDDINGS_FILE` → `qwen_image_embeddings.h5`

### 2. New Utility Module (`utils/h5_utils.py`)

Created comprehensive HDF5 utility module with:

**Classes:**
- `HDF5EmbeddingWriter` - Context manager for writing embeddings with compression

**Functions:**
- `save_text_embeddings_h5()` - Save simple text embeddings
- `save_image_embeddings_h5()` - Save image embeddings with viewpoint metadata
- `save_multimodal_text_embeddings_h5()` - Save English + Chinese text embeddings
- `load_text_embeddings_h5()` - Load text embeddings (all or by ID)
- `load_image_embeddings_h5()` - Load image embeddings with viewpoint structure
- `load_multimodal_text_embeddings_h5()` - Load English + Chinese embeddings
- `get_embedding_by_id()` - Query single embedding by objaverse ID
- `list_asset_ids_h5()` - List all asset IDs in file
- `get_h5_info()` - Get file metadata and structure info

**Features:**
- Gzip compression (level 4) for storage efficiency
- Chunked datasets for better I/O performance
- Support for querying by objaverse ID without loading full file
- Memory-mapped lazy loading for large datasets

### 3. SigLIP Embedding Script (`scripts/02_embed_siglip.py`)

**Changes:**
- Removed `numpy.lib.format` dependency
- Added `h5py` for direct HDF5 writing
- Refactored `embed_texts_to_memmap()` → `embed_texts()` (returns list)
- Refactored `embed_images_to_memmap()` → `embed_images()` (returns dict)
- Changed distributed strategy:
  - Each rank saves embeddings to temporary `.npz` files
  - Rank 0 merges all temporary files **directly into HDF5 incrementally**
  - **Memory efficient**: Processes one rank's data at a time, never loads all embeddings into memory
  - Automatic cleanup of temporary files
- Removed manual memmap allocation and management
- Simplified metadata handling (now embedded in HDF5)

**HDF5 Structure - Text:**
```
siglip_text_embeddings.h5
├── embeddings [n_assets, embedding_dim]
├── asset_ids [n_assets]
└── attrs: total_items, embedding_dim
```

**HDF5 Structure - Images:**
```
siglip_image_embeddings.h5
├── embeddings [total_viewpoints, embedding_dim]
├── metadata [n_assets] (compound: asset_id, start_idx, count)
└── attrs: total_viewpoints, total_assets, embedding_dim
```

### 4. Qwen Embedding Script (`scripts/03_embed_qwen.py`)

**Changes:**
- Removed `pickle` dependency
- Added `h5_utils` imports
- Updated `save_embeddings()` to use HDF5 functions
- Updated `load_existing_embeddings()` to read from HDF5
- English and Chinese embeddings now stored in single file with groups

**HDF5 Structure - Text:**
```
qwen_text_embeddings.h5
├── english/
│   ├── embeddings [n_assets, embedding_dim]
│   ├── asset_ids [n_assets]
│   └── attrs: total_items, embedding_dim
├── chinese/
│   ├── embeddings [n_assets, embedding_dim]
│   ├── asset_ids [n_assets]
│   └── attrs: total_items, embedding_dim
└── attrs: total_assets, embedding_dim
```

**HDF5 Structure - Images:**
```
qwen_image_embeddings.h5
├── embeddings [n_assets, embedding_dim]
├── asset_ids [n_assets]
└── attrs: total_items, embedding_dim
```

### 5. Database Population Script (`scripts/04_populate_database.py`)

**Changes:**
- Removed `pickle` dependency
- Added `h5_utils` imports
- Updated `populate_siglip_database()`:
  - Use `load_text_embeddings_h5()` for text
  - Use `load_image_embeddings_h5()` for images
- Updated `populate_qwen_database()`:
  - Use `load_multimodal_text_embeddings_h5()` for text
  - Use `load_text_embeddings_h5()` for images
- Updated file existence checks to use new HDF5 paths
- Database insertion logic remains unchanged (still uses dicts)

### 6. Dependencies (`requirements.txt`)

**Added:**
- `h5py>=3.8.0`

## Benefits

### 1. **Easy Querying by Objaverse ID**
- Asset IDs stored directly in HDF5 files
- Can query single or multiple embeddings without loading entire file
- No need for separate metadata JSON files

### 2. **Scalability for 2M+ Assets**
- Memory-mapped lazy loading
- Only loads requested data into memory
- Efficient for large-scale datasets

### 3. **Better Storage Efficiency**
- Gzip compression reduces file size
- Single file per embedding type (no separate metadata)
- Chunked storage optimizes I/O performance

### 4. **Unified Format**
- All embeddings use same HDF5 format
- Consistent API across SigLIP and Qwen
- Easier to maintain and extend

### 5. **Distributed Processing Support**
- Rank-specific temporary files during generation
- Clean merge process on rank 0
- **Incremental HDF5 writing**: Processes one rank at a time, never loads all embeddings
- Automatic cleanup of temporary files

### 6. **Memory Efficiency**
- **Critical for 2M+ assets**: Merging process never holds all embeddings in memory
- Two-pass approach: First collect metadata, then write data incrementally
- Each rank's data is written directly to HDF5 and immediately released from memory
- Can handle datasets of any size without OOM errors

## Migration Notes

- **No data conversion needed** - The `outputs/embeddings` directory doesn't exist yet, so this is a clean migration
- **Backward compatibility** - Old format completely replaced (full migration as requested)
- **Database schema unchanged** - Database tables and insertion logic remain the same

## Usage Examples

### Query Single Embedding
```python
from utils.h5_utils import get_embedding_by_id

# Get English text embedding
embedding = get_embedding_by_id(
    config.QWEN_TEXT_EMBEDDINGS_FILE,
    "some_asset_id",
    group_name="english"
)
```

### Load Specific Assets
```python
from utils.h5_utils import load_text_embeddings_h5

# Load embeddings for specific assets only
asset_ids = ["asset1", "asset2", "asset3"]
embeddings = load_text_embeddings_h5(
    config.SIGLIP_TEXT_EMBEDDINGS_FILE,
    asset_ids=asset_ids
)
```

### List All Assets
```python
from utils.h5_utils import list_asset_ids_h5

# Get all asset IDs in file
all_ids = list_asset_ids_h5(config.SIGLIP_TEXT_EMBEDDINGS_FILE)
```

### Get File Info
```python
from utils.h5_utils import get_h5_info

# Get metadata about HDF5 file
info = get_h5_info(config.SIGLIP_TEXT_EMBEDDINGS_FILE)
print(f"Total items: {info['file_attrs']['total_items']}")
print(f"Embedding dim: {info['file_attrs']['embedding_dim']}")
```

## Testing Recommendations

1. **Small-scale test**: Run with `MAX_ASSETS=100` to verify:
   - Distributed SigLIP embedding generation
   - Qwen embedding generation
   - HDF5 file creation and structure
   - Database population

2. **Query performance**: Test random access by ID:
   - Single embedding queries
   - Batch queries
   - Full file loading

3. **Data integrity**: Verify:
   - All asset IDs are preserved
   - Embedding values are correct
   - No data loss during merge

4. **Storage efficiency**: Compare:
   - File sizes with/without compression
   - Load times for different query patterns
   - Memory usage during operations

## Next Steps

1. Run embedding generation scripts with test data
2. Verify HDF5 file structure using `h5py` or `h5dump`
3. Test database population
4. Benchmark query performance
5. Scale up to full dataset (2M+ assets)

