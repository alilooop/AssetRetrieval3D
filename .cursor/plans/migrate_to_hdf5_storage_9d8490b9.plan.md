---
name: Migrate to HDF5 Storage
overview: Migrate all embedding storage from .npy memmap and pickle formats to HDF5 (.h5) for efficient random access by objaverse ID and better scalability for 2M+ assets.
todos:
  - id: update-config
    content: Update config.py with HDF5 file paths and remove old .npy/.pkl paths
    status: completed
  - id: create-h5-utils
    content: Create utils/h5_utils.py with helper functions for HDF5 operations
    status: completed
    dependencies:
      - update-config
  - id: update-siglip-script
    content: Refactor 02_embed_siglip.py to use HDF5 storage with distributed support
    status: completed
    dependencies:
      - create-h5-utils
  - id: update-qwen-script
    content: Refactor 03_embed_qwen.py to use HDF5 storage instead of pickle
    status: completed
    dependencies:
      - create-h5-utils
  - id: update-db-populator
    content: Update 04_populate_database.py to read from HDF5 files
    status: completed
    dependencies:
      - update-siglip-script
      - update-qwen-script
  - id: update-requirements
    content: Add h5py dependencies to requirements.txt
    status: completed
---

# Migrate Embedding Storage to HDF5

## Problem Summary

The current implementation has three main issues:

1. **SigLIP**: Generates `.npy` files but database populator expects `.pkl` files (mismatch)
2. **Querying difficulty**: Can't easily query embeddings by objaverse ID without loading metadata
3. **Qwen pickle files**: Not scalable for large datasets (2M+) and require full file loading

## Solution: HDF5 Format

Migrate all embedding storage to HDF5 (`.h5`) format with objaverse ID indexing for:

- Fast random access by asset ID
- Memory-mapped lazy loading
- Better scalability for large datasets
- Single unified format across all embedding types

## Implementation Plan

### 1. Update Configuration ([config.py](config.py))

**Changes:**

- Replace `.pkl` and `.npy` file paths with `.h5` equivalents
- Remove separate metadata JSON file paths (metadata will be stored in HDF5)
- Keep distributed processing configuration intact

**New paths:**

```python
# SigLip embeddings (HDF5 format)
SIGLIP_TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "siglip_text_embeddings.h5"
SIGLIP_IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "siglip_image_embeddings.h5"

# Qwen embeddings (HDF5 format)
QWEN_TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_text_embeddings.h5"
QWEN_IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_image_embeddings.h5"
```

### 2. Update SigLIP Embedding Script ([scripts/02_embed_siglip.py](scripts/02_embed_siglip.py))

**HDF5 Structure for Text:**

```
siglip_text_embeddings.h5
├── embeddings (2D dataset: [n_assets, embedding_dim])
└── asset_ids (1D string dataset: [n_assets])
```

**HDF5 Structure for Images:**

```
siglip_image_embeddings.h5
├── embeddings (2D dataset: [total_viewpoints, embedding_dim])
└── metadata (compound dataset with fields: asset_id, start_idx, count)
```

**Changes:**

- Replace `numpy.lib.format.open_memmap()` with `h5py.File()` and distributed writing support
- Use HDF5 datasets with appropriate chunking for distributed access
- Store asset IDs directly in HDF5 instead of separate JSON
- Maintain distributed multi-GPU support using MPI I/O or parallel HDF5
- Add helper functions to query embeddings by objaverse ID

**Key implementation notes:**

- Use `h5py` with `driver='mpio'` for distributed writing (requires h5py-mpi)
- Or use rank-specific temporary files and merge at the end
- Enable compression (gzip level 4) for storage efficiency

### 3. Update Qwen Embedding Script ([scripts/03_embed_qwen.py](scripts/03_embed_qwen.py))

**HDF5 Structure:**

```
qwen_text_embeddings.h5
├── english (group)
│   ├── embeddings (2D dataset: [n_assets, embedding_dim])
│   └── asset_ids (1D string dataset: [n_assets])
└── chinese (group)
    ├── embeddings (2D dataset: [n_assets, embedding_dim])
    └── asset_ids (1D string dataset: [n_assets])

qwen_image_embeddings.h5
├── embeddings (2D dataset: [n_assets, embedding_dim])
└── asset_ids (1D string dataset: [n_assets])
```

**Changes:**

- Replace `pickle.dump()` with HDF5 dataset creation
- Store English and Chinese embeddings in separate groups within same file
- Store asset IDs alongside embeddings
- Add helper functions to query by objaverse ID
- Enable compression (gzip level 4)

### 4. Update Database Population Script ([scripts/04_populate_database.py](scripts/04_populate_database.py))

**Changes in `populate_siglip_database()`:**

- Replace pickle loading (lines 309-313) with HDF5 reading:
  ```python
  with h5py.File(config.SIGLIP_TEXT_EMBEDDINGS_FILE, 'r') as f:
      text_embeddings = dict(zip(f['asset_ids'][:], f['embeddings'][:]))
  ```

- Update image loading to read from HDF5 with metadata table
- Keep database insertion logic mostly unchanged

**Changes in `populate_qwen_database()`:**

- Replace pickle loading (lines 345-352) with HDF5 reading from groups
- Keep database insertion logic mostly unchanged

### 5. Create Utility Module for HDF5 Operations

Create `utils/h5_utils.py` with helper functions:

- `save_embeddings_h5()`: Save embeddings dict to HDF5
- `load_embeddings_h5()`: Load embeddings by asset IDs from HDF5
- `get_embedding_by_id()`: Query single embedding by objaverse ID
- `get_embeddings_batch()`: Query multiple embeddings efficiently
- `list_asset_ids()`: Get all available asset IDs from HDF5 file

**Benefits:**

- Reusable across all scripts
- Consistent HDF5 access patterns
- Easy to extend for new embedding types

## Testing Considerations

1. Verify distributed writing works correctly with multiple GPUs
2. Test random access performance for single and batch queries
3. Validate data integrity (checksums before/after)
4. Ensure backward compatibility path exists (can read old data if needed)
5. Test with MAX_ASSETS limit for quick validation
6. Verify database population works end-to-end

## Dependencies

Add to requirements.txt:

```
h5py>=3.8.0
h5py-mpi>=3.8.0  # Optional: for true parallel HDF5 writing
```

## Migration Path

Since the embeddings haven't been generated yet (outputs/embeddings doesn't exist), this is a clean migration with no data conversion needed.