# 3D Asset Retrieval System

A comprehensive multi-modal retrieval system for searching through ~1 million 3D assets using text (English/Chinese) and images.

## Features

- 🔤 **Text Search**: Query in English or Chinese
- 🖼️ **Image Search**: Find similar 3D assets using images
- 🔄 **Cross-Modal Retrieval**: Search images with text or text with images
- 🤖 **Dual Algorithms**:
  - **SigLip**: Fast, English-only, per-image embeddings
  - **Qwen**: Multi-lingual, multi-image embeddings
- 📊 **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- 🌐 **Web Interface**: Beautiful Gradio UI with 3D model viewer
- 🚀 **REST API**: FastAPI backend for programmatic access

## Architecture

```
┌─────────────────┐
│ English Captions│
└────────┬────────┘
         │
         ├──► Translation ──► Chinese Captions
         │
         ├──► SigLip Embeddings ──┐
         │    - Text (EN)          │
         │    - Images (per-view)  │
         │                         │
         └──► Qwen Embeddings ─────┤
              - Text (EN + CN)     │
              - Images (multi)     │
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ PostgreSQL       │
                         │ + pgvector       │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  FastAPI Backend │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Gradio Frontend │
                         └──────────────────┘
```

## Prerequisites

- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- NVIDIA GPU (recommended for SigLip)
- DashScope API key (for Qwen)

## Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export DASHSCOPE_API_KEY="your-api-key"
   export DB_HOST="localhost"
   export DB_PORT="5432"
   export DB_USER="postgres"
   export DB_PASSWORD="your-password"
   ```

4. **Configure PostgreSQL**:
   - Install PostgreSQL
   - Install pgvector extension:
     ```sql
     CREATE EXTENSION vector;
     ```

## Usage

### Step 0: Configure Settings

Edit `config.py` to adjust settings:
- `MAX_ASSETS`: Set to a small number (e.g., 100) for testing
- Database credentials
- API keys
- Paths

### Step 1: Translate Captions (Optional but Recommended)

Translate English captions to Chinese:

```bash
python scripts/01_translate_captions.py
```

This creates `data/text_captions_cap3d_cn.json`.

**Note**: This uses the Qwen batch API and may take hours for the full dataset.

### Step 2: Generate Embeddings

#### SigLip Embeddings

```bash
python scripts/02_embed_siglip.py
```

This generates:
- Text embeddings (English only)
- Image embeddings (one per viewpoint)

#### Qwen Embeddings

```bash
python scripts/03_embed_qwen.py
```

This generates:
- Text embeddings (English and Chinese)
- Multi-image embeddings (8 images per asset)

**Note**: Can run both in parallel if you have resources.

### Step 3: Populate Database

```bash
python scripts/04_populate_database.py
```

This:
- Creates two databases: `siglip_embeddings` and `qwen_embeddings`
- Creates tables with pgvector columns
- Inserts all embeddings
- Creates vector indexes

### Step 4: Start Backend API

```bash
python backend/app.py
```

Or with uvicorn:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Step 5: Start Frontend

In a new terminal:

```bash
python frontend/gradio_app.py
```

The UI will be available at `http://localhost:7860`

## API Endpoints

### Text Search
```bash
POST /search/text
{
  "query": "a red car",
  "algorithm": "siglip",
  "language": "english",
  "cross_modal": false,
  "top_k": 10
}
```

### Image Search
```bash
POST /search/image
- file: image file
- algorithm: "siglip" or "qwen"
- cross_modal: true/false
- top_k: number of results
```

### Health Check
```bash
GET /health
```

## Configuration Options

### `config.py` Key Settings

- **MAX_ASSETS**: Limit number of assets to process (for debugging)
- **TRANSLATION_BATCH_SIZE**: Captions per translation batch (default: 1000)
- **EMBEDDING_BATCH_SIZE**: Batch size for embedding generation (default: 100)
- **QWEN_NUM_IMAGES**: Number of images per asset for Qwen (default: 8)
- **DEFAULT_TOP_K**: Default number of search results (default: 10)

### Algorithm Comparison

| Feature | SigLip | Qwen |
|---------|--------|------|
| Text Languages | English only | English + Chinese |
| Image Embeddings | One per viewpoint | Multi-image (8 views) |
| Speed | Fast | Slower (API calls) |
| Requires GPU | Yes (local) | No (API) |
| Cross-modal | Yes | Yes |

## Search Modes

### Inner-Modal Search
- **Text → Text**: Find assets with similar descriptions
- **Image → Image**: Find visually similar assets

### Cross-Modal Search
- **Text → Image**: Find images matching text description
- **Image → Text**: Find text descriptions matching image

## Debugging Tips

1. **Start Small**: Set `MAX_ASSETS=100` in config for testing
2. **Check Logs**: All scripts have detailed logging
3. **Verify Data**:
   - Run `python config.py` to validate configuration
   - Check if embedding files exist in `outputs/embeddings/`
4. **Test Database**:
   - Run `python utils/db_utils.py` to test connections
5. **API Testing**: Use the `/health` endpoint to check status

## Performance Optimization

- **Vector Indexes**: Automatically created with IVFFLAT
- **Connection Pooling**: Database connections are pooled
- **Batch Processing**: All operations support batching
- **GPU Acceleration**: SigLip uses GPU when available

## Troubleshooting

### "Connection refused" errors
- Ensure PostgreSQL is running
- Check database credentials in config
- Verify pgvector extension is installed

### "SigLip model loading failed"
- Ensure transformers and torch are installed
- Check GPU availability with `torch.cuda.is_available()`
- Try CPU mode if GPU unavailable

### "API key errors"
- Set DASHSCOPE_API_KEY environment variable
- Verify API key is valid

### "No images found"
- Check that `data/gobjaverse/` directory exists
- Verify image paths in the data

## File Structure

```
objaverse_retrieval/
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── README.md                # This file
│
├── data/                    # Data files
│   ├── text_captions_cap3d.json
│   ├── text_captions_cap3d_cn.json  (generated)
│   ├── gobjaverse/          # 3D asset images
│   └── gobjaverse_280k_index_to_objaverse.json
│
├── outputs/                 # Generated outputs
│   ├── embeddings/          # Saved embeddings
│   ├── translations/        # Translation results
│   └── batch_jsonl/         # API batch files
│
├── utils/                   # Utility modules
│   ├── data_loader.py
│   ├── image_utils.py
│   └── db_utils.py
│
├── scripts/                 # Processing scripts
│   ├── 01_translate_captions.py
│   ├── 02_embed_siglip.py
│   ├── 03_embed_qwen.py
│   └── 04_populate_database.py
│
├── backend/                 # FastAPI backend
│   ├── app.py
│   ├── embedding_service.py
│   └── vector_search.py
│
└── frontend/               # Gradio frontend
    └── gradio_app.py
```

## License

[Your License Here]

## Acknowledgments

- SigLip model from Google
- Qwen models from Alibaba Cloud
- Objaverse dataset
- pgvector extension

