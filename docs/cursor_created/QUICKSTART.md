# Quick Start Guide

Get the 3D Asset Retrieval System running in 5 steps!

## Prerequisites

1. ✅ PostgreSQL installed and running
2. ✅ Python 3.8+ installed
3. ✅ DashScope API key obtained
4. ✅ Data files in place:
   - `data/text_captions_cap3d.json`
   - `data/gobjaverse/` directory with images
   - `data/gobjaverse_280k_index_to_objaverse.json`

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy and edit the environment file:

```bash
cp env_example.txt .env
# Edit .env with your actual credentials
```

Or set environment variables:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export DB_USER="postgres"
export DB_PASSWORD="your-password"
export MAX_ASSETS="100"  # Start with 100 for testing
```

### 3. Setup Database

```bash
# Option A: Use script (Linux/Mac)
./setup_database.sh

# Option B: Manual setup
psql -U postgres -c "CREATE DATABASE siglip_embeddings;"
psql -U postgres -c "CREATE DATABASE qwen_embeddings;"
psql -U postgres -d siglip_embeddings -c "CREATE EXTENSION vector;"
psql -U postgres -d qwen_embeddings -c "CREATE EXTENSION vector;"
```

## Quick Test (Recommended for First Run)

Test with a small subset of data:

```bash
# Set MAX_ASSETS for testing
export MAX_ASSETS=100

# Skip translation for quick test (or run it)
# python scripts/01_translate_captions.py

# Generate embeddings
python scripts/02_embed_siglip.py    # Takes ~5-10 min for 100 assets
python scripts/03_embed_qwen.py      # Takes ~10-15 min for 100 assets

# Populate database
python scripts/04_populate_database.py
```

## Run the System

### Start Backend (Terminal 1)

```bash
python backend/app.py
# Or: ./run_backend.sh
```

Wait for message: `Uvicorn running on http://0.0.0.0:8000`

### Start Frontend (Terminal 2)

```bash
python frontend/gradio_app.py
# Or: ./run_frontend.sh
```

Wait for message with local URL.

### Open in Browser

Navigate to: `http://localhost:7860`

## Test the System

1. **Text Search**:
   - Enter: "a red car"
   - Algorithm: SigLip
   - Click "Search by Text"

2. **Image Search**:
   - Upload an image
   - Algorithm: SigLip
   - Click "Search by Image"

3. **Cross-Modal Search**:
   - Enable "Cross-Modal Search"
   - Try text→image or image→text search

## Full Production Run

Once testing is successful, process the full dataset:

```bash
# Unset MAX_ASSETS or set to 0
export MAX_ASSETS=0

# Run full processing pipeline
./run_all_processing.sh
```

⚠️ **Warning**: Full processing can take many hours or days:
- Translation: ~10-20 hours (depends on API)
- SigLip embeddings: ~2-4 hours (with GPU)
- Qwen embeddings: ~10-20 hours (API calls)

## Troubleshooting

### "Cannot connect to database"
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U postgres -h localhost -c "SELECT version();"
```

### "Model loading failed"
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
```

### "API key invalid"
```bash
# Verify API key is set
echo $DASHSCOPE_API_KEY

# Test API key
python -c "import os; import dashscope; print('Key set:', bool(os.getenv('DASHSCOPE_API_KEY')))"
```

### "No images found"
```bash
# Verify data directory structure
ls -la data/gobjaverse/0/ | head -20

# Check image paths
python -c "import config; from utils.image_utils import get_asset_image_paths; print(get_asset_image_paths('0/10005'))"
```

## API Usage Examples

### Text Search (curl)

```bash
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a red car",
    "algorithm": "siglip",
    "language": "english",
    "cross_modal": false,
    "top_k": 5
  }'
```

### Image Search (Python)

```python
import requests

with open('query_image.jpg', 'rb') as f:
    files = {'file': f}
    params = {'algorithm': 'siglip', 'top_k': 5}
    response = requests.post(
        'http://localhost:8000/search/image',
        files=files,
        params=params
    )
    
results = response.json()
print(f"Found {len(results['results'])} results")
```

## Next Steps

- Configure `BASE_URL_TEMPLATE` in `config.py` for actual 3D model downloads
- Adjust batch sizes in `config.py` for your hardware
- Explore cross-modal search capabilities
- Try both SigLip and Qwen algorithms to compare

## Support

For issues, check:
1. `README.md` for detailed documentation
2. Log outputs for error messages
3. Configuration in `config.py`

Happy searching! 🎨🔍

