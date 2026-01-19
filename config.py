"""
Central configuration for the 3D Asset Retrieval System.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ==================== Project Paths ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data files
CAPTIONS_FILE = DATA_DIR / "text_captions_cap3d.json"
CAPTIONS_CN_FILE = DATA_DIR / "text_captions_cap3d_cn.json"
GOBJAVERSE_DIR = DATA_DIR / "gobjaverse"
INDEX_MAPPING_FILE = DATA_DIR / "gobjaverse_index_to_objaverse.json"
INDEX_MAPPING_FILE_WITH_IMAGE = DATA_DIR / "gobjaverse_280k_index_to_objaverse.json"
HIGH_QUALITY_ASSETS_FILE = DATA_DIR / "kiui_gobj_merged.json"

# Output directories
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
TRANSLATION_DIR = OUTPUTS_DIR / "translations"
BATCH_JSONL_DIR = OUTPUTS_DIR / "batch_jsonl"

# Create output directories
for dir_path in [OUTPUTS_DIR, EMBEDDINGS_DIR, TRANSLATION_DIR, BATCH_JSONL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# SigLip embeddings output (HDF5 format)
SIGLIP_TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "siglip_text_embeddings.h5"
SIGLIP_IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "siglip_image_embeddings.h5"

# Qwen embeddings output (HDF5 format)
QWEN_TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_text_embeddings.h5"
QWEN_IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "qwen_image_embeddings.h5"

# ==================== API Configuration ====================
# Qwen/DashScope API configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Models
QWEN_TRANSLATION_MODEL = "qwen-plus"  # For translation tasks
QWEN_EMBEDDING_MODEL = "tongyi-embedding-vision-flash"  # For embeddings
SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"

# ==================== Database Configuration ====================
# PostgreSQL connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Database names
DB_NAME_SIGLIP = "siglip_embeddings"
DB_NAME_QWEN = "qwen_embeddings"

# Connection strings
DB_CONNECTION_SIGLIP = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_SIGLIP}"
DB_CONNECTION_QWEN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_QWEN}"

# ==================== Processing Configuration ====================
# Maximum number of assets to process (for debugging)
# Set to None to process all assets
MAX_ASSETS = int(os.getenv("MAX_ASSETS", "0")) or None

# Batch processing sizes
TRANSLATION_BATCH_SIZE = 1000  # Number of captions per JSONL file
EMBEDDING_BATCH_SIZE = 256     # Batch size for embedding generation
TEXT_EMBEDDING_API_BATCH_SIZE = 1000  # Number of texts per API batch file

# Qwen multi-image configuration
QWEN_NUM_IMAGES = 8  # Number of images to sample per asset for Qwen embedding

# ==================== Backend/Frontend Configuration ====================
# Backend API
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8001

# Frontend Gradio
FRONTEND_HOST = "0.0.0.0"
FRONTEND_PORT = 7864

# 3D Model download URL template
# Placeholder - will be configured with actual URL
BASE_URL_TEMPLATE = os.getenv("BASE_URL_TEMPLATE", "https://placeholder.com/models/{objaverse_id}.glb")

# ==================== Search Configuration ====================
# Vector similarity metric
SIMILARITY_METRIC = "cosine"  # Options: cosine, l2, inner_product

# Default search parameters
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

# ==================== Logging Configuration ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== Helper Functions ====================
def get_asset_image_dir(gobjaverse_id: str) -> Path:
    """
    Get the directory path for a given gobjaverse ID.
    
    Args:
        gobjaverse_id: Asset ID in format like "0/10002"
    
    Returns:
        Path to the asset's image directory
    """
    return GOBJAVERSE_DIR / gobjaverse_id

def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    
    if not DASHSCOPE_API_KEY:
        errors.append("DASHSCOPE_API_KEY environment variable not set")
    
    if not CAPTIONS_FILE.exists():
        errors.append(f"Captions file not found: {CAPTIONS_FILE}")
    
    if not GOBJAVERSE_DIR.exists():
        errors.append(f"Gobjaverse directory not found: {GOBJAVERSE_DIR}")
    
    if not INDEX_MAPPING_FILE.exists():
        errors.append(f"Index mapping file not found: {INDEX_MAPPING_FILE}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

if __name__ == "__main__":
    # Validate configuration when run directly
    try:
        validate_config()
        print("✓ Configuration is valid")
        print(f"  - Project root: {PROJECT_ROOT}")
        print(f"  - Max assets: {MAX_ASSETS or 'All'}")
        print(f"  - SigLip model: {SIGLIP_MODEL}")
        print(f"  - Qwen model: {QWEN_EMBEDDING_MODEL}")
    except ValueError as e:
        print(f"✗ {e}")

