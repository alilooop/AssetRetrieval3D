"""
Create database schemas and populate with embeddings.

This script:
1. Creates PostgreSQL databases if they don't exist
2. Enables pgvector extension
3. Creates tables for text and image embeddings
4. Loads embeddings from disk into memory
5. Inserts embeddings into database using efficient bulk operations
6. Creates vector indexes for fast similarity search
"""
import sys
import logging
from pathlib import Path
import numpy as np
import json
from psycopg2.extras import execute_values

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.db_utils import get_siglip_db, get_qwen_db, DatabaseManager
from utils.data_loader import DataLoader
from utils.h5_utils import (
    load_text_embeddings_h5,
    load_image_embeddings_h5,
    load_multimodal_text_embeddings_h5
)

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class DatabasePopulator:
    """Populate databases with embeddings."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.high_quality_ids = self._load_quality_flags()
        self.valid_asset_ids = self._load_valid_assets()
    
    def _load_valid_assets(self) -> set:
        """Load valid asset IDs."""
        if not config.INDEX_MAPPING_FILE.exists():
            logger.warning(f"Index mapping file not found: {config.INDEX_MAPPING_FILE}")
            return set()
            
        logger.info(f"Loading valid assets from {config.INDEX_MAPPING_FILE}...")
        with open(config.INDEX_MAPPING_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"Found {len(data)} valid assets")
        return set(data.keys())
    
    def _load_quality_flags(self) -> set:
        """Load high quality asset IDs."""
        if not config.HIGH_QUALITY_ASSETS_FILE.exists():
            logger.warning(f"High quality assets file not found: {config.HIGH_QUALITY_ASSETS_FILE}")
            return set()
            
        logger.info(f"Loading high quality assets from {config.HIGH_QUALITY_ASSETS_FILE}...")
        with open(config.HIGH_QUALITY_ASSETS_FILE, 'r') as f:
            ids = set(json.load(f))
        logger.info(f"Found {len(ids)} high quality assets")
        return ids
    
    def create_siglip_tables(self, db: DatabaseManager, embedding_dim: int):
        """
        Create tables for SigLip embeddings.
        
        Args:
            db: Database manager
            embedding_dim: Dimension of embeddings
        """
        logger.info(f"Creating SigLip tables (embedding_dim={embedding_dim})...")
        
        # Text embeddings table (English only)
        text_table_sql = f"""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            english_embedding vector({embedding_dim}),
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Image embeddings table (one row per image/viewpoint)
        image_table_sql = f"""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255),
            viewpoint_idx INTEGER,
            embedding vector({embedding_dim}),
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset_id, viewpoint_idx)
        );
        """
        
        db.execute_query(text_table_sql)
        db.execute_query(image_table_sql)

        # Ensure columns exist (in case tables existed without them)
        db.execute_query("ALTER TABLE text_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        db.execute_query("ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        
        # Create indexes for is_high_quality
        db.execute_query("CREATE INDEX IF NOT EXISTS text_is_high_quality_idx ON text_embeddings (is_high_quality);")
        db.execute_query("CREATE INDEX IF NOT EXISTS image_is_high_quality_idx ON image_embeddings (is_high_quality);")
        
        logger.info("SigLip tables created successfully")
    
    def create_qwen_tables(self, db: DatabaseManager, embedding_dim: int):
        """
        Create tables for Qwen embeddings.
        
        Args:
            db: Database manager
            embedding_dim: Dimension of embeddings
        """
        logger.info(f"Creating Qwen tables (embedding_dim={embedding_dim})...")
        
        # Text embeddings table (English and Chinese)
        text_table_sql = f"""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            english_embedding vector({embedding_dim}),
            chinese_embedding vector({embedding_dim}),
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Image embeddings table (one row per asset, multi-image embedding)
        image_table_sql = f"""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            embedding vector({embedding_dim}),
            num_images INTEGER,
            is_high_quality BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        db.execute_query(text_table_sql)
        db.execute_query(image_table_sql)

        # Ensure columns exist
        db.execute_query("ALTER TABLE text_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        db.execute_query("ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS is_high_quality BOOLEAN DEFAULT FALSE;")
        
        # Create indexes for is_high_quality
        db.execute_query("CREATE INDEX IF NOT EXISTS text_is_high_quality_idx ON text_embeddings (is_high_quality);")
        db.execute_query("CREATE INDEX IF NOT EXISTS image_is_high_quality_idx ON image_embeddings (is_high_quality);")
        
        logger.info("Qwen tables created successfully")
    
    def create_vector_indexes(self, db: DatabaseManager, algorithm: str):
        """
        Create vector indexes for fast similarity search.
        
        Args:
            db: Database manager
            algorithm: 'siglip' or 'qwen'
        """
        logger.info(f"Creating vector indexes for {algorithm}...")
        
        # Index on text embeddings
        text_index_sql = """
        CREATE INDEX IF NOT EXISTS text_english_embedding_idx 
        ON text_embeddings 
        USING ivfflat (english_embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        db.execute_query(text_index_sql)
        
        if algorithm == 'qwen':
            # Also index Chinese embeddings for Qwen
            text_cn_index_sql = """
            CREATE INDEX IF NOT EXISTS text_chinese_embedding_idx 
            ON text_embeddings 
            USING ivfflat (chinese_embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            db.execute_query(text_cn_index_sql)
        
        # Index on image embeddings
        image_index_sql = """
        CREATE INDEX IF NOT EXISTS image_embedding_idx 
        ON image_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        db.execute_query(image_index_sql)
        
        logger.info("Vector indexes created successfully")
    
    def insert_siglip_text_embeddings(
        self,
        db: DatabaseManager,
        filepath: Path
    ):
        """Insert SigLip text embeddings from HDF5 file."""
        logger.info(f"Loading SigLip text embeddings from {filepath}...")
        
        # Load all data into memory
        embeddings_dict = load_text_embeddings_h5(filepath)
        logger.info(f"Loaded {len(embeddings_dict)} embeddings. Preparing insertion...")
        
        data_to_insert = []
        for asset_id, embedding in embeddings_dict.items():
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids
            data_to_insert.append((asset_id, embedding.tolist(), is_hq))
            
        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO text_embeddings (asset_id, english_embedding, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id) DO UPDATE
                    SET english_embedding = EXCLUDED.english_embedding,
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000
                )
                conn.commit()
                logger.info(f"Inserted {len(data_to_insert)} SigLip text embeddings")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting text embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_siglip_image_embeddings(
        self,
        db: DatabaseManager,
        filepath: Path
    ):
        """Insert SigLip image embeddings from HDF5 file."""
        logger.info(f"Loading SigLip image embeddings from {filepath}...")
        
        embeddings_dict = load_image_embeddings_h5(filepath)
        logger.info(f"Loaded embeddings for {len(embeddings_dict)} assets. Preparing insertion...")
        
        data_to_insert = []
        for asset_id, embeddings_list in embeddings_dict.items():
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids
            for viewpoint_idx, embedding in enumerate(embeddings_list):
                data_to_insert.append((asset_id, viewpoint_idx, embedding.tolist(), is_hq))
                
        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO image_embeddings (asset_id, viewpoint_idx, embedding, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id, viewpoint_idx) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000
                )
                conn.commit()
                logger.info(f"Inserted {len(data_to_insert)} SigLip image embeddings")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting image embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_qwen_text_embeddings(
        self,
        db: DatabaseManager,
        filepath: Path
    ):
        """Insert Qwen text embeddings from HDF5 file."""
        logger.info(f"Loading Qwen text embeddings from {filepath}...")
        
        en_dict, cn_dict = load_multimodal_text_embeddings_h5(filepath)
        all_asset_ids = set(en_dict.keys()) | set(cn_dict.keys())
        logger.info(f"Loaded embeddings for {len(all_asset_ids)} assets. Preparing insertion...")
        
        data_to_insert = []
        for asset_id in all_asset_ids:
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids
            
            en_emb = en_dict.get(asset_id)
            cn_emb = cn_dict.get(asset_id)
            
            en_list = en_emb.tolist() if en_emb is not None and not np.all(en_emb == 0) else None
            cn_list = cn_emb.tolist() if cn_emb is not None and not np.all(cn_emb == 0) else None
            
            data_to_insert.append((asset_id, en_list, cn_list, is_hq))
            
        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO text_embeddings (asset_id, english_embedding, chinese_embedding, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id) DO UPDATE
                    SET english_embedding = COALESCE(EXCLUDED.english_embedding, text_embeddings.english_embedding),
                        chinese_embedding = COALESCE(EXCLUDED.chinese_embedding, text_embeddings.chinese_embedding),
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000
                )
                conn.commit()
                logger.info(f"Inserted {len(data_to_insert)} Qwen text embeddings")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting text embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_qwen_image_embeddings(
        self,
        db: DatabaseManager,
        filepath: Path
    ):
        """Insert Qwen multi-image embeddings from HDF5 file."""
        logger.info(f"Loading Qwen image embeddings from {filepath}...")
        
        # Qwen image embeddings are stored as 1 vector per asset (similar to text structure)
        embeddings_dict = load_text_embeddings_h5(filepath)
        logger.info(f"Loaded {len(embeddings_dict)} embeddings. Preparing insertion...")
        
        data_to_insert = []
        for asset_id, embedding in embeddings_dict.items():
            if asset_id not in self.valid_asset_ids:
                continue
            is_hq = asset_id in self.high_quality_ids
            data_to_insert.append((asset_id, embedding.tolist(), config.QWEN_NUM_IMAGES, is_hq))
            
        with db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO image_embeddings (asset_id, embedding, num_images, is_high_quality)
                    VALUES %s
                    ON CONFLICT (asset_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        num_images = EXCLUDED.num_images,
                        is_high_quality = EXCLUDED.is_high_quality
                    """,
                    data_to_insert,
                    page_size=2000
                )
                conn.commit()
                logger.info(f"Inserted {len(data_to_insert)} Qwen image embeddings")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting image embeddings: {e}")
                raise
            finally:
                cursor.close()

    def populate_siglip_database(self):
        """Populate SigLip database with embeddings."""
        logger.info("\n=== Populating SigLip Database ===")
        
        # Get embedding dimension
        import h5py
        with h5py.File(config.SIGLIP_TEXT_EMBEDDINGS_FILE, 'r') as f:
            embedding_dim = f.attrs['embedding_dim']
        
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Initialize database
        db = get_siglip_db()
        db.create_database_if_not_exists()
        db.enable_pgvector_extension()
        
        # Create tables
        self.create_siglip_tables(db, embedding_dim)
        
        # Insert embeddings (bulk)
        self.insert_siglip_text_embeddings(db, config.SIGLIP_TEXT_EMBEDDINGS_FILE)
        self.insert_siglip_image_embeddings(db, config.SIGLIP_IMAGE_EMBEDDINGS_FILE)
        
        # Create indexes
        self.create_vector_indexes(db, 'siglip')
        
        # Close connections
        db.close_pool()
        
        logger.info("✓ SigLip database populated successfully")
    
    def populate_qwen_database(self):
        """Populate Qwen database with embeddings."""
        logger.info("\n=== Populating Qwen Database ===")
        
        # Get embedding dimension
        import h5py
        with h5py.File(config.QWEN_TEXT_EMBEDDINGS_FILE, 'r') as f:
            embedding_dim = f.attrs['embedding_dim']
        
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Initialize database
        db = get_qwen_db()
        db.create_database_if_not_exists()
        db.enable_pgvector_extension()
        
        # Create tables
        self.create_qwen_tables(db, embedding_dim)
        
        # Insert embeddings (bulk)
        self.insert_qwen_text_embeddings(db, config.QWEN_TEXT_EMBEDDINGS_FILE)
        self.insert_qwen_image_embeddings(db, config.QWEN_IMAGE_EMBEDDINGS_FILE)
        
        # Create indexes
        self.create_vector_indexes(db, 'qwen')
        
        # Close connections
        db.close_pool()
        
        logger.info("✓ Qwen database populated successfully")

    def populate_qwen(self):
        """Populate qwen database."""
        if not config.QWEN_TEXT_EMBEDDINGS_FILE.exists():
            logger.error(f"Qwen embeddings not found: {config.QWEN_TEXT_EMBEDDINGS_FILE}")
            logger.error("Please run 03_embed_qwen.py first")
            return
        
        # Populate Qwen database
        self.populate_qwen_database()
        
        logger.info("\n✓ QWEN database populated successfully!")
    
    def populate_siglip(self):
        """Populate siglip database."""
        # Check if embedding files exist
        if not config.SIGLIP_TEXT_EMBEDDINGS_FILE.exists():
            logger.error(f"SigLip text embeddings not found: {config.SIGLIP_TEXT_EMBEDDINGS_FILE}")
            logger.error("Please run 02_embed_siglip.py first")
            return
        
        # Populate SigLip database
        self.populate_siglip_database()
        
        logger.info("\n✓ SIGLIP database populated successfully!")


def main():
    """Main entry point."""
    try:
        config.validate_config()
        
        populator = DatabasePopulator()
        populator.populate_qwen()
        
    except Exception as e:
        logger.error(f"Database population failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

