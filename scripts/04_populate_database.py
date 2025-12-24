"""
Create database schemas and populate with embeddings.

This script:
1. Creates PostgreSQL databases if they don't exist
2. Enables pgvector extension
3. Creates tables for text and image embeddings
4. Loads embeddings from disk
5. Inserts embeddings into database
6. Creates vector indexes for fast similarity search
"""
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.db_utils import get_siglip_db, get_qwen_db, DatabaseManager
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class DatabasePopulator:
    """Populate databases with embeddings."""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset_id, viewpoint_idx)
        );
        """
        
        db.execute_query(text_table_sql)
        db.execute_query(image_table_sql)
        
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Image embeddings table (one row per asset, multi-image embedding)
        image_table_sql = f"""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            asset_id VARCHAR(255) PRIMARY KEY,
            embedding vector({embedding_dim}),
            num_images INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        db.execute_query(text_table_sql)
        db.execute_query(image_table_sql)
        
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
        text_embeddings: Dict[str, np.ndarray]
    ):
        """Insert SigLip text embeddings."""
        logger.info(f"Inserting {len(text_embeddings)} SigLip text embeddings...")
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                for asset_id, embedding in tqdm(text_embeddings.items(), desc="Inserting text"):
                    # Convert numpy array to list for pgvector
                    embedding_list = embedding.tolist()
                    
                    cursor.execute(
                        """
                        INSERT INTO text_embeddings (asset_id, english_embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (asset_id) DO UPDATE
                        SET english_embedding = EXCLUDED.english_embedding
                        """,
                        (asset_id, embedding_list)
                    )
                
                conn.commit()
                logger.info("SigLip text embeddings inserted successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting text embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_siglip_image_embeddings(
        self,
        db: DatabaseManager,
        image_embeddings: Dict[str, List[np.ndarray]]
    ):
        """Insert SigLip image embeddings."""
        logger.info(f"Inserting SigLip image embeddings for {len(image_embeddings)} assets...")
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                for asset_id, embeddings_list in tqdm(image_embeddings.items(), desc="Inserting images"):
                    for viewpoint_idx, embedding in enumerate(embeddings_list):
                        embedding_list = embedding.tolist()
                        
                        cursor.execute(
                            """
                            INSERT INTO image_embeddings (asset_id, viewpoint_idx, embedding)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (asset_id, viewpoint_idx) DO UPDATE
                            SET embedding = EXCLUDED.embedding
                            """,
                            (asset_id, viewpoint_idx, embedding_list)
                        )
                
                conn.commit()
                logger.info("SigLip image embeddings inserted successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting image embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_qwen_text_embeddings(
        self,
        db: DatabaseManager,
        text_en_embeddings: Dict[str, np.ndarray],
        text_cn_embeddings: Dict[str, np.ndarray]
    ):
        """Insert Qwen text embeddings."""
        logger.info(f"Inserting Qwen text embeddings...")
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Get all asset IDs
                all_asset_ids = set(text_en_embeddings.keys()) | set(text_cn_embeddings.keys())
                
                for asset_id in tqdm(all_asset_ids, desc="Inserting text"):
                    en_embedding = text_en_embeddings.get(asset_id)
                    cn_embedding = text_cn_embeddings.get(asset_id)
                    
                    en_list = en_embedding.tolist() if en_embedding is not None else None
                    cn_list = cn_embedding.tolist() if cn_embedding is not None else None
                    
                    cursor.execute(
                        """
                        INSERT INTO text_embeddings (asset_id, english_embedding, chinese_embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (asset_id) DO UPDATE
                        SET english_embedding = COALESCE(EXCLUDED.english_embedding, text_embeddings.english_embedding),
                            chinese_embedding = COALESCE(EXCLUDED.chinese_embedding, text_embeddings.chinese_embedding)
                        """,
                        (asset_id, en_list, cn_list)
                    )
                
                conn.commit()
                logger.info("Qwen text embeddings inserted successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting text embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def insert_qwen_image_embeddings(
        self,
        db: DatabaseManager,
        image_embeddings: Dict[str, np.ndarray]
    ):
        """Insert Qwen multi-image embeddings."""
        logger.info(f"Inserting {len(image_embeddings)} Qwen image embeddings...")
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                for asset_id, embedding in tqdm(image_embeddings.items(), desc="Inserting images"):
                    embedding_list = embedding.tolist()
                    
                    cursor.execute(
                        """
                        INSERT INTO image_embeddings (asset_id, embedding, num_images)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (asset_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            num_images = EXCLUDED.num_images
                        """,
                        (asset_id, embedding_list, config.QWEN_NUM_IMAGES)
                    )
                
                conn.commit()
                logger.info("Qwen image embeddings inserted successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting image embeddings: {e}")
                raise
            finally:
                cursor.close()
    
    def populate_siglip_database(self):
        """Populate SigLip database with embeddings."""
        logger.info("\n=== Populating SigLip Database ===")
        
        # Load embeddings
        logger.info("Loading SigLip embeddings from disk...")
        with open(config.SIGLIP_TEXT_EMBEDDINGS_FILE, 'rb') as f:
            text_embeddings = pickle.load(f)
        
        with open(config.SIGLIP_IMAGE_EMBEDDINGS_FILE, 'rb') as f:
            image_embeddings = pickle.load(f)
        
        # Get embedding dimension
        embedding_dim = next(iter(text_embeddings.values())).shape[0]
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Initialize database
        db = get_siglip_db()
        db.create_database_if_not_exists()
        db.enable_pgvector_extension()
        
        # Create tables
        self.create_siglip_tables(db, embedding_dim)
        
        # Insert embeddings
        self.insert_siglip_text_embeddings(db, text_embeddings)
        self.insert_siglip_image_embeddings(db, image_embeddings)
        
        # Create indexes
        self.create_vector_indexes(db, 'siglip')
        
        # Close connections
        db.close_pool()
        
        logger.info("✓ SigLip database populated successfully")
    
    def populate_qwen_database(self):
        """Populate Qwen database with embeddings."""
        logger.info("\n=== Populating Qwen Database ===")
        
        # Load embeddings
        logger.info("Loading Qwen embeddings from disk...")
        with open(config.QWEN_TEXT_EN_EMBEDDINGS_FILE, 'rb') as f:
            text_en_embeddings = pickle.load(f)
        
        with open(config.QWEN_TEXT_CN_EMBEDDINGS_FILE, 'rb') as f:
            text_cn_embeddings = pickle.load(f)
        
        with open(config.QWEN_IMAGE_EMBEDDINGS_FILE, 'rb') as f:
            image_embeddings = pickle.load(f)
        
        # Get embedding dimension
        embedding_dim = next(iter(text_en_embeddings.values())).shape[0]
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Initialize database
        db = get_qwen_db()
        db.create_database_if_not_exists()
        db.enable_pgvector_extension()
        
        # Create tables
        self.create_qwen_tables(db, embedding_dim)
        
        # Insert embeddings
        self.insert_qwen_text_embeddings(db, text_en_embeddings, text_cn_embeddings)
        self.insert_qwen_image_embeddings(db, image_embeddings)
        
        # Create indexes
        self.create_vector_indexes(db, 'qwen')
        
        # Close connections
        db.close_pool()
        
        logger.info("✓ Qwen database populated successfully")
    
    def populate_all(self):
        """Populate both databases."""
        # Check if embedding files exist
        if not config.SIGLIP_TEXT_EMBEDDINGS_FILE.exists():
            logger.error(f"SigLip text embeddings not found: {config.SIGLIP_TEXT_EMBEDDINGS_FILE}")
            logger.error("Please run 02_embed_siglip.py first")
            return
        
        if not config.QWEN_TEXT_EN_EMBEDDINGS_FILE.exists():
            logger.error(f"Qwen embeddings not found: {config.QWEN_TEXT_EN_EMBEDDINGS_FILE}")
            logger.error("Please run 03_embed_qwen.py first")
            return
        
        # Populate SigLip database
        self.populate_siglip_database()
        
        # Populate Qwen database
        self.populate_qwen_database()
        
        logger.info("\n✓ All databases populated successfully!")


def main():
    """Main entry point."""
    try:
        config.validate_config()
        
        populator = DatabasePopulator()
        populator.populate_all()
        
    except Exception as e:
        logger.error(f"Database population failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

