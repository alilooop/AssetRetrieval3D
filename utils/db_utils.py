"""
Database utilities for PostgreSQL with pgvector.
"""
import logging
from contextlib import contextmanager
from typing import Optional, Generator
import psycopg2
from psycopg2 import pool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connections and operations."""
    
    def __init__(self, db_name: str, connection_string: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_name: Name of the database
            connection_string: PostgreSQL connection string (optional, built from config if not provided)
        """
        self.db_name = db_name
        
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{db_name}"
        
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        # Connect to default postgres database to create our database
        conn_string = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/postgres"
        
        try:
            conn = psycopg2.connect(conn_string)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_name,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database: {self.db_name}")
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                logger.info(f"Database {self.db_name} created successfully")
            else:
                logger.info(f"Database {self.db_name} already exists")
            
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def enable_pgvector_extension(self):
        """Enable the pgvector extension in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.info("pgvector extension enabled")
            except Exception as e:
                logger.error(f"Error enabling pgvector extension: {e}")
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def init_connection_pool(self, min_connections: int = 1, max_connections: int = 10):
        """
        Initialize connection pool.
        
        Args:
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                self.connection_string
            )
            logger.info(f"Connection pool initialized for {self.db_name}")
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a database connection from the pool.
        
        Yields:
            Database connection
        """
        if self.connection_pool is None:
            self.init_connection_pool()
        
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info(f"Connection pool closed for {self.db_name}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = False):
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
        
        Returns:
            Query results if fetch=True, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if fetch:
                    result = cursor.fetchall()
                    return result
                conn.commit()
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
        
        Returns:
            True if table exists
        """
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
        """
        result = self.execute_query(query, (table_name,), fetch=True)
        return result[0][0] if result else False


def get_siglip_db() -> DatabaseManager:
    """Get database manager for SigLip embeddings."""
    return DatabaseManager(config.DB_NAME_SIGLIP, config.DB_CONNECTION_SIGLIP)


def get_qwen_db() -> DatabaseManager:
    """Get database manager for Qwen embeddings."""
    return DatabaseManager(config.DB_NAME_QWEN, config.DB_CONNECTION_QWEN)


def test_connection(db_manager: DatabaseManager) -> bool:
    """
    Test database connection.
    
    Args:
        db_manager: DatabaseManager instance
    
    Returns:
        True if connection successful
    """
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logger.info(f"Connected to PostgreSQL: {version[0]}")
            cursor.close()
            return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test database utilities
    logging.basicConfig(level=logging.INFO)
    
    print("Testing SigLip database connection...")
    siglip_db = get_siglip_db()
    
    try:
        # Create database if needed
        siglip_db.create_database_if_not_exists()
        
        # Test connection
        if test_connection(siglip_db):
            print("✓ SigLip database connection successful")
        
        # Enable pgvector
        siglip_db.enable_pgvector_extension()
        print("✓ pgvector extension enabled")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        siglip_db.close_pool()
    
    print("\nTesting Qwen database connection...")
    qwen_db = get_qwen_db()
    
    try:
        qwen_db.create_database_if_not_exists()
        
        if test_connection(qwen_db):
            print("✓ Qwen database connection successful")
        
        qwen_db.enable_pgvector_extension()
        print("✓ pgvector extension enabled")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        qwen_db.close_pool()

