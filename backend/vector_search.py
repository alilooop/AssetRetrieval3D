"""
Vector search functionality for pgvector databases.
"""
import logging
from typing import List, Tuple, Dict
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db_utils import get_siglip_db, get_qwen_db
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class VectorSearch:
    """Handles vector similarity search in pgvector databases."""
    
    def __init__(self):
        self.siglip_db = get_siglip_db()
        self.qwen_db = get_qwen_db()
        self.data_loader = DataLoader()
        self.captions_en = self.data_loader.load_english_captions()
        self.captions_cn = self.data_loader.load_chinese_captions()
    
    def search_text_embeddings(
        self,
        query_embedding: np.ndarray,
        algorithm: str,
        language: str = "english",
        top_k: int = 10,
        high_quality_only: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Search text embeddings for similar assets.
        
        Args:
            query_embedding: Query embedding vector
            algorithm: 'siglip' or 'qwen'
            language: 'english' or 'chinese' (only for Qwen)
            top_k: Number of results to return
            high_quality_only: Filter for high quality assets
        
        Returns:
            List of (asset_id, similarity_score) tuples
        """
        db = self.siglip_db if algorithm == "siglip" else self.qwen_db
        
        # Determine which column to search
        if algorithm == "siglip":
            embedding_column = "english_embedding"
        else:
            embedding_column = "english_embedding" if language == "english" else "chinese_embedding"
        
        # Convert embedding to list for SQL
        embedding_list = query_embedding.tolist()
        
        # Build query
        where_clause = f"{embedding_column} IS NOT NULL"
        if high_quality_only:
            where_clause += " AND is_high_quality = TRUE"
            
        # Cosine similarity search using <=> operator
        query = f"""
        SELECT asset_id, 1 - ({embedding_column} <=> %s::vector) AS similarity
        FROM text_embeddings
        WHERE {where_clause}
        ORDER BY {embedding_column} <=> %s::vector
        LIMIT %s
        """
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (embedding_list, embedding_list, top_k))
            results = cursor.fetchall()
            cursor.close()
        
        return [(asset_id, float(similarity)) for asset_id, similarity in results]
    
    def search_image_embeddings_siglip(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        high_quality_only: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Search SigLip image embeddings (multiple per asset).
        Returns the best matching asset based on highest similarity across all viewpoints.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            high_quality_only: Filter for high quality assets
        
        Returns:
            List of (asset_id, similarity_score) tuples
        """
        db = self.siglip_db
        embedding_list = query_embedding.tolist()
        
        # Filter clause
        where_clause = "WHERE is_high_quality = TRUE" if high_quality_only else ""
        
        # Find best matching image for each asset, then rank assets
        query = f"""
        SELECT asset_id, MAX(1 - (embedding <=> %s::vector)) AS max_similarity
        FROM image_embeddings
        {where_clause}
        GROUP BY asset_id
        ORDER BY max_similarity DESC
        LIMIT %s
        """
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (embedding_list, top_k))
            results = cursor.fetchall()
            cursor.close()
        
        return [(asset_id, float(similarity)) for asset_id, similarity in results]
    
    def search_image_embeddings_qwen(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        high_quality_only: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Search Qwen image embeddings (one multi-image embedding per asset).
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            high_quality_only: Filter for high quality assets
        
        Returns:
            List of (asset_id, similarity_score) tuples
        """
        db = self.qwen_db
        embedding_list = query_embedding.tolist()
        
        # Filter clause
        where_clause = "WHERE is_high_quality = TRUE" if high_quality_only else ""
        
        query = f"""
        SELECT asset_id, 1 - (embedding <=> %s::vector) AS similarity
        FROM image_embeddings
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (embedding_list, embedding_list, top_k))
            results = cursor.fetchall()
            cursor.close()
        
        return [(asset_id, float(similarity)) for asset_id, similarity in results]
    
    def search(
        self,
        query_embedding: np.ndarray,
        algorithm: str,
        query_type: str,
        language: str = "english",
        cross_modal: bool = False,
        top_k: int = 10,
        high_quality_only: bool = False
    ) -> List[Dict]:
        """
        Perform vector search based on query type.
        
        Args:
            query_embedding: Query embedding vector
            algorithm: 'siglip' or 'qwen'
            query_type: 'text' or 'image'
            language: 'english' or 'chinese' (for text queries)
            cross_modal: If True, search opposite modality (text->image or image->text)
            top_k: Number of results to return
            high_quality_only: Filter for high quality assets
        
        Returns:
            List of result dictionaries with asset_id, similarity, caption, etc.
        """
        # Determine which embeddings to search
        if query_type == "text":
            if cross_modal:
                # Text query -> search image embeddings
                if algorithm == "siglip":
                    results = self.search_image_embeddings_siglip(query_embedding, top_k, high_quality_only)
                else:
                    results = self.search_image_embeddings_qwen(query_embedding, top_k, high_quality_only)
            else:
                # Text query -> search text embeddings
                results = self.search_text_embeddings(query_embedding, algorithm, language, top_k, high_quality_only)
        
        elif query_type == "image":
            if cross_modal:
                # Image query -> search text embeddings
                results = self.search_text_embeddings(query_embedding, algorithm, language, top_k, high_quality_only)
            else:
                # Image query -> search image embeddings
                if algorithm == "siglip":
                    results = self.search_image_embeddings_siglip(query_embedding, top_k, high_quality_only)
                else:
                    results = self.search_image_embeddings_qwen(query_embedding, top_k, high_quality_only)
        else:
            raise ValueError(f"Unsupported query_type: {query_type}")
        
        # Enrich results with captions
        enriched_results = []
        
        for asset_id, similarity in results:
            result = {
                "asset_id": asset_id,
                "similarity": similarity,
                "caption_en": self.captions_en.get(asset_id, ""),
                "caption_cn": self.captions_cn.get(asset_id, ""),
                "objaverse_id": self.data_loader.get_objaverse_id(asset_id)
            }
            enriched_results.append(result)
        
        return enriched_results
    
    def close(self):
        """Close database connections."""
        self.siglip_db.close_pool()
        self.qwen_db.close_pool()

