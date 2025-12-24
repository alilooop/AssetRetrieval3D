"""
Service for generating embeddings from queries.
"""
import logging
from typing import Optional
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import dashscope

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text and image queries."""
    
    def __init__(self):
        self.siglip_model = None
        self.siglip_processor = None
        self.device = None
    
    def load_siglip_model(self):
        """Load SigLip model for inference."""
        if self.siglip_model is None:
            logger.info(f"Loading SigLip model: {config.SIGLIP_MODEL}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.siglip_model = AutoModel.from_pretrained(
                config.SIGLIP_MODEL,
                device_map="auto"
            ).eval()
            
            self.siglip_processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
            logger.info("SigLip model loaded")
    
    def embed_text_siglip(self, text: str) -> np.ndarray:
        """
        Embed text using SigLip.
        
        Args:
            text: Input text string
        
        Returns:
            Normalized embedding vector
        """
        self.load_siglip_model()
        
        inputs = self.siglip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            embedding = self.siglip_model.get_text_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0]
    
    def embed_image_siglip(self, image: Image.Image) -> np.ndarray:
        """
        Embed image using SigLip.
        
        Args:
            image: PIL Image
        
        Returns:
            Normalized embedding vector
        """
        self.load_siglip_model()
        
        inputs = self.siglip_processor(
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embedding = self.siglip_model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0]
    
    def embed_text_qwen(self, text: str) -> np.ndarray:
        """
        Embed text using Qwen API.
        
        Args:
            text: Input text string
        
        Returns:
            Embedding vector
        """
        input_data = [{'text': text}]
        
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data
        )
        
        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get('embeddings', [])
            if embeddings:
                return np.array(embeddings[0]['embedding'], dtype=np.float32)
        
        raise RuntimeError(f"Failed to embed text with Qwen API")
    
    def embed_image_qwen(self, image_path: str) -> np.ndarray:
        """
        Embed image using Qwen API.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Embedding vector
        """
        input_data = [{'image': image_path}]
        
        resp = dashscope.MultiModalEmbedding.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            input=input_data
        )
        
        if resp.status_code == 200 and resp.output:
            embeddings = resp.output.get('embeddings', [])
            if embeddings:
                return np.array(embeddings[0]['embedding'], dtype=np.float32)
        
        raise RuntimeError(f"Failed to embed image with Qwen API")
    
    def embed_query(
        self,
        query: str = None,
        image: Image.Image = None,
        image_path: str = None,
        algorithm: str = "siglip"
    ) -> np.ndarray:
        """
        Embed a query (text or image) using specified algorithm.
        
        Args:
            query: Text query (optional)
            image: PIL Image (optional)
            image_path: Path to image file (optional, for Qwen)
            algorithm: 'siglip' or 'qwen'
        
        Returns:
            Embedding vector
        """
        if algorithm == "siglip":
            if query is not None:
                return self.embed_text_siglip(query)
            elif image is not None:
                return self.embed_image_siglip(image)
            else:
                raise ValueError("Either query or image must be provided")
        
        elif algorithm == "qwen":
            if query is not None:
                return self.embed_text_qwen(query)
            elif image_path is not None:
                return self.embed_image_qwen(image_path)
            elif image is not None:
                # Save image temporarily for Qwen API
                temp_path = "/tmp/query_image.png"
                image.save(temp_path)
                return self.embed_image_qwen(temp_path)
            else:
                raise ValueError("Either query or image must be provided")
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

