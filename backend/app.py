"""
FastAPI backend for 3D Asset Retrieval System.

Endpoints:
- GET /health - Health check
- POST /search/text - Search by text query
- POST /search/image - Search by image upload
"""
import logging
import sys
from pathlib import Path
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from backend.embedding_service import EmbeddingService
from backend.vector_search import VectorSearch

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="3D Asset Retrieval API",
    description="Multi-modal retrieval system for 3D assets using SigLip and Qwen embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService()
vector_search = VectorSearch()


# Request/Response models
class SearchResult(BaseModel):
    """Single search result."""
    asset_id: str
    similarity: float
    caption_en: str
    caption_cn: Optional[str] = None
    objaverse_id: Optional[str] = None
    model_url: Optional[str] = None


class TextSearchRequest(BaseModel):
    """Request for text-based search."""
    query: str = Field(..., description="Text query for search")
    algorithm: str = Field("siglip", description="Algorithm to use: 'siglip' or 'qwen'")
    language: str = Field("english", description="Query language: 'english' or 'chinese'")
    cross_modal: bool = Field(False, description="Enable cross-modal search (text->image)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")


class TextSearchResponse(BaseModel):
    """Response for text-based search."""
    results: List[SearchResult]
    query: str
    algorithm: str
    language: str
    cross_modal: bool


class ImageSearchResponse(BaseModel):
    """Response for image-based search."""
    results: List[SearchResult]
    algorithm: str
    cross_modal: bool


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "3D Asset Retrieval API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "siglip_model_loaded": embedding_service.siglip_model is not None,
        "databases": ["siglip_embeddings", "qwen_embeddings"]
    }


@app.post("/search/text", response_model=TextSearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search for 3D assets using text query.
    
    Args:
        request: Text search request with query, algorithm, and parameters
    
    Returns:
        Search results with asset IDs, similarities, and metadata
    """
    try:
        logger.info(f"Text search: query='{request.query[:50]}...', algorithm={request.algorithm}, "
                   f"language={request.language}, cross_modal={request.cross_modal}")
        
        # Validate inputs
        if request.algorithm not in ["siglip", "qwen"]:
            raise HTTPException(status_code=400, detail="Algorithm must be 'siglip' or 'qwen'")
        
        if request.language not in ["english", "chinese"]:
            raise HTTPException(status_code=400, detail="Language must be 'english' or 'chinese'")
        
        if request.algorithm == "siglip" and request.language == "chinese":
            raise HTTPException(status_code=400, detail="SigLip does not support Chinese text")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(
            query=request.query,
            algorithm=request.algorithm
        )
        
        # Search database
        results = vector_search.search(
            query_embedding=query_embedding,
            algorithm=request.algorithm,
            query_type="text",
            language=request.language,
            cross_modal=request.cross_modal,
            top_k=request.top_k
        )
        
        # Add model URLs
        for result in results:
            if result["objaverse_id"]:
                result["model_url"] = config.BASE_URL_TEMPLATE.format(
                    objaverse_id=result["objaverse_id"]
                )
        
        search_results = [SearchResult(**r) for r in results]
        
        logger.info(f"Found {len(search_results)} results")
        
        return TextSearchResponse(
            results=search_results,
            query=request.query,
            algorithm=request.algorithm,
            language=request.language,
            cross_modal=request.cross_modal
        )
    
    except Exception as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=ImageSearchResponse)
async def search_image(
    file: UploadFile = File(...),
    algorithm: str = Query("siglip", description="Algorithm: 'siglip' or 'qwen'"),
    cross_modal: bool = Query(False, description="Enable cross-modal search (image->text)"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results")
):
    """
    Search for 3D assets using image upload.
    
    Args:
        file: Uploaded image file
        algorithm: Algorithm to use ('siglip' or 'qwen')
        cross_modal: Enable cross-modal search
        top_k: Number of results to return
    
    Returns:
        Search results with asset IDs, similarities, and metadata
    """
    try:
        logger.info(f"Image search: filename={file.filename}, algorithm={algorithm}, "
                   f"cross_modal={cross_modal}")
        
        # Validate inputs
        if algorithm not in ["siglip", "qwen"]:
            raise HTTPException(status_code=400, detail="Algorithm must be 'siglip' or 'qwen'")
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(
            image=image,
            algorithm=algorithm
        )
        
        # Search database
        results = vector_search.search(
            query_embedding=query_embedding,
            algorithm=algorithm,
            query_type="image",
            cross_modal=cross_modal,
            top_k=top_k
        )
        
        # Add model URLs
        for result in results:
            if result["objaverse_id"]:
                result["model_url"] = config.BASE_URL_TEMPLATE.format(
                    objaverse_id=result["objaverse_id"]
                )
        
        search_results = [SearchResult(**r) for r in results]
        
        logger.info(f"Found {len(search_results)} results")
        
        return ImageSearchResponse(
            results=search_results,
            algorithm=algorithm,
            cross_modal=cross_modal
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    vector_search.close()


def main():
    """Run the FastAPI server."""
    logger.info(f"Starting FastAPI server on {config.BACKEND_HOST}:{config.BACKEND_PORT}")
    
    uvicorn.run(
        app,
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()

