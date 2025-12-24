"""
Gradio frontend for 3D Asset Retrieval System.

Features:
- Text input (auto-detect English/Chinese)
- Image upload
- Algorithm selection (SigLip/Qwen)
- Cross-modal search toggle
- 3D model viewer
- Results display with captions
"""
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import requests
import tempfile
import re

import gradio as gr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Backend API URL
API_BASE_URL = f"http://localhost:{config.BACKEND_PORT}"


def detect_language(text: str) -> str:
    """
    Detect if text is English or Chinese.
    
    Args:
        text: Input text
    
    Returns:
        'english' or 'chinese'
    """
    # Check for Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        return "chinese"
    return "english"


def search_by_text(
    query: str,
    algorithm: str,
    cross_modal: bool,
    top_k: int
) -> Tuple[str, str, str]:
    """
    Search by text query.
    
    Args:
        query: Text query
        algorithm: 'SigLip' or 'Qwen'
        cross_modal: Enable cross-modal search
        top_k: Number of results
    
    Returns:
        Tuple of (3D model path, results HTML, status message)
    """
    try:
        if not query.strip():
            return None, "", "Please enter a search query"
        
        # Detect language
        language = detect_language(query)
        
        # Map algorithm name
        algo_name = algorithm.lower()
        
        # Check if valid combination
        if algo_name == "siglip" and language == "chinese":
            return None, "", "Error: SigLip does not support Chinese text. Please use Qwen or enter English text."
        
        logger.info(f"Searching: query='{query[:50]}...', algorithm={algo_name}, language={language}")
        
        # Call API
        response = requests.post(
            f"{API_BASE_URL}/search/text",
            json={
                "query": query,
                "algorithm": algo_name,
                "language": language,
                "cross_modal": cross_modal,
                "top_k": top_k
            },
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("detail", "Unknown error")
            return None, "", f"Search failed: {error_msg}"
        
        results = response.json()
        
        # Format results
        if not results["results"]:
            return None, "", "No results found"
        
        # Get top result for 3D viewer
        top_result = results["results"][0]
        model_url = top_result.get("model_url")
        
        # Create results HTML
        results_html = format_results(results["results"], language)
        
        status_msg = f"Found {len(results['results'])} results (Language: {language.title()}, Algorithm: {algorithm}, Cross-modal: {cross_modal})"
        
        return model_url, results_html, status_msg
    
    except requests.exceptions.ConnectionError:
        return None, "", "Error: Cannot connect to backend API. Make sure the backend server is running."
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return None, "", f"Error: {str(e)}"


def search_by_image(
    image,
    algorithm: str,
    cross_modal: bool,
    top_k: int
) -> Tuple[str, str, str]:
    """
    Search by image upload.
    
    Args:
        image: PIL Image or file path
        algorithm: 'SigLip' or 'Qwen'
        cross_modal: Enable cross-modal search
        top_k: Number of results
    
    Returns:
        Tuple of (3D model path, results HTML, status message)
    """
    try:
        if image is None:
            return None, "", "Please upload an image"
        
        # Map algorithm name
        algo_name = algorithm.lower()
        
        logger.info(f"Searching by image: algorithm={algo_name}, cross_modal={cross_modal}")
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if hasattr(image, 'save'):
                image.save(tmp.name)
            else:
                # If it's already a file path
                import shutil
                shutil.copy(image, tmp.name)
            tmp_path = tmp.name
        
        # Call API
        with open(tmp_path, 'rb') as f:
            files = {'file': ('image.png', f, 'image/png')}
            params = {
                'algorithm': algo_name,
                'cross_modal': cross_modal,
                'top_k': top_k
            }
            response = requests.post(
                f"{API_BASE_URL}/search/image",
                files=files,
                params=params,
                timeout=60
            )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        if response.status_code != 200:
            error_msg = response.json().get("detail", "Unknown error")
            return None, "", f"Search failed: {error_msg}"
        
        results = response.json()
        
        # Format results
        if not results["results"]:
            return None, "", "No results found"
        
        # Get top result for 3D viewer
        top_result = results["results"][0]
        model_url = top_result.get("model_url")
        
        # Create results HTML
        results_html = format_results(results["results"], "english")
        
        status_msg = f"Found {len(results['results'])} results (Algorithm: {algorithm}, Cross-modal: {cross_modal})"
        
        return model_url, results_html, status_msg
    
    except requests.exceptions.ConnectionError:
        return None, "", "Error: Cannot connect to backend API. Make sure the backend server is running."
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return None, "", f"Error: {str(e)}"


def format_results(results: List[dict], language: str = "english") -> str:
    """
    Format search results as HTML.
    
    Args:
        results: List of search results
        language: Preferred language for captions
    
    Returns:
        HTML string
    """
    html = "<div style='font-family: Arial, sans-serif;'>"
    html += "<h3>Search Results:</h3>"
    
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        asset_id = result['asset_id']
        caption = result.get('caption_cn' if language == 'chinese' else 'caption_en', '')
        
        # Color code by similarity
        if similarity > 0.8:
            color = "#00aa00"
        elif similarity > 0.6:
            color = "#aa8800"
        else:
            color = "#aa0000"
        
        html += f"""
        <div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;'>
            <div style='font-weight: bold; color: {color};'>
                #{i} - Similarity: {similarity:.4f}
            </div>
            <div style='margin-top: 5px; color: #666;'>
                Asset ID: {asset_id}
            </div>
            <div style='margin-top: 5px;'>
                {caption}
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def create_ui():
    """Create Gradio UI."""
    
    with gr.Blocks(title="3D Asset Retrieval System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 3D Asset Retrieval System
        
        Search through millions of 3D assets using text or images with multi-modal embeddings.
        
        **Features:**
        - 🔤 Text search in English and Chinese
        - 🖼️ Image-based search
        - 🔄 Cross-modal retrieval (text↔image)
        - 🤖 Dual algorithms: SigLip and Qwen
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Search Options")
                
                # Algorithm selection
                algorithm = gr.Radio(
                    choices=["SigLip", "Qwen"],
                    value="SigLip",
                    label="Algorithm",
                    info="SigLip: English only, fast. Qwen: English + Chinese, slower."
                )
                
                # Cross-modal toggle
                cross_modal = gr.Checkbox(
                    label="Enable Cross-Modal Search",
                    value=False,
                    info="Search images with text or text with images"
                )
                
                # Top-K results
                top_k = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Number of Results",
                    info="Top-K results to return"
                )
                
                gr.Markdown("---")
                
                # Search by text
                gr.Markdown("### 🔤 Search by Text")
                text_query = gr.Textbox(
                    label="Text Query",
                    placeholder="Enter description in English or Chinese...",
                    lines=3
                )
                text_search_btn = gr.Button("🔍 Search by Text", variant="primary")
                
                gr.Markdown("---")
                
                # Search by image
                gr.Markdown("### 🖼️ Search by Image")
                image_query = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                image_search_btn = gr.Button("🔍 Search by Image", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                # Status message
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
                
                # 3D Model viewer
                model_3d = gr.Model3D(
                    label="Top Result - 3D Model",
                    height=400
                )
                
                # Results list
                results_display = gr.HTML(
                    label="All Results"
                )
        
        gr.Markdown("""
        ---
        ### 💡 Usage Tips:
        
        - **SigLip**: Best for English text and fast results
        - **Qwen**: Supports Chinese text and multi-image embeddings
        - **Cross-modal**: Find images from text descriptions or vice versa
        - **Top result**: Displayed in 3D viewer above
        - **Model URL**: Currently uses placeholder - configure BASE_URL_TEMPLATE in config
        
        ### 🔧 Notes:
        - Make sure the backend API is running (`python backend/app.py`)
        - Database must be populated with embeddings first
        - 3D model download depends on configured BASE_URL_TEMPLATE
        """)
        
        # Connect event handlers
        text_search_btn.click(
            fn=search_by_text,
            inputs=[text_query, algorithm, cross_modal, top_k],
            outputs=[model_3d, results_display, status_msg]
        )
        
        image_search_btn.click(
            fn=search_by_image,
            inputs=[image_query, algorithm, cross_modal, top_k],
            outputs=[model_3d, results_display, status_msg]
        )
    
    return demo


def main():
    """Launch Gradio app."""
    logger.info(f"Starting Gradio app on {config.FRONTEND_HOST}:{config.FRONTEND_PORT}")
    
    demo = create_ui()
    demo.launch(
        server_name=config.FRONTEND_HOST,
        server_port=config.FRONTEND_PORT,
        share=False
    )


if __name__ == "__main__":
    main()

