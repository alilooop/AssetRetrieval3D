"""
Backend Test Client for 3D Asset Retrieval System.

Tests the FastAPI backend using ground-truth data from the database.
"""
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

import requests

from utils.data_loader import DataLoader
from utils.image_utils import get_asset_image_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackendTestClient:
    """Client for testing the FastAPI backend service."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        """
        Initialize the test client.
        
        Args:
            backend_url: Base URL of the backend service
        """
        self.backend_url = backend_url.rstrip('/')
        self.data_loader = DataLoader()
        self.session = requests.Session()
        
        # Load captions and IDs
        logger.info("Loading captions and asset IDs...")
        self.captions_en = self.data_loader.load_english_captions()
        self.captions_cn = self.data_loader.load_chinese_captions()
        self.asset_ids = list(self.captions_en.keys())
        logger.info(f"Loaded {len(self.asset_ids)} assets")
    
    def health_check(self) -> Dict:
        """
        Check backend health.
        
        Returns:
            Health check response
        """
        try:
            response = self.session.get(f"{self.backend_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def search_text(
        self,
        query: str,
        algorithm: str = "siglip",
        language: str = "english",
        cross_modal: bool = False,
        top_k: int = 10
    ) -> Dict:
        """
        Perform text search.
        
        Args:
            query: Text query
            algorithm: 'siglip' or 'qwen'
            language: 'english' or 'chinese'
            cross_modal: Enable cross-modal search
            top_k: Number of results
        
        Returns:
            Search response
        """
        url = f"{self.backend_url}/search/text"
        payload = {
            "query": query,
            "algorithm": algorithm,
            "language": language,
            "cross_modal": cross_modal,
            "top_k": top_k
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise
    
    def search_image(
        self,
        image_path: Path,
        algorithm: str = "siglip",
        cross_modal: bool = False,
        top_k: int = 10
    ) -> Dict:
        """
        Perform image search.
        
        Args:
            image_path: Path to image file
            algorithm: 'siglip' or 'qwen'
            cross_modal: Enable cross-modal search
            top_k: Number of results
        
        Returns:
            Search response
        """
        url = f"{self.backend_url}/search/image"
        
        # Read image file
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            params = {
                'algorithm': algorithm,
                'cross_modal': cross_modal,
                'top_k': top_k
            }
            
            try:
                response = self.session.post(url, files=files, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Image search failed: {e}")
                raise
    
    def get_test_assets(self, num_samples: int = 3, require_images: bool = False) -> List[str]:
        """
        Get random asset IDs for testing.
        
        Args:
            num_samples: Number of assets to return
            require_images: If True, only return assets with images
        
        Returns:
            List of asset IDs
        """
        if require_images:
            # Use DataLoader to efficiently get assets with images
            asset_ids_with_images = self.data_loader.get_asset_ids_with_images()
            
            if not asset_ids_with_images:
                logger.warning("No assets with images found")
                return []
            
            num_samples = min(num_samples, len(asset_ids_with_images))
            return random.sample(asset_ids_with_images, num_samples)
        else:
            num_samples = min(num_samples, len(self.asset_ids))
            return random.sample(self.asset_ids, num_samples)
    
    def get_test_image(self, asset_id: str) -> Optional[Path]:
        """
        Get a test image path for an asset.
        
        Args:
            asset_id: Asset ID
        
        Returns:
            Path to first available image, or None if no images
        """
        image_paths = get_asset_image_paths(asset_id, max_images=1)
        return image_paths[0] if image_paths else None
    
    def print_separator(self, char: str = "=", length: int = 80):
        """Print a separator line."""
        print(char * length)
    
    def print_header(self, title: str):
        """Print a section header."""
        self.print_separator()
        print(f"  {title}")
        self.print_separator()
    
    def print_results(self, response: Dict, query_info: Dict):
        """
        Pretty-print search results.
        
        Args:
            response: API response
            query_info: Dictionary with query metadata
        """
        print()
        self.print_header(f"{query_info['type']} Search Results")
        
        # Print query info
        print(f"\nQuery Type: {query_info['type']}")
        print(f"Algorithm: {query_info['algorithm']}")
        print(f"Cross-modal: {query_info['cross_modal']}")
        
        if query_info['type'] == 'Text':
            print(f"Language: {query_info.get('language', 'N/A')}")
            print(f"Query: {query_info['query'][:100]}...")
        else:
            print(f"Image: {query_info['image_path']}")
        
        # Print results
        results = response.get('results', [])
        print(f"\nFound {len(results)} results:")
        print()
        
        for i, result in enumerate(results, 0):
            print(f"{i}. Asset ID: {result['asset_id']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Caption (EN): {result['caption_en'][:80]}...")
            if result.get('caption_cn'):
                print(f"   Caption (CN): {result['caption_cn'][:80]}...")
            if result.get('objaverse_id'):
                print(f"   Objaverse ID: {result['objaverse_id']}")
            if result.get('model_url'):
                print(f"   Model URL: {result['model_url']}")
            print()


class TestRunner:
    """Runs comprehensive tests on the backend."""
    
    def __init__(self, client: BackendTestClient, num_tests: int = 3, verbose: bool = False):
        """
        Initialize test runner.
        
        Args:
            client: Backend test client
            num_tests: Number of test samples per scenario
            verbose: Enable verbose output
        """
        self.client = client
        self.num_tests = num_tests
        self.verbose = verbose
        self.test_results = []
    
    def run_health_check(self) -> bool:
        """Run health check test."""
        print("\n" + "=" * 80)
        print("HEALTH CHECK")
        print("=" * 80)
        
        try:
            result = self.client.health_check()
            print("\n✓ Backend is healthy")
            print(f"  Status: {result.get('status')}")
            print(f"  SigLip model loaded: {result.get('siglip_model_loaded')}")
            print(f"  Databases: {result.get('databases')}")
            return True
        except Exception as e:
            print(f"\n✗ Health check failed: {e}")
            return False
    
    def run_text_search_tests(self, algorithm: str):
        """Run text search tests for given algorithm."""
        print("\n" + "=" * 80)
        print(f"TEXT SEARCH TESTS - {algorithm.upper()}")
        print("=" * 80)
        
        # Get test assets
        test_assets = self.client.get_test_assets(self.num_tests)
        
        # Test uni-modal search (English)
        print(f"\n--- Test 1: English Text → Text ({algorithm}) ---")
        asset_id = test_assets[0]
        query = self.client.captions_en[asset_id]
        
        try:
            response = self.client.search_text(
                query=query,
                algorithm=algorithm,
                language="english",
                cross_modal=False,
                top_k=5
            )
            
            query_info = {
                'type': 'Text',
                'algorithm': algorithm,
                'language': 'english',
                'cross_modal': False,
                'query': query
            }
            
            if self.verbose:
                self.client.print_results(response, query_info)
            else:
                results = response.get('results', [])
                print(f"✓ Found {len(results)} results")
                if results:
                    print(f"  Top result: {results[0]['asset_id']} (similarity: {results[0]['similarity']:.4f})")
        except Exception as e:
            print(f"✗ Test failed: {e}")
        
        # Test cross-modal search (English text -> image)
        print(f"\n--- Test 2: English Text → Image ({algorithm}) ---")
        asset_id = test_assets[1] if len(test_assets) > 1 else test_assets[0]
        query = self.client.captions_en[asset_id]
        
        try:
            response = self.client.search_text(
                query=query,
                algorithm=algorithm,
                language="english",
                cross_modal=True,
                top_k=5
            )
            
            query_info = {
                'type': 'Text',
                'algorithm': algorithm,
                'language': 'english',
                'cross_modal': True,
                'query': query
            }
            
            if self.verbose:
                self.client.print_results(response, query_info)
            else:
                results = response.get('results', [])
                print(f"✓ Found {len(results)} results")
                if results:
                    print(f"  Top result: {results[0]['asset_id']} (similarity: {results[0]['similarity']:.4f})")
        except Exception as e:
            print(f"✗ Test failed: {e}")
        
        # Test Chinese text search (only for Qwen)
        if algorithm == "qwen" and self.client.captions_cn:
            print("\n--- Test 3: Chinese Text → Text (qwen) ---")
            asset_id = test_assets[2] if len(test_assets) > 2 else test_assets[0]
            
            if asset_id in self.client.captions_cn:
                query = self.client.captions_cn[asset_id]
                
                try:
                    response = self.client.search_text(
                        query=query,
                        algorithm="qwen",
                        language="chinese",
                        cross_modal=False,
                        top_k=5
                    )
                    
                    query_info = {
                        'type': 'Text',
                        'algorithm': 'qwen',
                        'language': 'chinese',
                        'cross_modal': False,
                        'query': query
                    }
                    
                    if self.verbose:
                        self.client.print_results(response, query_info)
                    else:
                        results = response.get('results', [])
                        print(f"✓ Found {len(results)} results")
                        if results:
                            print(f"  Top result: {results[0]['asset_id']} (similarity: {results[0]['similarity']:.4f})")
                except Exception as e:
                    print(f"✗ Test failed: {e}")
            else:
                print("✗ No Chinese caption available for test asset")
    
    def run_image_search_tests(self, algorithm: str):
        """Run image search tests for given algorithm."""
        print("\n" + "=" * 80)
        print(f"IMAGE SEARCH TESTS - {algorithm.upper()}")
        print("=" * 80)
        
        # Get test assets with images
        test_assets = self.client.get_test_assets(self.num_tests, require_images=True)
        
        if not test_assets:
            print("✗ No assets with images found for testing")
            return
        
        # Test uni-modal search (image -> image)
        print(f"\n--- Test 1: Image → Image ({algorithm}) ---")
        asset_id = test_assets[0]
        image_path = self.client.get_test_image(asset_id)
        
        if image_path:
            try:
                response = self.client.search_image(
                    image_path=image_path,
                    algorithm=algorithm,
                    cross_modal=False,
                    top_k=5
                )
                
                query_info = {
                    'type': 'Image',
                    'algorithm': algorithm,
                    'cross_modal': False,
                    'image_path': str(image_path)
                }
                
                if self.verbose:
                    self.client.print_results(response, query_info)
                else:
                    results = response.get('results', [])
                    print(f"✓ Found {len(results)} results")
                    if results:
                        print(f"  Top result: {results[0]['asset_id']} (similarity: {results[0]['similarity']:.4f})")
            except Exception as e:
                print(f"✗ Test failed: {e}")
        else:
            print(f"✗ No image found for asset {asset_id}")
        
        # Test cross-modal search (image -> text)
        print(f"\n--- Test 2: Image → Text ({algorithm}) ---")
        asset_id = test_assets[1] if len(test_assets) > 1 else test_assets[0]
        image_path = self.client.get_test_image(asset_id)
        
        if image_path:
            try:
                response = self.client.search_image(
                    image_path=image_path,
                    algorithm=algorithm,
                    cross_modal=True,
                    top_k=5
                )
                
                query_info = {
                    'type': 'Image',
                    'algorithm': algorithm,
                    'cross_modal': True,
                    'image_path': str(image_path)
                }
                
                if self.verbose:
                    self.client.print_results(response, query_info)
                else:
                    results = response.get('results', [])
                    print(f"✓ Found {len(results)} results")
                    if results:
                        print(f"  Top result: {results[0]['asset_id']} (similarity: {results[0]['similarity']:.4f})")
            except Exception as e:
                print(f"✗ Test failed: {e}")
        else:
            print(f"✗ No image found for asset {asset_id}")
    
    def run_all_tests(self, algorithms: List[str], test_types: List[str]):
        """
        Run all tests.
        
        Args:
            algorithms: List of algorithms to test ('siglip', 'qwen')
            test_types: List of test types ('text', 'image')
        """
        # Health check first
        if not self.run_health_check():
            print("\n✗ Backend is not healthy. Aborting tests.")
            return
        
        # Run tests for each algorithm
        for algorithm in algorithms:
            if 'text' in test_types:
                self.run_text_search_tests(algorithm)
            
            if 'image' in test_types:
                self.run_image_search_tests(algorithm)
        
        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETED")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test client for 3D Asset Retrieval Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default settings
  python test_client.py
  
  # Test only SigLip algorithm
  python test_client.py --algorithm siglip
  
  # Test only image search
  python test_client.py --test-type image
  
  # Run with verbose output
  python test_client.py --verbose --num-tests 1
  
  # Test against remote backend
  python test_client.py --backend-url http://remote-server:8000
        """
    )
    
    parser.add_argument(
        '--backend-url',
        type=str,
        default='http://localhost:8000',
        help='Backend URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--num-tests',
        type=int,
        default=3,
        help='Number of test samples per scenario (default: 3)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['siglip', 'qwen', 'both'],
        default='both',
        help='Algorithm to test (default: both)'
    )
    
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['text', 'image', 'both'],
        default='both',
        help='Type of search to test (default: both)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output including all results'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine which algorithms to test
    if args.algorithm == 'both':
        algorithms = ['siglip', 'qwen']
    else:
        algorithms = [args.algorithm]
    
    # Determine which test types to run
    if args.test_type == 'both':
        test_types = ['text', 'image']
    else:
        test_types = [args.test_type]
    
    # Initialize client
    print(f"\nInitializing test client for backend: {args.backend_url}")
    try:
        client = BackendTestClient(backend_url=args.backend_url)
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(client, num_tests=args.num_tests, verbose=args.verbose)
    runner.run_all_tests(algorithms, test_types)


if __name__ == "__main__":
    main()

