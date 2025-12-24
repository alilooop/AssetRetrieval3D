"""
Translate English captions to Chinese using Qwen batch API.

This script:
1. Loads English captions
2. Splits them into batches
3. Creates JSONL files for batch API submission
4. Submits batch translation jobs
5. Monitors job status
6. Downloads and merges results
"""
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class CaptionTranslator:
    """Handles batch translation of captions using Qwen API."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL
        )
        self.data_loader = DataLoader()
    
    def create_translation_prompt(self, english_text: str) -> str:
        """Create translation prompt for a caption."""
        return f"请将以下英文描述翻译成中文，保持简洁准确：\n\n{english_text}\n\n只需要返回中文翻译，不要添加其他说明。"
    
    def create_batch_jsonl(
        self,
        captions: Dict[str, str],
        output_file: Path,
        start_idx: int = 0
    ) -> int:
        """
        Create a JSONL file for batch API submission.
        
        Args:
            captions: Dictionary of asset_id -> caption
            output_file: Path to output JSONL file
            start_idx: Starting custom_id index
        
        Returns:
            Number of entries created
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (asset_id, caption) in enumerate(captions.items()):
                custom_id = f"{start_idx + i}"
                
                entry = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config.QWEN_TRANSLATION_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional translator. Translate English to Chinese accurately and concisely."
                            },
                            {
                                "role": "user",
                                "content": self.create_translation_prompt(caption)
                            }
                        ],
                        "temperature": 0.3
                    }
                }
                
                # Store asset_id mapping for later
                entry["metadata"] = {"asset_id": asset_id}
                
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return len(captions)
    
    def split_captions_into_batches(
        self,
        captions: Dict[str, str],
        batch_size: int
    ) -> List[Path]:
        """
        Split captions into multiple JSONL files.
        
        Args:
            captions: Dictionary of asset_id -> caption
            batch_size: Number of captions per batch
        
        Returns:
            List of created JSONL file paths
        """
        config.BATCH_JSONL_DIR.mkdir(parents=True, exist_ok=True)
        
        caption_items = list(captions.items())
        jsonl_files = []
        
        for batch_idx in range(0, len(caption_items), batch_size):
            batch_captions = dict(caption_items[batch_idx:batch_idx + batch_size])
            
            output_file = config.BATCH_JSONL_DIR / f"translate_batch_{batch_idx // batch_size:04d}.jsonl"
            
            logger.info(f"Creating batch {batch_idx // batch_size}: {len(batch_captions)} captions")
            self.create_batch_jsonl(batch_captions, output_file, start_idx=batch_idx)
            
            jsonl_files.append(output_file)
        
        logger.info(f"Created {len(jsonl_files)} batch files")
        return jsonl_files
    
    def upload_file(self, file_path: Path) -> str:
        """Upload JSONL file to API."""
        logger.info(f"Uploading {file_path.name}...")
        file_object = self.client.files.create(file=file_path, purpose="batch")
        logger.info(f"Uploaded successfully. File ID: {file_object.id}")
        return file_object.id
    
    def create_batch_job(self, input_file_id: str) -> str:
        """Create batch translation job."""
        logger.info(f"Creating batch job for file {input_file_id}...")
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="72h"
        )
        logger.info(f"Batch job created. Batch ID: {batch.id}")
        return batch.id
    
    def check_job_status(self, batch_id: str) -> str:
        """Check batch job status."""
        batch = self.client.batches.retrieve(batch_id=batch_id)
        return batch.status
    
    def wait_for_completion(self, batch_id: str, poll_interval: int = 30):
        """
        Wait for batch job to complete.
        
        Args:
            batch_id: Batch job ID
            poll_interval: Seconds between status checks
        """
        logger.info(f"Waiting for batch {batch_id} to complete...")
        
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = self.check_job_status(batch_id)
            logger.info(f"Status: {status}")
            
            if status in ["completed", "failed", "expired", "cancelled"]:
                break
            
            time.sleep(poll_interval)
        
        if status == "failed":
            batch = self.client.batches.retrieve(batch_id)
            logger.error(f"Batch job failed: {batch.errors}")
            raise RuntimeError(f"Batch job {batch_id} failed")
        
        logger.info(f"Batch {batch_id} completed successfully")
    
    def download_results(self, batch_id: str, output_file: Path):
        """Download batch job results."""
        batch = self.client.batches.retrieve(batch_id=batch_id)
        
        if not batch.output_file_id:
            logger.warning(f"No output file for batch {batch_id}")
            return
        
        logger.info(f"Downloading results for batch {batch_id}...")
        content = self.client.files.content(batch.output_file_id)
        content.write_to_file(output_file)
        logger.info(f"Results saved to {output_file}")
    
    def parse_results(self, result_file: Path) -> Dict[str, str]:
        """
        Parse translation results from JSONL file.
        
        Args:
            result_file: Path to result JSONL file
        
        Returns:
            Dictionary mapping asset_id -> chinese_caption
        """
        translations = {}
        
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                result = json.loads(line)
                
                # Get asset_id from custom_id mapping
                custom_id = result.get('custom_id')
                
                # Extract translated text
                response = result.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices:
                    message = choices[0].get('message', {})
                    translated_text = message.get('content', '').strip()
                    
                    # We need to map back to asset_id
                    # For now, store by custom_id and we'll merge later
                    translations[custom_id] = translated_text
        
        return translations
    
    def save_batch_ids(self, batch_info: Dict, filepath: Path):
        """Save batch IDs and metadata to disk."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch IDs saved to {filepath}")
    
    def load_batch_ids(self, filepath: Path) -> Dict:
        """Load batch IDs and metadata from disk."""
        if not filepath.exists():
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def submit_all_batches(self, jsonl_files: List[Path]) -> Dict[str, Dict]:
        """
        Submit all batch jobs at once and save batch IDs.
        
        Args:
            jsonl_files: List of JSONL files to process
        
        Returns:
            Dictionary mapping batch_id -> {jsonl_file, result_file, status}
        """
        batch_ids_file = config.TRANSLATION_DIR / "batch_ids.json"
        
        # Load existing batch info if available
        batch_info = self.load_batch_ids(batch_ids_file)
        
        # Check which batches are already completed
        completed_batches = set()
        for batch_id, info in list(batch_info.items()):
            result_file = Path(info['result_file'])
            if result_file.exists():
                completed_batches.add(info['jsonl_file'])
                logger.info(f"Batch already completed: {result_file.name}")
        
        # Submit new batch jobs
        for jsonl_file in jsonl_files:
            jsonl_file_str = str(jsonl_file)
            
            # Skip if already processed
            if jsonl_file_str in completed_batches:
                continue
            
            # Check if batch job already exists
            existing_batch_id = None
            for batch_id, info in batch_info.items():
                if info['jsonl_file'] == jsonl_file_str:
                    existing_batch_id = batch_id
                    break
            
            if existing_batch_id:
                logger.info(f"Batch job already submitted for {jsonl_file.name}: {existing_batch_id}")
                continue
            
            # Upload and create batch job
            result_file = config.TRANSLATION_DIR / f"result_{jsonl_file.stem}.jsonl"
            
            try:
                file_id = self.upload_file(jsonl_file)
                batch_id = self.create_batch_job(file_id)
                
                # Store batch info
                batch_info[batch_id] = {
                    'jsonl_file': jsonl_file_str,
                    'result_file': str(result_file),
                    'status': 'submitted',
                    'file_id': file_id
                }
                
                # Save immediately after each submission
                self.save_batch_ids(batch_info, batch_ids_file)
                
            except Exception as e:
                logger.error(f"Failed to submit batch for {jsonl_file.name}: {e}")
                continue
        
        return batch_info
    
    def monitor_all_batches(self, batch_info: Dict[str, Dict], poll_interval: int = 30):
        """
        Monitor all batch jobs until completion.
        
        Args:
            batch_info: Dictionary of batch_id -> metadata
            poll_interval: Seconds between status checks
        """
        batch_ids_file = config.TRANSLATION_DIR / "batch_ids.json"
        pending_batches = set(batch_info.keys())
        
        logger.info(f"Monitoring {len(pending_batches)} batch jobs...")
        
        while pending_batches:
            for batch_id in list(pending_batches):
                try:
                    status = self.check_job_status(batch_id)
                    batch_info[batch_id]['status'] = status
                    
                    if status == "completed":
                        logger.info(f"✓ Batch {batch_id} completed")
                        
                        # Download results
                        result_file = Path(batch_info[batch_id]['result_file'])
                        if not result_file.exists():
                            self.download_results(batch_id, result_file)
                        
                        pending_batches.remove(batch_id)
                        
                    elif status in ["failed", "expired", "cancelled"]:
                        logger.error(f"✗ Batch {batch_id} {status}")
                        pending_batches.remove(batch_id)
                        
                        if status == "failed":
                            batch = self.client.batches.retrieve(batch_id)
                            logger.error(f"  Error details: {batch.errors}")
                    
                    else:
                        logger.info(f"Batch {batch_id}: {status}")
                    
                    # Save updated status
                    self.save_batch_ids(batch_info, batch_ids_file)
                    
                except Exception as e:
                    logger.error(f"Error checking batch {batch_id}: {e}")
            
            if pending_batches:
                logger.info(f"Waiting {poll_interval}s... ({len(pending_batches)} batches pending)")
                time.sleep(poll_interval)
    
    def merge_translations(
        self,
        result_files: List[Path],
        original_captions: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Merge translation results from multiple files.
        
        Args:
            result_files: List of result JSONL files
            original_captions: Original English captions
        
        Returns:
            Dictionary mapping asset_id -> chinese_caption
        """
        # First, create mapping from custom_id to asset_id
        custom_id_to_asset = {}
        asset_ids = list(original_captions.keys())
        
        for i, asset_id in enumerate(asset_ids):
            custom_id_to_asset[str(i)] = asset_id
        
        # Parse all results
        all_translations = {}
        
        for result_file in result_files:
            if not result_file.exists():
                logger.warning(f"Result file not found: {result_file}")
                continue
            
            translations = self.parse_results(result_file)
            
            # Map custom_ids back to asset_ids
            for custom_id, chinese_text in translations.items():
                if custom_id in custom_id_to_asset:
                    asset_id = custom_id_to_asset[custom_id]
                    all_translations[asset_id] = chinese_text
        
        logger.info(f"Merged {len(all_translations)} translations")
        return all_translations
    
    def translate_all(self):
        """Main method to translate all captions."""
        # Load English captions for ALL assets (even without images)
        captions_en = self.data_loader.load_english_captions()
        logger.info(f"Loaded {len(captions_en)} English captions")
        
        # Limit if MAX_ASSETS is set
        if config.MAX_ASSETS:
            asset_ids = list(captions_en.keys())[:config.MAX_ASSETS]
            captions_en = {aid: captions_en[aid] for aid in asset_ids}
            logger.info(f"Limited to {len(captions_en)} assets for processing")
        
        # Split into batches
        jsonl_files = self.split_captions_into_batches(
            captions_en,
            config.TRANSLATION_BATCH_SIZE
        )
        
        # Submit all batch jobs at once
        batch_info = self.submit_all_batches(jsonl_files)
        
        # Monitor all batches until completion
        self.monitor_all_batches(batch_info)
        
        # Collect result files
        result_files = [Path(info['result_file']) for info in batch_info.values()]
        
        # Merge results
        chinese_captions = self.merge_translations(result_files, captions_en)
        
        # Save merged results
        self.data_loader.save_chinese_captions(chinese_captions)
        
        logger.info("\n✓ Translation complete!")
        logger.info(f"  Total captions translated: {len(chinese_captions)}")
        logger.info(f"  Saved to: {config.CAPTIONS_CN_FILE}")


def main():
    """Main entry point."""
    try:
        config.validate_config()
        
        translator = CaptionTranslator()
        translator.translate_all()
        
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

