import pandas as pd
import json
import time
import os
import requests
import re
import random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
from pathlib import Path
import torch
import logging
from datetime import datetime
import sys

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "anthropic/claude-3.5-sonnet"

# ---------------------------------------------------------
# 0. LOGGING SETUP
# ---------------------------------------------------------

def setup_logging(output_dir):
    """Setup logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"classification_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"🚀 Logging initialized. Log file: {log_file}")
    return log_file

# ---------------------------------------------------------
# 1. COST TRACKER (UPDATED WITH LOGGING)
# ---------------------------------------------------------

class CostTracker:
    def __init__(self, budget_limit=1200.0):
        self.total_reviews_processed = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.budget_limit = budget_limit
        self.alert_threshold = budget_limit * 0.8  # Warn at 80%
        
        # Claude-3.5-Sonnet pricing (OpenRouter)
        self.input_price_per_million = 3.00   # $3.00 per 1M input tokens
        self.output_price_per_million = 15.00 # $15.00 per 1M output tokens
        
        # Estimated tokens per summarized review (conservative)
        self.avg_input_tokens_per_review = 250   # Summarized text + prompt
        self.avg_output_tokens_per_review = 50   # JSON response
    
    def check_budget(self):
        """Check if we're approaching or exceeding budget"""
        if self.total_cost >= self.budget_limit:
            return "EXCEEDED"
        elif self.total_cost >= self.alert_threshold:
            return "WARNING"
        else:
            return "OK"
    
    def update_batch(self, batch_size, success_count):
        """Update cost after processing a batch"""
        batch_input_tokens = batch_size * self.avg_input_tokens_per_review
        batch_output_tokens = success_count * self.avg_output_tokens_per_review
        
        input_cost = (batch_input_tokens / 1_000_000) * self.input_price_per_million
        output_cost = (batch_output_tokens / 1_000_000) * self.output_price_per_million
        batch_cost = input_cost + output_cost
        
        self.total_cost += batch_cost
        self.total_reviews_processed += success_count
        
        return batch_cost
    
    def get_cost_summary(self):
        """Get current cost summary"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        reviews_per_hour = self.total_reviews_processed / elapsed_hours if elapsed_hours > 0 else 0
        cost_per_review = self.total_cost / self.total_reviews_processed if self.total_reviews_processed > 0 else 0
        
        # Projection for 1.1M reviews
        projected_total_cost = (1100000 / self.total_reviews_processed) * self.total_cost if self.total_reviews_processed > 0 else 0
        
        return {
            "total_reviews": self.total_reviews_processed,
            "total_cost": self.total_cost,
            "cost_per_review": cost_per_review,
            "reviews_per_hour": reviews_per_hour,
            "projected_total_cost": projected_total_cost,
            "elapsed_hours": elapsed_hours,
            "budget_status": self.check_budget(),
            "remaining_budget": self.budget_limit - self.total_cost
        }
    
    def print_progress(self):
        """Print cost progress to terminal with budget warnings"""
        summary = self.get_cost_summary()
        
        status_emoji = "✅"
        if summary['budget_status'] == "WARNING":
            status_emoji = "⚠️"
        elif summary['budget_status'] == "EXCEEDED":
            status_emoji = "🚨"
        
        message = (f"💰 {status_emoji} COST: ${summary['total_cost']:.2f} / ${self.budget_limit:.2f} | "
                   f"{summary['total_reviews']:,} reviews | "
                   f"${summary['cost_per_review']:.4f} per review | "
                   f"Remaining: ${summary['remaining_budget']:.2f}")
        
        logging.info(message)
    
# ---------------------------------------------------------
# 2. ADAPTIVE BATCH SIZING (UPDATED WITH LOGGING)
# ---------------------------------------------------------

def adaptive_batch_size(current_batch_size, previous_success_rate):
    """Dynamically adjust batch size based on success rate"""
    if previous_success_rate > 0.9:
        new_size = min(20, current_batch_size + 2)  # Increase slowly
        logging.info(f"📈 Increasing batch size from {current_batch_size} to {new_size} (success rate: {previous_success_rate:.1%})")
        return new_size
    elif previous_success_rate > 0.7:
        logging.info(f"📊 Keeping batch size at {current_batch_size} (success rate: {previous_success_rate:.1%})")
        return current_batch_size  # Keep current
    else:
        new_size = max(4, current_batch_size - 5)  # Decrease on failure
        logging.info(f"📉 Decreasing batch size from {current_batch_size} to {new_size} (success rate: {previous_success_rate:.1%})")
        return new_size    

# Global cost tracker
cost_tracker = CostTracker()

# ---------------------------------------------------------
# 3. SYSTEM PROMPT (OPTIMIZED FOR BATCHING)
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an annotation model. Classify each pool review with this hierarchical schema:

1. social_encounter: "yes" or "no" (involves other people at the pool)
2. racial_connotations: "yes" , "no", "N/A" (references race/ethnicity/religion/culture)  
3. connotation_type: "discriminatory", "pleasant", "ambiguous", or "N/A"

**ONLY use these value combinations:**
- "no", "N/A", "N/A"
- "yes", "no", "N/A"  
- "yes", "yes", "discriminatory"
- "yes", "yes", "pleasant"
- "yes", "yes", "ambiguous"

Return ONLY a JSON array. Example:
[
  {
    "review_id": "review_1",
    "social_encounter": "yes",
    "racial_connotations": "no", 
    "connotation_type": "N/A"
  }
]
"""

# ---------------------------------------------------------
# 4. CORE FUNCTIONS (UPDATED WITH LOGGING)
# ---------------------------------------------------------

def extract_json_array_from_response(content):
    """Extract JSON array from response with multiple fallbacks"""
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON array
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None

def batch_classify_reviews(batch_data, batch_id):
    """Classify multiple reviews in one API call with robust error handling"""
    
    # Check budget before processing
    budget_status = cost_tracker.check_budget()
    if budget_status == "EXCEEDED":
        error_msg = f"🚨 BUDGET EXCEEDED: Skipping batch {batch_id} (${cost_tracker.total_cost:.2f} > ${cost_tracker.budget_limit:.2f})"
        logging.error(error_msg)
        return {
            "success": [],
            "failed": [{"review_id": rid, "error": "budget_exceeded"} for rid, _ in batch_data]
        }
    elif budget_status == "WARNING":
        warning_msg = f"⚠️  BUDGET WARNING: ${cost_tracker.total_cost:.2f} of ${cost_tracker.budget_limit:.2f}"
        logging.warning(warning_msg)
    
    logging.info(f"📦 Batch {batch_id}: Processing {len(batch_data)} reviews")
    
    # Randomize headers to avoid blocking
    referers = [
        "http://localhost:3000", "http://localhost:8080", "https://www.google.com",
        "https://github.com", "https://example.com", "http://127.0.0.1:3000"
    ]
    
    # Create batch prompt
    reviews_text = "\n\n".join([
        f"REVIEW {i+1} (ID: {review_id}):\n{review_text[:500]}..." if len(review_text) > 500 else f"REVIEW {i+1} (ID: {review_id}):\n{review_text}"
        for i, (review_id, review_text) in enumerate(batch_data)
    ])
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Classify these {len(batch_data)} pool reviews:

{reviews_text}

Return a JSON array with {len(batch_data)} objects in the same order as above, each with: review_id, social_encounter, racial_connotations, connotation_type"""}
    ]

    try:
        start_time = time.time()
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": random.choice(referers),
                "X-Title": f"Batch_{batch_id}_{random.randint(1000,9999)}"
            },
            json={
                "model": MODEL,
                "messages": msg,
                "temperature": 0.0,
                "max_tokens": 2000
            },
            timeout=120  # 2 minute timeout for batches
        )
        
        elapsed = time.time() - start_time
        
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            batch_results = extract_json_array_from_response(content)
            
            if batch_results and len(batch_results) == len(batch_data):
                # Success - all reviews processed
                success_msg = f"✅ Batch {batch_id}: All {len(batch_data)} reviews processed in {elapsed:.1f}s"
                logging.info(success_msg)
                
                # Update cost tracking
                batch_cost = cost_tracker.update_batch(len(batch_data), len(batch_data))
                logging.info(f"💰 Batch cost: ${batch_cost:.4f}")
                cost_tracker.print_progress()
                
                return {
                    "success": batch_results,
                    "failed": []
                }
            else:
                # Partial success - try to salvage
                partial_msg = f"⚠️ Batch {batch_id}: Partial success, salvaging..."
                logging.warning(partial_msg)
                result = salvage_batch_results(batch_data, content, batch_id)
                
                # Update cost tracking for partial success
                success_count = len(result["success"])
                batch_cost = cost_tracker.update_batch(len(batch_data), success_count)
                logging.info(f"💰 Batch cost: ${batch_cost:.4f} ({success_count}/{len(batch_data)} success)")
                cost_tracker.print_progress()
                
                return result
                
        elif r.status_code == 429:
            # Rate limited
            wait_time = random.uniform(10, 30)
            rate_limit_msg = f"⏳ Batch {batch_id}: Rate limited, waiting {wait_time:.1f}s"
            logging.warning(rate_limit_msg)
            time.sleep(wait_time)
            return batch_classify_reviews(batch_data, batch_id)  # Retry
            
        else:
            # API error
            error_msg = f"❌ Batch {batch_id}: API error {r.status_code}"
            logging.error(error_msg)
            
            # Still count the cost for failed batch (we used tokens for the request)
            batch_cost = cost_tracker.update_batch(len(batch_data), 0)
            logging.error(f"💰 Batch cost: ${batch_cost:.4f} (all failed)")
            cost_tracker.print_progress()
            
            return {
                "success": [],
                "failed": [{"review_id": rid, "error": f"api_error_{r.status_code}"} for rid, _ in batch_data]
            }
            
    except requests.exceptions.Timeout:
        timeout_msg = f"⏰ Batch {batch_id}: Timeout after 120s"
        logging.error(timeout_msg)
        
        # Count cost for timed out batch
        batch_cost = cost_tracker.update_batch(len(batch_data), 0)
        logging.error(f"💰 Batch cost: ${batch_cost:.4f} (timeout)")
        cost_tracker.print_progress()
        
        return {
            "success": [],
            "failed": [{"review_id": rid, "error": "timeout"} for rid, _ in batch_data]
        }
    except Exception as e:
        exception_msg = f"💥 Batch {batch_id}: Exception {e}"
        logging.error(exception_msg)
        
        # Count cost for crashed batch
        batch_cost = cost_tracker.update_batch(len(batch_data), 0)
        logging.error(f"💰 Batch cost: ${batch_cost:.4f} (exception)")
        cost_tracker.print_progress()
        
        return {
            "success": [],
            "failed": [{"review_id": rid, "error": f"exception_{str(e)}"} for rid, _ in batch_data]
        }

def salvage_batch_results(batch_data, content, batch_id):
    """Try to salvage individual results from a partial batch response"""
    success_results = []
    failed_reviews = []
    
    # Extract all JSON objects from response
    json_objects = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', content)
    
    extracted_results = []
    for json_str in json_objects:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "review_id" in obj:
                extracted_results.append(obj)
        except:
            continue
    
    logging.info(f"🔍 Batch {batch_id}: Found {len(extracted_results)} valid objects")
    
    # Match with original batch data
    processed_ids = set()
    for result in extracted_results:
        review_id = result["review_id"]
        for rid, text in batch_data:
            if rid == review_id:
                # Validate required fields
                if all(key in result for key in ["social_encounter", "racial_connotations", "connotation_type"]):
                    success_results.append(result)
                    processed_ids.add(rid)
                break
    
    # Mark unprocessed as failed
    for review_id, text in batch_data:
        if review_id not in processed_ids:
            failed_reviews.append({
                "review_id": review_id,
                "error": "not_found_or_invalid"
            })
    
    logging.info(f"📊 Batch {batch_id} salvaged: {len(success_results)} success, {len(failed_reviews)} failed")
    return {
        "success": success_results,
        "failed": failed_reviews
    }

def retry_failed_reviews_individual(failed_batch_data):
    """Retry failed reviews one by one with progress bar"""
    logging.info(f"🔄 Retrying {len(failed_batch_data)} failed reviews individually...")
    
    success_results = []
    still_failed = []
    
    for review_id, review_text in tqdm(failed_batch_data, desc="Individual retry"):
        result = classify_single_review((review_id, review_text))
        
        if result.get("status") == "success":
            success_results.append(result)
            # Update cost for successful individual retry
            cost_tracker.update_batch(1, 1)
        else:
            still_failed.append({
                "review_id": review_id,
                "error_type": result.get("status", "unknown"),
                "error_details": str(result.get("error_code", result.get("exception", ""))),
                "timestamp": time.time()
            })
            # Update cost for failed individual retry
            cost_tracker.update_batch(1, 0)
        
        time.sleep(random.uniform(1, 3))  # Rate limiting
    
    cost_tracker.print_progress()
    return success_results, still_failed

def retry_failed_reviews_batch(failed_batch_data, retry_batch_size=10, parallel_workers=2):
    """Retry failed reviews in small batches with parallel processing"""
    logging.info(f"🔄 Batch retrying {len(failed_batch_data)} failed reviews in batches of {retry_batch_size}...")
    
    # Create smaller batches for retry
    retry_batches = [failed_batch_data[i:i + retry_batch_size] for i in range(0, len(failed_batch_data), retry_batch_size)]
    
    success_results = []
    failed_results = []
    
    # Thread-safe collections
    import threading
    success_lock = threading.Lock()
    failed_lock = threading.Lock()
    
    def process_retry_batch(batch_data, batch_idx):
        """Process a single retry batch"""
        batch_id = f"retry_batch_{batch_idx}"
        try:
            result = batch_classify_reviews(batch_data, batch_id)
            return {
                "success": result["success"],
                "failed": result["failed"],
                "error": None
            }
        except Exception as e:
            logging.error(f"💥 Retry batch {batch_id} crashed: {e}")
            return {
                "success": [],
                "failed": [{"review_id": rid, "error": "retry_batch_crash"} for rid, _ in batch_data],
                "error": str(e)
            }
    
    # Process retry batches in parallel
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_batch = {
            executor.submit(process_retry_batch, batch_data, batch_idx): batch_idx
            for batch_idx, batch_data in enumerate(retry_batches)
        }
        
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                
                with success_lock:
                    success_results.extend(batch_result["success"])
                with failed_lock:
                    failed_results.extend(batch_result["failed"])
                
                batch_success = len(batch_result["success"])
                batch_failed = len(batch_result["failed"])
                logging.info(f"🔄 Retry batch: {batch_success} success, {batch_failed} failed")
                
            except Exception as e:
                logging.error(f"💥 Error processing retry batch result: {e}")
    
    logging.info(f"📊 Batch retry complete: {len(success_results)} success, {len(failed_results)} failed")
    return success_results, failed_results

def classify_single_review(review_data):
    """Fallback single-review classification"""
    review_id, review_text = review_data
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review_text}
    ]

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": f"http://localhost:{random.randint(1,10000000)}",
                "X-Title": "Single_Review_Fallback"
            },
            json={
                "model": MODEL,
                "messages": msg,
                "temperature": 0.0
            },
            timeout=120
        )

        if r.status_code != 200:
            return {"review_id": review_id, "status": "api_error", "error_code": r.status_code}
        
        content = r.json()["choices"][0]["message"]["content"]
        
        # Try to extract single JSON object
        try:
            json_result = json.loads(content)
        except:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_result = json.loads(json_match.group(0))
            else:
                return {"review_id": review_id, "status": "parse_error"}
        
        if not all(key in json_result for key in ["social_encounter", "racial_connotations", "connotation_type"]):
            return {"review_id": review_id, "status": "invalid_format"}
            
        json_result["review_id"] = review_id
        json_result["status"] = "success"
        return json_result

    except Exception as e:
        return {"review_id": review_id, "status": "request_error", "exception": str(e)}

def save_checkpoint(chunk_success, chunk_failed, output_dir, chunk_id):
    """Save checkpoint with all processed results so far"""
    checkpoint_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    
    checkpoint_data = {
        "chunk_id": chunk_id,
        "success": chunk_success,
        "failed": chunk_failed,
        "timestamp": time.time(),
        "cost_so_far": cost_tracker.total_cost,
        "reviews_processed_so_far": cost_tracker.total_reviews_processed
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logging.info(f"💾 Checkpoint saved: {checkpoint_file}")
    return checkpoint_file

def load_checkpoint(output_dir, chunk_id):
    """Load checkpoint if it exists"""
    checkpoint_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if checkpoint_file.exists():
        logging.info(f"🔄 Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def create_final_output(chunk_df, chunk_success, chunk_failed):
    """Create final output without checkpoint status columns"""
    # Create minimal output with only essential columns
    minimal_output = pd.DataFrame({
        'review_id': chunk_df['review_id'],
        'was_summarized': chunk_df.get('was_summarized', False)
    })
    
    # Add classification results
    if chunk_success:
        classification_map = {row['review_id']: row for row in chunk_success}
        
        for col in ['social_encounter', 'racial_connotations', 'connotation_type']:
            minimal_output[col] = minimal_output['review_id'].map(
                lambda x: classification_map.get(x, {}).get(col, 'MISSING')
            )
    else:
        # No successful classifications
        for col in ['social_encounter', 'racial_connotations', 'connotation_type']:
            minimal_output[col] = 'MISSING'
    
    # Mark failed reviews (no 'PENDING' or 'NOT_PROCESSED' status in final output)
    failed_ids = {failed['review_id'] for failed in chunk_failed}
    minimal_output.loc[minimal_output['review_id'].isin(failed_ids), 'classification_status'] = 'FAILED'
    minimal_output['classification_status'] = minimal_output['classification_status'].fillna('SUCCESS')
    
    return minimal_output

# ---------------------------------------------------------
# 5. CHUNK PROCESSING ENGINE (UPDATED WITH LOGGING)
# ---------------------------------------------------------
def process_chunk(chunk_df, chunk_id, output_dir, batch_size=20, parallel_batches=2):
    """Process a single chunk with parallel batch processing and parallel batch retries"""
    
    logging.info(f"📦 Processing chunk {chunk_id}: {len(chunk_df)} reviews (parallel_batches={parallel_batches})")
    
    # USE ONLY SUMMARIZED TEXT
    if 'summary' in chunk_df.columns and chunk_df['summary'].notna().any():
        text_column = 'summary'
        summarized_count = chunk_df['summary'].notna().sum()
        logging.info(f"📝 Using summarized text ({summarized_count}/{len(chunk_df)} summaries available)")
    else:
        logging.warning(f"⚠️ No summarized text found in chunk {chunk_id}, skipping classification")
        return None, None
    
    # Prepare review data - ONLY use summarized text
    review_data = [
        (row["review_id"], row[text_column]) 
        for _, row in chunk_df.iterrows() 
        if pd.notna(row[text_column]) and str(row[text_column]).strip()
    ]
    
    if not review_data:
        logging.warning(f"⚠️ No valid summarized reviews in chunk {chunk_id}")
        return None, None
    
    logging.info(f"📊 {len(review_data)} summarized reviews to classify")
    
    # Check for existing checkpoint
    checkpoint_data = load_checkpoint(output_dir, chunk_id)
    if checkpoint_data:
        logging.info(f"🔄 Found checkpoint, resuming from previous progress...")
        chunk_success = checkpoint_data["success"]
        chunk_failed = checkpoint_data["failed"]
        
        # Get already processed review IDs
        processed_ids = {item['review_id'] for item in chunk_success} | {item['review_id'] for item in chunk_failed}
        
        # Filter out already processed reviews
        review_data = [(rid, text) for rid, text in review_data if rid not in processed_ids]
        logging.info(f"📊 {len(review_data)} reviews remaining to process")
        
        if not review_data:
            logging.info(f"✅ Chunk {chunk_id} already completed in checkpoint")
            return chunk_success, chunk_failed
    else:
        chunk_success = []
        chunk_failed = []
    
    # Create batches
    batches = [review_data[i:i + batch_size] for i in range(0, len(review_data), batch_size)]
    
    logging.info(f"🎯 Processing {len(batches)} batches with {parallel_batches} parallel workers")
    
    # Thread-safe collections for parallel processing
    import threading
    success_lock = threading.Lock()
    failed_lock = threading.Lock()
    
    def process_single_batch(batch_data, batch_idx):
        """Process a single batch and return results with batch info"""
        batch_full_id = f"{chunk_id}_{batch_idx}"
        try:
            result = batch_classify_reviews(batch_data, batch_full_id)
            return {
                "batch_idx": batch_idx,
                "batch_id": batch_full_id,
                "success": result["success"],
                "failed": result["failed"],
                "error": None
            }
        except Exception as e:
            logging.error(f"💥 Batch {batch_full_id} crashed in executor: {e}")
            return {
                "batch_idx": batch_idx,
                "batch_id": batch_full_id,
                "success": [],
                "failed": [{"review_id": rid, "error": "executor_crash"} for rid, _ in batch_data],
                "error": str(e)
            }
    
    # Process batches in parallel
    completed_batches = 0
    with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
        # Submit all batches to the executor
        future_to_batch = {
            executor.submit(process_single_batch, batch_data, batch_idx): batch_idx
            for batch_idx, batch_data in enumerate(batches)
        }
        
        # Process completed batches as they finish
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc=f"Chunk {chunk_id}"):
            batch_idx = future_to_batch[future]
            completed_batches += 1
            
            try:
                batch_result = future.result()
                
                # Thread-safe updates to shared collections
                with success_lock:
                    chunk_success.extend(batch_result["success"])
                with failed_lock:
                    chunk_failed.extend(batch_result["failed"])
                
                # Log batch completion
                batch_success_count = len(batch_result["success"])
                batch_failed_count = len(batch_result["failed"])
                
                if batch_result["error"]:
                    logging.error(f"❌ Batch {batch_result['batch_id']}: {batch_success_count} success, {batch_failed_count} failed - Error: {batch_result['error']}")
                else:
                    logging.info(f"✅ Batch {batch_result['batch_id']}: {batch_success_count} success, {batch_failed_count} failed")
                
                # Save checkpoint every 5 completed batches
                if completed_batches % 5 == 0:
                    checkpoint_file = save_checkpoint(chunk_success, chunk_failed, output_dir, chunk_id)
                    processed_count = len(chunk_success) + len(chunk_failed)
                    logging.info(f"💾 Checkpoint progress: {processed_count}/{len(review_data)} reviews processed")
                    
                    # Memory cleanup after checkpoint
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
            except Exception as e:
                logging.error(f"💥 Unexpected error processing batch result: {e}")
                continue
    
    # BATCH RETRY for failed reviews (10 per batch, 2 parallel workers)
    if chunk_failed:
        logging.info(f"🔄 Batch retrying {len(chunk_failed)} failed reviews in chunk {chunk_id}...")
        failed_review_data = []
        for failed in chunk_failed:
            for rid, text in review_data:
                if rid == failed["review_id"]:
                    failed_review_data.append((rid, text))
                    break
        
        retry_success, retry_failed = retry_failed_reviews_batch(
            failed_review_data, 
            retry_batch_size=10, 
            parallel_workers=2
        )
        chunk_success.extend(retry_success)
        chunk_failed = retry_failed
    
    # Save final chunk results
    if chunk_success or chunk_failed:
        final_output = create_final_output(chunk_df, chunk_success, chunk_failed)
        final_file = output_dir / f"classified_chunk_{chunk_id}.parquet"
        final_output.to_parquet(final_file)
        
        # Clean up checkpoint
        checkpoint_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logging.info(f"🧹 Cleaned up checkpoint for chunk {chunk_id}")
        
        logging.info(f"💾 Saved final chunk: {final_file.name}")
        
        # Memory cleanup
        del final_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Save chunk summary
        success_rate = len(chunk_success) / len(review_data) if review_data else 0
        summary_msg = f"✅ Chunk {chunk_id} complete: {len(chunk_success)} success, {len(chunk_failed)} failed ({success_rate:.1%})"
        logging.info(summary_msg)
        
        return chunk_success, chunk_failed
    
    return [], []
def get_processed_review_ids(processed_file, failed_file):
    """Get set of review IDs that have already been processed"""
    processed_ids = set()
    
    # Load processed reviews
    if processed_file.exists():
        processed_df = pd.read_csv(processed_file)
        if 'review_id' in processed_df.columns:
            processed_ids.update(processed_df['review_id'].tolist())
            logging.info(f"📊 Loaded {len(processed_ids)} processed reviews from {processed_file}")
    
    # Load failed reviews
    if failed_file.exists():
        failed_df = pd.read_csv(failed_file)
        if 'review_id' in failed_df.columns:
            failed_ids = set(failed_df['review_id'].tolist())
            processed_ids.update(failed_ids)
            logging.info(f"📊 Loaded {len(failed_ids)} failed reviews from {failed_file}")
    
    return processed_ids

def main():
    # Configuration
    CHUNK_SIZE = 5000  # Process in chunks of 5000
    BATCH_SIZE = 20
    PARALLEL_BATCHES = 2
    BUDGET_LIMIT = 500.0
    
    # File paths
    input_file = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/reprocessing/unprocessed_rerun.parquet")
    processed_file = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/reprocessing/reprocessed_success.csv")
    failed_file = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/reprocessing/reprocessed_failed.csv")
    output_dir = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/reprocessing/classification_chunks/")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging FIRST
    log_file = setup_logging(output_dir)
    
    # Initialize cost tracker with budget
    global cost_tracker
    cost_tracker = CostTracker(budget_limit=BUDGET_LIMIT)
    
    logging.info("🚀 SINGLE FILE CHUNK CLASSIFIER - Starting...")
    logging.info(f"💰 BUDGET SET: ${BUDGET_LIMIT:.2f}")
    logging.info(f"📦 CHUNK SIZE: {CHUNK_SIZE:,} reviews")
    logging.info(f"📁 Log file: {log_file}")
    
    # Load single parquet file
    logging.info(f"📖 Loading single parquet file: {input_file}")
    all_reviews_df = pd.read_parquet(input_file)
    logging.info(f"📊 Loaded {len(all_reviews_df):,} total reviews")
    
    # Get already processed review IDs
    processed_ids = get_processed_review_ids(processed_file, failed_file)
    
    # Filter out already processed reviews
    if processed_ids:
        unprocessed_df = all_reviews_df[~all_reviews_df['review_id'].isin(processed_ids)]
        logging.info(f"📊 Filtered to {len(unprocessed_df):,} unprocessed reviews")
    else:
        unprocessed_df = all_reviews_df
        logging.info(f"📊 No previously processed reviews found, processing all {len(unprocessed_df):,} reviews")
    
    if len(unprocessed_df) == 0:
        logging.info("✅ All reviews have already been processed!")
        return
    
    # Split into chunks
    chunks = [unprocessed_df[i:i + CHUNK_SIZE] for i in range(0, len(unprocessed_df), CHUNK_SIZE)]
    logging.info(f"🔪 Split into {len(chunks)} chunks of {CHUNK_SIZE} reviews")
    
    # Progress file - THIS IS THE IMPORTANT PART THAT WAS MISSING
    progress_file = output_dir / "classification_progress.json"
    
    # Load progress
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        processed_chunk_ids = set(progress.get("processed_chunk_ids", []))
        all_success = progress.get("all_success", [])
        all_failed = progress.get("all_failed", [])
        BATCH_SIZE = progress.get("current_batch_size", 20)
        
        # Initialize cost tracker with previous progress
        cost_tracker.total_reviews_processed = len(all_success)
        cost_tracker.total_cost = progress.get("total_cost", 0.0)
        
        logging.info(f"🔄 Resuming: {len(processed_chunk_ids)} chunks already processed")
        logging.info(f"📦 Resuming with batch size: {BATCH_SIZE}")
        logging.info(f"💰 Previous cost: ${cost_tracker.total_cost:.2f}")
        logging.info(f"📊 Previous reviews processed: {cost_tracker.total_reviews_processed:,}")
    else:
        processed_chunk_ids = set()
        all_success = []
        all_failed = []
        logging.info("🆕 Starting fresh classification")
    
    # Track success rates for adaptive sizing
    chunk_success_rates = []
    
    # Process chunks
    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_id = f"chunk_{chunk_idx:04d}"
        
        if chunk_id in processed_chunk_ids:
            logging.info(f"⏭️  Skipping already processed {chunk_id}")
            continue
        
        try:
            chunk_success, chunk_failed = process_chunk(
                chunk_df, chunk_id, output_dir, BATCH_SIZE, PARALLEL_BATCHES
            )
            
            if chunk_success is not None:  # Chunk was processed
                # Update results
                all_success.extend(chunk_success)
                all_failed.extend(chunk_failed)
                processed_chunk_ids.add(chunk_id)
                
                # Calculate chunk success rate for adaptive sizing
                chunk_review_count = len(chunk_success) + len(chunk_failed)
                if chunk_review_count > 0:
                    chunk_success_rate = len(chunk_success) / chunk_review_count
                    chunk_success_rates.append(chunk_success_rate)
                    
                    # Keep only last 3 chunks for moving average
                    if len(chunk_success_rates) > 3:
                        chunk_success_rates.pop(0)
                    
                    # Calculate average success rate
                    avg_success_rate = sum(chunk_success_rates) / len(chunk_success_rates) if chunk_success_rates else 0
                    
                    # Adapt batch size based on performance
                    if chunk_success_rates:  # Only adapt if we have success rates
                        BATCH_SIZE = adaptive_batch_size(BATCH_SIZE, avg_success_rate)
                
                # Save progress with cost information - THIS IS CRUCIAL
                progress = {
                    "processed_chunk_ids": list(processed_chunk_ids),
                    "all_success": all_success,
                    "all_failed": all_failed,
                    "total_cost": cost_tracker.total_cost,
                    "total_reviews_processed": cost_tracker.total_reviews_processed,
                    "current_batch_size": BATCH_SIZE,
                    "parallel_batches": PARALLEL_BATCHES,
                    "last_update": time.time(),
                    "chunk_success_rates": chunk_success_rates,
                    "avg_success_rate": avg_success_rate if chunk_success_rates else 0
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                
                logging.info(f"💾 Progress saved: {len(processed_chunk_ids)} chunks processed")
                logging.info(f"📊 Current stats: {len(all_success):,} success, {len(all_failed):,} failed")
                
                # Clean memory between chunks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
        except Exception as e:
            logging.error(f"💥 Failed to process {chunk_id}: {e}")
            continue
        
        # Small break between chunks
        time.sleep(random.uniform(2, 5))
    
    # Combine all classified chunks into final output
    logging.info("🔗 Combining all classified chunks...")
    classified_chunks = list(output_dir.glob("classified_chunk_*.parquet"))
    
    if classified_chunks:
        final_df = pd.concat([pd.read_parquet(f) for f in classified_chunks], ignore_index=True)
        
        # Append to existing processed file or create new
        if processed_file.exists():
            existing_df = pd.read_csv(processed_file)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
            logging.info(f"📊 Appended to existing processed file: {len(final_df):,} total rows")
        else:
            logging.info(f"📊 Creating new processed file: {len(final_df):,} rows")
        
        final_df.to_csv(processed_file, index=False)
        logging.info(f"💾 Final output saved: {processed_file}")
        
        # Save failed reviews
        if all_failed:
            failed_df = pd.DataFrame(all_failed)
            if failed_file.exists():
                existing_failed = pd.read_csv(failed_file)
                failed_df = pd.concat([existing_failed, failed_df], ignore_index=True)
                logging.info(f"📊 Appended to existing failed file: {len(failed_df):,} total failed reviews")
            else:
                logging.info(f"📊 Creating new failed file: {len(failed_df):,} failed reviews")
                
            failed_df.to_csv(failed_file, index=False)
            logging.info(f"📄 Failed reviews saved: {failed_file}")
        
        # Cleanup progress file
        if progress_file.exists():
            progress_file.unlink()
            logging.info("🧹 Cleaned up progress file")
        
        # Final summary with cost
        total_processed = len(all_success) + len(all_failed)
        success_rate = len(all_success) / total_processed if total_processed > 0 else 0
        cost_summary = cost_tracker.get_cost_summary()
        
        final_summary = (
            f"\n🎉 CLASSIFICATION COMPLETE!\n"
            f"✅ Successful: {len(all_success):,}\n"
            f"❌ Failed: {len(all_failed):,}\n"
            f"📊 Success rate: {success_rate:.1%}\n"
            f"📁 Processed chunks: {len(classified_chunks)}\n"
            f"💰 TOTAL COST: ${cost_tracker.total_cost:.2f}\n"
            f"   Cost per review: ${cost_summary['cost_per_review']:.4f}\n"
            f"   Reviews per hour: {cost_summary['reviews_per_hour']:.0f}\n"
            f"   Total time: {cost_summary['elapsed_hours']:.1f} hours\n"
            f"   Final batch size: {BATCH_SIZE}"
        )
        logging.info(final_summary)
        
        # Also save final summary to a separate file
        summary_file = output_dir / "classification_summary.json"
        final_summary_data = {
            "total_success": len(all_success),
            "total_failed": len(all_failed),
            "success_rate": success_rate,
            "total_chunks_processed": len(classified_chunks),
            "total_cost": cost_tracker.total_cost,
            "cost_per_review": cost_summary['cost_per_review'],
            "reviews_per_hour": cost_summary['reviews_per_hour'],
            "total_time_hours": cost_summary['elapsed_hours'],
            "final_batch_size": BATCH_SIZE,
            "completion_time": datetime.now().isoformat()
        }
        with open(summary_file, 'w') as f:
            json.dump(final_summary_data, f, indent=2)
        logging.info(f"📄 Final summary saved: {summary_file}")
        
    else:
        logging.error("❌ No classified chunks found!")

if __name__ == "__main__":
    main()