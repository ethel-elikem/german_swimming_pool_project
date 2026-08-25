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

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"🚀 API key loaded successfully")

MODEL = "anthropic/claude-3.5-sonnet"

# ---------------------------------------------------------
# 0. COST TRACKER
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
        
        print(f"💰 {status_emoji} COST: ${summary['total_cost']:.2f} / ${self.budget_limit:.2f} | "
              f"{summary['total_reviews']:,} reviews | "
              f"${summary['cost_per_review']:.4f} per review | "
              f"Remaining: ${summary['remaining_budget']:.2f}")
    
# ---------------------------------------------------------
# 0.5 ADAPTIVE BATCH SIZING
# ---------------------------------------------------------

def adaptive_batch_size(current_batch_size, previous_success_rate):
    """Dynamically adjust batch size based on success rate"""
    if previous_success_rate > 0.9:
        new_size = min(20, current_batch_size + 2)  # Increase slowly
        print(f"📈 Increasing batch size from {current_batch_size} to {new_size} (success rate: {previous_success_rate:.1%})")
        return new_size
    elif previous_success_rate > 0.7:
        print(f"📊 Keeping batch size at {current_batch_size} (success rate: {previous_success_rate:.1%})")
        return current_batch_size  # Keep current
    else:
        new_size = max(4, current_batch_size - 5)  # Decrease on failure
        print(f"📉 Decreasing batch size from {current_batch_size} to {new_size} (success rate: {previous_success_rate:.1%})")
        return new_size    
    

# Global cost tracker
cost_tracker = CostTracker()

# ---------------------------------------------------------
# 1. SYSTEM PROMPT (OPTIMIZED FOR BATCHING)
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an annotation model. Classify each pool review with this schema:

1. social_encounter: "yes" or "no" (involves other people at the pool)
2. racial_connotations: "yes" or "no" (references race/ethnicity/religion/culture)  
3. connotation_type: "discriminatory", "pleasant", "ambiguous", or "N/A"

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
# 2. CORE FUNCTIONS (SAME AS BEFORE)
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
        print(f"🚨 BUDGET EXCEEDED: Skipping batch {batch_id} (${cost_tracker.total_cost:.2f} > ${cost_tracker.budget_limit:.2f})")
        return {
            "success": [],
            "failed": [{"review_id": rid, "error": "budget_exceeded"} for rid, _ in batch_data]
        }
    elif budget_status == "WARNING":
        print(f"⚠️  BUDGET WARNING: ${cost_tracker.total_cost:.2f} of ${cost_tracker.budget_limit:.2f}")
    
    
    print(f"📦 Batch {batch_id}: Processing {len(batch_data)} reviews")
    
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
                print(f"✅ Batch {batch_id}: All {len(batch_data)} reviews processed in {elapsed:.1f}s")
                
                # Update cost tracking
                batch_cost = cost_tracker.update_batch(len(batch_data), len(batch_data))
                print(f"💰 Batch cost: ${batch_cost:.4f}")
                cost_tracker.print_progress()
                
                return {
                    "success": batch_results,
                    "failed": []
                }
            else:
                # Partial success - try to salvage
                print(f"⚠️ Batch {batch_id}: Partial success, salvaging...")
                result = salvage_batch_results(batch_data, content, batch_id)
                
                # Update cost tracking for partial success
                success_count = len(result["success"])
                batch_cost = cost_tracker.update_batch(len(batch_data), success_count)
                print(f"💰 Batch cost: ${batch_cost:.4f} ({success_count}/{len(batch_data)} success)")
                cost_tracker.print_progress()
                
                return result
                
        elif r.status_code == 429:
            # Rate limited
            wait_time = random.uniform(10, 30)
            print(f"⏳ Batch {batch_id}: Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            return batch_classify_reviews(batch_data, batch_id)  # Retry
            
        else:
            # API error
            print(f"❌ Batch {batch_id}: API error {r.status_code}")
            
            # Still count the cost for failed batch (we used tokens for the request)
            batch_cost = cost_tracker.update_batch(len(batch_data), 0)
            print(f"💰 Batch cost: ${batch_cost:.4f} (all failed)")
            cost_tracker.print_progress()
            
            return {
                "success": [],
                "failed": [{"review_id": rid, "error": f"api_error_{r.status_code}"} for rid, _ in batch_data]
            }
            
    except requests.exceptions.Timeout:
        print(f"⏰ Batch {batch_id}: Timeout after 120s")
        
        # Count cost for timed out batch
        batch_cost = cost_tracker.update_batch(len(batch_data), 0)
        print(f"💰 Batch cost: ${batch_cost:.4f} (timeout)")
        cost_tracker.print_progress()
        
        return {
            "success": [],
            "failed": [{"review_id": rid, "error": "timeout"} for rid, _ in batch_data]
        }
    except Exception as e:
        print(f"💥 Batch {batch_id}: Exception {e}")
        
        # Count cost for crashed batch
        batch_cost = cost_tracker.update_batch(len(batch_data), 0)
        print(f"💰 Batch cost: ${batch_cost:.4f} (exception)")
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
    
    print(f"🔍 Batch {batch_id}: Found {len(extracted_results)} valid objects")
    
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
    
    print(f"📊 Batch {batch_id} salvaged: {len(success_results)} success, {len(failed_reviews)} failed")
    return {
        "success": success_results,
        "failed": failed_reviews
    }

def retry_failed_reviews_individual(failed_batch_data):
    """Retry failed reviews one by one with progress bar"""
    print(f"🔄 Retrying {len(failed_batch_data)} failed reviews individually...")
    
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

def save_chunk_checkpoint(chunk_df, chunk_success, chunk_failed, output_dir, chunk_file, checkpoint_interval=50):
    """Save checkpoint with all processed results so far"""
    # Create output with current progress
    minimal_output = pd.DataFrame({
        'review_id': chunk_df['review_id'],
        'was_summarized': chunk_df.get('was_summarized', False)
    })
    
    # Add classification results for processed reviews
    if chunk_success:
        classification_map = {row['review_id']: row for row in chunk_success}
        
        for col in ['social_encounter', 'racial_connotations', 'connotation_type']:
            minimal_output[col] = minimal_output['review_id'].map(
                lambda x: classification_map.get(x, {}).get(col, 'NOT_PROCESSED')
            )
    else:
        # No successful classifications yet
        for col in ['social_encounter', 'racial_connotations', 'connotation_type']:
            minimal_output[col] = 'NOT_PROCESSED'
    
    # Mark status
    processed_ids = {success['review_id'] for success in chunk_success} | {failed['review_id'] for failed in chunk_failed}
    minimal_output['classification_status'] = minimal_output['review_id'].map(
        lambda x: 'PROCESSED' if x in processed_ids else 'PENDING'
    )
    
    # Save checkpoint (overwrites previous checkpoint)
    checkpoint_file = output_dir / f"checkpoint_{chunk_file.name}"
    minimal_output.to_parquet(checkpoint_file)
    
    return len(processed_ids)

def extract_results_from_dataframe(df):
    """Extract success/failure results from checkpoint dataframe"""
    success_results = []
    failed_reviews = []
    
    processed_mask = df['classification_status'] == 'PROCESSED'
    processed_df = df[processed_mask]
    
    for _, row in processed_df.iterrows():
        if row['social_encounter'] != 'NOT_PROCESSED':
            success_results.append({
                'review_id': row['review_id'],
                'social_encounter': row['social_encounter'],
                'racial_connotations': row['racial_connotations'],
                'connotation_type': row['connotation_type']
            })
        else:
            failed_reviews.append({
                'review_id': row['review_id'],
                'error': 'processing_failed'
            })
    
    return success_results, failed_reviews

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
# 3. CHUNK-BASED PROCESSING ENGINE
# ---------------------------------------------------------
def process_chunk(chunk_file, output_dir, batch_size=20, parallel_batches=2):
    """Process a single chunk file with periodic checkpointing"""
    chunk_id = chunk_file.stem.replace("chunk_", "")
    print(f"\n📦 Processing chunk {chunk_id}: {chunk_file.name}")
    
    # Load chunk data
    chunk_df = pd.read_parquet(chunk_file)
    
    # USE ONLY SUMMARIZED TEXT
    if 'summary' in chunk_df.columns and chunk_df['summary'].notna().any():
        text_column = 'summary'
        summarized_count = chunk_df['summary'].notna().sum()
        print(f"   📝 Using summarized text ({summarized_count}/{len(chunk_df)} summaries available)")
    else:
        print(f"   ⚠️ No summarized text found in chunk {chunk_id}, skipping classification")
        return None, None
    
    # Prepare review data - ONLY use summarized text
    review_data = [
        (row["review_id"], row[text_column]) 
        for _, row in chunk_df.iterrows() 
        if pd.notna(row[text_column]) and str(row[text_column]).strip()
    ]
    
    if not review_data:
        print(f"   ⚠️ No valid summarized reviews in chunk {chunk_id}")
        return None, None
    
    print(f"   📊 {len(review_data)} summarized reviews to classify")
    # Load existing progress from checkpoint if any
    
    checkpoint_file = output_dir / f"checkpoint_{chunk_file.name}"
    final_file = output_dir / f"classified_{chunk_file.name}"
    chunk_success = []
    chunk_failed = []
    processed_batches = set()

    # Check if chunk is already completed
    if final_file.exists():
        print(f"   ✅ Chunk {chunk_id} already completed, skipping...")
        return extract_results_from_dataframe(pd.read_parquet(final_file))

    # Check if we have a checkpoint to resume from
    elif checkpoint_file.exists():
        print(f"   🔄 Found checkpoint, loading previous progress...")
        checkpoint_df = pd.read_parquet(checkpoint_file)
        
        # Extract already processed results from checkpoint
        processed_mask = checkpoint_df['classification_status'] == 'PROCESSED'
        processed_reviews = checkpoint_df[processed_mask]
        
        processed_count = len(processed_reviews)
        print(f"   📊 Checkpoint shows {processed_count} reviews already processed")
        print(f"   ⚠️  Will reprocess from last checkpoint position")
        
        # Calculate where to resume
        batches_to_skip = processed_count // batch_size
        processed_batches = set(range(batches_to_skip))
        
    # Create batches
    batches = [review_data[i:i + batch_size] for i in range(0, len(review_data), batch_size)]
    
    # Load existing progress from checkpoint if any
    checkpoint_file = output_dir / f"checkpoint_{chunk_file.name}"
    chunk_success = []
    chunk_failed = []
    processed_batches = set()
    
    if checkpoint_file.exists():
        print(f"   🔄 Found checkpoint, loading previous progress...")
        checkpoint_df = pd.read_parquet(checkpoint_file)
        
        # Extract already processed results from checkpoint
        processed_mask = checkpoint_df['classification_status'] == 'PROCESSED'
        processed_reviews = checkpoint_df[processed_mask]
        
        # We can't fully recover the original classification results from checkpoint alone
        # So we'll note the progress but need to reprocess from last checkpoint
        processed_count = len(processed_reviews)
        print(f"   📊 Checkpoint shows {processed_count} reviews already processed")
        print(f"   ⚠️  Will reprocess from last checkpoint position")
        
        # Calculate where to resume
        batches_to_skip = processed_count // batch_size
        processed_batches = set(range(batches_to_skip))
    
    # Track batches to process
    batches_to_process = [i for i in range(len(batches)) if i not in processed_batches]
    
    if not batches_to_process:
        print(f"   ✅ All batches already processed for chunk {chunk_id}")
        # Load final results from checkpoint
        final_output = pd.read_parquet(checkpoint_file)
        output_file = output_dir / f"classified_{chunk_file.name}"
        final_output.to_parquet(output_file)
        print(f"💾 Saved final chunk: {output_file.name}")
        checkpoint_file.unlink()  # Clean up checkpoint
        return extract_results_from_dataframe(final_output)
    
    print(f"   🎯 Processing {len(batches_to_process)} batches ({len(batches_to_process) * batch_size} reviews)")
    
    # Process batches with periodic checkpointing
    total_processed = len(processed_batches) * batch_size
    for batch_idx in tqdm(batches_to_process, desc=f"Chunk {chunk_id}"):
        batch_data = batches[batch_idx]
        batch_id = f"{chunk_id}_{batch_idx}"
        
        try:
            result = batch_classify_reviews(batch_data, batch_id)
            chunk_success.extend(result["success"])
            chunk_failed.extend(result["failed"])
            
            processed_batches.add(batch_idx)
            total_processed += len(batch_data)
            
            # ✅ PERIODIC CHECKPOINT (every 100 reviews)
            if total_processed % 100 == 0:
                processed_count = save_chunk_checkpoint(
                    chunk_df, chunk_success, chunk_failed, output_dir, chunk_file
                )
                print(f"💾 Checkpoint saved: {processed_count}/{len(review_data)} reviews processed")
            
            # 🧹 CLEAR MEMORY AFTER EACH BATCH
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                    
        except Exception as e:
            print(f"💥 Batch {batch_id} crashed: {e}")
            chunk_failed.extend([{"review_id": rid, "error": "batch_crash"} for rid, _ in batch_data])
    
    # Individual retry for failed reviews
    if chunk_failed:
        print(f"🔄 Retrying {len(chunk_failed)} failed reviews in chunk {chunk_id}...")
        failed_review_data = []
        for failed in chunk_failed:
            for rid, text in review_data:
                if rid == failed["review_id"]:
                    failed_review_data.append((rid, text))
                    break
        
        retry_success, retry_failed = retry_failed_reviews_individual(failed_review_data)
        chunk_success.extend(retry_success)
        chunk_failed = retry_failed
    
    # Save final chunk results
    if chunk_success or chunk_failed:
        # Use checkpoint function to create final output
        processed_count = save_chunk_checkpoint(chunk_df, chunk_success, chunk_failed, output_dir, chunk_file)
        
        # Rename checkpoint to final file
        checkpoint_file = output_dir / f"checkpoint_{chunk_file.name}"
        final_file = output_dir / f"classified_{chunk_file.name}"
        
        if checkpoint_file.exists():
            checkpoint_file.rename(final_file)
        else:
            # If no checkpoint exists (unlikely), create final file directly
            final_output = create_final_output(chunk_df, chunk_success, chunk_failed)
            final_output.to_parquet(final_file)
        
        print(f"💾 Saved final chunk: {final_file.name}")
        
        # 🗑️ DELETE CHECKPOINT FILE AFTER SUCCESSFUL COMPLETION
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"🧹 Cleaned up checkpoint for chunk {chunk_id}")
        
        # 🧹 FINAL MEMORY CLEANUP
        del chunk_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Save chunk summary
        success_rate = len(chunk_success) / len(review_data) if review_data else 0
        print(f"✅ Chunk {chunk_id} complete: {len(chunk_success)} success, {len(chunk_failed)} failed ({success_rate:.1%})")
        
        return chunk_success, chunk_failed
    
    return [], []

def main():
    print("🚀 INCREMENTAL CHUNK CLASSIFIER - Starting...")
    
    # Configuration - START SMALL
    BATCH_SIZE = 20
    PARALLEL_BATCHES = 2
    BUDGET_LIMIT = 1200.0  # ⬅️ ADD THIS
    
    # Initialize cost tracker with budget
    global cost_tracker
    cost_tracker = CostTracker(budget_limit=BUDGET_LIMIT)  # ⬅️ ADD THIS
    
    print(f"💰 BUDGET SET: ${BUDGET_LIMIT:.2f} (will stop if exceeded)")
    
    # Track success rates for adaptive sizing
    chunk_success_rates = []
    
    # File paths - MODIFIED TO USE SUMMARIZATION CHECKPOINT FOLDER
    summarization_checkpoint_dir = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/text_reviews_summarized_25112025_checkpoints_992ed0a2")
    classification_output_dir = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/classified_chunks/")
    final_output_path = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/cs_srt_labeled_26112025.csv")
    failed_path = final_output_path.with_suffix("").with_name(final_output_path.stem + "_failed.csv")
    progress_file = classification_output_dir / "classification_progress.json"
    
    # Create output directory
    classification_output_dir.mkdir(exist_ok=True)
    
    # Print initial cost estimate
    print(f"💰 INITIAL COST ESTIMATE: ~$1,500 for 1.1M summarized reviews")
    print(f"   (Based on Claude-3.5-Sonnet: $3/M input + $15/M output tokens)")
    
    # Load progress
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        processed_chunks = set(progress.get("processed_chunks", []))
        all_success = progress.get("all_success", [])
        all_failed = progress.get("all_failed", [])
        BATCH_SIZE = progress.get("current_batch_size", 20)  # ⬅️ RESUME BATCH SIZE
        print(f"🔄 Resuming: {len(processed_chunks)} chunks already processed")
        print(f"📦 Resuming with batch size: {BATCH_SIZE}")
        
        # Initialize cost tracker with previous progress
        cost_tracker.total_reviews_processed = len(all_success)
        cost_tracker.total_cost = progress.get("total_cost", 0.0)
    else:
        processed_chunks = set()
        all_success = []
        all_failed = []
        print("🆕 Starting fresh classification with batch size 8")
    
    # Monitor for new chunks
    print(f"🔍 Monitoring {summarization_checkpoint_dir} for new chunks...")
    
    while True:
        # Find all chunk files
        chunk_files = list(summarization_checkpoint_dir.glob("chunk_*.parquet"))
        unprocessed_chunks = [f for f in chunk_files if f.stem not in processed_chunks]
        
        if not unprocessed_chunks:
            print("⏳ No new chunks found. Waiting 60 seconds...")
            time.sleep(60)
            continue
        
        # Process new chunks
        print(f"🎯 Found {len(unprocessed_chunks)} new chunks to process")
        
        for chunk_file in sorted(unprocessed_chunks, key=lambda x: int(x.stem.replace("chunk_", ""))):
            chunk_id = chunk_file.stem
            
            try:
                chunk_success, chunk_failed = process_chunk(
                    chunk_file, classification_output_dir, BATCH_SIZE, PARALLEL_BATCHES
                )
                
                if chunk_success is not None:  # Chunk was processed
                    # Update results
                    all_success.extend(chunk_success)
                    all_failed.extend(chunk_failed)
                    processed_chunks.add(chunk_id)
                    
                    # Calculate chunk success rate for adaptive sizing
                    chunk_review_count = len(chunk_success) + len(chunk_failed)
                    if chunk_review_count > 0:
                        chunk_success_rate = len(chunk_success) / chunk_review_count
                        chunk_success_rates.append(chunk_success_rate)
                        
                        # Keep only last 3 chunks for moving average
                        if len(chunk_success_rates) > 3:
                            chunk_success_rates.pop(0)
                        
                        # Calculate average success rate
                        avg_success_rate = sum(chunk_success_rates) / len(chunk_success_rates)
                        
                        # Adapt batch size based on performance
                        BATCH_SIZE = adaptive_batch_size(BATCH_SIZE, avg_success_rate)
                    
                    # Save progress with cost information
                    progress = {
                        "processed_chunks": list(processed_chunks),
                        "all_success": all_success,
                        "all_failed": all_failed,
                        "total_cost": cost_tracker.total_cost,
                        "total_reviews_processed": cost_tracker.total_reviews_processed,
                        "current_batch_size": BATCH_SIZE,
                        "last_update": time.time()
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
                    
                    print(f"💾 Progress saved: {len(processed_chunks)} chunks processed")
                    
                    # Clean memory
                    gc.collect()
                
            except Exception as e:
                print(f"💥 Failed to process chunk {chunk_id}: {e}")
                continue
            
            # Small break between chunks
            time.sleep(random.uniform(2, 5))
        
        # Check if summarization is complete (progress.json doesn't exist = done)
        progress_json = summarization_checkpoint_dir / "progress.json"
        if not progress_json.exists():
            print("🎉 Summarization appears complete! Processing final chunks...")
            
            # One final check for any remaining chunks
            final_chunks = list(summarization_checkpoint_dir.glob("chunk_*.parquet"))
            remaining_chunks = [f for f in final_chunks if f.stem not in processed_chunks]
            
            if remaining_chunks:
                print(f"🔍 Found {len(remaining_chunks)} final chunks")
                continue
            else:
                print("✅ All chunks processed! Generating final output...")
                break
        
        print("🔄 Waiting for more chunks...")
        time.sleep(30)  # Check every 30 seconds for new chunks
    
    # Combine all classified chunks into final output
    print("🔗 Combining all classified chunks...")
    classified_chunks = list(classification_output_dir.glob("classified_chunk_*.parquet"))
    
    if classified_chunks:
        final_df = pd.concat([pd.read_parquet(f) for f in classified_chunks], ignore_index=True)
        final_df.to_csv(final_output_path, index=False)
        print(f"💾 Final output saved: {final_output_path}")
        
        # Save failed reviews
        if all_failed:
            failed_df = pd.DataFrame(all_failed)
            failed_df.to_csv(failed_path, index=False)
            print(f"📄 Failed reviews saved: {failed_path}")
        
        # Cleanup
        if progress_file.exists():
            progress_file.unlink()
        
        # Final summary with cost
        total_processed = len(all_success) + len(all_failed)
        success_rate = len(all_success) / total_processed if total_processed > 0 else 0
        cost_summary = cost_tracker.get_cost_summary()
        
        print(f"\n🎉 CLASSIFICATION COMPLETE!")
        print(f"✅ Successful: {len(all_success):,}")
        print(f"❌ Failed: {len(all_failed):,}")
        print(f"📊 Success rate: {success_rate:.1%}")
        print(f"📁 Classified chunks: {len(classified_chunks)}")
        print(f"💰 TOTAL COST: ${cost_tracker.total_cost:.2f}")
        print(f"   Cost per review: ${cost_summary['cost_per_review']:.4f}")
        print(f"   Reviews per hour: {cost_summary['reviews_per_hour']:.0f}")
        print(f"   Total time: {cost_summary['elapsed_hours']:.1f} hours")
    else:
        print("❌ No classified chunks found!")

if __name__ == "__main__":
    main()