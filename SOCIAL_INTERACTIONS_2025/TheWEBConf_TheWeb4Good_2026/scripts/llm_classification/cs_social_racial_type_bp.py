import pandas as pd
import json
import time
import os
import requests
import re
import random
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"API key loaded successfully")

MODEL = "anthropic/claude-3.5-sonnet"

# ---------------------------------------------------------
# 1. SYSTEM INSTRUCTIONS (SENT ONCE)
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an annotation model. Classify each pool review with the following schema:

1. social_encounter: "yes" or "no"
2. racial_connotations: "yes" or "no"
3. connotation_type: "discriminatory", "pleasant", "ambiguous", or "N/A"

Rules:
- "social_encounter" = yes if the review involves other people at the pool.
- "racial_connotations" = yes if it references groups (race/ethnicity/religion/culture).
- If racial_connotations = no → connotation_type = "N/A".

Return ONLY JSON. Example output:
{
  "social_encounter": "yes",
  "racial_connotations": "no",
  "connotation_type": "N/A"
}
"""

# ---------------------------------------------------------
# 2. JSON EXTRACTION
# ---------------------------------------------------------

def extract_json_from_response(content):
    """Extract JSON from various response formats"""
    # Try to parse directly first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all else fails, return error
    print(f"Could not extract JSON from response: {content}")
    return {"social_encounter": "error", "racial_connotations": "error", "connotation_type": "error"}

# ---------------------------------------------------------
# 3. CLASSIFICATION FUNCTION (THREAD-SAFE)
# ---------------------------------------------------------

def classify_single_review(review_data):
    """Send only the review text. System prompt is reused."""
    review_id, review_text = review_data
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review_text}
    ]
    request_id = random.randint(1,1000000)
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": f"http://localhost:{request_id}",
                "X-Title": "Review Classifier"
            },
            json={
                "model": MODEL,
                "messages": msg,
                "temperature": 0.0
            },
            timeout=120
        )

        if r.status_code != 200:
            if r.status_code == 429:  # Rate limit error
                print(f"Rate limit exceeded for {MODEL}. Waiting about 10 seconds...")
                time.sleep(random.uniform(2, 5))  # Randomized wait to avoid repeated hits
                return classify_single_review((review_id, review_text))  # Retry
            print(f"API Error {r.status_code}: {r.text[:200]}")
            return {"review_id": review_id, "status": "api_error", "error_code": r.status_code}
        
        content = r.json()["choices"][0]["message"]["content"]
        json_result = extract_json_from_response(content)
        
        if json_result is None:
            return {"review_id": review_id, "status": "parse_error", "raw_response": content[:500]}
            
        json_result["review_id"] = review_id
        json_result["status"] = "success"
        return json_result

    except Exception as e:
        print(f"Request Exception for review {review_id}: {e}")
        return {"review_id": review_id, "status": "request_error", "exception": str(e)}

# ---------------------------------------------------------
# 4. PARALLEL PROCESSING WITH THREADPOOL
# ---------------------------------------------------------

def process_batch(review_batch, max_workers=5):
    """Process a batch of reviews in parallel"""
    results = []
    failed_reviews = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_review = {
            executor.submit(classify_single_review, (review_id, review_text)): review_id 
            for review_id, review_text in review_batch
        }
        
        # Process completed tasks as they finish
        for future in tqdm(as_completed(future_to_review), total=len(review_batch), desc="Processing batch"):
            try:
                result = future.result()
                if result.get("status") == "success":
                    print(f"Successfully processed review {result['review_id']}")
                    results.append(result)
                else:
                    print(f"Failed to process review {result['review_id']}: {result['status']}")
                    failed_reviews.append({
                        "review_id": result["review_id"],
                        "trans_review": next(text for rid, text in review_batch if rid == result["review_id"]),
                        "error_type": result["status"],
                        "error_details": str(result.get("error_code", result.get("exception", result.get("raw_response", "")))),
                        "timestamp": time.time()
                    })
            except Exception as e:
                review_id = future_to_review[future]
                print(f"Unexpected error processing review {review_id}: {e}")
                failed_reviews.append({
                    "review_id": review_id,
                    "trans_review": next(text for rid, text in review_batch if rid == review_id),
                    "error_type": "unexpected_error",
                    "error_details": str(e),
                    "timestamp": time.time()
                })
    
    return results, failed_reviews

def main():
    input_df = pd.read_parquet(
        "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/collected_reviews_25112025.parquet"
    )

    input_df = input_df.dropna(subset=["trans_review"]).reset_index(drop=True)
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)
    #input_df = input_df.head(30)  # For testing, limit to first 200 reviews
    print(f"Loaded {len(input_df)} reviews, shuffled for processing")

    output_path = "/home/ethel/Desktop/WEBCONF_25/llm_processing_25112025/cs_srt_classification_26112025.csv"
    checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")
    failed_reviews_path = output_path.replace(".csv", "_failed_reviews.csv")

    # Configuration for parallel processing
    BATCH_SIZE = 100 # Process 100 reviews at a time
    MAX_WORKERS = 5   # Number of concurrent API calls (adjust based on rate limits)
    
    results = []
    failed_reviews = []
    processed_ids = set()

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        cp = pd.read_csv(checkpoint_path)
        results = cp.to_dict("records")
        processed_ids = set(cp["review_id"].tolist())
        print(f"Resuming from checkpoint with {len(processed_ids)} completed reviews.")

    # Load failed reviews if exists
    if os.path.exists(failed_reviews_path):
        failed_df = pd.read_csv(failed_reviews_path)
        failed_reviews = failed_df.to_dict("records")
        print(f"Loaded {len(failed_reviews)} previously failed reviews.")

    # Prepare unprocessed reviews
    unprocessed_reviews = [
        (row["review_id"], row["trans_review"]) 
        for _, row in input_df.iterrows() 
        if row["review_id"] not in processed_ids
    ]
    
    print(f"Processing {len(unprocessed_reviews)} unprocessed reviews in parallel...")

    # Process in batches
    total_batches = (len(unprocessed_reviews) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(unprocessed_reviews))
        batch = unprocessed_reviews[start_idx:end_idx]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} reviews) with {MAX_WORKERS} workers...")
        
        batch_results, batch_failed = process_batch(batch, MAX_WORKERS)
        
        results.extend(batch_results)
        failed_reviews.extend(batch_failed)
        processed_ids.update([r["review_id"] for r in batch_results])
        processed_ids.update([f["review_id"] for f in batch_failed])

        # Save progress after each batch
        if results:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        if failed_reviews:
            pd.DataFrame(failed_reviews).to_csv(failed_reviews_path, index=False)
            
        print(f"Batch {batch_num + 1} complete: {len(batch_results)} success, {len(batch_failed)} failed")
        
        # Add delay between batches to avoid overwhelming the API
        if batch_num < total_batches - 1:  # Don't sleep after the last batch
            sleep_time = random.uniform(2, 5)
            print(f"Waiting {sleep_time:.1f} seconds before next batch...")
            time.sleep(sleep_time)

    # Final saves
    if results:
        final = input_df.merge(pd.DataFrame(results), on="review_id")
        final.to_csv(output_path, index=False)
        print(f"Successfully processed {len(results)} reviews.")

    if failed_reviews:
        pd.DataFrame(failed_reviews).to_csv(failed_reviews_path, index=False)
        print(f"Failed to process {len(failed_reviews)} reviews. Saved to {failed_reviews_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up.")

    # Summary
    success_count = len(results) if results else 0
    failure_count = len(failed_reviews) if failed_reviews else 0
    print(f"\nProcessing complete:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failure_count}")
    if success_count + failure_count > 0:
        print(f"📊 Success rate: {success_count/(success_count + failure_count)*100:.1f}%")

if __name__ == "__main__":
    main()