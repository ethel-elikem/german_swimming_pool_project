import pandas as pd
import json
import time
import os
import requests
import re
import random
from dotenv import load_dotenv
from tqdm import tqdm

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
# 3. CLASSIFICATION FUNCTION (NO RETRIES)
# ---------------------------------------------------------

def classify_single_review(review_text):
    """Send only the review text. System prompt is reused."""

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
            }
        )

        if r.status_code != 200:
            if r.status_code == 429:  # Rate limit error
                print(f"Rate limit exceeded for {MODEL}. Waiting  about 10 seconds...")
                time.sleep(random.uniform(2, 10))  # Randomized wait to avoid repeated hits
                return classify_single_review(review_text)  # Retry
            print(f"API Error {r.status_code}: {r.text[:200]}")
            return {"status": "api_error", "error_code": r.status_code}
        
    
        content = r.json()["choices"][0]["message"]["content"]
        json_result = extract_json_from_response(content)
        
        if json_result is None:
            return {"status": "parse_error", "raw_response": content[:500]}
            
        return json_result

    except Exception as e:
        print(f"Request Exception: {e}")
        return {"status": "request_error", "exception": str(e)}


# ---------------------------------------------------------
# 4. MAIN PROCESSING LOOP
# ---------------------------------------------------------

def main():
    input_df = pd.read_parquet(
        "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/collected_reviews_25112025.parquet"
    )

    input_df = input_df.dropna(subset=["trans_review"]).reset_index(drop=True)
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(input_df)} reviews, shuffled for processing")

    output_path = "/home/ethel/Desktop/WEBCONF_25/llm_processing_25112025/cs_srt_classification_26112025.csv"
    checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")
    failed_reviews_path = output_path.replace(".csv", "_failed_reviews.csv")

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

    with tqdm(total=len(input_df), initial=len(processed_ids)) as pbar:
        for _, row in input_df.iterrows():
            rid = row["review_id"]
            if rid in processed_ids:
                continue

            review = row["trans_review"]
            res = classify_single_review(review)

            # Check if the response is an error
            if "status" in res and res["status"] != "success":
                # This is a failed request
                failed_entry = {
                    "review_id": rid,
                    "trans_review": review,
                    "error_type": res["status"],
                    "error_details": str(res.get("error_code", res.get("exception", res.get("raw_response", "")))),
                    "timestamp": time.time()
                }
                failed_reviews.append(failed_entry)
                print(f"Failed review {rid}: {res['status']}")
                
            else:
                # This is a successful classification
                res["review_id"] = rid
                res["status"] = "success"
                results.append(res)
                processed_ids.add(rid)

            # Save progress every 50 successful reviews OR every 10 failed reviews
            if len(results) % 50 == 0 or len(failed_reviews) % 10 == 0:
                if results:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                if failed_reviews:
                    pd.DataFrame(failed_reviews).to_csv(failed_reviews_path, index=False)

            pbar.update(1)
            #ramdomize sleep to avoid rate limits   
            time.sleep(random.uniform(2, 5))  # Randomized rate limiting between 2 and 5  seconds

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
    print(f"📊 Success rate: {success_count/(success_count + failure_count)*100:.1f}%")


if __name__ == "__main__":
    main()