import pandas as pd
import json
import time
import os
import requests
import re
from dotenv import load_dotenv
from tqdm import tqdm
import random

# Load API key from .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"API key loaded successfully")

# Define the models you want to use with their OpenRouter identifiers
MODEL = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.5-pro",  # Top Gemini model
    "google/gemini-2.5-flash",    # Second top Gemini model
    "deepseek/deepseek-chat",   # Top DeepSeek model
    "deepseek/deepseek-r1-0528",     # Second top DeepSeek model
]
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

def classify_review(review_text, model_name):
    """Classify a review using the OpenRouter API with the specified model"""
    
    prompt = f"""
    Classify the following pool review according to the same criteria that human annotators used.

    ---

    ### Step 1: Identify Social Encounter / Interaction
    Check whether the review partly or wholly describes a **social encounter or interaction** — an opinion, experience, or observation involving **others** (staff or clients) at the pool.

    If not, output:
    {{"social_encounter": "no"}}

    **Examples (no social encounter):**
    - "The entrance fee is too expensive."
    - "There was mold everywhere and the toilets were very dirty."
    - "A great spot to relax and enjoy the sun."

    **Examples (social encounter):**
    - "The receptionist was very friendly."
    - "There were too many people, I could not swim."
    - "There were a lot of migrants there."

    ---

    ### Step 2: Racial/Ethnic/Religious Connotation
    If a social encounter exists, check whether it contains **racial, ethnic, cultural, or religious connotations** — explicit or implicit references, attitudes, or observations about groups.

    If none, output:
    {{"social_encounter": "yes", "racial_connotations": "no"}}

    **Examples (racial/ethnic/religious connotation = yes):**
    - "There were a lot of migrants present."
    - "It is a great pool, no one looked at me strangely for wearing a burkini."
    - "It feels like I was in a mosque! Why are people allowed into the pool with full clothing?"
    - "The diversity and multiculturalism at this pool is beautiful!"

    **Examples (no racial connotation):**
    - "The pool is not disability friendly."
    - "People just don’t know how to behave."
    - "The clientele is very youthful."

    ---

    ### Step 3: Subclassification (if racial/ethnic/religious connotation = yes)
    If the review includes such connotations, classify it as one of the following:

    1. **Discriminatory** — the review describes discrimination or uses derogatory or exclusionary language.
    **Examples:**
    - "I was yelled at by the lifeguard to go change my swimwear or leave the pool because I was in a burkini."
    - "Too many Arabs who do not know how to behave."
    - "I felt uncomfortable and discriminated against because I was in a bikini, people stared at me so much I had to leave."

    2. **Pleasant** — the review expresses positive or welcoming experiences regarding race/ethnicity/religion.
    **Example:**
    - "It is a great pool, no one looked at me strangely for wearing a burkini."

    3. **Ambiguous** — the review refers to race/ethnicity/religion but is neutral, observational, or unclear in sentiment.
    **Examples:**
    - "There were a lot of migrants present."
    - "The employees barely speak any German."

    ---

    ### Output Format (JSON only)
    Return **only** valid JSON, in this exact structure:

    {{
    "social_encounter": "yes|no",
    "racial_connotations": "yes|no",
    "connotation_type": "discriminatory|pleasant|ambiguous|N/A"
    }}

    ---

    ### Review to classify: "{review_text}"

    IMPORTANT: Return ONLY valid JSON in this exact format:
    {{"social_encounter": "yes|no", "racial_connotations": "yes|no", "connotation_type": "discriminatory|pleasant|ambiguous|N/A"}}

    Do not include any additional text, explanations, or code blocks.
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Review Classifier"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
        )
        
        # Check if request was successful
        if response.status_code != 200:
            if response.status_code == 429:  # Rate limit error
                print(f"Rate limit exceeded for {model_name}. Waiting 10 seconds...")
                time.sleep(10)
                return classify_review(review_text, model_name)  # Retry
            print(f"API Error: {response.status_code} - {response.text}")
            return {"social_encounter": "error", "racial_connotations": "error", "connotation_type": "error"}
        
        result = response.json()
        
        # Extract content
        content = result['choices'][0]['message']['content'].strip()
        
        # Extract JSON from response
        return extract_json_from_response(content)
                    
    except Exception as e:
        print(f"Error classifying review with {model_name}: {e}")
        return {"social_encounter": "error", "racial_connotations": "error", "connotation_type": "error"}

# def main():
#     Read input CSV
#     input_df = pd.read_csv("/home/ethel/Documents/RESEARCH/swimming_pools_data_16gsc/annotations_12092025/100_allsc_swimreviews_12092025.csv")
    
#     Shuffle the dataset
#     input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     print(f"Loaded {len(input_df)} reviews, shuffled for processing")
    
#     Create safe filename for model
#     safe_model_name = MODEL.replace("/", "_").replace(":", "_")
#     output_path = f"/home/ethel/Documents/RESEARCH/swimming_pools_data_16gsc/fs_annotations_13102025/100_swimreviews_{safe_model_name}_labels.csv"
    
#     Check if we have existing progress
#     checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")
#     start_index = 0
#     results = []
    
#     if os.path.exists(checkpoint_path):
#         print(f"Found checkpoint file: {checkpoint_path}")
#         checkpoint_df = pd.read_csv(checkpoint_path)
#         start_index = len(checkpoint_df)
#         results = checkpoint_df.to_dict('records')
#         print(f"Resuming from index {start_index}")
    
#     Process reviews with progress bar
#     with tqdm(total=len(input_df), initial=start_index, desc="Classifying reviews") as pbar:
#         for index in range(start_index, len(input_df)):
#             row = input_df.iloc[index]
#             review_id = row['review_id']
#             review_text = row['trans_review']
        
#             Classify review
#             classification = classify_review(review_text, MODEL)
            
#             classification['review_id'] = review_id
#             classification['model'] = MODEL
#             results.append(classification)
            
#             Update progress bar
#             pbar.update(1)
#             pbar.set_postfix({"Last ID": review_id})
            
#             Save checkpoint every 50 classifications
#             if (index + 1) % 50 == 0:
#                 Convert current results to DataFrame and save
#                 results_df = pd.DataFrame(results)
#                 results_df.to_csv(checkpoint_path, index=False)
#                 print(f"\nCheckpoint saved at index {index + 1}")
            
#             Add delay to respect API rate limits
#             time.sleep(1)
    
#     Final save
#     results_df = pd.DataFrame(results)
#     output_df = input_df.merge(results_df, on='review_id')
#     output_df.to_csv(output_path, index=False)
    
#     Remove checkpoint file after successful completion
#     if os.path.exists(checkpoint_path):
#         os.remove(checkpoint_path)
    
#     print(f"Classification complete. Results saved to {output_path}")


def main():
    # Read input CSV
    input_df = pd.read_csv(
        "/home/ethel/Documents/RESEARCH/swimming_pools_data_16gsc/annotations_12092025/100_allsc_swimreviews_12092025.csv"
    )

    # Shuffle the dataset
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(input_df)} reviews, shuffled for processing")

    for model in MODEL:  # MODEL should be a list of model names
        safe_model_name = model.replace("/", "_").replace(":", "_")
        output_path = (
            f"/home/ethel/Documents/RESEARCH/swimming_pools_data_16gsc/fs_annotations_13102025/allsc_swimreviews_fs_{safe_model_name}_labels.csv"
        )
        checkpoint_path = output_path.replace(".csv", "_checkpoint.csv")

        results = []

        # Track already processed IDs
        processed_ids = set()
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint file: {checkpoint_path}")
            checkpoint_df = pd.read_csv(checkpoint_path)
            results = checkpoint_df.to_dict("records")
            processed_ids = set(checkpoint_df["review_id"].tolist())
            print(f"Resuming, {len(processed_ids)} reviews already processed")

        with tqdm(total=len(input_df), initial=len(processed_ids), desc=f"Classifying with {model}") as pbar:
            for _, row in input_df.iterrows():
                review_id = row["review_id"]

                # Skip if already done
                if review_id in processed_ids:
                    continue

                review_text = row["trans_review"]

                # Classify with current model
                classification = classify_review(review_text, model)
                classification["review_id"] = review_id
                classification["model"] = model
                results.append(classification)
                processed_ids.add(review_id)

                pbar.update(1)
                pbar.set_postfix({"Last ID": review_id})

                # Save checkpoint every 50
                if len(results) % 50 == 0:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    print(f"\nCheckpoint saved with {len(results)} records")

                time.sleep(1)  # adjustable rate limit

        # Final save
        results_df = pd.DataFrame(results)
        output_df = input_df.merge(results_df, on="review_id")
        output_df.to_csv(output_path, index=False)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        print(f"Classification complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()