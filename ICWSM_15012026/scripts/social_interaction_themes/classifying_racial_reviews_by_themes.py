import csv
import json
import time
import os
import re
import logging
from datetime import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import random
from typing import List, Dict, Any
import pandas as pd

# -------------------------------
# CONFIGURATION - MODIFIED FOR CLASSIFICATION
# -------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

MODEL = "anthropic/claude-3.5-sonnet"
PATH = "/Users/i2scg/Documents/workstation_repos_30122025/ICWSM_15012026/working_data/racial_theme_analysis_20260106_221456/"
INPUT_CSV = os.path.join(PATH,"racial_review_1.csv")  # the racial reviews
THEMES_CSV = os.path.join(PATH, "aggregated_themes_claude-3.5-sonnet.csv")  # Your themes file
BATCH_SIZE = 20  # CHANGED: Reduced from 50 to 20 as requested

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(PATH) / f"theme_classification_{timestamp}"
output_dir.mkdir(exist_ok=True)

# Output files
CLASSIFICATION_RESULTS_CSV = output_dir / f"theme_classified_reviews_{MODEL.split('/')[-1]}.csv"
BATCH_CLASSIFICATIONS_JSON = output_dir / f"batch_theme_classifications_{MODEL.split('/')[-1]}.json"
LOG_FILE = output_dir / "theme_classification.log"
COST_LOG_FILE = output_dir / "theme_classification_costs.csv"

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
def setup_logging():
    """Setup logging to both file and console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
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
    
    logging.info(f"🚀 Theme classification initialized. Output directory: {output_dir}")
    logging.info(f"📦 Batch size: {BATCH_SIZE} reviews per API call")
    return LOG_FILE

# ---------------------------------------------------------
# COST TRACKER (adapted for classification)
# ---------------------------------------------------------
class ClassificationCostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        
        # Claude-3.5-Sonnet pricing on OpenRouter
        self.input_price_per_million = 3.00    # $3.00 per 1M input tokens
        self.output_price_per_million = 15.00  # $15.00 per 1M output tokens
        
        # Create cost log file
        with open(COST_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'batch_id', 'batch_size', 'input_tokens', 'output_tokens', 'cost', 'cumulative_cost'])
    
    def update_cost(self, batch_id: int, batch_size: int, input_tokens: int, output_tokens: int):
        """Update cost tracking for a batch"""
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
        batch_cost = input_cost + output_cost
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += batch_cost
        
        # Log to CSV
        with open(COST_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                batch_id,
                batch_size,
                input_tokens,
                output_tokens,
                round(batch_cost, 6),
                round(self.total_cost, 6)
            ])
        
        return batch_cost
    
    def get_summary(self):
        """Get cost summary"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost,
            'elapsed_hours': elapsed_hours
        }
    
    def print_summary(self):
        """Print cost summary"""
        summary = self.get_summary()
        logging.info("=" * 60)
        logging.info("💰 CLASSIFICATION COST SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total Input Tokens:  {summary['total_input_tokens']:,}")
        logging.info(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        logging.info(f"Total Cost:          ${summary['total_cost']:.6f}")
        logging.info(f"Elapsed Time:        {summary['elapsed_hours']:.2f} hours")
        if summary['total_cost'] > 0:
            logging.info(f"Cost per review:     ${summary['total_cost']/4000:.4f}")
        logging.info("=" * 60)

# Global cost tracker
cost_tracker = ClassificationCostTracker()

# ---------------------------------------------------------
# LOAD THEMES AND CREATE SYSTEM PROMPT
# ---------------------------------------------------------
def load_themes(themes_csv_path: str) -> List[Dict[str, str]]:
    """Load themes from CSV file and add 'Other' theme"""
    themes = []
    try:
        with open(themes_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                # Parse keywords
                keywords_str = row['aggregated_keywords']
                keywords = [k.strip() for k in keywords_str.split(',')]
                
                themes.append({
                    'theme_id': i,
                    'theme_name': row['theme_name'],
                    'description': row['description'],
                    'keywords': keywords,
                    'frequency': row['overall_frequency']
                })
    except Exception as e:
        logging.error(f"Error loading themes: {e}")
        return []
    
    # ADD "OTHER" THEME AS REQUESTED
    themes.append({
        'theme_id': len(themes) + 1,
        'theme_name': 'Other',
        'description': 'Reviews that do not fit into any of the specific thematic categories above',
        'keywords': [],
        'frequency': 'occasional'
    })
    
    logging.info(f"Loaded {len(themes)} themes from {themes_csv_path} (including 'Other' theme)")
    return themes


def create_classification_prompt(themes: List[Dict[str, str]]) -> str:
    """Create the system prompt for theme classification"""
    
    # Build themes text
    themes_text = "\n\n".join([
        f"{theme['theme_id']}. {theme['theme_name']}\n"
        f"   Description: {theme['description']}\n"
        f"   Keywords: {', '.join(theme['keywords'][:5]) if theme['keywords'] else 'N/A'}"
        for theme in themes
    ])
    
    prompt = f"""You are a social researcher analyzing customer reviews of swimming pools in Germany. 
Your task is to classify reviews into thematic categories.

AVAILABLE THEMES ({len(themes)} total, including 'Other' for reviews that don't fit specific categories):
{themes_text}

IMPORTANT INSTRUCTIONS:
1. For EACH review, select the TOP 3 most relevant themes (ranked in order of relevance)
2. Rank themes from most relevant (1st) to least relevant (3rd)
3. If a review has fewer than 3 relevant themes, only list the relevant ones
4. You may use the "Other" theme for reviews that don't fit any specific category
5. Use EXACT theme names as listed above
6. DO NOT add new themes - only use the themes provided

OUTPUT FORMAT (JSON object):
{{
  "batch_id": 1,
  "classifications": [
    {{
      "review_id": "123",
      "ranked_themes": [
        "Cultural and Racial Discrimination",
        "Staff Behavior and Treatment",
        "Language and Communication"
      ]
    }},
    {{
      "review_id": "124",
      "ranked_themes": [
        "Other"
      ]
    }},
    ...
  ]
}}

Return ONLY valid JSON. No additional text or explanations."""
    
    return prompt

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def load_reviews_for_classification(csv_path: str) -> List[Dict[str, str]]:
    """Load reviews from CSV file - assuming columns: review_id, review_text"""
    reviews = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row.get('review_text') or row.get('trans_review'):
                    review_text = row.get('review_text') or row.get('trans_review')
                    reviews.append({
                        'review_id': row.get('review_id', f"review_{i+1}"),
                        'review_text': review_text[:1500]  # Limit length
                    })
    except Exception as e:
        logging.error(f"Error loading reviews: {e}")
        # Try alternative approach
        df = pd.read_csv(csv_path, encoding='utf-8')
        for i, row in df.iterrows():
            review_text = row.get('review_text', row.get('trans_review', ''))
            reviews.append({
                'review_id': row.get('review_id', f"review_{i+1}"),
                'review_text': str(review_text)[:1500]
            })
    
    logging.info(f"Loaded {len(reviews)} reviews from {csv_path}")
    return reviews

def extract_json_from_response(content: str) -> Dict:
    """Extract JSON from LLM response with multiple fallbacks"""
    try:
        # Try direct JSON parsing
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    json_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # ```json {...} ```
        r'```\s*(\{.*?\})\s*```',           # ``` {...} ```
        r'(\{.*\})',                         # Any JSON object
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                return result
            except (json.JSONDecodeError, AttributeError):
                continue
    
    logging.warning(f"Could not extract JSON from response. Content: {content[:200]}...")
    return {}

def estimate_tokens(text: str) -> int:
    """Estimate token count"""
    return len(text) // 4

def query_llm(messages: List[Dict[str, str]], max_tokens: int = 4000) -> Dict[str, Any]:
    """Query OpenRouter API with error handling and retry"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": random.choice([
            "http://localhost:3000",
            "https://github.com",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
        ])
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
                
                return {
                    'content': content,
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            elif response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                logging.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"API error {response.status_code}: {response.text}")
                if attempt < 2:
                    time.sleep(10)
                    
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout on attempt {attempt + 1}")
            if attempt < 2:
                time.sleep(20)
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < 2:
                time.sleep(10)
    
    return {'content': '', 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

# ---------------------------------------------------------
# BATCH CLASSIFICATION PROCESSING
# ---------------------------------------------------------
def process_classification_batch(batch_reviews: List[Dict[str, str]], batch_id: int, 
                               themes: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, Any]]:
    """Process a single batch of reviews and classify themes"""
    logging.info(f"Processing classification batch {batch_id} with {len(batch_reviews)} reviews")
    
    # Prepare reviews text for the batch
    reviews_text = "\n\n".join([
        f"Review ID: {review['review_id']}\nText: {review['review_text']}"
        for i, review in enumerate(batch_reviews)
    ])
    
    # Estimate input tokens for cost tracking
    estimated_input_tokens = estimate_tokens(system_prompt + reviews_text)
    
    # Query LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please classify these {len(batch_reviews)} reviews:\n\n{reviews_text}"}
    ]
    
    result = query_llm(messages, max_tokens=3000)
    
    if not result['content']:
        logging.error(f"Failed to get response for batch {batch_id}")
        return []
    
    # Update cost tracking
    input_tokens = result['input_tokens'] or estimated_input_tokens
    output_tokens = result['output_tokens'] or 1000
    batch_cost = cost_tracker.update_cost(batch_id, len(batch_reviews), input_tokens, output_tokens)
    
    logging.info(f"Batch {batch_id}: {input_tokens:,} in, {output_tokens:,} out, cost: ${batch_cost:.6f}")
    
    # Extract classifications from response
    classification_data = extract_json_from_response(result['content'])
    
    if not classification_data:
        logging.warning(f"No classifications extracted from batch {batch_id}")
        # Return empty classifications for each review
        return [
            {
                'review_id': review['review_id'],
                'ranked_themes': [],
                'batch_id': batch_id,
                'error': 'No classification extracted'
            }
            for review in batch_reviews
        ]
    
    # Validate and process classifications
    processed_classifications = []
    classifications = classification_data.get('classifications', [])
    
    # Create mapping of review_id to classification
    classification_map = {str(c.get('review_id', '')).strip(): c for c in classifications}
    
    for review in batch_reviews:
        review_id = str(review['review_id']).strip()
        if review_id in classification_map:
            classification = classification_map[review_id]
            ranked_themes = classification.get('ranked_themes', [])
            
            # Validate themes exist in our theme list
            valid_themes = []
            theme_names = [t['theme_name'] for t in themes]
            for theme in ranked_themes:
                if theme in theme_names:
                    valid_themes.append(theme)
                else:
                    logging.warning(f"Batch {batch_id}, Review {review_id}: Unknown theme '{theme}' - ignoring")
            
            # CHANGED: NO LONGER ENFORCING CORE RACIAL THEMES
            # Just take what LLM provided, up to 3 themes
            valid_themes = valid_themes[:3]  # Limit to max 3
            
            # CONFIDENCE ASSIGNMENT HAPPENS HERE
            confidence_scores = assign_confidence_scores(valid_themes)
            
            processed_classifications.append({
                'review_id': review_id,
                'ranked_themes': valid_themes,
                'confidence_scores': confidence_scores,  # Added here
                'batch_id': batch_id,
                'theme_count': len(valid_themes),
                # CHANGED: Removed has_core_theme check
            })
        else:
            # No classification found for this review
            logging.warning(f"Batch {batch_id}: No classification for review {review_id}")
            processed_classifications.append({
                'review_id': review_id,
                'ranked_themes': [],
                'batch_id': batch_id,
                'error': 'Review not classified in batch output'
            })
    
    logging.info(f"Batch {batch_id}: Processed {len(processed_classifications)} classifications")
    return processed_classifications


def assign_confidence_scores(ranked_themes: List[str]) -> Dict[str, float]:
    """
    🔥 CONFIDENCE ASSIGNMENT HAPPENS HERE
    
    Assign confidence scores based on ranking position.
    This is our heuristic: 1st rank = highest confidence, 2nd = medium, 3rd = lower
    
    Parameters:
    - ranked_themes: List of themes in rank order [1st, 2nd, 3rd]
    
    Returns:
    - Dictionary mapping theme_name -> confidence_score
    """
    if not ranked_themes:
        return {}
    
    # 🎯 CONFIDENCE SCORING HEURISTICS
    # We use simple rank-based scoring:
    # 1st place (most relevant): 1.0
    # 2nd place: 0.7  
    # 3rd place: 0.4
    # Any beyond 3rd: 0.3 (shouldn't happen due to earlier limit)
    
    scores = [1.0, 0.7, 0.4]
    confidence_dict = {}
    
    for i, theme in enumerate(ranked_themes):
        if i < len(scores):
            confidence_dict[theme] = scores[i]
        else:
            confidence_dict[theme] = 0.3  # Fallback for any additional themes
    
    logging.debug(f"Assigned confidence scores: {confidence_dict}")
    return confidence_dict


def save_batch_classifications(all_classifications: List[List[Dict]], output_path: str):
    """Save batch classifications to JSON for backup"""
    # Flatten the list
    flat_classifications = []
    for batch in all_classifications:
        flat_classifications.extend(batch)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat_classifications, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved {len(flat_classifications)} classifications to {output_path}")

def create_final_dataframe(all_classifications: List[List[Dict]], themes: List[Dict[str, str]]) -> pd.DataFrame:
    """Create final DataFrame with one row per review"""
    # Flatten classifications
    flat_classifications = []
    for batch in all_classifications:
        flat_classifications.extend(batch)
    
    # Prepare data for DataFrame
    rows = []
    for classification in flat_classifications:
        ranked_themes = classification.get('ranked_themes', [])
        confidence_scores = classification.get('confidence_scores', {})
        
        # Get up to 3 themes
        theme_1 = ranked_themes[0] if len(ranked_themes) > 0 else ''
        theme_2 = ranked_themes[1] if len(ranked_themes) > 1 else ''
        theme_3 = ranked_themes[2] if len(ranked_themes) > 2 else ''
        
        # Get corresponding confidences
        conf_1 = confidence_scores.get(theme_1, 0.0) if theme_1 else 0.0
        conf_2 = confidence_scores.get(theme_2, 0.0) if theme_2 else 0.0
        conf_3 = confidence_scores.get(theme_3, 0.0) if theme_3 else 0.0
        
        row = {
            'review_id': classification['review_id'],
            'batch_id': classification['batch_id'],
            'theme_1': theme_1,
            'confidence_1': conf_1,
            'theme_2': theme_2,
            'confidence_2': conf_2,
            'theme_3': theme_3,
            'confidence_3': conf_3,
            'all_themes': ' | '.join(ranked_themes),
            'theme_count': len(ranked_themes),
            'error': classification.get('error', '')
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add theme distribution statistics
    theme_counts = {}
    for classification in flat_classifications:
        for theme in classification.get('ranked_themes', []):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    logging.info(f"Total reviews classified: {len(df)}")
    logging.info(f"Theme distribution: {dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))}")
    
    return df

# ---------------------------------------------------------
# MAIN CLASSIFICATION FUNCTION
# ---------------------------------------------------------
def main():
    """Main execution function for theme classification"""
    # Setup logging
    setup_logging()
    logging.info("Starting theme classification")
    
    # Load themes
    themes = load_themes(THEMES_CSV)
    if not themes:
        logging.error("No themes loaded. Exiting.")
        return
    
    logging.info(f"Loaded {len(themes)} themes including 'Other'")
    
    # Create classification prompt
    system_prompt = create_classification_prompt(themes)
    
    # Load reviews
    reviews = load_reviews_for_classification(INPUT_CSV)
    if not reviews:
        logging.error("No reviews loaded. Exiting.")
        return
    
    logging.info(f"Total reviews to classify: {len(reviews)}")
    
    # Split into batches of size 20
    batches = []
    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i:i + BATCH_SIZE]
        batches.append(batch)
    
    logging.info(f"Created {len(batches)} batches of size {BATCH_SIZE}")
    
    # Process each batch
    all_classifications = []
    
    for batch_id, batch in enumerate(batches, 1):
        logging.info(f"Processing batch {batch_id}/{len(batches)}")
        
        # Process batch and get classifications
        batch_classifications = process_classification_batch(
            batch, batch_id, themes, system_prompt
        )
        all_classifications.append(batch_classifications)
        
        # Save intermediate results every 10 batches
        if batch_id % 10 == 0:
            temp_output = output_dir / f"temp_classifications_batch_{batch_id}.json"
            save_batch_classifications(all_classifications, str(temp_output))
            cost_tracker.print_summary()
            
            # Calculate progress
            processed_count = sum(len(b) for b in all_classifications)
            logging.info(f"Progress: {processed_count}/{len(reviews)} reviews processed ({processed_count/len(reviews)*100:.1f}%)")
        
        # Avoid rate limiting
        time.sleep(2)
    
    # Save all batch classifications
    save_batch_classifications(all_classifications, str(BATCH_CLASSIFICATIONS_JSON))
    
    # Create final DataFrame
    final_df = create_final_dataframe(all_classifications, themes)
    
    # Save to CSV
    final_df.to_csv(str(CLASSIFICATION_RESULTS_CSV), index=False, encoding='utf-8')
    
    # Final cost summary
    cost_tracker.print_summary()
    
    # Print summary statistics
    print_summary_statistics(final_df, themes)
    
    logging.info("Classification complete!")
    print(f"\n{'='*60}")
    print(f"🎯 CLASSIFICATION COMPLETE!")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📊 Results CSV: {CLASSIFICATION_RESULTS_CSV}")
    print(f"💾 Batch Classifications: {BATCH_CLASSIFICATIONS_JSON}")
    print(f"💰 Total Cost: ${cost_tracker.total_cost:.6f}")
    print(f"{'='*60}")

def print_summary_statistics(df: pd.DataFrame, themes: List[Dict[str, str]]):
    """Print summary statistics of classification results"""
    logging.info("=" * 60)
    logging.info("📊 CLASSIFICATION SUMMARY STATISTICS")
    logging.info("=" * 60)
    
    # Basic stats
    total_reviews = len(df)
    reviews_with_themes = len(df[df['theme_count'] > 0])
    avg_themes_per_review = df['theme_count'].mean()
    
    logging.info(f"Total reviews: {total_reviews}")
    logging.info(f"Reviews with at least one theme: {reviews_with_themes} ({reviews_with_themes/total_reviews:.1%})")
    logging.info(f"Average themes per review: {avg_themes_per_review:.2f}")
    logging.info(f"Reviews with 3 themes: {len(df[df['theme_count'] == 3])} ({len(df[df['theme_count'] == 3])/total_reviews:.1%})")
    
    # Theme frequency
    theme_freq = {}
    for idx, row in df.iterrows():
        for theme_col in ['theme_1', 'theme_2', 'theme_3']:
            theme = row[theme_col]
            if theme and pd.notna(theme):
                theme_freq[theme] = theme_freq.get(theme, 0) + 1
    
    logging.info("\n📈 Theme Frequency (out of all reviews):")
    for theme_name, count in sorted(theme_freq.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_reviews) * 100
        logging.info(f"  {theme_name:<40} {count:>5} ({percentage:>5.1f}%)")
    
    # Confidence statistics
    logging.info("\n🎯 Average Confidence Scores:")
    conf_stats = {}
    for theme in theme_freq.keys():
        conf_1 = df[df['theme_1'] == theme]['confidence_1'].mean()
        conf_2 = df[df['theme_2'] == theme]['confidence_2'].mean()
        conf_3 = df[df['theme_3'] == theme]['confidence_3'].mean()
        
        # Weighted average
        total = (df['theme_1'] == theme).sum() + (df['theme_2'] == theme).sum() + (df['theme_3'] == theme).sum()
        if total > 0:
            weighted_conf = (
                (df[df['theme_1'] == theme]['confidence_1'].sum() * 1.0 +
                 df[df['theme_2'] == theme]['confidence_2'].sum() * 0.7 +
                 df[df['theme_3'] == theme]['confidence_3'].sum() * 0.4) / total
            )
            conf_stats[theme] = weighted_conf
    
    for theme, avg_conf in sorted(conf_stats.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {theme:<40} {avg_conf:.3f}")
    
    # Most common theme combinations
    logging.info("\n🤝 Most Common Theme Combinations (top 10):")
    combinations = {}
    for idx, row in df.iterrows():
        themes_in_review = [row['theme_1'], row['theme_2'], row['theme_3']]
        themes_in_review = [t for t in themes_in_review if t and pd.notna(t)]
        if len(themes_in_review) >= 2:
            # Sort to avoid duplicates
            combo = tuple(sorted(themes_in_review))
            combinations[combo] = combinations.get(combo, 0) + 1
    
    # Top 10 combinations
    for combo, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  {combo}: {count} reviews")
    
    logging.info("=" * 60)

# ---------------------------------------------------------
# COST ESTIMATION FOR CLASSIFICATION
# ---------------------------------------------------------
def estimate_classification_cost():
    """Estimate total cost for classifying 4,000 reviews with batch size 20"""
    print("\n📊 COST ESTIMATION for 4,000 reviews (batch size: 20)")
    print("=" * 60)
    
    # Assumptions
    avg_review_length = 500  # characters
    reviews_per_batch = BATCH_SIZE  # 20
    total_reviews = 4000
    total_batches = total_reviews // reviews_per_batch + (1 if total_reviews % reviews_per_batch else 0)
    
    # Themes text length (approx) - now 12 themes with "Other"
    themes_text_length = 3500  # characters for all 12 themes with descriptions
    
    # Token counts (1 token ≈ 4 chars)
    tokens_per_review = avg_review_length // 4  # ~125 tokens
    system_prompt_tokens = themes_text_length // 4  # ~875 tokens
    
    # Per batch calculation
    reviews_text_per_batch = reviews_per_batch * avg_review_length
    input_tokens_per_batch = system_prompt_tokens + (reviews_text_per_batch // 4) + 200  # overhead
    output_tokens_per_batch = 200 * reviews_per_batch  # ~200 tokens per review output (JSON)
    
    # Total
    total_input_tokens = total_batches * input_tokens_per_batch
    total_output_tokens = total_batches * output_tokens_per_batch
    
    # Prices
    input_price = 3.00 / 1_000_000
    output_price = 15.00 / 1_000_000
    
    # Calculate costs
    input_cost = total_input_tokens * input_price
    output_cost = total_output_tokens * output_price
    total_cost = input_cost + output_cost
    
    print(f"Assumptions:")
    print(f"  Reviews: {total_reviews:,}")
    print(f"  Batch size: {reviews_per_batch}")
    print(f"  Total batches: {total_batches}")
    print(f"  Themes: 12 (11 from CSV + 'Other')")
    print()
    print(f"Estimated tokens per batch ({reviews_per_batch} reviews):")
    print(f"  Input:  {input_tokens_per_batch:,} tokens")
    print(f"  Output: {output_tokens_per_batch:,} tokens")
    print()
    print(f"Total for {total_batches} batches:")
    print(f"  Input:  {total_input_tokens:,} tokens (${input_cost:.2f})")
    print(f"  Output: {total_output_tokens:,} tokens (${output_cost:.2f})")
    print()
    print(f"💰 TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print(f"📊 Range likely: ${total_cost * 0.8:.2f} - ${total_cost * 1.3:.2f}")
    print(f"🎯 Cost per review: ${total_cost/total_reviews:.4f}")
    print("=" * 60)
    
    return total_cost

if __name__ == "__main__":
    # First, show cost estimate
    estimated_cost = estimate_classification_cost()
    
    # Ask for confirmation
    response = input(f"\n💡 Estimated cost: ${estimated_cost:.2f}. Continue with classification? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Classification cancelled.")