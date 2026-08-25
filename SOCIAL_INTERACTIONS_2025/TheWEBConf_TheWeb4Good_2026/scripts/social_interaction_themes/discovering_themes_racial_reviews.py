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

# -------------------------------
# CONFIGURATION
# -------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

MODEL = "anthropic/claude-3.5-sonnet"
PATH = "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/"
INPUT_CSV = os.path.join(PATH, "racial_reviews_23122025.csv")
BATCH_SIZE = 100

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(PATH) / f"racial_theme_analysis_{timestamp}"
output_dir.mkdir(exist_ok=True)

# Output files
BATCH_THEMES_CSV = output_dir / f"batch_themes_{MODEL.split('/')[-1]}.csv"
AGGREGATED_THEMES_CSV = output_dir / f"aggregated_themes_{MODEL.split('/')[-1]}.csv"
LOG_FILE = output_dir / "analysis.log"
COST_LOG_FILE = output_dir / "cost_tracking.csv"

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
    
    logging.info(f"🚀 Logging initialized. Output directory: {output_dir}")
    return LOG_FILE

# ---------------------------------------------------------
# COST TRACKER
# ---------------------------------------------------------
class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        
        # Claude-3.5-Sonnet pricing on OpenRouter (as of Oct 2024)
        self.input_price_per_million = 3.00    # $3.00 per 1M input tokens
        self.output_price_per_million = 15.00  # $15.00 per 1M output tokens
        
        # Create cost log file
        with open(COST_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'batch_id', 'input_tokens', 'output_tokens', 'cost', 'cumulative_cost'])
    
    def update_cost(self, batch_id: int, input_tokens: int, output_tokens: int):
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
            'input_cost': (self.total_input_tokens / 1_000_000) * self.input_price_per_million,
            'output_cost': (self.total_output_tokens / 1_000_000) * self.output_price_per_million,
            'elapsed_hours': elapsed_hours,
            'cost_per_input_token': self.input_price_per_million / 1_000_000,
            'cost_per_output_token': self.output_price_per_million / 1_000_000
        }
    
    def print_summary(self):
        """Print cost summary"""
        summary = self.get_summary()
        logging.info("=" * 60)
        logging.info("💰 COST SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total Input Tokens:  {summary['total_input_tokens']:,}")
        logging.info(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        logging.info(f"Total Cost:          ${summary['total_cost']:.6f}")
        logging.info(f"  - Input Cost:      ${summary['input_cost']:.6f}")
        logging.info(f"  - Output Cost:     ${summary['output_cost']:.6f}")
        logging.info(f"Elapsed Time:        {summary['elapsed_hours']:.2f} hours")
        logging.info(f"Cost per 1K reviews (projected): ${(summary['total_cost'] / 8):.2f}")
        logging.info("=" * 60)

# Global cost tracker
cost_tracker = CostTracker()

# ---------------------------------------------------------
# SYSTEM PROMPT (for batch analysis)
# ---------------------------------------------------------
SYSTEM_PROMPT_BATCH = """You are a social researcher analyzing public reviews of swimming pools in Germany.
Your task is to identify recurring social themes across the reviews.

For each batch of reviews, analyze the text and identify 10-15 of the most recurring themes.

For each theme, provide:
1. theme_name: A concise name for the theme (2-4 words)
2. description: A brief description (1-2 sentences)
3. keywords: Key terms or phrases from reviews that represent this theme (comma-separated)
4. frequency_estimate: Your estimate of how often this theme appears in this batch (e.g., "very common", "common", "occasional")

Focus on social interactions, behaviors, and experiences mentioned in the reviews.
Return ONLY a valid JSON array of objects with those four keys. No additional text."""

# ---------------------------------------------------------
# SYSTEM PROMPT (for aggregation)
# ---------------------------------------------------------
SYSTEM_PROMPT_AGGREGATE = """You are a social researcher analyzing theme data from swimming pool reviews.
You have been given theme data from multiple batches of reviews (about 8000 reviews total).

Your task is to aggregate and consolidate these themes to identify the top 10-15 overall themes.

Some themes may be similar but worded differently. Your job is to:
1. Merge similar themes (where the meaning is essentially the same)
2. Create unified theme names and descriptions
3. Combine the keywords from similar themes
4. Estimate overall frequency based on batch frequencies

For each final theme, provide:
1. theme_name: A concise, unified name (2-4 words)
2. description: Clear description of what this theme represents
3. aggregated_keywords: All relevant keywords from merged themes (comma-separated, remove duplicates)
4. overall_frequency: Estimate of overall prevalence (e.g., "extremely common", "very common", "common", "occasional", "rare")
5. source_themes: List of original theme names that were merged into this one

Return ONLY a valid JSON array of objects with those five keys. No additional text."""

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def load_reviews(csv_path: str) -> List[Dict[str, str]]:
    """Load reviews from CSV file"""
    reviews = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('trans_review'):
                reviews.append({
                    'review_id': row.get('review_id', ''),
                    'review_text': row['trans_review']
                })
    logging.info(f"Loaded {len(reviews)} reviews from {csv_path}")
    return reviews

def extract_json_from_response(content: str) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response with multiple fallbacks"""
    try:
        # Try direct JSON parsing
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    json_patterns = [
        r'```(?:json)?\s*(\[.*?\])\s*```',  # ```json [...] ```
        r'```\s*(\[.*?\])\s*```',           # ``` [...] ```
        r'(\[.*\])',                         # Any JSON array
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, AttributeError):
                continue
    
    logging.warning(f"Could not extract JSON from response. Content: {content[:200]}...")
    return []

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters for English)"""
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
# BATCH PROCESSING
# ---------------------------------------------------------
def process_batch(batch_reviews: List[Dict[str, str]], batch_id: int) -> List[Dict[str, Any]]:
    """Process a single batch of reviews and extract themes"""
    logging.info(f"Processing batch {batch_id} with {len(batch_reviews)} reviews")
    
    # Prepare reviews text
    reviews_text = "\n\n".join([
        f"Review {i+1} (ID: {review['review_id']}):\n{review['review_text'][:1000]}"
        for i, review in enumerate(batch_reviews)
    ])
    
    # Estimate input tokens for cost tracking
    estimated_input_tokens = estimate_tokens(SYSTEM_PROMPT_BATCH + reviews_text)
    
    # Query LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BATCH},
        {"role": "user", "content": f"Please analyze these {len(batch_reviews)} reviews:\n\n{reviews_text}"}
    ]
    
    result = query_llm(messages, max_tokens=2000)
    
    if not result['content']:
        logging.error(f"Failed to get response for batch {batch_id}")
        return []
    
    # Update cost tracking
    input_tokens = result['input_tokens'] or estimated_input_tokens
    output_tokens = result['output_tokens'] or 1000  # Conservative estimate
    batch_cost = cost_tracker.update_cost(batch_id, input_tokens, output_tokens)
    
    logging.info(f"Batch {batch_id}: {input_tokens} input tokens, {output_tokens} output tokens, cost: ${batch_cost:.6f}")
    
    # Extract themes from response
    themes = extract_json_from_response(result['content'])
    
    if not themes:
        logging.warning(f"No themes extracted from batch {batch_id}")
        return []
    
    # Add batch metadata to each theme
    for theme in themes:
        theme['batch_id'] = batch_id
        theme['batch_size'] = len(batch_reviews)
        theme['review_ids'] = [r['review_id'] for r in batch_reviews]
    
    logging.info(f"Batch {batch_id}: Extracted {len(themes)} themes")
    return themes

def save_batch_themes(all_batch_themes: List[Dict[str, Any]]):
    """Save batch themes to CSV"""
    if not all_batch_themes:
        logging.warning("No themes to save")
        return
    
    # Flatten the list of themes
    flat_themes = []
    for batch_themes in all_batch_themes:
        for theme in batch_themes:
            flat_themes.append(theme)
    
    # Prepare CSV data
    csv_data = []
    for theme in flat_themes:
        csv_data.append({
            'batch_id': theme.get('batch_id', ''),
            'theme_name': theme.get('theme_name', ''),
            'description': theme.get('description', ''),
            'keywords': theme.get('keywords', ''),
            'frequency_estimate': theme.get('frequency_estimate', ''),
            'batch_size': theme.get('batch_size', ''),
            'review_count': len(theme.get('review_ids', []))
        })
    
    # Write to CSV
    with open(BATCH_THEMES_CSV, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    logging.info(f"Saved {len(csv_data)} batch themes to {BATCH_THEMES_CSV}")

# ---------------------------------------------------------
# AGGREGATION PROCESSING
# ---------------------------------------------------------
def aggregate_all_themes(all_batch_themes: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Aggregate themes from all batches"""
    logging.info("Starting aggregation of all themes")
    
    # Collect all themes for aggregation
    all_themes_for_aggregation = []
    for batch_themes in all_batch_themes:
        for theme in batch_themes:
            all_themes_for_aggregation.append({
                'theme_name': theme.get('theme_name', ''),
                'description': theme.get('description', ''),
                'keywords': theme.get('keywords', ''),
                'frequency_estimate': theme.get('frequency_estimate', ''),
                'batch_id': theme.get('batch_id', '')
            })
    
    logging.info(f"Collected {len(all_themes_for_aggregation)} themes for aggregation")
    
    # Prepare prompt for aggregation
    themes_json = json.dumps(all_themes_for_aggregation, indent=2)
    
    # Estimate tokens for cost tracking
    estimated_input_tokens = estimate_tokens(SYSTEM_PROMPT_AGGREGATE + themes_json)
    
    # Query LLM for aggregation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGGREGATE},
        {"role": "user", "content": f"Here are all the themes from {len(all_batch_themes)} batches (approximately {len(all_themes_for_aggregation)} themes):\n\n{themes_json}"}
    ]
    
    result = query_llm(messages, max_tokens=3000)
    
    if not result['content']:
        logging.error("Failed to get aggregation response")
        return []
    
    # Update cost tracking for aggregation
    input_tokens = result['input_tokens'] or estimated_input_tokens
    output_tokens = result['output_tokens'] or 1500  # Conservative estimate
    aggregation_cost = cost_tracker.update_cost('aggregation', input_tokens, output_tokens)
    
    logging.info(f"Aggregation: {input_tokens} input tokens, {output_tokens} output tokens, cost: ${aggregation_cost:.6f}")
    
    # Extract aggregated themes
    aggregated_themes = extract_json_from_response(result['content'])
    
    if not aggregated_themes:
        logging.warning("No aggregated themes extracted")
        return []
    
    logging.info(f"Aggregation complete: {len(aggregated_themes)} final themes")
    return aggregated_themes

def save_aggregated_themes(aggregated_themes: List[Dict[str, Any]]):
    """Save aggregated themes to CSV"""
    if not aggregated_themes:
        logging.warning("No aggregated themes to save")
        return
    
    # Prepare CSV data
    csv_data = []
    for theme in aggregated_themes:
        csv_data.append({
            'theme_name': theme.get('theme_name', ''),
            'description': theme.get('description', ''),
            'aggregated_keywords': theme.get('aggregated_keywords', ''),
            'overall_frequency': theme.get('overall_frequency', ''),
            'source_themes': '; '.join(theme.get('source_themes', [])),
            'source_theme_count': len(theme.get('source_themes', []))
        })
    
    # Write to CSV
    with open(AGGREGATED_THEMES_CSV, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    logging.info(f"Saved {len(csv_data)} aggregated themes to {AGGREGATED_THEMES_CSV}")

# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def main():
    """Main execution function"""
    # Setup logging
    setup_logging()
    logging.info("Starting theme analysis")
    
    # Load reviews
    reviews = load_reviews(INPUT_CSV)
    if not reviews:
        logging.error("No reviews loaded. Exiting.")
        return
    
    # Split into batches
    batches = []
    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i:i + BATCH_SIZE]
        batches.append(batch)
    
    logging.info(f"Created {len(batches)} batches of size {BATCH_SIZE}")
    
    # Process each batch
    all_batch_themes = []
    
    for batch_id, batch in enumerate(batches):
        logging.info(f"Processing batch {batch_id + 1}/{len(batches)}")
        
        # Process batch and extract themes
        batch_themes = process_batch(batch, batch_id + 1)
        all_batch_themes.append(batch_themes)
        
        # Save intermediate results every 10 batches
        if (batch_id + 1) % 10 == 0:
            save_batch_themes(all_batch_themes)
            cost_tracker.print_summary()
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Save all batch themes
    save_batch_themes(all_batch_themes)
    
    # Aggregate all themes
    aggregated_themes = aggregate_all_themes(all_batch_themes)
    
    # Save aggregated themes
    save_aggregated_themes(aggregated_themes)
    
    # Final cost summary
    cost_tracker.print_summary()
    
    logging.info("Analysis complete!")
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Themes: {BATCH_THEMES_CSV}")
    print(f"Aggregated Themes: {AGGREGATED_THEMES_CSV}")
    print(f"Cost Log: {COST_LOG_FILE}")
    print(f"Total Cost: ${cost_tracker.total_cost:.6f}")
    print(f"{'='*60}")

# ---------------------------------------------------------
# COST ESTIMATION
# ---------------------------------------------------------
def estimate_total_cost():
    """Estimate total cost for processing 8000+ reviews"""
    print("\n📊 COST ESTIMATION for 8000+ reviews")
    print("=" * 50)
    
    # Assumptions
    avg_review_length = 500  # characters
    reviews_per_batch = 100
    total_batches = 85  # 8500 reviews / 100
    
    # Token counts (1 token ≈ 4 chars for English)
    tokens_per_review = avg_review_length // 4  # ~125 tokens
    system_prompt_tokens = len(SYSTEM_PROMPT_BATCH) // 4  # ~500 tokens
    user_prompt_overhead = 100  # Additional prompt text
    
    # Per batch calculation
    input_tokens_per_batch = system_prompt_tokens + user_prompt_overhead + (reviews_per_batch * tokens_per_review)
    output_tokens_per_batch = 1500  # Conservative estimate for 10-15 themes
    
    # Total batch processing
    total_input_tokens = total_batches * input_tokens_per_batch
    total_output_tokens = total_batches * output_tokens_per_batch
    
    # Aggregation step
    aggregation_input_tokens = 20000  # All themes JSON
    aggregation_output_tokens = 3000   # Aggregated themes
    
    # Prices
    input_price = 3.00 / 1_000_000
    output_price = 15.00 / 1_000_000
    
    # Calculate costs
    batch_input_cost = total_input_tokens * input_price
    batch_output_cost = total_output_tokens * output_price
    aggregation_input_cost = aggregation_input_tokens * input_price
    aggregation_output_cost = aggregation_output_tokens * output_price
    
    total_cost = batch_input_cost + batch_output_cost + aggregation_input_cost + aggregation_output_cost
    
    print(f"Estimated tokens per batch:")
    print(f"  Input:  {input_tokens_per_batch:,}")
    print(f"  Output: {output_tokens_per_batch:,}")
    print()
    print(f"Total tokens for {total_batches} batches:")
    print(f"  Input:  {total_input_tokens:,} (${batch_input_cost:.2f})")
    print(f"  Output: {total_output_tokens:,} (${batch_output_cost:.2f})")
    print()
    print(f"Aggregation step:")
    print(f"  Input:  {aggregation_input_tokens:,} (${aggregation_input_cost:.2f})")
    print(f"  Output: {aggregation_output_tokens:,} (${aggregation_output_cost:.2f})")
    print()
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print(f"Range likely: ${total_cost * 0.7:.2f} - ${total_cost * 1.3:.2f}")
    print("=" * 50)
    
    return total_cost

if __name__ == "__main__":
    # First, show cost estimate
    estimate_total_cost()
    
    # Ask for confirmation
    response = input("\nContinue with analysis? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Analysis cancelled.")