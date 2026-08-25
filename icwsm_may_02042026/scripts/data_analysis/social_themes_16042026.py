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
from typing import List, Dict, Any, Optional, Set
import pandas as pd

# -------------------------------
# CONFIGURATION
# -------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

MODEL = "anthropic/claude-3.5-sonnet"
PATH = "/Users/i2scg/Documents/workstation_repos_30122025/ICWSM_15012026/working_data/racial_theme_analysis_20260106_221456/"
INPUT_CSV = os.path.join(PATH, "racial_review_1.csv")
THEMES_CSV = os.path.join(PATH, "aggregated_themes_claude-3.5-sonnet.csv")
BATCH_SIZE = 20

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(PATH) / f"theme_classification_{timestamp}"
output_dir.mkdir(exist_ok=True)

# Output files
MODEL_SLUG = MODEL.split("/")[-1]
CLASSIFICATION_RESULTS_CSV  = output_dir / f"theme_classified_reviews_{MODEL_SLUG}.csv"
BATCH_CLASSIFICATIONS_JSON  = output_dir / f"batch_theme_classifications_{MODEL_SLUG}.json"
CHECKPOINT_FILE             = output_dir / "checkpoint.json"
LOG_FILE                    = output_dir / "theme_classification.log"
COST_LOG_FILE               = output_dir / "theme_classification_costs.csv"


# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
def setup_logging() -> Path:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logging.info(f"Theme classification initialised. Output: {output_dir}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    return LOG_FILE


# ---------------------------------------------------------
# COST TRACKER
# ---------------------------------------------------------
class CostTracker:
    # Claude-3.5-Sonnet pricing on OpenRouter
    INPUT_PRICE_PER_M  = 3.00   # $ per 1M input tokens
    OUTPUT_PRICE_PER_M = 15.00  # $ per 1M output tokens

    def __init__(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost          = 0.0
        self.start_time          = time.time()

        with open(COST_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "batch_id", "batch_size",
                 "input_tokens", "output_tokens", "cost", "cumulative_cost"]
            )

    def update(self, batch_id: int, batch_size: int,
               input_tokens: int, output_tokens: int) -> float:
        input_cost  = (input_tokens  / 1_000_000) * self.INPUT_PRICE_PER_M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_M
        batch_cost  = input_cost + output_cost

        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost          += batch_cost

        with open(COST_LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                batch_id, batch_size,
                input_tokens, output_tokens,
                round(batch_cost, 6), round(self.total_cost, 6),
            ])
        return batch_cost

    def print_summary(self, total_reviews: Optional[int] = None):
        elapsed = (time.time() - self.start_time) / 3600
        logging.info("=" * 60)
        logging.info("COST SUMMARY")
        logging.info(f"  Input tokens  : {self.total_input_tokens:,}")
        logging.info(f"  Output tokens : {self.total_output_tokens:,}")
        logging.info(f"  Total cost    : ${self.total_cost:.6f}")
        logging.info(f"  Elapsed       : {elapsed:.2f} h")
        if total_reviews and self.total_cost > 0:
            logging.info(f"  Cost/review   : ${self.total_cost / total_reviews:.4f}")
        logging.info("=" * 60)


cost_tracker = CostTracker()


# ---------------------------------------------------------
# CHECKPOINT  (resume support)
# ---------------------------------------------------------
def load_checkpoint() -> Dict[str, Any]:
    """Return checkpoint dict or empty defaults."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        logging.info(f"Resuming from checkpoint: {len(cp['done_ids'])} reviews already done.")
        return cp
    return {"done_ids": [], "classifications": []}


def save_checkpoint(done_ids: List[str], classifications: List[Dict]):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"done_ids": done_ids, "classifications": classifications}, f)


# ---------------------------------------------------------
# LOAD THEMES
# ---------------------------------------------------------
def load_themes(themes_csv_path: str) -> List[Dict[str, Any]]:
    themes = []
    with open(themes_csv_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            themes.append({
                "theme_id":   i,
                "theme_name": row["theme_name"],
                "description": row["description"],
                "keywords":   [k.strip() for k in row["aggregated_keywords"].split(",")],
            })

    # Add catch-all
    themes.append({
        "theme_id":    len(themes) + 1,
        "theme_name":  "Other",
        "description": "Reviews that do not fit into any specific thematic category above.",
        "keywords":    [],
    })

    logging.info(f"Loaded {len(themes)} themes (including 'Other')")
    return themes


def build_system_prompt(themes: List[Dict[str, Any]]) -> str:
    themes_text = "\n\n".join(
        f"{t['theme_id']}. {t['theme_name']}\n"
        f"   Description: {t['description']}\n"
        f"   Keywords: {', '.join(t['keywords'][:5]) or 'N/A'}"
        for t in themes
    )
    return f"""You are a social researcher analysing customer reviews of swimming pools in Germany.
Classify each review into the thematic categories below.

AVAILABLE THEMES ({len(themes)} total):
{themes_text}

INSTRUCTIONS:
1. For EACH review select the TOP 3 most relevant themes, ranked most-to-least relevant.
2. If fewer than 3 themes apply, list only those that do.
3. Use "Other" for reviews that fit no specific category.
4. Use EXACT theme names as listed. Do NOT invent new themes.

OUTPUT — return ONLY valid JSON, no extra text:
{{
  "classifications": [
    {{"review_id": "123", "ranked_themes": ["Theme A", "Theme B", "Theme C"]}},
    {{"review_id": "124", "ranked_themes": ["Other"]}}
  ]
}}"""


# ---------------------------------------------------------
# LOAD REVIEWS
# ---------------------------------------------------------
def load_reviews(csv_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Accept either column name
    text_col = "review_text" if "review_text" in df.columns else "trans_review"
    id_col   = "review_id"   if "review_id"   in df.columns else None

    reviews = []
    for i, row in df.iterrows():
        reviews.append({
            "review_id":   str(row[id_col]) if id_col else f"review_{i+1}",
            "review_text": str(row[text_col])[:1500],
        })

    logging.info(f"Loaded {len(reviews)} reviews from {csv_path}")
    return reviews


# ---------------------------------------------------------
# LLM QUERY  (smart rate-limit handling, no unconditional sleep)
# ---------------------------------------------------------
def query_llm(messages: List[Dict[str, str]], max_tokens: int = 3000) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": random.choice([
            "http://localhost:3000", "https://github.com",
            "http://127.0.0.1:3000", "http://localhost:8080",
        ]),
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    backoff = 5  # seconds; doubles on each 429
    for attempt in range(4):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=180,
            )

            if resp.status_code == 200:
                data    = resp.json()
                usage   = data.get("usage", {})
                return {
                    "content":       data["choices"][0]["message"]["content"],
                    "input_tokens":  usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }

            if resp.status_code == 429:
                # Respect Retry-After header if present, else exponential backoff
                retry_after = int(resp.headers.get("Retry-After", backoff))
                logging.warning(f"Rate limited. Waiting {retry_after}s (attempt {attempt+1})")
                time.sleep(retry_after)
                backoff = min(backoff * 2, 120)
                continue

            logging.error(f"API error {resp.status_code}: {resp.text[:200]}")
            if attempt < 3:
                time.sleep(backoff)

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout on attempt {attempt+1}")
            time.sleep(backoff)
        except Exception as e:
            logging.error(f"Unexpected error attempt {attempt+1}: {e}")
            time.sleep(backoff)

    return {"content": "", "input_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------
def extract_json(content: str) -> Dict:
    """Try direct parse, then strip markdown fences, then regex fallback."""
    for text in [
        content,
        re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip(),
    ]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    logging.warning(f"JSON extraction failed. Preview: {content[:200]}")
    return {}


# ---------------------------------------------------------
# RANK → SCORE  (honest naming)
# ---------------------------------------------------------
RANK_SCORES = [1.0, 0.7, 0.4]

def rank_scores(ranked_themes: List[str]) -> Dict[str, float]:
    """
    Assign positional rank scores (not statistical confidence).
    1st place = 1.0, 2nd = 0.7, 3rd = 0.4.
    These reflect ranking order only, not model certainty.
    """
    return {
        theme: RANK_SCORES[i] if i < len(RANK_SCORES) else 0.3
        for i, theme in enumerate(ranked_themes)
    }


# ---------------------------------------------------------
# BATCH PROCESSING
# ---------------------------------------------------------
def process_batch(
    batch: List[Dict[str, str]],
    batch_id: int,
    valid_theme_names: Set[str],
    system_prompt: str,
    unknown_theme_counts: Dict[str, int],
) -> List[Dict[str, Any]]:

    reviews_text = "\n\n".join(
        f"Review ID: {r['review_id']}\nText: {r['review_text']}"
        for r in batch
    )

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": f"Classify these {len(batch)} reviews:\n\n{reviews_text}"},
    ]

    result = query_llm(messages)

    if not result["content"]:
        logging.error(f"Batch {batch_id}: empty response — marking all as errors")
        return [
            {"review_id": r["review_id"], "ranked_themes": [],
             "rank_scores": {}, "batch_id": batch_id, "error": "empty_response"}
            for r in batch
        ]

    # Track actual token usage (no estimation needed)
    batch_cost = cost_tracker.update(
        batch_id, len(batch),
        result["input_tokens"], result["output_tokens"],
    )
    logging.info(
        f"Batch {batch_id}: {result['input_tokens']:,} in / "
        f"{result['output_tokens']:,} out / ${batch_cost:.6f}"
    )

    data = extract_json(result["content"])
    classification_map = {
        str(c.get("review_id", "")).strip(): c
        for c in data.get("classifications", [])
    }

    processed = []
    for review in batch:
        rid = str(review["review_id"]).strip()
        clf = classification_map.get(rid)

        if not clf:
            logging.warning(f"Batch {batch_id}: no classification for review {rid}")
            processed.append({
                "review_id": rid, "ranked_themes": [],
                "rank_scores": {}, "batch_id": batch_id,
                "error": "missing_from_output",
            })
            continue

        # Validate theme names; track hallucinations
        valid, invalid = [], []
        for t in clf.get("ranked_themes", []):
            (valid if t in valid_theme_names else invalid).append(t)

        for t in invalid:
            unknown_theme_counts[t] = unknown_theme_counts.get(t, 0) + 1
            logging.debug(f"Unknown theme ignored: '{t}'")

        valid = valid[:3]
        processed.append({
            "review_id":    rid,
            "ranked_themes": valid,
            "rank_scores":  rank_scores(valid),
            "batch_id":     batch_id,
            "theme_count":  len(valid),
            "error":        "",
        })

    return processed


# ---------------------------------------------------------
# FLATTEN + DATAFRAME
# ---------------------------------------------------------
def to_dataframe(all_classifications: List[Dict]) -> pd.DataFrame:
    rows = []
    for c in all_classifications:
        rt = c.get("ranked_themes", [])
        rs = c.get("rank_scores", {})
        rows.append({
            "review_id":    c["review_id"],
            "batch_id":     c["batch_id"],
            "theme_1":      rt[0] if len(rt) > 0 else "",
            "rank_score_1": rs.get(rt[0], 0.0) if len(rt) > 0 else 0.0,
            "theme_2":      rt[1] if len(rt) > 1 else "",
            "rank_score_2": rs.get(rt[1], 0.0) if len(rt) > 1 else 0.0,
            "theme_3":      rt[2] if len(rt) > 2 else "",
            "rank_score_3": rs.get(rt[2], 0.0) if len(rt) > 2 else 0.0,
            "all_themes":   " | ".join(rt),
            "theme_count":  len(rt),
            "error":        c.get("error", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# SUMMARY STATISTICS  (fixed double-weighting)
# ---------------------------------------------------------
def print_summary(df: pd.DataFrame, total_reviews: int):
    logging.info("=" * 60)
    logging.info("CLASSIFICATION SUMMARY")
    logging.info("=" * 60)

    classified = len(df[df["theme_count"] > 0])
    logging.info(f"Total reviews       : {total_reviews}")
    logging.info(f"Successfully classified: {classified} ({classified/total_reviews:.1%})")
    logging.info(f"Avg themes/review   : {df['theme_count'].mean():.2f}")
    logging.info(f"Reviews with 3 themes: {(df['theme_count']==3).sum()} ({(df['theme_count']==3).mean():.1%})")

    # Theme frequency (count across all three columns)
    theme_freq: Dict[str, int] = {}
    for col in ["theme_1", "theme_2", "theme_3"]:
        for t in df[col][df[col] != ""]:
            theme_freq[t] = theme_freq.get(t, 0) + 1

    logging.info("\nTheme Frequency:")
    for theme, count in sorted(theme_freq.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {theme:<42} {count:>5}  ({count/total_reviews:5.1%})")

    # Average rank score per theme — simple mean of assigned rank scores
    # (no re-weighting; rank_scores already encode position)
    theme_score_sum:   Dict[str, float] = {}
    theme_score_count: Dict[str, int]   = {}

    for _, row in df.iterrows():
        for theme_col, score_col in [
            ("theme_1", "rank_score_1"),
            ("theme_2", "rank_score_2"),
            ("theme_3", "rank_score_3"),
        ]:
            t = row[theme_col]
            if t:
                theme_score_sum[t]   = theme_score_sum.get(t, 0.0)   + row[score_col]
                theme_score_count[t] = theme_score_count.get(t, 0)   + 1

    logging.info("\nAverage Rank Score per Theme:")
    for theme in sorted(theme_score_sum, key=lambda x: theme_score_sum[x]/theme_score_count[x], reverse=True):
        avg = theme_score_sum[theme] / theme_score_count[theme]
        logging.info(f"  {theme:<42} {avg:.3f}")

    # Top 10 co-occurrence pairs
    combos: Dict[tuple, int] = {}
    for _, row in df.iterrows():
        themes = [row[c] for c in ["theme_1","theme_2","theme_3"] if row[c]]
        if len(themes) >= 2:
            combo = tuple(sorted(themes))
            combos[combo] = combos.get(combo, 0) + 1

    logging.info("\nTop 10 Theme Co-occurrences:")
    for combo, count in sorted(combos.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  {combo}  →  {count}")

    logging.info("=" * 60)


# ---------------------------------------------------------
# COST ESTIMATE (pre-run)
# ---------------------------------------------------------
def estimate_cost(total_reviews: int = 4000) -> float:
    batches      = -(-total_reviews // BATCH_SIZE)  # ceiling division
    # System prompt ~875 tokens + 20 reviews * ~125 tokens each + 200 overhead
    in_per_batch = 875 + BATCH_SIZE * 125 + 200
    out_per_batch = BATCH_SIZE * 200

    total_in  = batches * in_per_batch
    total_out = batches * out_per_batch
    cost = (total_in / 1e6) * CostTracker.INPUT_PRICE_PER_M \
         + (total_out / 1e6) * CostTracker.OUTPUT_PRICE_PER_M

    print(f"\nCost Estimate  ({total_reviews:,} reviews, batch {BATCH_SIZE})")
    print(f"  Batches        : {batches}")
    print(f"  Input tokens   : {total_in:,}  (${(total_in/1e6)*CostTracker.INPUT_PRICE_PER_M:.2f})")
    print(f"  Output tokens  : {total_out:,}  (${(total_out/1e6)*CostTracker.OUTPUT_PRICE_PER_M:.2f})")
    print(f"  Estimated total: ${cost:.2f}  (range ${cost*0.8:.2f}–${cost*1.3:.2f})")
    print(f"  Per review     : ${cost/total_reviews:.4f}")
    return cost


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    setup_logging()
    logging.info("Starting theme classification")

    themes = load_themes(THEMES_CSV)
    valid_theme_names: Set[str] = {t["theme_name"] for t in themes}
    system_prompt = build_system_prompt(themes)

    reviews = load_reviews(INPUT_CSV)
    if not reviews:
        logging.error("No reviews loaded. Exiting.")
        return

    # Resume support: skip already-processed review IDs
    checkpoint       = load_checkpoint()
    done_ids: Set[str] = set(checkpoint["done_ids"])
    all_classifications: List[Dict] = checkpoint["classifications"]

    pending = [r for r in reviews if r["review_id"] not in done_ids]
    logging.info(f"Reviews total: {len(reviews)} | Already done: {len(done_ids)} | Pending: {len(pending)}")

    if not pending:
        logging.info("All reviews already classified. Skipping API calls.")
    else:
        batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
        logging.info(f"Batches to process: {len(batches)}")

        unknown_theme_counts: Dict[str, int] = {}

        for batch_id, batch in enumerate(batches, start=len(done_ids)//BATCH_SIZE + 1):
            logging.info(f"Batch {batch_id} / {len(batches) + batch_id - 1}  ({len(batch)} reviews)")

            results = process_batch(batch, batch_id, valid_theme_names, system_prompt, unknown_theme_counts)
            all_classifications.extend(results)

            # Update checkpoint
            done_ids.update(r["review_id"] for r in batch)
            save_checkpoint(list(done_ids), all_classifications)

            # Progress log every 10 batches
            if batch_id % 10 == 0:
                cost_tracker.print_summary(len(done_ids))
                logging.info(f"Progress: {len(done_ids)}/{len(reviews)} ({len(done_ids)/len(reviews):.1%})")

            # No unconditional sleep — rate limiting is handled inside query_llm

        # Report hallucinated theme names
        if unknown_theme_counts:
            logging.warning(f"Unknown themes ignored during run: {unknown_theme_counts}")

    # Persist final results
    with open(BATCH_CLASSIFICATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_classifications, f, indent=2, ensure_ascii=False)

    df = to_dataframe(all_classifications)
    df.to_csv(CLASSIFICATION_RESULTS_CSV, index=False, encoding="utf-8")

    cost_tracker.print_summary(len(reviews))
    print_summary(df, len(reviews))

    logging.info("Classification complete!")
    print(f"\n{'='*60}")
    print(f"DONE — Output directory : {output_dir}")
    print(f"Results CSV             : {CLASSIFICATION_RESULTS_CSV}")
    print(f"Batch JSON              : {BATCH_CLASSIFICATIONS_JSON}")
    print(f"Total cost              : ${cost_tracker.total_cost:.6f}")
    print(f"{'='*60}")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    estimated = estimate_cost()
    if input(f"\nEstimated cost ${estimated:.2f}. Continue? (y/n): ").strip().lower() == "y":
        main()
    else:
        print("Cancelled.")