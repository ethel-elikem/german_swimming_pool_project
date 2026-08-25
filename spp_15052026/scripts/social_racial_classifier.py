import pandas as pd
import json
import time
import os
import requests
import re
import random
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
from pathlib import Path
import logging
from datetime import datetime
import sys

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "anthropic/claude-sonnet-4.6"   # claude-3.7-sonnet on OpenRouter


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> Path:
    """Write timestamped logs to both console and a file under output_dir/logs/."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"classification_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:          # clear any stale handlers
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(f"Logging initialised → {log_file}")
    return log_file


# ══════════════════════════════════════════════════════════════════════════════
# 1. COST TRACKER
#    Uses *actual* token counts returned by the API instead of estimates.
# ══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    # claude-3.7-sonnet pricing on OpenRouter (same as 3.5-sonnet)
    INPUT_PRICE_PER_M  = 3.00   # $ per 1 M input tokens
    OUTPUT_PRICE_PER_M = 15.00  # $ per 1 M output tokens

    def __init__(self, budget_limit: float = 500.0):
        self.total_reviews_processed = 0
        self.total_input_tokens      = 0
        self.total_output_tokens     = 0
        self.total_cost              = 0.0
        self.start_time              = time.time()
        self.budget_limit            = budget_limit
        self._lock                   = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────

    def record_call(self, input_tokens: int, output_tokens: int, reviews_in_batch: int):
        """Thread-safe update from actual API usage block."""
        cost = (input_tokens  / 1_000_000) * self.INPUT_PRICE_PER_M \
             + (output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_M
        with self._lock:
            self.total_input_tokens      += input_tokens
            self.total_output_tokens     += output_tokens
            self.total_cost              += cost
            self.total_reviews_processed += reviews_in_batch
        return cost

    def budget_status(self) -> str:
        with self._lock:
            if self.total_cost >= self.budget_limit:
                return "EXCEEDED"
            if self.total_cost >= self.budget_limit * 0.80:
                return "WARNING"
            return "OK"

    def summary(self) -> dict:
        with self._lock:
            elapsed_h = (time.time() - self.start_time) / 3600
            n         = self.total_reviews_processed
            return {
                "total_reviews"      : n,
                "total_cost"         : round(self.total_cost, 4),
                "cost_per_review"    : round(self.total_cost / n, 6) if n else 0,
                "reviews_per_hour"   : round(n / elapsed_h, 1)       if elapsed_h else 0,
                "elapsed_hours"      : round(elapsed_h, 3),
                "budget_status"      : self.budget_status(),
                "remaining_budget"   : round(self.budget_limit - self.total_cost, 4),
                "total_input_tokens" : self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            }

    def log_progress(self):
        s      = self.summary()
        icon   = {"OK": "✅", "WARNING": "⚠️", "EXCEEDED": "🚨"}.get(s["budget_status"], "")
        logging.info(
            f"{icon} COST ${s['total_cost']:.4f} / ${self.budget_limit:.2f} | "
            f"{s['total_reviews']:,} reviews | "
            f"${s['cost_per_review']:.6f}/review | "
            f"remaining ${s['remaining_budget']:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an annotation model. Classify each pool review with exactly two fields:

1. social_encounter   : "yes" | "no"
2. racial_connotations: "yes" | "no"

RULE: if social_encounter=no, racial_connotations must be "no".
Only reviews that mention other people can contain racialised content.

\u2501\u2501\u2501 SOCIAL ENCOUNTER \u2501\u2501\u2501
YES \u2014 any reference to other visitors or staff as actors in the experience,
      including crowd composition, behaviour, or atmosphere.
  "The receptionist was very friendly and helped us find a locker."
  "There were too many people, I could not swim a single lap."
  "The atmosphere felt hostile, people were staring at us."
  "A staff member asked us to leave the area near the slides."
NO \u2014 review concerns only infrastructure, pricing, cleanliness, or
     personal experience with no other people referenced.
  "The entrance fee is too expensive for what you get."
  "There was mould on the walls of the changing rooms."
  "A great spot to relax in the sun, beautiful facility."
  "The lockers were broken and the showers had no hot water."

\u2501\u2501\u2501 RACIALISED REVIEW \u2501\u2501\u2501
YES \u2014 review contains racial, ethnic, national, religious, or cultural
      markers, whether explicit, implicit, via coded language, or proxy.
      This includes: direct naming of groups, euphemisms, nostalgia framing,
      references to religious dress, language as an ethnic boundary marker,
      or positive celebration of diversity.
  "Too many Arabs who do not know how to behave in public spaces."
  "The pool has changed a lot in recent years, it is no longer the same clientele."
  "Why are people allowed to swim in full clothing? It should not be permitted."
  "The staff barely speak any German, very frustrating."
  "The diversity and multiculturalism here is wonderful."
NO \u2014 social complaint or observation with no racial, ethnic, religious,
     or cultural framing whatsoever.
  "People just do not know how to behave."
  "The pool is not disability friendly at all."
  "The clientele is very youthful, lots of teenagers."
  "Very crowded on weekends, hard to find a lane."

Return ONLY a valid JSON array \u2014 no prose, no markdown fences.
Example output for 3 reviews:
[
  {"review_id": "r1", "social_encounter": "no",  "racial_connotations": "no"},
  {"review_id": "r2", "social_encounter": "yes", "racial_connotations": "no"},
  {"review_id": "r3", "social_encounter": "yes", "racial_connotations": "yes"}
]"""


# ══════════════════════════════════════════════════════════════════════════════
# 3. JSON EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"review_id", "social_encounter", "racial_connotations"}

VALID_COMBOS = {
    ("no",  "no"),
    ("yes", "no"),
    ("yes", "yes"),
}

def _validate_and_normalise(obj: dict) -> dict | None:
    """
    Return a normalised dict if obj is a valid classification, else None.
    Handles minor casing differences (e.g. "Yes" -> "yes").
    """
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return None

    se  = str(obj["social_encounter"]).strip().lower()
    rc  = str(obj["racial_connotations"]).strip().lower()

    # accept n/a as "no" for racial_connotations — model may still output it
    if rc in ("n/a", "na", "n_a"):
        rc = "no"

    combo = (se, rc)
    if combo not in VALID_COMBOS:
        logging.warning(f"Invalid combo {combo} for review_id={obj.get('review_id')}, skipping")
        return None

    return {
        "review_id"          : str(obj["review_id"]),
        "social_encounter"   : se,
        "racial_connotations": rc,
    }


def extract_json_array(content: str) -> list | None:
    """Try several strategies to extract a JSON array from model output."""
    # 1. Direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Bare array anywhere in the string
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def salvage_objects(content: str, batch_data: list, batch_id: str) -> tuple[list, list]:
    """
    When the model returns fewer objects than expected, extract whatever is valid.
    Returns (success_list, failed_list).
    """
    raw_objects = re.findall(r"\{[^{}]+\}", content, re.DOTALL)
    extracted   = []
    for raw in raw_objects:
        try:
            obj = json.loads(raw)
            if "review_id" in obj:
                extracted.append(obj)
        except json.JSONDecodeError:
            continue

    logging.info(f"🔍 Batch {batch_id}: salvaged {len(extracted)} raw objects from partial response")

    valid_ids = set()
    success   = []
    for obj in extracted:
        normed = _validate_and_normalise(obj)
        if normed:
            for rid, _ in batch_data:
                if rid == normed["review_id"] and rid not in valid_ids:
                    success.append(normed)
                    valid_ids.add(rid)
                    break

    failed = [
        {"review_id": rid, "error": "salvage_missing"}
        for rid, _ in batch_data if rid not in valid_ids
    ]
    logging.info(f"📊 Batch {batch_id} salvage: {len(success)} ok, {len(failed)} failed")
    return success, failed


# ══════════════════════════════════════════════════════════════════════════════
# 4. API CALL — single batch, with bounded retry on rate-limit
# ══════════════════════════════════════════════════════════════════════════════

cost_tracker = CostTracker()   # replaced in main()

MAX_RATE_LIMIT_RETRIES = 6


def _call_api(messages: list, batch_id: str, attempt: int = 0) -> requests.Response | None:
    """
    Make one API call.  Retries on 429 up to MAX_RATE_LIMIT_RETRIES times
    with exponential backoff.  Returns None if all retries exhausted.
    """
    if attempt >= MAX_RATE_LIMIT_RETRIES:
        logging.error(f"Batch {batch_id}: rate-limit retries exhausted")
        return None

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type" : "application/json",
                "HTTP-Referer" : "http://localhost:3000",
                "X-Title"      : f"classifier_{batch_id}",
            },
            json={
                "model"      : MODEL,
                "messages"   : messages,
                "temperature": 0.0,
                "max_tokens" : 1500,
            },
            timeout=120,
        )
    except requests.exceptions.Timeout:
        logging.error(f"Batch {batch_id}: timeout (attempt {attempt+1})")
        return None
    except Exception as exc:
        logging.error(f"Batch {batch_id}: request exception {exc}")
        return None

    if r.status_code == 429:
        wait = min(2 ** attempt * 5 + random.uniform(0, 3), 120)
        logging.warning(f"Batch {batch_id}: rate-limited, waiting {wait:.1f}s (attempt {attempt+1})")
        time.sleep(wait)
        return _call_api(messages, batch_id, attempt + 1)

    return r


def classify_batch(batch_data: list, batch_id: str) -> dict:
    """
    Classify one batch of (review_id, review_text) pairs.
    Returns {"success": [...], "failed": [...]}.
    """
    # Budget guard
    status = cost_tracker.budget_status()
    if status == "EXCEEDED":
        logging.error(f"Batch {batch_id}: BUDGET EXCEEDED, skipping")
        return {
            "success": [],
            "failed" : [{"review_id": rid, "error": "budget_exceeded"} for rid, _ in batch_data],
        }
    if status == "WARNING":
        logging.warning(f"Batch {batch_id}: budget at WARNING level")

    # Build prompt — truncate individual reviews at 500 chars to keep context tight
    reviews_text = "\n\n".join(
        f"REVIEW {i+1} (ID: {rid}):\n{text[:500]}{'...' if len(text) > 500 else ''}"
        for i, (rid, text) in enumerate(batch_data)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Classify these {len(batch_data)} pool reviews:\n\n"
            f"{reviews_text}\n\n"
            f"Return a JSON array with exactly {len(batch_data)} objects, "
            f"one per review in the same order, each with: "
            f"review_id, social_encounter, racial_connotations"},
    ]

    t0 = time.time()
    r  = _call_api(messages, batch_id)

    if r is None:
        return {
            "success": [],
            "failed" : [{"review_id": rid, "error": "api_unreachable"} for rid, _ in batch_data],
        }

    elapsed = time.time() - t0

    if r.status_code != 200:
        logging.error(f"Batch {batch_id}: HTTP {r.status_code}")
        return {
            "success": [],
            "failed" : [{"review_id": rid, "error": f"http_{r.status_code}"} for rid, _ in batch_data],
        }

    body = r.json()

    # ── Record ACTUAL token usage ──────────────────────────────────────────
    usage        = body.get("usage", {})
    input_tok    = usage.get("prompt_tokens",     0)
    output_tok   = usage.get("completion_tokens", 0)
    call_cost    = cost_tracker.record_call(input_tok, output_tok, len(batch_data))
    logging.info(
        f"Batch {batch_id}: {elapsed:.1f}s | "
        f"in={input_tok} out={output_tok} | cost=${call_cost:.5f}"
    )
    cost_tracker.log_progress()

    content = body["choices"][0]["message"]["content"]
    arr     = extract_json_array(content)

    # ── Validate each item ────────────────────────────────────────────────
    if arr and len(arr) == len(batch_data):
        success, failed = [], []
        for obj in arr:
            normed = _validate_and_normalise(obj)
            if normed:
                success.append(normed)
            else:
                # find which review_id this was for
                rid = str(obj.get("review_id", "unknown"))
                failed.append({"review_id": rid, "error": "invalid_combo"})
        if failed:
            logging.warning(f"Batch {batch_id}: {len(failed)} items had invalid combos")
        return {"success": success, "failed": failed}

    # ── Partial / malformed response → try to salvage ─────────────────────
    logging.warning(f"Batch {batch_id}: expected {len(batch_data)} objects, got {len(arr) if arr else 0}")
    success, failed = salvage_objects(content, batch_data, batch_id)
    return {"success": success, "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-REVIEW FALLBACK (for stubborn failures)
# ══════════════════════════════════════════════════════════════════════════════

def classify_single(review_id: str, review_text: str) -> dict | None:
    """
    Send one review on its own.  Returns a normed dict on success, None on failure.
    Used only as a last-resort retry for reviews that failed batch classification.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Classify this single pool review:\n\nREVIEW (ID: {review_id}):\n{review_text[:500]}\n\n"
            "Return a JSON array with exactly 1 object."},
    ]
    r = _call_api(messages, f"single_{review_id}")
    if r is None or r.status_code != 200:
        return None

    body      = r.json()
    usage     = body.get("usage", {})
    cost_tracker.record_call(usage.get("prompt_tokens", 0),
                             usage.get("completion_tokens", 0), 1)

    content = body["choices"][0]["message"]["content"]
    arr     = extract_json_array(content)
    if arr:
        normed = _validate_and_normalise(arr[0])
        if normed:
            return normed

    # Also try bare object
    m = re.search(r"\{[^{}]+\}", content, re.DOTALL)
    if m:
        try:
            normed = _validate_and_normalise(json.loads(m.group(0)))
            if normed:
                return normed
        except json.JSONDecodeError:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(chunk_id: str, output_dir: Path,
                    success: list, failed: list) -> Path:
    ckpt = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    data = {
        "chunk_id"    : chunk_id,
        "success"     : success,
        "failed"      : failed,
        "timestamp"   : time.time(),
        "cost_so_far" : cost_tracker.summary()["total_cost"],
    }
    with open(ckpt, "w") as f:
        json.dump(data, f)
    logging.info(f"💾 Checkpoint saved: {ckpt.name} "
                 f"({len(success)} ok, {len(failed)} failed so far)")
    return ckpt


def load_checkpoint(chunk_id: str, output_dir: Path) -> dict | None:
    ckpt = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt.exists():
        logging.info(f"🔄 Checkpoint found for chunk {chunk_id}, resuming…")
        with open(ckpt) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHUNK PROCESSOR
#    Sequential batches (no threading) for accuracy.
#    Retries failures first as small batches, then one-by-one as last resort.
# ══════════════════════════════════════════════════════════════════════════════

def process_chunk(chunk_df: pd.DataFrame, chunk_id: str,
                  output_dir: Path, batch_size: int = 10) -> tuple[list, list]:
    """
    Classify all reviews in chunk_df.  Returns (success_list, failed_list).
    Saves a parquet file classified_chunk_{chunk_id}.parquet on completion.
    """
    # Already done?
    final_file = output_dir / f"classified_chunk_{chunk_id}.parquet"
    if final_file.exists():
        try:
            existing = pd.read_parquet(final_file)
            expected_ids = set(chunk_df["review_id"].astype(str))
            existing_ids = set(existing["review_id"].astype(str)) if "review_id" in existing.columns else set()
            if len(existing) == len(chunk_df) and existing_ids == expected_ids:
                logging.info(f"⏭  Chunk {chunk_id} already complete, skipping")
                return [], []   # caller reads from the parquet if it needs the data

            logging.warning(
                f"⚠️  Chunk {chunk_id} exists but is incomplete/mismatched "
                f"({len(existing)}/{len(chunk_df)} rows); reprocessing"
            )
        except Exception as exc:
            logging.warning(f"⚠️  Chunk {chunk_id} exists but could not be read ({exc}); reprocessing")

    # ── Determine text source per review (summary or raw fallback) ───────
    # Reviews >100 chars should have been summarised upstream.
    # Reviews ≤100 chars are sent as-is.
    # If a long review somehow has no summary, we still fall back to raw
    # text rather than silently dropping it, but we warn loudly.

    has_summary_col = "summary" in chunk_df.columns

    review_data         = []   # (review_id, text)  — what gets sent to the API
    summarised_lengths  = []   # char lengths of summary texts
    raw_lengths         = []   # char lengths of raw texts used as fallback
    missing_summary_ids = []   # reviews that needed a summary but didn't have one

    for _, row in chunk_df.iterrows():
        rid      = str(row["review_id"])
        raw_text = str(row.get("text", row.get("review_text", row.get("review", ""))))
        raw_len  = len(raw_text.strip())

        if has_summary_col and pd.notna(row.get("summary")) and str(row["summary"]).strip():
            # Use the pre-computed summary
            text = str(row["summary"]).strip()
            summarised_lengths.append(len(text))
            review_data.append((rid, text))

        elif raw_len > 0:
            # No summary available — use raw text
            if raw_len > 100:
                # This review should have been summarised but wasn't — flag it
                missing_summary_ids.append(rid)
            raw_lengths.append(raw_len)
            review_data.append((rid, raw_text.strip()))
        # else: completely empty row — skip silently

    # ── Log length statistics ─────────────────────────────────────────────
    n_summarised = len(summarised_lengths)
    n_raw        = len(raw_lengths)
    n_total      = len(review_data)

    if n_summarised:
        avg_s  = sum(summarised_lengths) / n_summarised
        min_s  = min(summarised_lengths)
        max_s  = max(summarised_lengths)
        logging.info(
            f"📝 Chunk {chunk_id} | SUMMARIES  : {n_summarised:,} reviews | "
            f"avg {avg_s:.1f} chars | min {min_s} | max {max_s}"
        )
    else:
        logging.info(f"📝 Chunk {chunk_id} | SUMMARIES  : 0 (no summary column or all null)")

    if n_raw:
        avg_r  = sum(raw_lengths) / n_raw
        min_r  = min(raw_lengths)
        max_r  = max(raw_lengths)
        logging.info(
            f"📄 Chunk {chunk_id} | RAW TEXTS  : {n_raw:,} reviews | "
            f"avg {avg_r:.1f} chars | min {min_r} | max {max_r}"
        )
        if missing_summary_ids:
            logging.warning(
                f"⚠️  Chunk {chunk_id} | {len(missing_summary_ids)} reviews are >100 chars "
                f"but have NO summary — sending raw text. "
                f"First 5 IDs: {missing_summary_ids[:5]}"
            )

    # Estimated input tokens for this chunk (rough but useful for cost sanity-check)
    # System prompt ≈200 tokens, framing ≈50, each char ≈0.25 tokens
    est_chars_per_call  = (sum(summarised_lengths) + sum(raw_lengths)) / max(n_total / 15, 1)
    est_input_tok_call  = 250 + est_chars_per_call * 0.25
    est_output_tok_call = 15 * 70   # ~70 tokens per JSON object
    est_cost_chunk      = (n_total / 15) * (
        (est_input_tok_call  / 1_000_000) * CostTracker.INPUT_PRICE_PER_M +
        (est_output_tok_call / 1_000_000) * CostTracker.OUTPUT_PRICE_PER_M
    )
    logging.info(
        f"💰 Chunk {chunk_id} | ESTIMATE   : {n_total:,} reviews | "
        f"~{est_input_tok_call:.0f} input tok/call | "
        f"~{est_output_tok_call} output tok/call | "
        f"est. chunk cost ${est_cost_chunk:.4f}"
    )

    if not review_data:
        logging.warning(f"Chunk {chunk_id}: no valid text found at all, skipping")
        return None, None

    logging.info(f"📦 Chunk {chunk_id}: {n_total} reviews ready | batch_size={batch_size}")

    # ── Resume from checkpoint if available ───────────────────────────────
    ckpt = load_checkpoint(chunk_id, output_dir)
    if ckpt:
        success = ckpt["success"]
        failed  = ckpt["failed"]
        done_ids = {r["review_id"] for r in success} | {r["review_id"] for r in failed}
        review_data = [(rid, text) for rid, text in review_data if rid not in done_ids]
        logging.info(f"Chunk {chunk_id}: {len(done_ids)} already done, {len(review_data)} remaining")
    else:
        success, failed = [], []

    # ── Pass 1: sequential batches ────────────────────────────────────────
    batches     = [review_data[i:i + batch_size] for i in range(0, len(review_data), batch_size)]
    total_done  = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Chunk {chunk_id}")):
        result   = classify_batch(batch, f"{chunk_id}_b{batch_idx}")
        success.extend(result["success"])
        failed.extend(result["failed"])
        total_done += len(batch)

        # Checkpoint every 50 reviews
        if total_done % 50 == 0:
            save_checkpoint(chunk_id, output_dir, success, failed)

        # Brief pause to be kind to the API
        time.sleep(0.5)

    # ── Pass 2: retry failures as small batches of 5 ─────────────────────
    if failed:
        logging.info(f"Chunk {chunk_id}: pass-2 retry — {len(failed)} failures as batches of 5")
        id_to_text   = dict(review_data)
        retry_data   = [(f["review_id"], id_to_text[f["review_id"]])
                        for f in failed if f["review_id"] in id_to_text]
        still_failed = []

        for i in range(0, len(retry_data), 5):
            batch    = retry_data[i:i + 5]
            result   = classify_batch(batch, f"{chunk_id}_retry_b{i//5}")
            success.extend(result["success"])
            still_failed.extend(result["failed"])
            time.sleep(0.5)

        failed = still_failed

    # ── Pass 3: one-by-one for anything still failing ─────────────────────
    if failed:
        logging.info(f"Chunk {chunk_id}: pass-3 individual retry — {len(failed)} reviews")
        id_to_text   = dict(review_data)
        truly_failed = []

        for item in tqdm(failed, desc=f"Chunk {chunk_id} individual retry"):
            rid  = item["review_id"]
            text = id_to_text.get(rid, "")
            if not text:
                truly_failed.append({"review_id": rid, "error": "text_not_found"})
                continue

            result = classify_single(rid, text)
            if result:
                success.append(result)
                logging.info(f"  ✅ Individual retry succeeded for {rid}")
            else:
                truly_failed.append({"review_id": rid, "error": "individual_retry_failed"})
                logging.warning(f"  ❌ Individual retry failed for {rid}")
            time.sleep(1.0)   # extra caution on individual calls

        failed = truly_failed

    # ── Save final parquet ────────────────────────────────────────────────
    _save_chunk_parquet(chunk_df, chunk_id, success, failed, output_dir)

    # Clean up checkpoint
    ckpt_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt_file.exists():
        ckpt_file.unlink()

    gc.collect()

    rate = len(success) / max(len(review_data), 1)
    logging.info(f"✅ Chunk {chunk_id} done: {len(success)} success, "
                 f"{len(failed)} failed ({rate:.1%})")
    return success, failed


def _save_chunk_parquet(chunk_df: pd.DataFrame, chunk_id: str,
                        success: list, failed: list, output_dir: Path):
    """Build and persist the classified parquet for this chunk."""
    cls_map     = {r["review_id"]: r for r in success}
    failed_ids  = {r["review_id"] for r in failed}

    out = pd.DataFrame({"review_id": chunk_df["review_id"].astype(str)})
    if "was_summarized" in chunk_df.columns:
        out["was_summarized"] = chunk_df["was_summarized"].values

    for col in ["social_encounter", "racial_connotations"]:
        out[col] = out["review_id"].map(lambda x: cls_map.get(x, {}).get(col, "MISSING"))

    out["classification_status"] = out["review_id"].apply(
        lambda x: "FAILED" if x in failed_ids else ("SUCCESS" if x in cls_map else "MISSING")
    )

    path = output_dir / f"classified_chunk_{chunk_id}.parquet"
    out.to_parquet(path, index=False)
    logging.info(f"💾 {path.name} written ({len(out):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROGRESS FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"processed_chunk_ids": [], "total_cost": 0.0,
            "total_reviews_processed": 0, "current_batch_size": 10}


def save_progress(progress_file: Path, processed_ids: set,
                  batch_size: int):
    s = cost_tracker.summary()
    data = {
        "processed_chunk_ids"    : list(processed_ids),
        "total_cost"             : s["total_cost"],
        "total_reviews_processed": s["total_reviews"],
        "current_batch_size"     : batch_size,
        "last_update"            : datetime.now().isoformat(),
    }
    with open(progress_file, "w") as f:
        json.dump(data, f, indent=2)


def get_already_processed_ids(processed_file: Path, failed_file: Path) -> set:
    """Read only the review_id column — efficient even for millions of rows."""
    ids: set = set()
    for fp in (processed_file, failed_file):
        if fp.exists():
            df  = pd.read_csv(fp, usecols=["review_id"])
            ids.update(df["review_id"].astype(str).tolist())
            logging.info(f"Loaded {len(df):,} IDs from {fp.name}")
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 9. LENGTH ANALYSIS  (standalone — no API calls, no cost)
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: list, p: float) -> float:
    """Simple percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def run_length_analysis(all_df: pd.DataFrame, output_dir: Path,
                        raw_col: str) -> None:
    """
    Scan the whole dataframe and report character-length statistics for:
      • reviews that have a summary   (sent as summary)
      • reviews without a summary     (sent as raw text)
      • the subset of raw reviews >100 chars  (missing summaries — cost risk)

    Writes a JSON report to output_dir/length_analysis.json and logs a
    human-readable table.  No API calls are made.
    """
    logging.info("📏 LENGTH ANALYSIS — scanning all reviews (no API calls)…")

    has_summary_col = "summary" in all_df.columns

    summarised_lens      = []
    raw_short_lens       = []   # raw ≤100 chars (expected — no summary needed)
    raw_long_lens        = []   # raw >100 chars (unexpected — summary missing)
    missing_summary_ids  = []

    for _, row in all_df.iterrows():
        rid      = str(row["review_id"])
        raw_text = str(row.get(raw_col, "")).strip()
        raw_len  = len(raw_text)

        summary_text = ""
        if has_summary_col and pd.notna(row.get("summary")):
            summary_text = str(row["summary"]).strip()

        if summary_text:
            summarised_lens.append(len(summary_text))
        elif raw_len > 100:
            raw_long_lens.append(raw_len)
            missing_summary_ids.append(rid)
        else:
            raw_short_lens.append(raw_len)

    def stats(vals: list) -> dict:
        if not vals:
            return {"count": 0, "avg": 0, "min": 0, "max": 0,
                    "p25": 0, "p50": 0, "p75": 0, "p95": 0}
        s = sorted(vals)
        return {
            "count": len(s),
            "avg"  : round(sum(s) / len(s), 1),
            "min"  : s[0],
            "max"  : s[-1],
            "p25"  : _percentile(s, 25),
            "p50"  : _percentile(s, 50),
            "p75"  : _percentile(s, 75),
            "p95"  : _percentile(s, 95),
        }

    ss  = stats(summarised_lens)
    srs = stats(raw_short_lens)
    srl = stats(raw_long_lens)

    # ── Estimate cost impact ───────────────────────────────────────────────
    # chars → tokens at ~0.25 tok/char; output fixed at ~70 tok/review
    total_reviews  = ss["count"] + srs["count"] + srl["count"]
    avg_input_chars = (
        (ss["avg"]  * ss["count"]  +
         srs["avg"] * srs["count"] +
         srl["avg"] * srl["count"])
        / max(total_reviews, 1)
    )
    avg_input_tok_per_call  = 250 + (avg_input_chars * 0.25 * 15)  # 15 reviews/call
    avg_output_tok_per_call = 15 * 70
    n_calls = total_reviews / 15
    est_cost = n_calls * (
        (avg_input_tok_per_call  / 1_000_000) * CostTracker.INPUT_PRICE_PER_M +
        (avg_output_tok_per_call / 1_000_000) * CostTracker.OUTPUT_PRICE_PER_M
    )

    sep = "─" * 62
    logging.info(f"\n{sep}")
    logging.info(f"  LENGTH ANALYSIS REPORT  ({total_reviews:,} reviews total)")
    logging.info(sep)
    logging.info(f"  {'Group':<30} {'N':>8} {'avg':>6} {'min':>5} {'p50':>5} {'p95':>5} {'max':>6}")
    logging.info(f"  {sep}")
    logging.info(f"  {'Summaries (sent as-is)':<30} "
                 f"{ss['count']:>8,} {ss['avg']:>6.1f} {ss['min']:>5} "
                 f"{ss['p50']:>5} {ss['p95']:>5} {ss['max']:>6}")
    logging.info(f"  {'Raw ≤100 chars (expected)':<30} "
                 f"{srs['count']:>8,} {srs['avg']:>6.1f} {srs['min']:>5} "
                 f"{srs['p50']:>5} {srs['p95']:>5} {srs['max']:>6}")
    logging.info(f"  {'Raw >100 chars (⚠ no summary)':<30} "
                 f"{srl['count']:>8,} {srl['avg']:>6.1f} {srl['min']:>5} "
                 f"{srl['p50']:>5} {srl['p95']:>5} {srl['max']:>6}")
    logging.info(sep)
    logging.info(f"  Avg input chars / review : {avg_input_chars:.1f}")
    logging.info(f"  Est. input tok / call    : {avg_input_tok_per_call:.0f}  "
                 f"(system prompt + 15 reviews)")
    logging.info(f"  Est. output tok / call   : {avg_output_tok_per_call}")
    logging.info(f"  Est. total API calls     : {n_calls:,.0f}")
    logging.info(f"  Est. total cost          : ${est_cost:,.2f}  "
                 f"(claude-3.7-sonnet, batch=15)")
    logging.info(sep)

    if missing_summary_ids:
        logging.warning(
            f"  ⚠️  {len(missing_summary_ids):,} reviews are >100 chars but have NO summary.\n"
            f"      These will be sent as raw text and cost significantly more.\n"
            f"      Consider running summarisation on them first.\n"
            f"      First 10 IDs: {missing_summary_ids[:10]}"
        )
    else:
        logging.info("  ✅ All reviews >100 chars have summaries — no cost surprises expected.")

    logging.info(sep)

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "total_reviews"         : total_reviews,
        "summarised"            : ss,
        "raw_short_le100"       : srs,
        "raw_long_gt100_no_sum" : {**srl, "review_ids_sample": missing_summary_ids[:50]},
        "cost_estimate"         : {
            "batch_size"              : 15,
            "model"                   : MODEL,
            "avg_input_chars_review"  : round(avg_input_chars, 1),
            "est_input_tok_per_call"  : round(avg_input_tok_per_call),
            "est_output_tok_per_call" : avg_output_tok_per_call,
            "est_total_calls"         : round(n_calls),
            "est_total_cost_usd"      : round(est_cost, 4),
        },
        "generated_at": datetime.now().isoformat(),
    }
    report_path = output_dir / "length_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"  📄 Full report saved → {report_path}\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLASSIFY  (the actual API pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def run_classify(all_df: pd.DataFrame, output_dir: Path,
                 processed_file: Path, failed_file: Path,
                 chunk_size: int, batch_size: int, budget: float) -> None:
    """Full classification pipeline."""
    global cost_tracker
    cost_tracker = CostTracker(budget_limit=budget)

    progress_file = output_dir / "classification_progress.json"
    prog          = load_progress(progress_file)
    processed_chunk_ids = set(prog["processed_chunk_ids"])
    batch_size          = prog.get("current_batch_size", batch_size)

    cost_tracker.total_cost              = prog.get("total_cost", 0.0)
    cost_tracker.total_reviews_processed = prog.get("total_reviews_processed", 0)

    logging.info(f"Budget: ${budget} | Chunk: {chunk_size:,} | "
                 f"Batch: {batch_size} | Done chunks: {len(processed_chunk_ids)}")

    # Filter already processed
    done_ids = get_already_processed_ids(processed_file, failed_file)
    if done_ids:
        all_df = all_df[~all_df["review_id"].isin(done_ids)]
        logging.info(f"{len(all_df):,} reviews remain after filtering {len(done_ids):,} done")

    if len(all_df) == 0:
        logging.info("All reviews already processed. Done.")
        return

    chunks      = [all_df.iloc[i:i + chunk_size] for i in range(0, len(all_df), chunk_size)]
    all_success : list = []
    all_failed  : list = []

    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_id = f"chunk_{chunk_idx:04d}"

        if chunk_id in processed_chunk_ids:
            logging.info(f"⏭  Skipping {chunk_id}")
            continue

        if cost_tracker.budget_status() == "EXCEEDED":
            logging.error("Budget exceeded — stopping early")
            break

        try:
            c_success, c_failed = process_chunk(
                chunk_df.reset_index(drop=True), chunk_id, output_dir, batch_size
            )
        except Exception as exc:
            logging.error(f"Chunk {chunk_id} crashed: {exc}", exc_info=True)
            continue

        if c_success is None:
            continue

        all_success.extend(c_success)
        all_failed.extend(c_failed)
        processed_chunk_ids.add(chunk_id)
        save_progress(progress_file, processed_chunk_ids, batch_size)
        logging.info(f"Progress: {len(processed_chunk_ids)}/{len(chunks)} chunks | "
                     f"{len(all_success):,} ok | {len(all_failed):,} failed")
        gc.collect()
        time.sleep(random.uniform(1, 3))

    # ── Combine → CSV ──────────────────────────────────────────────────────
    classified = sorted(output_dir.glob("classified_chunk_*.parquet"))
    if not classified:
        logging.error("No classified chunk files found.")
        return

    logging.info(f"Combining {len(classified)} chunk parquets…")
    final_df = pd.concat([pd.read_parquet(f) for f in classified], ignore_index=True)
    if processed_file.exists():
        final_df = pd.concat([pd.read_csv(processed_file), final_df], ignore_index=True)
    final_df.to_csv(processed_file, index=False)
    logging.info(f"✅ {len(final_df):,} rows → {processed_file}")

    if all_failed:
        failed_df = pd.DataFrame(all_failed)
        if failed_file.exists():
            failed_df = pd.concat([pd.read_csv(failed_file), failed_df], ignore_index=True)
        failed_df.to_csv(failed_file, index=False)
        logging.info(f"📄 {len(failed_df):,} failed → {failed_file}")

    if progress_file.exists():
        progress_file.unlink()

    s = cost_tracker.summary()
    logging.info(
        f"\n{'='*60}\n"
        f"CLASSIFICATION COMPLETE\n"
        f"  Successful     : {len(all_success):,}\n"
        f"  Failed         : {len(all_failed):,}\n"
        f"  Success rate   : {len(all_success)/max(len(all_success)+len(all_failed),1):.1%}\n"
        f"  Total cost     : ${s['total_cost']:.4f}\n"
        f"  Cost/review    : ${s['cost_per_review']:.6f}\n"
        f"  Input tokens   : {s['total_input_tokens']:,}\n"
        f"  Output tokens  : {s['total_output_tokens']:,}\n"
        f"  Reviews/hour   : {s['reviews_per_hour']:.0f}\n"
        f"  Elapsed        : {s['elapsed_hours']:.2f} h\n"
        f"{'='*60}"
    )
    summary_path = output_dir / "classification_summary.json"
    with open(summary_path, "w") as f:
        json.dump({**s, "completion_time": datetime.now().isoformat(),
                   "total_success": len(all_success),
                   "total_failed" : len(all_failed)}, f, indent=2)
    logging.info(f"Summary → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="classifier_v3",
        description=(
            "Pool review classifier.\n\n"
            "MODES:\n"
            "  lengths   — scan the dataset and report character-length stats +\n"
            "              cost estimate. No API calls, runs in seconds.\n"
            "  classify  — run the full classification pipeline (API calls).\n"
            "  both      — run length analysis first, then classify.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode",
        choices=["lengths", "classify", "both"],
        help="What to run (see above).",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    DEFAULT_BASE = (
        "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/"
        "TheWEBConf_TheWeb4Good_2026/working_data/reprocessing"
    )
    parser.add_argument(
        "--input", default=None,
        help=f"Path to input parquet file. Default: <base>/unprocessed_rerun.parquet",
    )
    parser.add_argument(
        "--base", default=DEFAULT_BASE,
        help=f"Base directory for all paths. Default: {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for chunk parquets, logs, checkpoints. "
             "Default: <base>/classification_chunks/",
    )
    parser.add_argument(
        "--raw-col", default="text",
        help="Column name for the original review text (used when no summary). "
             "Default: text",
    )

    # ── Classification tuning ──────────────────────────────────────────────
    parser.add_argument(
        "--batch-size", type=int, default=15,
        help="Reviews per API call (default: 15).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5_000,
        help="Reviews per processing chunk (default: 5000).",
    )
    parser.add_argument(
        "--budget", type=float, default=500.0,
        help="Hard budget cap in USD (default: 500).",
    )

    return parser.parse_args()


def main():
    args       = parse_args()
    base       = Path(args.base)
    input_file = Path(args.input) if args.input else base / "unprocessed_rerun.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else base / "classification_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_file = base / "reprocessed_success.csv"
    failed_file    = base / "reprocessed_failed.csv"

    setup_logging(output_dir)
    logging.info(f"🚀 classifier_v3 | mode={args.mode} | input={input_file}")

    # ── Load data once (shared by both modes) ─────────────────────────────
    logging.info(f"Loading {input_file} …")
    all_df = pd.read_parquet(input_file)
    all_df["review_id"] = all_df["review_id"].astype(str)
    logging.info(f"{len(all_df):,} reviews loaded")

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.mode in ("lengths", "both"):
        run_length_analysis(all_df, output_dir, raw_col=args.raw_col)

    if args.mode in ("classify", "both"):
        run_classify(
            all_df,
            output_dir     = output_dir,
            processed_file = processed_file,
            failed_file    = failed_file,
            chunk_size     = args.chunk_size,
            batch_size     = args.batch_size,
            budget         = args.budget,
        )


if __name__ == "__main__":
    main()