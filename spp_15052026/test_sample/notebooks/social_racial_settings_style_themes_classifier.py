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
MODEL = "anthropic/claude-3.7-sonnet"   # claude-3.7-sonnet on OpenRouter

TRUNCATION_LIMIT = 700  # chars — reviews longer than this are truncated before API call


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
        self._lock                   = threading.RLock()

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

SYSTEM_PROMPT = """You are an annotation model. Classify each pool review using the hierarchical schema below.
Work through Steps 1–4 in order. Each step is only applicable when the prior condition is met.
             
---------------------
STEP 1 — SOCIAL ENCOUNTER  (field: "social_encounter")
---------------------
Does the review describe, evaluate, or imply an interaction with — or an
impression shaped by — other people at the pool (users or staff)?

  YES — the review references other people: their presence, behaviour,
        appearance, language, or the atmosphere they create.

  NO  — the review concerns only the physical environment, facilities,
        pricing, or the reviewer's solitary experience.
        
---------------------
STEP 2 — RACIALISED SOCIAL ENCOUNTER  (field: "racial_connotations")
Only applicable when social_encounter = "yes". Otherwise set to "N/A".
---------------------
Is the social encounter filtered through a racial, ethnic, cultural, or
religious lens? Explicit slurs are not required — proxies, euphemisms, and
coded language all count. Positive racialisation also counts.

  YES — the review links people or behaviour to race, ethnicity, national/
        regional origin, religion, or cultural markers (dress, language,
        food, customs).

  NO  — the encounter is described without any such framing. Generic
        complaints about noise, behaviour, age, gender, or disability
        without racial coding are NO.

  N/A — social_encounter = "no".

---------------------
STEP 3 — SETTING  (field: "setting")
Only applicable when racial_connotations = "yes". Otherwise set to "N/A".
---------------------
What is the context of the racialised content? Choose the best fit:

  INCIDENT   — the review describes a specific event or interaction that
               occurred at the pool. The account is bounded in time.

  ATMOSPHERE — the review expresses a general impression or ongoing opinion
               about the social environment at the pool. No single
               event is the focus.

  BOTH       — the review contains both a specific incident AND a general
               atmospheric impression, each with racial framing.

  OTHER      — the review is racialised but does not fit the above three
               categories. Append a brief note (≤ 8 words) explaining why.

  UNSPECIFIED — the racialised content is too vague to assign any category
                above.

 ---------------------
STEP 4 - NATURE OF RACIALISED CONTENT  (field: "style_of_racialisation") 
Only applicable when racial_connotations = "yes". Otherwise set to "N/A".
------------------------------------
How is the racialised content expressed in the review? 
    EXPLICIT — the review uses direct references to race, ethnicity, nationality, religion, or culture.
      This includes overt slurs, stereotypes, or positive affirmations linked to these identities.

    IMPLICIT — the review conveys racialised content through coded language, euphemisms, 
    or proxies without direct mention of race, ethnicity, nationality, religion, or culture. 
    This can include dog whistles, stereotypes, or any language that implies racialised meaning without explicit identifiers.
               

---------------------
STEP 5 — RACIALISED THEMES  (field: "racialised_themes")
Only applicable when racial_connotations = "yes". Otherwise set to [].
---------------------
Identify every distinct theme that is racialised in the review — i.e.
explicitly or implicitly linked to a racial, ethnic, religious, or
cultural group. Assign up to 3 themes.

Each label must be:
  • 1–2 words (snake_case)
  • derived from the review content, not imposed from a fixed list
  • only included if it carries racial/ethnic/cultural framing

---------------------
VALID COMBINATIONS
---------------------
  social_encounter=no
    → racial_connotations="N/A", setting="N/A", style_of_racialisation="N/A", racialised_themes=[]

  social_encounter=yes, racial_connotations=no
    → setting="N/A", style_of_racialisation="N/A", racialised_themes=[]

  social_encounter=yes, racial_connotations=yes
    → setting= one of [incident, atmosphere, both, other, unspecified]
    → style_of_racialisation= one of [explicit, implicit]
    → racialised_themes= [1–3 labels]

---------------------
OUTPUT FORMAT
---------------------
Return ONLY a valid JSON array — no prose, no markdown fences.
Each object must include all four fields.

[
  {"review_id": "r1", "social_encounter": "no",  "racial_connotations": "N/A", "setting": "N/A", "style_of_racialisation": "N/A", "racialised_themes": []},
  {"review_id": "r2", "social_encounter": "yes", "racial_connotations": "no",  "setting": "N/A", "style_of_racialisation": "N/A", "racialised_themes": []},
  {"review_id": "r3", "social_encounter": "yes", "racial_connotations": "yes", "setting": "atmosphere", "style_of_racialisation": "implicit", "racialised_themes": ["crowd_composition", "belonging"]}
]"""


# ══════════════════════════════════════════════════════════════════════════════
# 3. JSON EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"review_id", "social_encounter", "racial_connotations","setting", "style_of_racialisation", "racialised_themes"}

def _validate_and_normalise(obj: dict) -> dict | None:
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return None

    se  = str(obj["social_encounter"]).strip().lower()
    rc  = str(obj["racial_connotations"]).strip().lower()
    st  = str(obj["setting"]).strip().lower()
    sor = str(obj["style_of_racialisation"]).strip().lower()

    # normalise N/A variants (compare lowercase, store as canonical "N/A")
    rc  = "N/A" if rc  in ("n/a", "na", "n_a") else rc
    st  = "N/A" if st  in ("n/a", "na", "n_a") else st
    sor = "N/A" if sor in ("n/a", "na", "n_a") else sor

    if se == "no":
        if not (rc == "N/A" and st == "N/A" and sor == "N/A"):
            return None
    elif se == "yes" and rc == "no":
        if not (st == "N/A" and sor == "N/A"):
            return None
    elif se == "yes" and rc == "yes":
        valid_settings = {"incident", "atmosphere", "both", "other", "unspecified"}
        valid_sor      = {"explicit", "implicit", "N/A"}
        if st not in valid_settings or sor not in valid_sor:
            return None
    else:
        return None
   

    # ── Validate / clean racialised_themes ────────────────────────────────
    raw_themes = obj.get("racialised_themes", [])

    if isinstance(raw_themes, str):
        try:
            raw_themes = json.loads(raw_themes)
        except (json.JSONDecodeError, TypeError):
            raw_themes = [t.strip() for t in raw_themes.split(",") if t.strip()]

    if not isinstance(raw_themes, list):
        raw_themes = []

    themes = [str(t).strip().lower() for t in raw_themes if str(t).strip()]

    # Enforce consistency with combo
    if rc != "yes" and themes:
        logging.warning(
            f"review_id={obj.get('review_id')}: racialised_themes non-empty "
            f"but racial_connotations={rc!r} — clearing themes"
        )
        themes = []

    if rc == "yes" and not themes:
        logging.warning(
            f"review_id={obj.get('review_id')}: racial_connotations=yes "
            f"but racialised_themes is empty — assigning ['other']"
        )
        themes = ["other"]

    return {
        "review_id"          : str(obj["review_id"]),
        "social_encounter"   : se,
        "racial_connotations": rc,
        "setting"            : st,
        "style_of_racialisation": sor,
        "racialised_themes"  : themes,
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
                "max_tokens" : 2000,   # increased to accommodate racialised_themes field
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
    Reviews longer than TRUNCATION_LIMIT chars are truncated before sending.
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

    # Build prompt — truncate reviews at TRUNCATION_LIMIT chars
    review_lines = []
    n_truncated  = 0
    for i, (rid, text) in enumerate(batch_data):
        if len(text) > TRUNCATION_LIMIT:
            display = text[:TRUNCATION_LIMIT] + "…"
            n_truncated += 1
        else:
            display = text
        review_lines.append(f"REVIEW {i+1} (ID: {rid}):\n{display}")

    if n_truncated:
        logging.debug(
            f"Batch {batch_id}: {n_truncated}/{len(batch_data)} reviews "
            f"truncated to {TRUNCATION_LIMIT} chars"
        )

    reviews_text = "\n\n".join(review_lines)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Classify these {len(batch_data)} pool reviews:\n\n"
            f"{reviews_text}\n\n"
            f"Return a JSON array with exactly {len(batch_data)} objects, "
            f"one per review in the same order, each with: "
            f"review_id, social_encounter, racial_connotations, "
            f"setting, racialised_themes"},
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
        try:
            error_payload = json.dumps(r.json(), ensure_ascii=False)[:800]
        except Exception:
            error_payload = (r.text or "")[:800]
        logging.error(f"Batch {batch_id}: HTTP {r.status_code} | response={error_payload}")
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
    text = review_text[:TRUNCATION_LIMIT] + ("…" if len(review_text) > TRUNCATION_LIMIT else "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Classify this single pool review:\n\nREVIEW (ID: {review_id}):\n{text}\n\n"
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
# ══════════════════════════════════════════════════════════════════════════════

def process_chunk(chunk_df: pd.DataFrame, chunk_id: str,
                  output_dir: Path, batch_size: int = 10, chunk_suffix: str = "") -> tuple[list, list]:
    """
    Classify all reviews in chunk_df.  Returns (success_list, failed_list).
    Saves a parquet file classified_chunk_{chunk_id}{chunk_suffix}.parquet on completion.
    """
    # Already done?
    review_data = []
    skipped_failures = []
    for _, row in chunk_df.iterrows():
        rid  = str(row["review_id"])
        text = str(row.get("trans_review", "")).strip()
        if not text:
            skipped_failures.append({"review_id": rid, "error": "text_not_found"})
            continue
        if len(text) > TRUNCATION_LIMIT:
            text = text[:TRUNCATION_LIMIT] + "…"
        review_data.append((rid, text))

    n_total = len(chunk_df)

    n_over_limit = sum(1 for _, t in review_data if len(t) > TRUNCATION_LIMIT)
    if n_over_limit:
        logging.info(
            f"✂️  Chunk {chunk_id}: {n_over_limit:,} texts exceed {TRUNCATION_LIMIT} chars "
            f"and will be truncated before sending to the API"
        )

    if not review_data:
        logging.warning(f"Chunk {chunk_id}: no valid text found; marking {len(skipped_failures):,} reviews failed")
        _save_chunk_parquet(chunk_df, chunk_id, [], skipped_failures, output_dir)
        return [], skipped_failures

    logging.info(f"📦 Chunk {chunk_id}: {n_total} reviews ready | batch_size={batch_size}")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    ckpt = load_checkpoint(chunk_id, output_dir)
    if ckpt:
        success  = ckpt["success"]
        failed   = ckpt["failed"]
        done_ids = {r["review_id"] for r in success} | {r["review_id"] for r in failed}
        review_data = [(rid, text) for rid, text in review_data if rid not in done_ids]
        logging.info(f"Chunk {chunk_id}: {len(done_ids)} already done, {len(review_data)} remaining")
    else:
        success, failed = [], []

    # ── Pass 1: sequential batches ────────────────────────────────────────
    batches    = [review_data[i:i + batch_size] for i in range(0, len(review_data), batch_size)]
    total_done = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Chunk {chunk_id}")):
        result   = classify_batch(batch, f"{chunk_id}_b{batch_idx}")
        success.extend(result["success"])
        failed.extend(result["failed"])
        total_done += len(batch)

        if total_done % 50 == 0:
            save_checkpoint(chunk_id, output_dir, success, failed)

        time.sleep(0)

    # ── Pass 2: retry failures as small batches of 5 ─────────────────────
    if failed:
        logging.info(f"Chunk {chunk_id}: pass-2 retry — {len(failed)} failures as batches of 5")
        id_to_text   = dict(review_data)
        retry_data   = [(f["review_id"], id_to_text[f["review_id"]])
                        for f in failed if f["review_id"] in id_to_text]
        still_failed = []

        for i in range(0, len(retry_data), 5):
            batch  = retry_data[i:i + 5]
            result = classify_batch(batch, f"{chunk_id}_retry_b{i//5}")
            success.extend(result["success"])
            still_failed.extend(result["failed"])
            time.sleep(0)

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
            time.sleep(1.0)

        failed = truly_failed

    # ── Save final parquet ────────────────────────────────────────────────
    final_failed_map = {str(item["review_id"]): item for item in (failed + skipped_failures)}
    final_failed = list(final_failed_map.values())
    _save_chunk_parquet(chunk_df, chunk_id, success, final_failed, output_dir, chunk_suffix="")

    ckpt_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt_file.exists():
        ckpt_file.unlink()

    gc.collect()

    rate = len(success) / max(len(chunk_df), 1)
    logging.info(f"✅ Chunk {chunk_id} done: {len(success)} success, "
                 f"{len(final_failed)} failed ({rate:.1%})")
    return success, final_failed


def _save_chunk_parquet(chunk_df: pd.DataFrame, chunk_id: str,
                        success: list, failed: list, output_dir: Path, chunk_suffix: str = ""):
    """Build and persist the classified parquet for this chunk."""
    cls_map    = {r["review_id"]: r for r in success}
    failed_map = {str(r["review_id"]): str(r.get("error", "failed")) for r in failed}
    success_ids = set(cls_map)

    out = pd.DataFrame({"review_id": chunk_df["review_id"].astype(str)})
    if "was_summarized" in chunk_df.columns:
        out["was_summarized"] = chunk_df["was_summarized"].values

    for col in ["social_encounter", "racial_connotations", "setting", "style_of_racialisation"]:
        out[col] = out["review_id"].map(lambda x: cls_map.get(x, {}).get(col))

    # racialised_themes stored as a JSON string in the parquet
    out["racialised_themes"] = out["review_id"].map(
        lambda x: json.dumps(cls_map.get(x, {}).get("racialised_themes", [])) if x in success_ids else None
    )

    out["classification_status"] = out["review_id"].apply(
        lambda x: "SUCCESS" if x in success_ids else "FAILED"
    )

    out["classification_error"] = out["review_id"].map(
        lambda x: failed_map.get(x) if x not in success_ids else None
    )

    path = output_dir / f"classified_chunk_{chunk_id}{chunk_suffix}.parquet"
    out.to_parquet(path, index=False)
    logging.info(f"💾 {path.name} written ({len(out):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROGRESS FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {
        "processed_chunk_ids": [],
        "total_cost": 0.0,
        "total_reviews_processed": 0,
    }


def save_progress(progress_file: Path, processed_ids: set, batch_size: int):
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


def get_already_successfully_processed_ids(processed_file: Path, output_dir: Path) -> set:
    ids: set = set()
    sources: list[str] = []

    if processed_file.exists():
        df = pd.read_csv(processed_file)
        if "classification_status" in df.columns:
            df = df[df["classification_status"].astype(str).str.upper() == "SUCCESS"]
        ids.update(df["review_id"].astype(str).tolist())
        sources.append(f"{processed_file.name} ({len(df):,} SUCCESS rows)")

    for fp in sorted(output_dir.glob("classified_chunk_*.parquet")):
        df = pd.read_parquet(fp)
        if "classification_status" in df.columns:
            df = df[df["classification_status"].astype(str).str.upper() == "SUCCESS"]
        ids.update(df["review_id"].astype(str).tolist())
        sources.append(f"{fp.name} ({len(df):,} SUCCESS rows)")

    if sources:
        logging.info("Loaded successful review_ids from: " + ", ".join(sources))
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# 9. LENGTH / CHARACTER DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _bracket_distribution(lengths: list[int]) -> dict:
    """
    Count reviews falling into character-length brackets.
    Brackets are chosen to be meaningful for our 700-char truncation threshold.
    """
    brackets = [
        (0,   100,  "0–100"),
        (101,  250,  "101–250"),
        (251,  500,  "251–500"),
        (501,  700,  "501–700"),
        (701,  None, "701+"),
      
    ]
    counts = {}
    for lo, hi, label in brackets:
        if hi is None:
            counts[label] = sum(1 for l in lengths if l > lo)
        else:
            counts[label] = sum(1 for l in lengths if lo <= l <= hi)
    return counts


def run_length_analysis(all_df: pd.DataFrame, output_dir: Path,
                        raw_col: str) -> None:
    """
    Scan the whole dataframe and report character-length statistics and
    bracket distributions for:
      • reviews that have a summary   (sent as summary)
      • reviews without a summary     (sent as raw text, ≤700 chars — expected)
      • reviews without a summary     (sent as raw text, >700 chars — cost risk)

    Also reports how many reviews would be truncated at TRUNCATION_LIMIT chars.
    Writes output_dir/length_analysis.json.
    """
    logging.info("📏 LENGTH ANALYSIS — scanning all reviews (no API calls)…")

    has_summary_col = "summary" in all_df.columns

    summarised_lens     = []
    raw_short_lens      = []   # raw ≤100 chars
    raw_long_lens       = []   # raw >100 chars, no summary
    missing_summary_ids = []

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

    # Effective lengths (what actually gets passed to the API)
    all_effective = summarised_lens + raw_short_lens + raw_long_lens

    def stats(vals: list) -> dict:
        if not vals:
            return {"count": 0, "avg": 0, "min": 0, "max": 0,
                    "p25": 0, "p50": 0, "p75": 0, "p95": 0,
                    "bracket_distribution": {}}
        s = sorted(vals)
        return {
            "count"               : len(s),
            "avg"                 : round(sum(s) / len(s), 1),
            "min"                 : s[0],
            "max"                 : s[-1],
            "p25"                 : _percentile(s, 25),
            "p50"                 : _percentile(s, 50),
            "p75"                 : _percentile(s, 75),
            "p95"                 : _percentile(s, 95),
            "bracket_distribution": _bracket_distribution(s),
        }

    ss  = stats(summarised_lens)
    srs = stats(raw_short_lens)
    srl = stats(raw_long_lens)
    sa  = stats(all_effective)

    n_would_truncate = sum(1 for l in all_effective if l > TRUNCATION_LIMIT)
    pct_truncate     = n_would_truncate / max(len(all_effective), 1) * 100

    # ── Cost estimate ──────────────────────────────────────────────────────
    total_reviews = sa["count"]
    avg_sent_chars = (
        sum(min(l, TRUNCATION_LIMIT) for l in all_effective) / max(total_reviews, 1)
    )
    avg_input_tok_per_call  = 250 + (avg_sent_chars * 0.25 * 5)
    avg_output_tok_per_call = 5 * 90   # ~90 tokens per object (themes add tokens)
    n_calls   = total_reviews / 5
    est_cost  = n_calls * (
        (avg_input_tok_per_call  / 1_000_000) * CostTracker.INPUT_PRICE_PER_M +
        (avg_output_tok_per_call / 1_000_000) * CostTracker.OUTPUT_PRICE_PER_M
    )

    # ── Console report ─────────────────────────────────────────────────────
    sep = "─" * 70
    logging.info(f"\n{sep}")
    logging.info(f"  LENGTH ANALYSIS REPORT  ({total_reviews:,} reviews total)")
    logging.info(sep)

    # Per-group summary table
    logging.info(
        f"  {'Group':<34} {'N':>8} {'avg':>6} {'p50':>5} {'p75':>5} {'p95':>5} {'max':>6}"
    )
    logging.info(f"  {sep}")
    for label, s in [
        ("Summaries (sent as-is)",         ss),
        ("Raw ≤700 chars (expected)",       srs),
        ("Raw >700 chars (⚠ no summary)",  srl),
        ("ALL EFFECTIVE LENGTHS",           sa),
    ]:
        logging.info(
            f"  {label:<34} {s['count']:>8,} {s['avg']:>6.1f} "
            f"{s['p50']:>5} {s['p75']:>5} {s['p95']:>5} {s['max']:>6}"
        )

    # Bracket distribution for all effective lengths
    logging.info(f"\n  CHARACTER-LENGTH BRACKET DISTRIBUTION (effective texts sent to API)")
    logging.info(f"  {sep}")
    logging.info(f"  {'Bracket':<14} {'Count':>9} {'%':>7}  {'Bar'}")
    logging.info(f"  {sep}")
    brackets = sa.get("bracket_distribution", {})
    total    = max(sa["count"], 1)
    for bracket, count in brackets.items():
        pct  = count / total * 100
        bar  = "█" * int(pct / 2)
        marker = " ← truncation threshold" if bracket == "501–700" else (
                 " ← TRUNCATED" if bracket in ("701–1000", "1001–2000", "2001+") else "")
        logging.info(f"  {bracket:<14} {count:>9,} {pct:>6.1f}%  {bar}{marker}")

    logging.info(sep)
    logging.info(f"  Truncation limit         : {TRUNCATION_LIMIT} chars")
    logging.info(f"  Reviews to be truncated  : {n_would_truncate:,} ({pct_truncate:.1f}%)")
    logging.info(f"  Avg chars sent / review  : {avg_sent_chars:.1f}  (after truncation)")
    logging.info(f"  Est. input tok / call    : {avg_input_tok_per_call:.0f}  (system + 5 reviews)")
    logging.info(f"  Est. output tok / call   : {avg_output_tok_per_call}  (themes field adds tokens)")
    logging.info(f"  Est. total API calls     : {n_calls:,.0f}")
    logging.info(f"  Est. total cost          : ${est_cost:,.2f}  (claude-3.7-sonnet, batch=5)")
    logging.info(sep)

    # if missing_summary_ids:
    #     logging.warning(
    #         f"  ⚠️  {len(missing_summary_ids):,} reviews are >700 chars but have NO summary.\n"
    #         f"      First 10 IDs: {missing_summary_ids[:10]}"
    #     )
    # else:
    #     logging.info("  ✅ All reviews >700chars have summaries — no cost surprises expected.")
    # logging.info(sep)

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "total_reviews"         : total_reviews,
        "truncation_limit"      : TRUNCATION_LIMIT,
        "n_would_truncate"      : n_would_truncate,
        "pct_would_truncate"    : round(pct_truncate, 2),
        "summarised"            : ss,
        "raw_short_le100"       : srs,
        "raw_long_gt100_no_sum" : {**srl, "review_ids_sample": missing_summary_ids[:50]},
        "all_effective"         : sa,
        "cost_estimate"         : {
            "batch_size"              : 5,
            "model"                   : MODEL,
            "truncation_limit"        : TRUNCATION_LIMIT,
            "avg_sent_chars_review"   : round(avg_sent_chars, 1),
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
# 10. CLASSIFY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_classify(all_df: pd.DataFrame, output_dir: Path,
                 processed_file: Path, failed_file: Path,
                 chunk_size: int, batch_size: int, budget: float) -> None:
    global cost_tracker
    cost_tracker = CostTracker(budget_limit=budget)

    logging.info(f"Budget: ${budget} | Chunk: {chunk_size:,} | "
                 f"Batch: {batch_size}")

    # Detect if old chunks exist; if so, use timestamp suffix for new chunks to avoid overwrites
    existing_chunks = list(output_dir.glob("classified_chunk_*.parquet"))
    chunk_suffix = ""
    if existing_chunks:
        chunk_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Detected {len(existing_chunks)} existing chunk parquets; using suffix '{chunk_suffix}' for new chunks")

    done_ids = get_already_successfully_processed_ids(processed_file, output_dir)
    if done_ids:
        all_df = all_df[~all_df["review_id"].isin(done_ids)]
        logging.info(f"{len(all_df):,} reviews remain after filtering {len(done_ids):,} successful review_ids")

    if len(all_df) == 0:
        logging.info("All reviews already processed. Done.")
        return

    chunks      = [all_df.iloc[i:i + chunk_size] for i in range(0, len(all_df), chunk_size)]
    all_success : list = []
    all_failed  : list = []

    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_id = f"chunk_{chunk_idx:04d}"

        if cost_tracker.budget_status() == "EXCEEDED":
            logging.error("Budget exceeded — stopping early")
            break

        try:
            c_success, c_failed = process_chunk(
                chunk_df.reset_index(drop=True), chunk_id, output_dir, batch_size, chunk_suffix
            )
        except Exception as exc:
            logging.error(f"Chunk {chunk_id} crashed: {exc}", exc_info=True)
            continue

        if c_success is None:
            continue

        all_success.extend(c_success)
        all_failed.extend(c_failed)
        logging.info(f"Progress: {chunk_idx + 1}/{len(chunks)} chunks | "
                     f"{len(all_success):,} ok | {len(all_failed):,} failed")
        gc.collect()
        time.sleep(random.uniform(1, 3))

    # ── Combine chunk parquets → CSV ───────────────────────────────────────
    classified = sorted(output_dir.glob("classified_chunk_*.parquet"))
    if not classified:
        logging.error("No classified chunk files found.")
        return

    logging.info(f"Combining {len(classified)} chunk parquets…")
    final_df = pd.concat([pd.read_parquet(f) for f in classified], ignore_index=True)

    success_df = final_df[final_df["classification_status"] == "SUCCESS"].copy()
    if processed_file.exists():
        existing_success = pd.read_csv(processed_file)
        if "classification_status" in existing_success.columns:
            existing_success = existing_success[
                existing_success["classification_status"].astype(str).str.upper() == "SUCCESS"
            ]
        success_df = pd.concat([existing_success, success_df], ignore_index=True)
    success_df = success_df.drop_duplicates(subset=["review_id"], keep="last")
    success_df.to_csv(processed_file, index=False)
    logging.info(f"✅ {len(success_df):,} successful rows → {processed_file}")

    failed_df = final_df[final_df["classification_status"] == "FAILED"].copy()
    if failed_file.exists():
        existing_failed = pd.read_csv(failed_file)
        if "classification_status" in existing_failed.columns:
            existing_failed = existing_failed[
                existing_failed["classification_status"].astype(str).str.upper() == "FAILED"
            ]
        failed_df = pd.concat([existing_failed, failed_df], ignore_index=True)
    failed_df = failed_df[~failed_df["review_id"].isin(success_df["review_id"])]
    failed_df = failed_df.drop_duplicates(subset=["review_id"], keep="last")
    failed_df.to_csv(failed_file, index=False)
    logging.info(f"📄 {len(failed_df):,} failed rows → {failed_file}")

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
        prog="classifier_v4",
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

    DEFAULT_BASE = (
        "/home/ethel/Desktop/swimming_pool_project/spp_15052026/test_sample"
    )
    parser.add_argument("--input",      default=None)
    parser.add_argument("--base",       default=DEFAULT_BASE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-col",    default="trans_review",
                        help="Column name for the original review text. Default: text")
    parser.add_argument("--batch-size", type=int,   default=5)
    parser.add_argument("--chunk-size", type=int,   default=5000)
    parser.add_argument("--budget",     type=float, default=500.0)

    return parser.parse_args()


def main():
    args       = parse_args()
    base       = Path(args.base)
    input_file = Path(args.input) if args.input else base / "unprocessed_rerun.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else base / "classification_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_file = base / "processed_success.csv"
    failed_file    = base / "processed_failed.csv"

    setup_logging(output_dir)
    logging.info(
        f"🚀 classifier_v4 | mode={args.mode} | input={input_file} | "
        f"truncation_limit={TRUNCATION_LIMIT}"
    )

    logging.info(f"Loading {input_file} …")
    all_df = pd.read_parquet(input_file)
    all_df["review_id"] = all_df["review_id"].astype(str)
    logging.info(f"{len(all_df):,} reviews loaded")

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