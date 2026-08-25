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
MODEL = "anthropic/claude-4.6-sonnet"   # claude-3.5 and 3.7-sonnet no longer on OpenRouter 11.05.2026

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
    for h in logger.handlers[:]:
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
# ══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    INPUT_PRICE_PER_M  = 3.00
    OUTPUT_PRICE_PER_M = 15.00

    def __init__(self, budget_limit: float = 500.0):
        self.total_reviews_processed = 0
        self.total_input_tokens      = 0
        self.total_output_tokens     = 0
        self.total_cost              = 0.0
        self.start_time              = time.time()
        self.budget_limit            = budget_limit
        self._lock                   = threading.RLock()

    def record_call(self, input_tokens: int, output_tokens: int, reviews_in_batch: int):
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

SYSTEM_PROMPT = """You are an annnotation model. Classify each Google Maps swimming pool review using the following schema. 
 
SOCIAL ENCOUNTER (field: "social")    
YES — the review references other people(users/staff): their presence, behaviour,  
appearance, language, or the atmosphere they create.  

NO — the review concerns only the physical environment, facilities,  
pricing, or the reviewer's solitary experience. 

OUTPUT FORMAT (STRICT):
- Return ONLY a JSON array
- No markdown, no explanatory text
- Each object: {"review_id": string, "social": "yes"|"no"}
- Maintain EXACT review order
- Never add, remove, or reorder items
"""


# ══════════════════════════════════════════════════════════════════════════════
# 3. JSON EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"review_id", "social"}

def _validate_and_normalise(obj: dict) -> dict | None:
    if not REQUIRED_KEYS.issubset(obj.keys()):
        missing = REQUIRED_KEYS - set(obj.keys())
        logging.debug(f"Validation failed: missing keys {missing} in {obj.get('review_id', 'unknown')}")
        return None
    
    se = str(obj["social"]).strip().lower()
    if se not in {"yes", "no"}:
        logging.debug(f"Validation failed: invalid social value '{se}' for review_id {obj.get('review_id')}")
        return None
    
    return {
        "review_id": str(obj["review_id"]),
        "social": se,
    }

def extract_json_array(content: str) -> list | None:
    content = content.strip()
    
    # Fix trailing commas before closing brackets (common model output issue)
    content = re.sub(r',\s*\]', ']', content)
    # NOTE: do NOT replace single quotes with double quotes — apostrophes in
    # review text that leaks into output would corrupt valid JSON

    # 1. Direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    for pattern in code_block_patterns:
        m = re.search(pattern, content)
        if m:
            try:
                result = json.loads(m.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                # Try to fix truncated JSON inside fence
                fixed = m.group(1)
                if fixed.count('[') > fixed.count(']'):
                    fixed += ']'
                try:
                    result = json.loads(fixed)
                    if isinstance(result, list):
                        logging.debug("Fixed missing closing bracket in fenced block")
                        return result
                except json.JSONDecodeError:
                    pass

    # 3. Bracket-counting to find top-level array (handles content before/after)
    stack = []
    for i, char in enumerate(content):
        if char == '[':
            stack.append(i)
        elif char == ']' and stack:
            start = stack.pop()
            if not stack:  # top-level array complete
                candidate = content[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    # Try to salvage by trimming to last complete object
                    last_obj_end = candidate.rfind('}')
                    if last_obj_end > 0:
                        fixed = candidate[:last_obj_end+1] + ']'
                        try:
                            result = json.loads(fixed)
                            if isinstance(result, list):
                                logging.debug("Fixed truncated array by trimming to last complete object")
                                return result
                        except json.JSONDecodeError:
                            pass

    return None
def salvage_objects(content: str, batch_data: list, batch_id: str) -> tuple[list, list]:
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
# 4. API CALL
# ══════════════════════════════════════════════════════════════════════════════

cost_tracker = CostTracker()

MAX_ATTEMPTS = 3   # 1 original attempt + 2 retries on 429
    

def _call_api(messages: list, batch_id: str, attempt: int = 0) -> requests.Response | None:
    if attempt >= MAX_ATTEMPTS:
        logging.error(f"Batch {batch_id}: max attempts ({MAX_ATTEMPTS}) exhausted")
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
                "max_tokens" : 2000,
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
        logging.warning(f"Batch {batch_id}: rate-limited, waiting {wait:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
        time.sleep(wait)
        return _call_api(messages, batch_id, attempt + 1)

    return r


def classify_batch(batch_data: list, batch_id: str) -> dict:
    """
    Classify one batch of (review_id, review_text) pairs.
    Returns {"success": [...], "failed": [...]}.
    Does NOT retry — retries are handled globally in run_classify.
    """
    status = cost_tracker.budget_status()
    if status == "EXCEEDED":
        logging.error(f"Batch {batch_id}: BUDGET EXCEEDED, skipping")
        return {
            "success": [],
            "failed" : [{"review_id": rid, "error": "budget_exceeded"} for rid, _ in batch_data],
        }
    if status == "WARNING":
        logging.warning(f"Batch {batch_id}: budget at WARNING level")

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
    user_message = f"Classify these {len(batch_data)} reviews in order:\n\n{reviews_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
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

    usage      = body.get("usage", {})
    input_tok  = usage.get("prompt_tokens",     0)
    output_tok = usage.get("completion_tokens", 0)
    call_cost  = cost_tracker.record_call(input_tok, output_tok, len(batch_data))
    logging.info(
        f"Batch {batch_id}: {elapsed:.1f}s | "
        f"in={input_tok} out={output_tok} | cost=${call_cost:.5f}"
    )
    cost_tracker.log_progress()

    content = body["choices"][0]["message"]["content"]
    arr     = extract_json_array(content)

    if arr and len(arr) == len(batch_data):
        sent_ids     = [rid for rid, _ in batch_data]
        returned_ids = [str(obj.get("review_id")) for obj in arr]

        if returned_ids != sent_ids:
            n_mismatches = sum(r != s for r, s in zip(returned_ids, sent_ids))
            logging.warning(
                f"Batch {batch_id}: {n_mismatches} review_id mismatch(es) — "
                f"sent={sent_ids} | got={returned_ids} — correcting by position"
            )
            for obj, (expected_id, _) in zip(arr, batch_data):
                obj["review_id"] = expected_id

        success, failed = [], []
        for obj in arr:
            normed = _validate_and_normalise(obj)
            if normed:
                success.append(normed)
            else:
                rid = str(obj.get("review_id", "unknown"))
                failed.append({"review_id": rid, "error": "invalid_value"})
        if failed:
            logging.warning(f"Batch {batch_id}: {len(failed)} items had invalid value")
        return {"success": success, "failed": failed}

    logging.warning(f"Batch {batch_id}: expected {len(batch_data)} objects, got {len(arr) if arr else 0}")
    success, failed = salvage_objects(content, batch_data, batch_id)
    return {"success": success, "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-REVIEW FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def classify_single(review_id: str, review_text: str) -> dict | None:
    """
    Send one review on its own.
    Used only in the final global retry pass — called once per review, no further retries.
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

    body  = r.json()
    usage = body.get("usage", {})
    cost_tracker.record_call(usage.get("prompt_tokens", 0),
                             usage.get("completion_tokens", 0), 1)

    content = body["choices"][0]["message"]["content"]
    arr     = extract_json_array(content)
    if arr:
        normed = _validate_and_normalise(arr[0])
        if normed:
            if normed.get("review_id") != review_id:
                logging.warning(
                    f"single_{review_id}: model returned review_id={normed.get('review_id')} — correcting to requested id"
                )
                normed["review_id"] = review_id
            return normed

    m = re.search(r"\{[^{}]+\}", content, re.DOTALL)
    if m:
        try:
            normed = _validate_and_normalise(json.loads(m.group(0)))
            if normed:
                if normed.get("review_id") != review_id:
                    logging.warning(
                        f"single_{review_id}: model returned review_id={normed.get('review_id')} — correcting to requested id"
                    )
                    normed["review_id"] = review_id
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


def load_checkpoint_rows(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    success_rows: list[dict] = []
    failed_rows: list[dict] = []
    sources: list[str] = []

    for fp in sorted(output_dir.glob("checkpoint_chunk_*.json")):
        try:
            with open(fp) as f:
                ckpt = json.load(f)
        except Exception as exc:
            logging.warning(f"Skipping unreadable checkpoint {fp.name}: {exc}")
            continue

        success = ckpt.get("success", []) or []
        failed = ckpt.get("failed", []) or []

        for row in success:
            item = dict(row)
            item.setdefault("classification_status", "SUCCESS")
            item.setdefault("classification_error", None)
            item.setdefault("failure_batch_id", None)
            item.setdefault("failure_root_cause", None)
            success_rows.append(item)

        for row in failed:
            item = dict(row)
            item.setdefault("classification_status", "FAILED")
            if "classification_error" not in item and "error" in item:
                item["classification_error"] = item.get("error")
            if "failure_batch_id" not in item and "batch_id" in item:
                item["failure_batch_id"] = item.get("batch_id")
            if "failure_root_cause" not in item and "root_cause" in item:
                item["failure_root_cause"] = item.get("root_cause")
            failed_rows.append(item)

        if success or failed:
            sources.append(f"{fp.name} ({len(success):,} ok, {len(failed):,} failed)")

    return pd.DataFrame(success_rows), pd.DataFrame(failed_rows), sources


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHUNK PROCESSOR  (Pass 1 only — no retries)
# ══════════════════════════════════════════════════════════════════════════════

def process_chunk(chunk_df: pd.DataFrame, chunk_id: str,
                  output_dir: Path, batch_size: int = 5,
                  chunk_suffix: str = "") -> tuple[list, list]:
    """
    Pass 1: classify all reviews in chunk_df sequentially.
    Returns (success_list, failed_list) — failures are NOT retried here;
    they bubble up to run_classify for the global retry cascade.
    """
    review_data      = []
    skipped_failures = []

    for _, row in chunk_df.iterrows():
        rid  = str(row["review_id"])
        text = str(row.get("trans_review", "")).strip()
        if not text:
            skipped_failures.append({"review_id": rid, "error": "text_not_found"})
            continue
        review_data.append((rid, text))

    n_over_limit = sum(1 for _, t in review_data if len(t) > TRUNCATION_LIMIT)
    if n_over_limit:
        logging.info(
            f"✂️  Chunk {chunk_id}: {n_over_limit:,} texts exceed {TRUNCATION_LIMIT} chars "
            f"and will be truncated before sending to the API"
        )

    if not review_data:
        logging.warning(f"Chunk {chunk_id}: no valid text found; "
                        f"marking {len(skipped_failures):,} reviews failed")
        _save_chunk_parquet(chunk_df, chunk_id, [], skipped_failures, output_dir, chunk_suffix)
        return [], skipped_failures

    logging.info(f"📦 Chunk {chunk_id}: {len(chunk_df)} reviews ready | batch_size={batch_size}")

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

    # ── Pass 1: sequential batches, no retries ────────────────────────────
    batches    = [review_data[i:i + batch_size] for i in range(0, len(review_data), batch_size)]
    total_done = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Chunk {chunk_id}")):
        result = classify_batch(batch, f"{chunk_id}_b{batch_idx}")
        success.extend(result["success"])
        failed.extend(result["failed"])
        total_done += len(batch)

        if total_done % 50 == 0:
            save_checkpoint(chunk_id, output_dir, success, failed)

        time.sleep(0)

    # Include any reviews that had no text
    failed.extend(skipped_failures)

    # ── Save chunk parquet (preliminary — may be updated after global retries) ──
    _save_chunk_parquet(chunk_df, chunk_id, success, failed, output_dir, chunk_suffix)

    # Clean up checkpoint now that the chunk pass is done
    ckpt_file = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt_file.exists():
        ckpt_file.unlink()

    gc.collect()

    rate = len(success) / max(len(chunk_df), 1)
    logging.info(f"📦 Chunk {chunk_id} pass-1 done: {len(success)} success, "
                 f"{len(failed)} failed ({rate:.1%} success rate)")
    return success, failed


def _save_chunk_parquet(chunk_df: pd.DataFrame, chunk_id: str,
                        success: list, failed: list,
                        output_dir: Path, chunk_suffix: str = ""):
    cls_map    = {r["review_id"]: r for r in success}
    failed_map = {str(r["review_id"]): str(r.get("error", "failed")) for r in failed}
    success_ids = set(cls_map)

    out = pd.DataFrame({"review_id": chunk_df["review_id"].astype(str)})
    if "was_summarized" in chunk_df.columns:
        out["was_summarized"] = chunk_df["was_summarized"].values

    for col in ["social"]:
        out[col] = out["review_id"].map(lambda x: cls_map.get(x, {}).get(col))

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
# 8. GLOBAL RETRY CASCADE  (called once after all chunks complete)
# ══════════════════════════════════════════════════════════════════════════════

def run_global_retries(
    all_failed: list,
    id_to_text: dict,
    output_dir: Path,
    initial_batch_size: int = 5,
) -> tuple[list, list]:
    """
    Three-pass global retry cascade, run once after all primary chunks finish.

      Pass 2 — batch retry at initial_batch_size (same as primary run)
      Pass 3 — batch retry at max(1, initial_batch_size // 2)
      Pass 4 — individual retry, exactly once per review, no further retries

    Returns (recovered_success, truly_failed).
    """
    # Strip reviews whose text was never found — no point retrying those
    retryable = [f for f in all_failed if f["review_id"] in id_to_text]
    permanent = [f for f in all_failed if f["review_id"] not in id_to_text]
    if permanent:
        logging.info(f"Global retries: {len(permanent)} reviews have no text — skipping them")

    if not retryable:
        logging.info("Global retries: nothing retryable.")
        return [], permanent

    recovered: list = []

    # ── Pass 2: full batch size ────────────────────────────────────────────
    logging.info(
        f"\n{'─'*60}\n"
        f"  GLOBAL PASS 2 — batch retry | "
        f"{len(retryable)} failures | batch_size={initial_batch_size}\n"
        f"{'─'*60}"
    )
    retry_data   = [(f["review_id"], id_to_text[f["review_id"]]) for f in retryable]
    still_failed = []

    for i in tqdm(range(0, len(retry_data), initial_batch_size),
                  desc="Global pass-2"):
        batch  = retry_data[i:i + initial_batch_size]
        result = classify_batch(batch, f"gretry2_b{i // initial_batch_size}")
        recovered.extend(result["success"])
        still_failed.extend(result["failed"])
        time.sleep(0)

    logging.info(f"Global pass-2 done: {len(recovered)} recovered, "
                 f"{len(still_failed)} still failing")

    # ── Pass 3: half batch size ────────────────────────────────────────────
    if still_failed:
        half_batch = max(1, initial_batch_size // 2)
        logging.info(
            f"\n{'─'*60}\n"
            f"  GLOBAL PASS 3 — half-batch retry | "
            f"{len(still_failed)} failures | batch_size={half_batch}\n"
            f"{'─'*60}"
        )
        retry_data   = [(f["review_id"], id_to_text[f["review_id"]])
                        for f in still_failed if f["review_id"] in id_to_text]
        still_failed2 = []

        for i in tqdm(range(0, len(retry_data), half_batch),
                      desc="Global pass-3"):
            batch  = retry_data[i:i + half_batch]
            result = classify_batch(batch, f"gretry3_b{i // half_batch}")
            recovered.extend(result["success"])
            still_failed2.extend(result["failed"])
            time.sleep(0)

        logging.info(f"Global pass-3 done: {len(recovered)} total recovered, "
                     f"{len(still_failed2)} still failing")
        still_failed = still_failed2

    # ── Pass 4: individual, exactly once ──────────────────────────────────
    if still_failed:
        logging.info(
            f"\n{'─'*60}\n"
            f"  GLOBAL PASS 4 — individual retry (once) | "
            f"{len(still_failed)} reviews\n"
            f"{'─'*60}"
        )
        truly_failed = []

        for item in tqdm(still_failed, desc="Global pass-4 individual"):
            rid  = item["review_id"]
            text = id_to_text.get(rid, "")
            if not text:
                truly_failed.append({"review_id": rid, "error": "text_not_found"})
                continue

            result = classify_single(rid, text)
            if result:
                recovered.append(result)
                logging.info(f"  ✅ Individual retry succeeded: {rid}")
            else:
                truly_failed.append({"review_id": rid, "error": "individual_retry_failed"})
                logging.warning(f"  ❌ Individual retry failed: {rid}")
            time.sleep(1.0)

        still_failed = truly_failed

    truly_failed = still_failed + permanent
    logging.info(
        f"\nGlobal retry cascade complete: "
        f"{len(recovered)} recovered | {len(truly_failed)} permanently failed"
    )
    return recovered, truly_failed


# ══════════════════════════════════════════════════════════════════════════════
# 9. PROGRESS FILE HELPERS
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
    ids: set  = set()
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
# 10. LENGTH / CHARACTER DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _bracket_distribution(lengths: list[int]) -> dict:
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


def run_length_analysis(all_df: pd.DataFrame, output_dir: Path, raw_col: str) -> None:
    logging.info("📏 LENGTH ANALYSIS — scanning all reviews (no API calls)…")

    has_summary_col = "summary" in all_df.columns
    summarised_lens     = []
    raw_short_lens      = []
    raw_long_lens       = []
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

    total_reviews = sa["count"]
    avg_sent_chars = (
        sum(min(l, TRUNCATION_LIMIT) for l in all_effective) / max(total_reviews, 1)
    )
    avg_input_tok_per_call  = 250 + (avg_sent_chars * 0.25 * 5)
    avg_output_tok_per_call = 5 * 25
    n_calls   = total_reviews / 5
    est_cost  = n_calls * (
        (avg_input_tok_per_call  / 1_000_000) * CostTracker.INPUT_PRICE_PER_M +
        (avg_output_tok_per_call / 1_000_000) * CostTracker.OUTPUT_PRICE_PER_M
    )

    sep = "─" * 70
    logging.info(f"\n{sep}")
    logging.info(f"  LENGTH ANALYSIS REPORT  ({total_reviews:,} reviews total)")
    logging.info(sep)
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
# 11. CLASSIFY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_classify(all_df: pd.DataFrame, output_dir: Path,
                 processed_file: Path, failed_file: Path,
                 chunk_size: int, batch_size: int, budget: float) -> None:
    global cost_tracker
    cost_tracker = CostTracker(budget_limit=budget)

    logging.info(f"Budget: ${budget} | Chunk: {chunk_size:,} | Batch: {batch_size}")

    existing_chunks = list(output_dir.glob("classified_chunk_*.parquet"))
    chunk_suffix = ""
    if existing_chunks:
        chunk_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(
            f"Detected {len(existing_chunks)} existing chunk parquets; "
            f"using suffix '{chunk_suffix}' for new chunks"
        )

    done_ids = get_already_successfully_processed_ids(processed_file, output_dir)
    if done_ids:
        all_df = all_df[~all_df["review_id"].isin(done_ids)]
        logging.info(
            f"{len(all_df):,} reviews remain after filtering "
            f"{len(done_ids):,} successful review_ids"
        )

    if len(all_df) == 0:
        logging.info("All reviews already processed. Done.")
        return

    # Build a global id→text lookup used by the retry cascade
    id_to_text: dict = {}
    for _, row in all_df.iterrows():
        rid  = str(row["review_id"])
        text = str(row.get("trans_review", "")).strip()
        if text:
            id_to_text[rid] = text

    chunks      = [all_df.iloc[i:i + chunk_size] for i in range(0, len(all_df), chunk_size)]
    all_success : list = []
    all_failed  : list = []

    # ── Pass 1: process every chunk (no retries inside) ───────────────────
    logging.info(
        f"\n{'═'*60}\n"
        f"  PASS 1 — primary classification | "
        f"{len(all_df):,} reviews across {len(chunks)} chunks\n"
        f"{'═'*60}"
    )
    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_id = f"chunk_{chunk_idx:04d}"

        if cost_tracker.budget_status() == "EXCEEDED":
            logging.error("Budget exceeded — stopping early")
            break

        try:
            c_success, c_failed = process_chunk(
                chunk_df.reset_index(drop=True),
                chunk_id,
                output_dir,
                batch_size,
                chunk_suffix,
            )
        except Exception as exc:
            logging.error(f"Chunk {chunk_id} crashed: {exc}", exc_info=True)
            continue

        if c_success is None:
            continue

        all_success.extend(c_success)
        all_failed.extend(c_failed)
        logging.info(
            f"Progress: {chunk_idx + 1}/{len(chunks)} chunks | "
            f"{len(all_success):,} ok | {len(all_failed):,} failed"
        )
        gc.collect()
        

    # ── Global retry cascade (Passes 2–4) ─────────────────────────────────
    if all_failed:
        logging.info(
            f"\n{'═'*60}\n"
            f"  GLOBAL RETRY CASCADE | {len(all_failed):,} failures to recover\n"
            f"{'═'*60}"
        )
        recovered, truly_failed = run_global_retries(
            all_failed      = all_failed,
            id_to_text      = id_to_text,
            output_dir      = output_dir,
            initial_batch_size = batch_size,
        )
        all_success.extend(recovered)
        all_failed = truly_failed
    else:
        logging.info("No failures — skipping global retry cascade.")

    # ── Combine chunk parquets → CSV ───────────────────────────────────────
    classified = sorted(output_dir.glob("classified_chunk_*.parquet"))
    if not classified:
        logging.error("No classified chunk files found.")
        return

    logging.info(f"Combining {len(classified)} chunk parquets…")
    final_df = pd.concat([pd.read_parquet(f) for f in classified], ignore_index=True)

    # Overwrite FAILED rows with retry successes
    recovered_ids = {r["review_id"] for r in all_success}
    final_df.loc[
        (final_df["classification_status"] == "FAILED") &
        (final_df["review_id"].isin(recovered_ids)),
        "classification_status"
    ] = "SUCCESS"

    ckpt_success_df, ckpt_failed_df, ckpt_sources = load_checkpoint_rows(output_dir)
    if ckpt_sources:
        logging.info(
            f"Reconciling leftover checkpoints: {len(ckpt_success_df):,} success, "
            f"{len(ckpt_failed_df):,} failed"
        )
        logging.info("Checkpoint sources:\n" + "\n".join(f"     • {s}" for s in ckpt_sources))

    if not ckpt_success_df.empty:
        ckpt_success_df = ckpt_success_df.reindex(columns=final_df.columns, fill_value=None)
        ckpt_success_df["classification_status"] = ckpt_success_df["classification_status"].fillna("SUCCESS")
    if not ckpt_failed_df.empty:
        ckpt_failed_df = ckpt_failed_df.reindex(columns=final_df.columns, fill_value=None)
        ckpt_failed_df["classification_status"] = ckpt_failed_df["classification_status"].fillna("FAILED")

    if not ckpt_success_df.empty or not ckpt_failed_df.empty:
        final_df = pd.concat([final_df, ckpt_success_df, ckpt_failed_df], ignore_index=True, sort=False)
        final_df["classification_status"] = final_df["classification_status"].astype(str).str.upper()
        final_df["_status_priority"] = (final_df["classification_status"] == "SUCCESS").astype(int)
        final_df = final_df.sort_values(["review_id", "_status_priority"]).drop_duplicates(
            subset=["review_id"], keep="last"
        )
        final_df = final_df.drop(columns=["_status_priority"])

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
                   "total_failed" : len(all_failed),
                   "checkpoint_success": len(ckpt_success_df),
                   "checkpoint_failed" : len(ckpt_failed_df)}, f, indent=2)
    logging.info(f"Summary → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="classifier_v5",
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
        "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data_annotation/stage_1_social"
    )
    parser.add_argument("--input",      default=None)
    parser.add_argument("--base",       default=DEFAULT_BASE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-col",    default="trans_review",
                        help="Column name for the original review text. Default: trans_review")
    parser.add_argument("--batch-size", type=int,   default=5)
    parser.add_argument("--chunk-size", type=int,   default=10000)
    parser.add_argument("--budget",     type=float, default=1000.0)

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
        f"🚀 classifier_v5 | mode={args.mode} | input={input_file} | "
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