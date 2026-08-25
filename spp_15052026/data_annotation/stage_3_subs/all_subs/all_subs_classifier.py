"""
classifier_v6.py  —  Pool-review racial classifier
====================================================
Improvements over v5
---------------------
* Resumption check happens BEFORE any work starts (not mid-pipeline)
* main() ordering fixed (args parsed first)
* API keys passed explicitly — no mutable global
* Async queue-based logging — 20 threads no longer fight over the log lock
* tqdm parallelism fixed (position= per worker, disabled in CI)
* time.sleep(0) removed; real inter-batch pause configurable
* Checkpoint frequency raised to every 5 reviews
* gc.collect() removed (Python GC handles it)
* CostTracker no longer reassigned as a global mid-run
* All magic numbers lifted to named constants at the top
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS  (edit here — nowhere else)
# ══════════════════════════════════════════════════════════════════════════════

MODEL            = "anthropic/claude-sonnet-4-6"
TRUNCATION_LIMIT = 700        # chars — reviews longer than this are trimmed
MAX_API_ATTEMPTS = 3          # 1 original + 2 retries on 429
INTER_BATCH_SLEEP= 0.0        # seconds between batches inside a chunk (set >0 to rate-limit)
CHECKPOINT_EVERY = 5          # reviews between checkpoint saves

INPUT_PRICE_PER_M  = 3.00     # USD per 1 M input tokens
OUTPUT_PRICE_PER_M = 15.00    # USD per 1 M output tokens

DEFAULT_BASE = str(Path(__file__).resolve().parent)

DEFAULT_INPUT = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data/social_reviews.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOGGING  (async queue — threads never block each other)
# ══════════════════════════════════════════════════════════════════════════════

_log_listener: QueueListener | None = None


def setup_logging(output_dir: Path) -> Path:
    """
    Route all log records through a single-threaded QueueListener so worker
    threads never contend on the log lock.
    Returns the path of the log file.
    """
    global _log_listener

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    log_queue = queue.Queue(-1)
    _log_listener = QueueListener(log_queue, fh, ch, respect_handler_level=True)
    _log_listener.start()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(QueueHandler(log_queue))

    logging.info(f"Logging initialised → {log_file}")
    return log_file


def stop_logging() -> None:
    if _log_listener:
        _log_listener.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  COST TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    def __init__(self, budget_limit: float = 500.0):
        self.budget_limit            = budget_limit
        self.total_reviews_processed = 0
        self.total_input_tokens      = 0
        self.total_output_tokens     = 0
        self.total_cost              = 0.0
        self.start_time              = time.time()
        self._lock                   = threading.RLock()

    def record_call(self, input_tokens: int, output_tokens: int,
                    reviews_in_batch: int) -> float:
        cost = (input_tokens  / 1_000_000) * INPUT_PRICE_PER_M \
             + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
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

    def log_progress(self) -> None:
        s    = self.summary()
        icon = {"OK": "✅", "WARNING": "⚠️", "EXCEEDED": "🚨"}.get(
            s["budget_status"], ""
        )
        logging.info(
            f"{icon} COST ${s['total_cost']:.4f} / ${self.budget_limit:.2f} | "
            f"{s['total_reviews']:,} reviews | "
            f"${s['cost_per_review']:.6f}/review | "
            f"remaining ${s['remaining_budget']:.4f}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 2.5. CHUNK LOGGER (for parallel traceability)
# ══════════════════════════════════════════════════════════════════════════════

class ChunkLogger:
    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
    
    def info(self, msg):
        logging.info(f"[{self.chunk_id}] {msg}")
    
    def warning(self, msg):
        logging.warning(f"[{self.chunk_id}] {msg}")
    
    def error(self, msg):
        logging.error(f"[{self.chunk_id}] {msg}")
    
    def debug(self, msg):
        logging.debug(f"[{self.chunk_id}] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.6. API KEY TRACKER (per-key credit exhaustion tracking)
# ══════════════════════════════════════════════════════════════════════════════

class ApiKeyTracker:
    """
    Track which API keys have exhausted credits (HTTP 402).
    Allows intelligent key rotation that skips exhausted keys.
    Periodically re-enables keys in case credits are refilled.
    """
    def __init__(self, api_keys: list[str]):
        self.api_keys = api_keys
        self.exhausted = set()  # Keys that hit 402
        self.exhausted_at = {}  # {key: timestamp}
        self.lock = threading.Lock()
        self.recheck_interval = 300  # Re-enable after 5 minutes
    
    def mark_exhausted(self, api_key: str) -> None:
        """Mark a key as exhausted (hit 402)."""
        with self.lock:
            self.exhausted.add(api_key)
            self.exhausted_at[api_key] = time.time()
            key_suffix = api_key[-6:] if len(api_key) > 6 else "???"
            logging.warning(f"API key …{key_suffix} marked EXHAUSTED (402 credit limit)")
    
    def try_reenable_keys(self) -> None:
        """Periodically re-enable keys that might have had credits refilled."""
        with self.lock:
            now = time.time()
            to_reenable = []
            for key in list(self.exhausted):
                if now - self.exhausted_at.get(key, now) > self.recheck_interval:
                    to_reenable.append(key)
            
            for key in to_reenable:
                self.exhausted.discard(key)
                key_suffix = key[-6:] if len(key) > 6 else "???"
                logging.info(f"Re-enabling API key …{key_suffix} (possible credit refill)")
    
    def get_available_key(self, prefer_idx: int = 0) -> str | None:
        """
        Get an available key (not exhausted). Start from prefer_idx and rotate.
        Returns None if all keys exhausted.
        """
        self.try_reenable_keys()
        
        with self.lock:
            available = [k for k in self.api_keys if k not in self.exhausted]
            if not available:
                logging.error("ALL API keys exhausted (402). Waiting for credit refills...")
                return None
            
            # Round-robin from prefer_idx among available keys
            idx = prefer_idx % len(available)
            return available[idx]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_COMBINED = """You are an annotation model. Classify each Google Maps swimming pool review on four dimensions.
 
Racialisation: attribution of characteristics, behaviours, attitudes, or presence to race, ethnicity,
nationality, culture, or religion — explicit, implicit, or coded; any tone.
 
INPUT: JSON array of {"review_id": string, "text": string}.
 
OUTPUT FORMAT (strict):
- Return ONLY a JSON array; no markdown, no commentary
- Each object must have these fields; preserve input order
{
    "review_id": string,
    "theme": string | [string,...], // up to 3 themes allowed (primary, secondary, tertiary)
    "connotation": string, // one of: Positive | Negative | Neutral
    "approach": "yes" | "no",
    "context": "yes" | "no"
}
 
ALLOWED THEME LABELS (use ONLY these exact labels, up to 3):
- Staff
- Crowd
- Belonging
- Language
- Clothing
- Hygiene
- Security
- Safety
- Access
- Other

If multiple themes apply, return an array of these labels (max 3). Do not invent or free-text other labels.

---
1. THEME — what/who is being racialised, not merely mentioned
---
 
Staff Pool staff racialised as individuals or a group — composition,
                features, conduct, attitude, or behaviour.
  ✓ "The Slavic-sounding lifeguard is disrespectful"
  ✓ "A true German receptionist — so polite and efficient"
  ✓ "The pool staff are racist"
  ✗ "Too many foreigners at the pool" → Crowd Composition (users, not staff)
  ✗ "The receptionist was rude" → no racialisation
  ✗ "The pool is poorly managed" → no racialisation
 
Crowd Users racialised — who is present, absent, or how the pool
Composition population is characterised in racial/ethnic/national terms.
  ✓ "Germans don't belong here anymore; the place is full of foreigners"
  ✓ "It's a beautiful pool, but there are too many Arabs"
  ✓ "There were people of many backgrounds there"
  ✗ "The pool was very crowded on Saturday" → no racialisation
  ✗ "The staff are all foreigners" → Staff (workforce, not user demographic)
  ✗ "I felt unwelcome as a foreigner" → Belonging
 
Belonging Racialisation of cultural fit, acceptance, or entitlement to be
                at the pool — felt by reviewer or projected onto others.
  ✓ "As a Muslim female, I did not feel accepted here"
  ✓ "I felt like an outsider, being the only Black person there"
  ✓ "As a German, I felt displaced"
  ✗ "The pool was unwelcoming — long queues and rude staff" → no racialisation
  ✗ "There are too many foreigners here" → Crowd Composition
  ✗ "I was denied entry because of my skin colour" → Access/Entry
 
Language Racialisation of what language is spoken or not, by whom —
                staff, clientele, or signage.
  ✓ "The staff only speak German"
  ✓ "All you hear at this pool is fluent Turkish"
  ✓ "The rules were only posted in German — clearly not for everyone"
  ✗ "The announcements were too quiet to hear" → no racialisation
  ✗ "Staff were uncommunicative and unhelpful" → no racialisation
  ✗ "The signage was confusing" → no racialisation
 
Clothing Swimwear or attire racialised; includes unequal dress-code
                enforcement along racial, cultural, or religious lines.
  ✓ "I was not allowed to wear my burka"
  ✓ "White women in bikinis were fine but my modest swimwear was flagged"
  ✓ "People were allowed into the pool fully clothed from head to toe"
  ✗ "The changing rooms were too small" → no racialisation
  ✗ "I was asked to shower before entering" → Hygiene / Etiquette
  ✗ "The dress code was strictly enforced" → no racialisation
 
Hygiene Cleanliness of people, bodies, or sanitary practices racialised.
  ✓ "Dirty Turkish people everywhere"
  ✓ "These foreigners don't shower before entering — no hygiene culture"
  ✓ "All the Slavic people I saw were tidy"
  ✗ "They bring food to the pool edge" → Pool Etiquette
  ✗ "The pool itself was dirty" → no racialisation
  ✗ "They speak only Turkish" → language
 
Pool Behaviour, attitude, or adherence to pool norms racialised.
Etiquette If about cleanliness specifically, use Hygiene.
  ✓ "They do not know German sauna culture"
  ✓ "These foreigners are loud and have no respect for others"
  ✓ "French people are so loud and disruptive"
  ✗ "They are dirty and don't shower before entering" → Hygiene
  ✗ "I was not allowed to wear my burka" → Clothing
  ✗ "The pool rules are not clearly posted" → no racialisation
 
Security Security procedures or enforcement racialised — who is checked,
                how, and whether applied equally.
  ✓ "The guard searched us multiple times, but not the Germans ahead of us"
  ✓ "Security followed us around because they heard us speak French"
  ✓ "We were profiled at the entrance — clearly because of how we look"
  ✗ "I felt unsafe around the group of men" → Safety
  ✗ "The security guard was rude to everyone" → no racialisation
  ✗ "There weren't enough security staff" → no racialisation
 
Safety Personal safety or threat perception racialised — feeling unsafe
                or attributing danger/criminal behaviour to a racialised group.
  ✓ "It doesn't feel safe with so many southern men gawking at you"
  ✓ "An Arab boy stole my daughter's toy"
  ✓ "The Turkish men who come in groups make the pool feel threatening"
  ✗ "The security guard searched us unfairly" → Security
  ✗ "The pool area was slippery and dangerous" → no racialisation
  ✗ "The lifeguard wasn't paying attention" → no racialisation
 
Access/Entry Entry or access racialised — denied, restricted, delayed, or
                informally excluded on racial/ethnic/cultural/religious grounds.
  ✓ "We were denied access because of our skin colour"
  ✓ "They let the white families in immediately, but made us wait"
  ✓ "Muslims are not allowed to use the sauna, apparently"
  ✗ "The pool was genuinely at capacity" → no racialisation
  ✗ "I was not allowed to wear my burka" → Clothing
  ✗ "I was denied entry" → no racialisation (no racial attribution)
 
Other Racialisation present and attributable but fits none of the above.
                Flag for review.

 
---
2. CONNOTATION — tone of the racialised content only (not the review overall)
---
Apply multiple values if the review contains distinct racialised statements of different tones.
 
Positive Racialisation is favourable — positive stereotypes, flattering
            attributions, or appreciation linked to racial/ethnic/cultural identity.
  ✓ "The German lifeguard is so patient with the kids"
  ✓ "All the Slavic people I saw were very neat and tidy"
  ✓ "As a Muslim female I felt very welcomed and respected here"
  ✗ "The receptionist was very helpful" → no racial attribution
  ✗ "The staff were friendly to everyone" → no racial attribution
  ✗ "Great facilities and good opening hours" → no racial dimension
 
Negative Racialisation is unfavourable — us-vs-them dynamic, harm/threat
            attributed to a group, or discrimination framed in racial terms.
  ✓ "Dirty Turkish people everywhere — no hygiene culture"
  ✓ "It doesn't feel safe with so many southern men gawking at you"
  ✓ "Germans no longer have a place here — too many foreigners"
  ✗ "The queue was long and the staff were slow" → no racial attribution
  ✗ "The changing rooms were dirty" → no racialisation
  ✗ "The lifeguard was rude to everyone" → no racial dimension
 
Neutral Racialisation present but tone is indeterminate — factual or
            demographic observation; no qualifying adjectives conveying approval
            or disapproval.
  ✓ "The lifeguard speaks English and German"
  ✓ "There were people of many backgrounds at the pool"
  ✓ "The pool now employs foreigners"
  ✗ "The German lifeguard is so patient" → Positive (quality attributed)
  ✗ "Too many foreigners" → Negative (unfavourable attribution)
  ✗ "These foreigners don't know how to behave" → Negative
 
---
3. APPROACH — does the review name the group directly?
---
 
yes A racial, ethnic, national, cultural, or religious identity is directly
        named.
  ✓ "There are too many Turkish people at the pool"
  ✓ "As a Muslim female I did not feel welcome"
  ✓ "Afghan families were not checked at the security gate"
  ✗ "They were allowed in fully clothed from head to toe" → no (implies Muslim)
  ✗ "There is no room for locals here anymore" → no (coded ethnic in-group)
  ✗ "The staff member had a strong accent and was rude" → no (implies foreign)
 
no No group is directly named — racialisation conveyed through coded
        language, cultural shorthand, regional implication, or inference.
  ✓ "They were allowed in fully clothed from head to toe" (implies Muslim dress)
  ✓ "There is no room for locals here anymore" (implies ethnic in-group)
  ✓ "The staff member had a strong accent and was rude" (implies foreign)
  ✗ "There are too many Turkish people at the pool" → yes (named)
  ✗ "As a Muslim I did not feel welcome" → yes (named)
  ✗ "Afghan families were waved through security" → yes (named)
 
---
4. CONTEXT — is the racialisation tied to a specific event?
---
Apply both values if the review contains both an incident and a generalisation.
 
yes Racialisation is attached to a specific event, interaction, or
        occurrence — something happened.
  ✓ "The receptionist was racist when she found out I am Ukrainian"
  ✓ "The guard searched us multiple times but not the Germans ahead of us"
  ✓ "We were turned away at the door because we don't speak German"
  ✗ "The staff are racist" → no (characterisation, no event described)
  ✗ "Too many foreigners — it doesn't feel like home" → no (atmosphere)
  ✗ "Turkish people don't follow the rules" → no (generalisation)
 
no Racialisation is a generalisation or judgement about a person or group
        not tied to a specific event; includes atmospheric observations.
  ✓ "Turkish people don't know how to follow pool rules"
  ✓ "These foreigners have no hygiene culture"
  ✓ "The staff are racist"
  ✗ "The receptionist was racist when she found out I am Ukrainian" → yes
  ✗ "Afghan families were not checked at the security gate" → yes (event)
  ✗ "We were turned away at the door because we don't speak German" → yes
"""

# ══════════════════════════════════════════════════════════════════════════════
# 4.  JSON EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"review_id", "theme", "connotation", "approach", "context"}


def _normalize_failure_root_cause(error: str, batch_id: str | None = None) -> str:
    error_l = str(error).lower()
    if error_l == "text_not_found":
        return "missing_input_text"
    if error_l == "budget_exceeded":
        return "budget_exhausted"
    if error_l == "api_unreachable":
        return "api_unreachable"
    if error_l == "missing_content":
        return "missing_model_content"
    if error_l.startswith("http_402"):
        return "insufficient_credits"
    if error_l.startswith("http_404"):
        return "endpoint_not_found"
    if error_l == "salvage_missing":
        return "partial_response_salvage_failed"
    if error_l == "invalid_value":
        return "invalid_model_value"
    if error_l == "individual_retry_failed":
        return "individual_retry_exhausted" if batch_id and batch_id.startswith("single_") else "retry_exhausted"
    return error_l or "unknown"


def _validate_and_normalise(obj: dict) -> dict | None:
    # Ensure required keys present
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return None

    rid = str(obj.get("review_id"))

    # Theme: allow up to 3 themes. Accept list or delimiter-separated string
    theme = obj.get("theme")
    themes: list[str] = []
    if isinstance(theme, list):
        themes = [str(t).strip() for t in theme if str(t).strip()]
    elif isinstance(theme, str):
        # split on comma or slash if multiple provided
        parts = [p.strip() for p in re.split(r"[,/]|\s+/\s+", theme) if p.strip()]
        themes = parts
    if not themes:
        return None
    # limit to 3 themes
    themes = themes[:3]

    # Connotation must be a single label
    con = obj.get("connotation")
    if isinstance(con, list):
        con_val = str(con[0]).strip()
    else:
        con_val = str(con).strip()
    if not con_val:
        return None
    con_val = con_val.title()
    if con_val not in {"Positive", "Negative", "Neutral"}:
        return None

    approach = str(obj.get("approach") or "").strip().lower()
    if approach not in {"yes", "no"}:
        return None

    context = str(obj.get("context") or "").strip().lower()
    if context not in {"yes", "no"}:
        return None

    return {
        "review_id": rid,
        "theme": themes if len(themes) > 1 else themes[0],
        "connotation": con_val,
        "approach": approach,
        "context": context,
    }


def extract_json_array(content: str | None) -> list | None:
    if not content:
        return None

    content = content.strip()
    content = re.sub(r",\s*\]", "]", content)   # fix trailing commas

    # 1. Direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        m = re.search(pattern, content)
        if m:
            candidate = m.group(1)
            # attempt direct parse, then try adding missing bracket
            for text in (candidate, candidate + "]"):
                try:
                    result = json.loads(text)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue

    # 3. Bracket-counting — find the outermost array
    stack: list[int] = []
    for i, char in enumerate(content):
        if char == "[":
            stack.append(i)
        elif char == "]" and stack:
            start = stack.pop()
            if not stack:
                candidate = content[start : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    last = candidate.rfind("}")
                    if last > 0:
                        try:
                            result = json.loads(candidate[: last + 1] + "]")
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            pass
    return None


def salvage_objects(
    content: str, batch_data: list[tuple[str, str]], batch_id: str
) -> tuple[list, list]:
    """Extract whatever valid objects we can from a malformed response."""
    raw_objects = re.findall(r"\{[^{}]+\}", content, re.DOTALL)
    extracted: list[dict] = []
    for raw in raw_objects:
        try:
            obj = json.loads(raw)
            if "review_id" in obj:
                extracted.append(obj)
        except json.JSONDecodeError:
            continue

    logging.info(
        f"🔍 Batch {batch_id}: salvaged {len(extracted)} raw objects from partial response"
    )

    valid_ids: set[str] = set()
    success: list[dict] = []
    for obj in extracted:
        normed = _validate_and_normalise(obj)
        if normed:
            for rid, _ in batch_data:
                if rid == normed["review_id"] and rid not in valid_ids:
                    success.append(normed)
                    valid_ids.add(rid)
                    break

    failed = [
        {
            "review_id": rid,
            "error": "salvage_missing",
            "batch_id": batch_id,
            "root_cause": _normalize_failure_root_cause("salvage_missing", batch_id),
        }
        for rid, _ in batch_data
        if rid not in valid_ids
    ]
    logging.info(
        f"📊 Batch {batch_id} salvage: {len(success)} ok, {len(failed)} failed"
    )
    return success, failed


def _extract_response_content(body: dict, batch_id: str) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        logging.error(f"Batch {batch_id}: missing choices in API response")
        return None

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        logging.error(f"Batch {batch_id}: missing message in API response")
        return None

    content = message.get("content")
    if content is None:
        logging.error(f"Batch {batch_id}: missing message content in API response")
        return None
    if not isinstance(content, str):
        content = str(content)
    return content


# ══════════════════════════════════════════════════════════════════════════════
# 5.  API LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _call_api(
    messages: list, batch_id: str, api_key: str, attempt: int = 0
) -> requests.Response | None:
    if attempt >= MAX_API_ATTEMPTS:
        logging.error(f"Batch {batch_id}: max attempts ({MAX_API_ATTEMPTS}) exhausted")
        return None

    payload = {
        "model"      : MODEL,
        "messages"   : messages,
        "temperature": 0.0,
        "max_tokens" : 2000,
    }

    try:
        # Ensure API key is ASCII-safe
        auth_header = f"Bearer {api_key.strip()}"
        auth_header.encode("ascii")  # Will raise UnicodeEncodeError if non-ASCII
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": auth_header,
            },
            json=payload,
            timeout=120,
        )

    except requests.exceptions.Timeout:
        logging.error(f"Batch {batch_id}: request timed out (attempt {attempt + 1})")
        return None
    except UnicodeEncodeError as exc:
        logging.error(
            f"Batch {batch_id}: API key contains non-ASCII characters (corrupted key). "
            f"Error: {exc!r}"
        )
        return None
    except Exception as exc:
        logging.error(f"Batch {batch_id}: request exception {exc!r}")
        return None

    if response.status_code == 429:
        wait = min(2 ** attempt * 5 + random.uniform(0, 3), 120)
        logging.warning(
            f"Batch {batch_id}: rate-limited, waiting {wait:.1f}s "
            f"(attempt {attempt + 1}/{MAX_API_ATTEMPTS})"
        )
        time.sleep(wait)
        return _call_api(messages, batch_id, api_key, attempt + 1)

    return response


def classify_batch(
    batch_data: list[tuple[str, str]],
    batch_id: str,
    api_key: str,
    tracker: CostTracker,
    key_tracker: ApiKeyTracker | None = None,
) -> dict:
    """
    Classify one batch of (review_id, review_text) pairs.
    Returns {"success": [...], "failed": [...]}.
    """
    status = tracker.budget_status()
    if status == "EXCEEDED":
        logging.error(f"Batch {batch_id}: BUDGET EXCEEDED — skipping")
        return {
            "success": [],
            "failed" : [{
                "review_id": rid,
                "error": "budget_exceeded",
                "batch_id": batch_id,
                "root_cause": _normalize_failure_root_cause("budget_exceeded", batch_id),
            }
                        for rid, _ in batch_data],
        }
    if status == "WARNING":
        logging.warning(f"Batch {batch_id}: budget at WARNING level")

    # Build prompt
    review_lines: list[str] = []
    n_truncated = 0
    for i, (rid, text) in enumerate(batch_data):
        if len(text) > TRUNCATION_LIMIT:
            display = text[:TRUNCATION_LIMIT] + "…"
            n_truncated += 1
        else:
            display = text
        review_lines.append(f"REVIEW {i + 1} (ID: {rid}):\n{display}")

    if n_truncated:
        logging.debug(
            f"Batch {batch_id}: {n_truncated}/{len(batch_data)} reviews "
            f"truncated to {TRUNCATION_LIMIT} chars"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMBINED},
        {
            "role"   : "user",
            "content": (
                f"Classify these {len(batch_data)} reviews in order:\n\n"
                + "\n\n".join(review_lines)
            ),
        },
    ]

    t0       = time.time()
    response = _call_api(messages, batch_id, api_key)
    elapsed  = time.time() - t0

    if response is None:
        return {
            "success": [],
            "failed" : [{
                "review_id": rid,
                "error": "api_unreachable",
                "batch_id": batch_id,
                "root_cause": _normalize_failure_root_cause("api_unreachable", batch_id),
            }
                        for rid, _ in batch_data],
        }

    if response.status_code != 200:
        try:
            error_payload = json.dumps(response.json(), ensure_ascii=False)[:800]
        except Exception:
            error_payload = (response.text or "")[:800]
        logging.error(
            f"Batch {batch_id}: HTTP {response.status_code} | {error_payload}"
        )
        # Mark key as exhausted if it hit 402 (insufficient credits)
        if response.status_code == 402 and key_tracker is not None:
            key_tracker.mark_exhausted(api_key)
        return {
            "success": [],
            "failed" : [{
                "review_id": rid,
                "error": f"http_{response.status_code}",
                "batch_id": batch_id,
                "root_cause": _normalize_failure_root_cause(f"http_{response.status_code}", batch_id),
            }
                        for rid, _ in batch_data],
        }

    body       = response.json()
    usage      = body.get("usage", {})
    input_tok  = usage.get("prompt_tokens",     0)
    output_tok = usage.get("completion_tokens", 0)
    call_cost  = tracker.record_call(input_tok, output_tok, len(batch_data))
    logging.info(
        f"Batch {batch_id}: {elapsed:.1f}s | "
        f"in={input_tok} out={output_tok} | cost=${call_cost:.5f}"
    )
    tracker.log_progress()

    content = _extract_response_content(body, batch_id)
    if content is None:
        return {
            "success": [],
            "failed" : [{
                "review_id": rid,
                "error": "missing_content",
                "batch_id": batch_id,
                "root_cause": _normalize_failure_root_cause("missing_content", batch_id),
            }
                        for rid, _ in batch_data],
        }

    arr     = extract_json_array(content)

    if arr and len(arr) == len(batch_data):
        sent_ids     = [rid for rid, _ in batch_data]
        returned_ids = [str(obj.get("review_id")) for obj in arr]

        if returned_ids != sent_ids:
            n_mis = sum(r != s for r, s in zip(returned_ids, sent_ids))
            logging.warning(
                f"Batch {batch_id}: {n_mis} review_id mismatch(es) — "
                f"correcting by position"
            )
            for obj, (expected_id, _) in zip(arr, batch_data):
                obj["review_id"] = expected_id

        success, failed = [], []
        for obj in arr:
            normed = _validate_and_normalise(obj)
            if normed:
                success.append(normed)
            else:
                failed.append(
                    {
                        "review_id": str(obj.get("review_id", "unknown")),
                        "error"    : "invalid_value",
                        "batch_id"  : batch_id,
                        "root_cause": _normalize_failure_root_cause("invalid_value", batch_id),
                    }
                )
        if failed:
            logging.warning(
                f"Batch {batch_id}: {len(failed)} items had invalid value"
            )
        return {"success": success, "failed": failed}

    # Count mismatch — try to salvage
    logging.warning(
        f"Batch {batch_id}: expected {len(batch_data)} objects, "
        f"got {len(arr) if arr else 0}"
    )
    return dict(zip(("success", "failed"), salvage_objects(content, batch_data, batch_id)))


def classify_single(
    review_id: str, review_text: str, api_key: str, tracker: CostTracker,
    key_tracker: ApiKeyTracker | None = None,
) -> dict | None:
    """Single-review fallback used in the global pass-4 individual retry."""
    text = (
        review_text[:TRUNCATION_LIMIT] + "…"
        if len(review_text) > TRUNCATION_LIMIT
        else review_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMBINED},
        {
            "role"   : "user",
            "content": (
                f"Classify this single pool review:\n\n"
                f"REVIEW (ID: {review_id}):\n{text}\n\n"
                "Return a JSON array with exactly 1 object."
            ),
        },
    ]
    response = _call_api(messages, f"single_{review_id}", api_key)
    if response is None or response.status_code != 200:
        # Mark key as exhausted if it hit 402
        if response is not None and response.status_code == 402 and key_tracker is not None:
            key_tracker.mark_exhausted(api_key)
        return None

    body  = response.json()
    usage = body.get("usage", {})
    tracker.record_call(
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), 1
    )
    content = _extract_response_content(body, f"single_{review_id}")
    if content is None:
        return None

    arr = extract_json_array(content)
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
# 6.  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    chunk_id: str, output_dir: Path,
    success: list, failed: list,
    tracker: CostTracker,
) -> None:
    ckpt = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    data = {
        "chunk_id"  : chunk_id,
        "success"   : success,
        "failed"    : failed,
        "timestamp" : time.time(),
        "cost_so_far": tracker.summary()["total_cost"],
    }
    with open(ckpt, "w") as f:
        json.dump(data, f)
    logging.info(
        f"💾 Checkpoint saved: {ckpt.name} "
        f"({len(success)} ok, {len(failed)} failed so far)"
    )


def load_checkpoint(chunk_id: str, output_dir: Path) -> dict | None:
    ckpt = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt.exists():
        logging.info(f"🔄 Checkpoint found for chunk {chunk_id} — resuming…")
        with open(ckpt) as f:
            return json.load(f)
    return None


def delete_checkpoint(chunk_id: str, output_dir: Path) -> None:
    ckpt = output_dir / f"checkpoint_chunk_{chunk_id}.json"
    if ckpt.exists():
        ckpt.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RESUMPTION  — collect already-successful review_ids BEFORE any API calls
# ══════════════════════════════════════════════════════════════════════════════

def collect_done_ids(processed_file: Path, output_dir: Path) -> set[str]:
    """
    Scan every available source of completed work and return the set of
    review_ids that have already been classified successfully.

    Called once at the very start of run_classify(), before any chunks
    are dispatched, so we never re-classify a review that's already done.

    Sources checked (in order):
      1. processed_success.csv  — the master success CSV
            2. classified_chunks_merged.parquet  — merged chunk output, if present
            3. classified_chunk_*.parquet  — individual chunk outputs
            4. checkpoint_chunk_*.json  — in-progress checkpoints from a crashed run
    """
    done: set[str] = set()
    sources: list[str] = []

    # 1. Master success CSV
    if processed_file.exists():
        df = pd.read_csv(processed_file, usecols=lambda c: c in
                         {"review_id", "classification_status"})
        if "classification_status" in df.columns:
            df = df[df["classification_status"].str.upper() == "SUCCESS"]
        done.update(df["review_id"].astype(str).tolist())
        sources.append(f"{processed_file.name} ({len(df):,} rows)")

    # 2. Merged chunk parquet, if the chunk files have already been cleaned up
    merged_file = output_dir / "classified_chunks_merged.parquet"
    if merged_file.exists():
        df = pd.read_parquet(merged_file)
        if "classification_status" in df.columns:
            df = df[df["classification_status"].str.upper() == "SUCCESS"]
        done.update(df["review_id"].astype(str).tolist())
        sources.append(f"{merged_file.name} ({len(df):,} rows)")

    # 3. Chunk parquets
    for fp in sorted(output_dir.glob("classified_chunk_*.parquet")):
        df = pd.read_parquet(fp)   # read once
        if "classification_status" in df.columns:
            df = df[df["classification_status"].str.upper() == "SUCCESS"]
        ids = df["review_id"].astype(str).tolist()
        done.update(ids)
        sources.append(f"{fp.name} ({len(ids):,} rows)")

    # 4. Existing checkpoints — partial in-progress work
    for fp in sorted(output_dir.glob("checkpoint_chunk_*.json")):
        with open(fp) as f:
            ckpt = json.load(f)
        ids = [r["review_id"] for r in ckpt.get("success", [])]
        done.update(ids)
        if ids:
            sources.append(f"{fp.name} ({len(ids):,} success rows)")

    if sources:
        logging.info(
            f"✅ Resumption: {len(done):,} already-successful review_ids loaded from:\n"
            + "\n".join(f"     • {s}" for s in sources)
        )
    else:
        logging.info("Resumption: no prior results found — starting fresh.")

    return done


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
# 8.  CHUNK PARQUET WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _save_chunk_parquet(
    chunk_df: pd.DataFrame,
    chunk_id: str,
    success: list,
    failed: list,
    output_dir: Path,
    chunk_suffix: str = "",
) -> None:
    cls_map = {r["review_id"]: r for r in success}
    failed_map = {str(r["review_id"]): r for r in failed}
    success_ids = set(cls_map)

    out = pd.DataFrame({"review_id": chunk_df["review_id"].astype(str)})
    if "was_summarized" in chunk_df.columns:
        out["was_summarized"] = chunk_df["was_summarized"].values

    # Map classifier fields
    out["theme"] = out["review_id"].map(
        lambda x: cls_map.get(x, {}).get("theme") if x in success_ids else None
    )
    out["connotation"] = out["review_id"].map(
        lambda x: cls_map.get(x, {}).get("connotation") if x in success_ids else None
    )
    out["approach"] = out["review_id"].map(
        lambda x: cls_map.get(x, {}).get("approach") if x in success_ids else None
    )
    out["context"] = out["review_id"].map(
        lambda x: cls_map.get(x, {}).get("context") if x in success_ids else None
    )

    out["classification_status"] = out["review_id"].apply(
        lambda x: "SUCCESS" if x in success_ids else "FAILED"
    )
    out["classification_error"] = out["review_id"].map(
        lambda x: failed_map.get(x, {}).get("error") if x not in success_ids else None
    )
    out["failure_batch_id"] = out["review_id"].map(
        lambda x: failed_map.get(x, {}).get("batch_id") if x not in success_ids else None
    )
    out["failure_root_cause"] = out["review_id"].map(
        lambda x: failed_map.get(x, {}).get("root_cause") if x not in success_ids else None
    )

    path = output_dir / f"classified_chunk_{chunk_id}{chunk_suffix}.parquet"
    out.to_parquet(path, index=False)
    logging.info(f"💾 {path.name} written ({len(out):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CHUNK PROCESSOR  (Pass 1 — sequential batches, no retries)
# ══════════════════════════════════════════════════════════════════════════════

def process_chunk(
    chunk_df: pd.DataFrame,
    chunk_id: str,
    output_dir: Path,
    batch_size: int,
    chunk_suffix: str,
    api_key: str,
    tracker: CostTracker,
    key_tracker: ApiKeyTracker | None = None,
    tqdm_position: int = 0,
) -> tuple[list, list]:
    """
    Pass 1: classify all reviews in chunk_df sequentially.
    Failures bubble up to the global retry cascade in run_classify().
    """
    logger = ChunkLogger(chunk_id)
    
    review_data:      list[tuple[str, str]] = []
    skipped_failures: list[dict]            = []

    for _, row in chunk_df.iterrows():
        rid  = str(row["review_id"])
        text = str(row.get("trans_review", "")).strip()
        if not text:
            skipped_failures.append({
                "review_id": rid,
                "error": "text_not_found",
                "batch_id": chunk_id,
                "root_cause": _normalize_failure_root_cause("text_not_found", chunk_id),
            })
        else:
            review_data.append((rid, text))

    n_over = sum(1 for _, t in review_data if len(t) > TRUNCATION_LIMIT)
    if n_over:
        logger.info(f" {n_over:,} texts exceed {TRUNCATION_LIMIT} chars and will be truncated")

    if not review_data:
        logger.warning(f"No valid text found; marking {len(skipped_failures):,} reviews failed")
        _save_chunk_parquet(chunk_df, chunk_id, [], skipped_failures,
                            output_dir, chunk_suffix)
        return [], skipped_failures

    logger.info(f" {len(chunk_df):,} reviews | batch_size={batch_size}")
    
    # Resume from checkpoint if one exists
    ckpt = load_checkpoint(chunk_id, output_dir)
    if ckpt:
        success   = ckpt["success"]
        failed    = ckpt["failed"]
        done_ids  = {r["review_id"] for r in success} | {r["review_id"] for r in failed}
        review_data = [(rid, text) for rid, text in review_data if rid not in done_ids]
        logger.info(f"{len(done_ids):,} already done, {len(review_data):,} remaining")
    else:
        success, failed = [], []

    batches    = [review_data[i : i + batch_size]
                  for i in range(0, len(review_data), batch_size)]
    total_done = 0
    reviews_since_checkpoint = 0

    for batch_idx, batch in enumerate(
        tqdm(batches,  desc=f"[{chunk_id}] slot={tqdm_position}",
             position=tqdm_position, leave=True, dynamic_ncols=True)
    ):
        result = classify_batch(batch, f"{chunk_id}_b{batch_idx}", api_key, tracker, key_tracker)
        success.extend(result["success"])
        failed.extend(result["failed"])
        total_done += len(batch)

        reviews_since_checkpoint += len(batch)
        if reviews_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(chunk_id, output_dir, success, failed, tracker)
            reviews_since_checkpoint = 0

        if INTER_BATCH_SLEEP > 0:
            time.sleep(INTER_BATCH_SLEEP)

    failed.extend(skipped_failures)

    _save_chunk_parquet(chunk_df, chunk_id, success, failed, output_dir, chunk_suffix)
    logger.info(f"Chunk {chunk_id}: deleting checkpoint")
    delete_checkpoint(chunk_id, output_dir)

    rate = len(success) / max(len(chunk_df), 1)
    logger.info(
        f" Chunk {chunk_id} pass-1 done: {len(success):,} success, "
        f"{len(failed):,} failed ({rate:.1%})"
    )
    return success, failed


# ══════════════════════════════════════════════════════════════════════════════
# 10. GLOBAL RETRY CASCADE  (runs once after all pass-1 chunks finish)
# ══════════════════════════════════════════════════════════════════════════════
INDIVIDUAL_RETRY_SLEEP = 1.0   # seconds between pass-4 individual retries 
def run_global_retries(
    all_failed        : list,
    output_dir       : Path,
    id_to_text        : dict[str, str],
    initial_batch_size: int,
    api_keys          : list[str],   # ← now takes all keys
    tracker           : CostTracker,
    key_tracker       : ApiKeyTracker,  # ← now accepts shared key_tracker
    ) -> tuple[list, list]:
    """
    Three-pass retry cascade with per-key credit tracking:
      Pass 2 — batch at initial_batch_size
      Pass 3 — batch at initial_batch_size // 2
      Pass 4 — individual (one shot per review, no further retries)
    Returns (recovered_success, truly_failed).
    """
    retryable = [f for f in all_failed if f["review_id"] in id_to_text]
    permanent = [f for f in all_failed if f["review_id"] not in id_to_text]

    if not retryable:
        logging.info("Global retries: nothing retryable.")
        return [], permanent

    recovered: list[dict] = []
    recovered_lock = threading.Lock()

    def _run_pass_parallel(
        items: list[tuple[str, str]], pass_label: str, bs: int
    ) -> list[dict]:
        batches    = [items[i : i + bs] for i in range(0, len(items), bs)]
        still_lock = threading.Lock()
        still      = []

        def _do_batch(batch_idx: int, batch: list) -> None:
            key = key_tracker.get_available_key(batch_idx)
            if key is None:
                logging.error(f"Pass {pass_label} batch {batch_idx}: ALL keys exhausted, skipping")
                with still_lock:
                    still.extend([{"review_id": rid, "error": "all_keys_exhausted"} for rid, _ in batch])
                return
            result = classify_batch(batch, f"{pass_label}_b{batch_idx}", key, tracker, key_tracker)
            with recovered_lock:
                recovered.extend(result["success"])
            with still_lock:
                still.extend(result["failed"])
            if INTER_BATCH_SLEEP > 0:
                time.sleep(INTER_BATCH_SLEEP)

        with ThreadPoolExecutor(max_workers=len(api_keys)) as ex:
            futures = {
                ex.submit(_do_batch, i, b): i
                for i, b in enumerate(batches)
            }
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=pass_label):
                try:
                    f.result()
                except Exception as exc:
                    logging.error(f"{pass_label} batch crashed: {exc}")
        return still

    sep = "─" * 60

    # If batch size is already 1, batch-level retries add no value.
    # Go straight to individual retries.
    if initial_batch_size <= 1:
        logging.info(
            f"\n{sep}\n  GLOBAL PASS 2/3 skipped | initial batch_size=1 "
            f"→ moving directly to PASS 4 individual retries\n{sep}"
        )
        still_failed = retryable
    else:
        # Pass 2 — full batch, all keys
        logging.info(f"\n{sep}\n  GLOBAL PASS 2 | {len(retryable):,} failures "
                     f"| batch_size={initial_batch_size} | keys={len(api_keys)}\n{sep}")
        retry_pairs  = [(f["review_id"], id_to_text[f["review_id"]]) for f in retryable]
        still_failed = _run_pass_parallel(retry_pairs, "gretry2", initial_batch_size)

    # Pass 3 — half batch, all keys (skip if unchanged, e.g. initial batch_size=1)
    if still_failed and initial_batch_size > 1:
        half = max(1, initial_batch_size // 2)
        if half == initial_batch_size:
            logging.info(
                f"Skipping GLOBAL PASS 3 because halved batch size remains {half}"
            )
        else:
            logging.info(f"\n{sep}\n  GLOBAL PASS 3 | {len(still_failed):,} failures "
                         f"| batch_size={half} | keys={len(api_keys)}\n{sep}")
            retry_pairs  = [(f["review_id"], id_to_text[f["review_id"]])
                            for f in still_failed if f["review_id"] in id_to_text]
            still_failed = _run_pass_parallel(retry_pairs, "gretry3", half)

    # Pass 4 — individual, round-robin keys
    if still_failed:
        logging.info(f"\n{sep}\n  GLOBAL PASS 4 — individual | "
                     f"{len(still_failed):,} reviews\n{sep}")
        truly_failed: list[dict] = []
        tf_lock = threading.Lock()

        def _do_single(item: dict, item_idx: int) -> None:
            rid    = item["review_id"]
            text   = id_to_text.get(rid, "")
            if not text:
                with tf_lock:
                    truly_failed.append({
                        "review_id": rid,
                        "error": "text_not_found",
                        "batch_id": f"single_{rid}",
                        "root_cause": _normalize_failure_root_cause("text_not_found", f"single_{rid}"),
                    })
                return
            
            # Get an available key (skip exhausted ones)
            key = key_tracker.get_available_key(item_idx)
            if key is None:
                logging.error(f"Pass 4 item {rid}: ALL keys exhausted, marking failed")
                with tf_lock:
                    truly_failed.append({
                        "review_id": rid,
                        "error": "all_keys_exhausted",
                        "batch_id": f"single_{rid}",
                        "root_cause": _normalize_failure_root_cause("insufficient_credits", f"single_{rid}"),
                    })
                return
            
            result = classify_single(rid, text, key, tracker, key_tracker)
            if result:
                with recovered_lock:
                    recovered.append(result)
                logging.info(f"  ✅ Individual retry succeeded: {rid}")
            else:
                with tf_lock:
                    truly_failed.append(
                        {
                            "review_id": rid,
                            "error": "individual_retry_failed",
                            "batch_id": f"single_{rid}",
                            "root_cause": _normalize_failure_root_cause("individual_retry_failed", f"single_{rid}"),
                        }
                    )
                logging.warning(f"  ❌ Individual retry failed: {rid}")
            time.sleep(INDIVIDUAL_RETRY_SLEEP)   # see Fix 2

        with ThreadPoolExecutor(max_workers=len(api_keys)) as ex:
            futures = {
                ex.submit(_do_single, item, i): i
                for i, item in enumerate(still_failed)
            }
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="Global pass-4"):
                try:
                    f.result()
                except Exception as exc:
                    logging.error(f"Pass-4 item crashed: {exc}")
        still_failed = truly_failed

    truly_failed = still_failed + permanent
    logging.info(
        f"\nGlobal retry cascade complete: "
        f"{len(recovered):,} recovered | {len(truly_failed):,} permanently failed"
    )
    return recovered, truly_failed

# ══════════════════════════════════════════════════════════════════════════════
# 11. FINAL CSV MERGE
# ══════════════════════════════════════════════════════════════════════════════

def _merge_results(
    output_dir    : Path,
    processed_file: Path,
    failed_file   : Path,
    all_success   : list,
    all_failed    : list,
    tracker       : CostTracker,
) -> None:
    """Combine all chunk parquets, apply retry results, write master CSVs."""
    classified = sorted(output_dir.glob("classified_chunk_*.parquet"))
    if not classified:
        logging.error("No classified chunk files found — nothing to merge.")
        return

    logging.info(f"Merging {len(classified)} chunk parquets…")
    chunks_read = []
    for fp in classified:
        chunk = pd.read_parquet(fp)
        chunks_read.append(chunk)
        if len(chunks_read) >= 50:          # flush every 50 files
            chunks_read = [pd.concat(chunks_read, ignore_index=True)]
    final_df = pd.concat(chunks_read, ignore_index=True) if chunks_read else pd.DataFrame()
    
    # Stamp retry-recovered rows as SUCCESS
    recovered_ids = {r["review_id"] for r in all_success}
    mask = (
        (final_df["classification_status"] == "FAILED") &
        (final_df["review_id"].isin(recovered_ids))
    )
    final_df.loc[mask, "classification_status"] = "SUCCESS"

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

    # Apply recovered classifier fields from retry passes
    recovered_map = {r["review_id"]: r for r in all_success}

    def _fill_field(row, field):
        if row["classification_status"] == "SUCCESS" and pd.isna(row.get(field)):
            return recovered_map.get(row["review_id"], {}).get(field)
        return row.get(field)

    for f in ("theme", "connotation", "approach", "context"):
        final_df[f] = final_df.apply(lambda r, ff=f: _fill_field(r, ff), axis=1)

    # ── Write success CSV ──────────────────────────────────────────────────
    success_df = final_df[final_df["classification_status"] == "SUCCESS"].copy()
    if processed_file.exists():
        existing = pd.read_csv(processed_file)
        if "classification_status" in existing.columns:
            existing = existing[existing["classification_status"].str.upper() == "SUCCESS"]
        success_df = pd.concat([existing, success_df], ignore_index=True)
    success_df = success_df.drop_duplicates(subset=["review_id"], keep="last")
    success_df.to_csv(processed_file, index=False)
    logging.info(f"✅ {len(success_df):,} successful rows → {processed_file}")

    # ── Write failed CSV ───────────────────────────────────────────────────
    failed_df = final_df[final_df["classification_status"] == "FAILED"].copy()
    if failed_file.exists():
        existing_f = pd.read_csv(failed_file)
        if "classification_status" in existing_f.columns:
            existing_f = existing_f[existing_f["classification_status"].str.upper() == "FAILED"]
        failed_df = pd.concat([existing_f, failed_df], ignore_index=True)
    failed_df = failed_df[~failed_df["review_id"].isin(success_df["review_id"])]
    failed_df = failed_df.drop_duplicates(subset=["review_id"], keep="last")
    if "failure_root_cause" not in failed_df.columns:
        failed_df["failure_root_cause"] = None
    if "classification_error" in failed_df.columns:
        mask = failed_df["failure_root_cause"].isna() & failed_df["classification_error"].notna()
        failed_df.loc[mask, "failure_root_cause"] = failed_df.loc[mask, "classification_error"].map(
            lambda x: _normalize_failure_root_cause(x, None)
        )
    failed_df.to_csv(failed_file, index=False)
    logging.info(f"📄 {len(failed_df):,} failed rows → {failed_file}")

    # ── Summary ────────────────────────────────────────────────────────────
    s = tracker.summary()
    logging.info(
        f"\n{'='*60}\n"
        f"CLASSIFICATION COMPLETE\n"
        f"  Successful   : {len(success_df):,}\n"
        f"  Failed       : {len(failed_df):,}\n"
        f"  Success rate : {len(success_df)/max(len(success_df)+len(failed_df),1):.1%}\n"
        f"  Total cost   : ${s['total_cost']:.4f}\n"
        f"  Cost/review  : ${s['cost_per_review']:.6f}\n"
        f"  Input tokens : {s['total_input_tokens']:,}\n"
        f"  Output tokens: {s['total_output_tokens']:,}\n"
        f"  Reviews/hour : {s['reviews_per_hour']:.0f}\n"
        f"  Elapsed      : {s['elapsed_hours']:.2f} h\n"
        f"{'='*60}"
    )
    summary_path = output_dir / "classification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {**s,
             "completion_time": datetime.now().isoformat(),
             "total_success"  : len(success_df),
             "total_failed"   : len(failed_df),
             "checkpoint_success": len(ckpt_success_df),
             "checkpoint_failed" : len(ckpt_failed_df)},
            f, indent=2,
        )
    logging.info(f"Summary → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
    # ── Clean up classified chunk files after successful consolidation ────
    logging.info("Cleaning up classified chunk files…")
    for fp in classified:
        try:
            fp.unlink()
            logging.debug(f"Deleted {fp.name}")
        except Exception as e:
            logging.warning(f"Failed to delete {fp.name}: {e}")
    logging.info(f"Cleanup complete: removed {len(classified)} chunk files")

    checkpoint_files = sorted(output_dir.glob("checkpoint_chunk_*.json"))
    if checkpoint_files:
        logging.info("Cleaning up leftover checkpoint files…")
        for fp in checkpoint_files:
            try:
                fp.unlink()
                logging.debug(f"Deleted {fp.name}")
            except Exception as e:
                logging.warning(f"Failed to delete {fp.name}: {e}")
        logging.info(f"Cleanup complete: removed {len(checkpoint_files)} checkpoint files")


def run_classify(
    all_df        : pd.DataFrame,
    output_dir    : Path,
    processed_file: Path,
    failed_file   : Path,
    chunk_size    : int,
    batch_size    : int,
    budget        : float,
    api_keys      : list[str],
) -> None:
    tracker = CostTracker(budget_limit=budget)
    key_tracker = ApiKeyTracker(api_keys)  # ← NEW: track per-key credit exhaustion
    n_keys  = len(api_keys)

    logging.info(
        f"Budget: ${budget} | Chunk: {chunk_size:,} | "
        f"Batch: {batch_size} | Keys: {n_keys} | Key rotation: intelligent (skip exhausted keys)"
    )

    # ── RESUMPTION CHECK — happens here, before any chunks are dispatched ──
    done_ids = collect_done_ids(processed_file, output_dir)
    if done_ids:
        before = len(all_df)
        all_df = all_df[~all_df["review_id"].isin(done_ids)].reset_index(drop=True)
        #run a fraction to seeif it works
        #all_df = all_df.sample(frac=0.001).reset_index(drop=True)
        logging.info(
            f"Filtered out {before - len(all_df):,} already-successful reviews; "
            f"{len(all_df):,} remaining"
        )

    if len(all_df) == 0:
        logging.info("All reviews already classified. Nothing to do.")
        return

    # Build id→text lookup used by the retry cascade
    id_to_text: dict[str, str] = {}
    for _, row in all_df.iterrows():
        rid  = str(row["review_id"])
        text = str(row.get("trans_review", "")).strip()
        if text:
            id_to_text[rid] = text

    # Suffix new chunk files if older ones already exist (avoids clobbering)
    existing_chunks = list(output_dir.glob("classified_chunk_*.parquet"))
    chunk_suffix    = (
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if existing_chunks else ""
    )
    if existing_chunks:
        logging.info(
            f"Detected {len(existing_chunks)} existing chunk parquets; "
            f"new chunks will use suffix '{chunk_suffix}'"
        )

    chunks = [all_df.iloc[i : i + chunk_size]
              for i in range(0, len(all_df), chunk_size)]

    # ── Pass 1: parallel chunks, one key per worker ────────────────────────
    logging.info(
        f"\n{'═'*60}\n"
        f"  PASS 1 — parallel classification\n"
        f"  {len(all_df):,} reviews | {len(chunks)} chunks | {n_keys} workers\n"
        f"{'═'*60}"
    )

    all_success: list[dict] = []
    all_failed : list[dict] = []
    lock = threading.Lock()

  
    _tqdm_slots = list(range(1, n_keys + 1))  # Position 0 reserved for main bar
    _slot_lock      = threading.Lock()
    _available_slots: list[int] = list(_tqdm_slots)

    def _acquire_slot() -> int:
        with _slot_lock:
            return _available_slots.pop(0) if _available_slots else 0

    def _release_slot(slot: int) -> None:
        with _slot_lock:
            _available_slots.append(slot)

    def _run_chunk(chunk_idx: int, chunk_df: pd.DataFrame) -> tuple[list, list]:
        api_key  = api_keys[chunk_idx % n_keys]
        chunk_id = f"chunk_{chunk_idx:04d}"
        slot     = _acquire_slot()
        logging.info(f"[{chunk_id}] START → key …{api_key[-6:]} | slot {slot}")
        try:
            return process_chunk(
                chunk_df.reset_index(drop=True),
                chunk_id,
                output_dir,
                batch_size,
                chunk_suffix,
                api_key,
                tracker,
                key_tracker=key_tracker,
                tqdm_position=slot,
            )
        finally:
            _release_slot(slot)
            logging.info(f"✅ [{chunk_id}] FINISHED | slot {slot}") 
   
    with ThreadPoolExecutor(max_workers=n_keys) as executor:
        futures = {
            executor.submit(_run_chunk, idx, df): idx
            for idx, df in enumerate(chunks)
        }
        
        # Create progress bar AFTER futures are defined
        chunk_pbar = tqdm(total=len(chunks), desc="Overall chunks", position=0, leave=True)
        completed = 0
        
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                c_success, c_failed = future.result()
            except Exception as exc:
                logging.error(f"Chunk {chunk_idx} crashed: {exc}", exc_info=True)
                chunk_pbar.update(1)  # Still update progress on failure
                continue

            with lock:
                all_success.extend(c_success)
                all_failed.extend(c_failed)
                completed += 1
                
            # Update progress bar
            chunk_pbar.update(1)
            chunk_pbar.set_postfix({
                'success': len(all_success),
                'failed': len(all_failed)
            })
            
            logging.info(
                f"Progress: {completed}/{len(chunks)} chunks done | "
                f"{len(all_success):,} ok | {len(all_failed):,} failed"
            )
        
        chunk_pbar.close()

    # ── Global retry cascade ───────────────────────────────────────────────
    if all_failed:
        logging.info(
            f"\n{'═'*60}\n"
            f"  GLOBAL RETRY CASCADE | {len(all_failed):,} failures\n"
            f"{'═'*60}"
        )
        recovered, truly_failed = run_global_retries(
            all_failed         = all_failed,
            id_to_text         = id_to_text,
            output_dir         = output_dir,
            initial_batch_size = batch_size,
            api_keys           = api_keys,
            tracker            = tracker,
            key_tracker        = key_tracker,
        )
        all_success.extend(recovered)
        all_failed = truly_failed
    else:
        logging.info("No failures — skipping global retry cascade.")

    # ── Merge everything into master CSVs ─────────────────────────────────
    _merge_results(
        output_dir,
        processed_file,
        failed_file,
        all_success,
        all_failed,
        tracker,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 13. LENGTH / COST ANALYSIS  (no API calls)
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _bracket_distribution(lengths: list[int]) -> dict:
    brackets = [
        (0,    100,  "0–100"),
        (101,  250,  "101–250"),
        (251,  500,  "251–500"),
        (501,  700,  "501–700"),
        (701,  None, "701+"),
    ]
    return {
        label: (
            sum(1 for l in lengths if l > lo)
            if hi is None
            else sum(1 for l in lengths if lo <= l <= hi)
        )
        for lo, hi, label in brackets
    }


def run_length_analysis(all_df: pd.DataFrame, output_dir: Path, raw_col: str) -> None:
    logging.info("📏 LENGTH ANALYSIS — scanning all reviews (no API calls)…")

    has_summary = "summary" in all_df.columns
    summarised_lens: list[int] = []
    raw_short_lens : list[int] = []
    raw_long_lens  : list[int] = []
    missing_ids    : list[str] = []

    for _, row in all_df.iterrows():
        rid      = str(row["review_id"])
        raw_text = str(row.get(raw_col, "")).strip()
        raw_len  = len(raw_text)
        summary  = (
            str(row["summary"]).strip()
            if has_summary and pd.notna(row.get("summary"))
            else ""
        )

        if summary:
            summarised_lens.append(len(summary))
        elif raw_len > 100:
            raw_long_lens.append(raw_len)
            missing_ids.append(rid)
        else:
            raw_short_lens.append(raw_len)

    all_effective = summarised_lens + raw_short_lens + raw_long_lens

    def stats(vals: list) -> dict:
        if not vals:
            return {k: 0 for k in
                    ("count","avg","min","max","p25","p50","p75","p95",
                     "bracket_distribution")}
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

    sa  = stats(all_effective)
    ss  = stats(summarised_lens)
    srs = stats(raw_short_lens)
    srl = stats(raw_long_lens)

    total    = sa["count"]
    n_trunc  = sum(1 for l in all_effective if l > TRUNCATION_LIMIT)
    pct_trunc= n_trunc / max(total, 1) * 100

    avg_sent = sum(min(l, TRUNCATION_LIMIT) for l in all_effective) / max(total, 1)
    avg_in   = 250 + avg_sent * 0.25 * 5
    avg_out  = 1 * 25
    n_calls  = total / 1
    est_cost = n_calls * (
        (avg_in  / 1_000_000) * INPUT_PRICE_PER_M +
        (avg_out / 1_000_000) * OUTPUT_PRICE_PER_M
    )

    sep = "─" * 70
    logging.info(f"\n{sep}")
    logging.info(f"  LENGTH ANALYSIS REPORT  ({total:,} reviews)")
    logging.info(sep)
    logging.info(f"  {'Group':<34} {'N':>8} {'avg':>6} {'p50':>5} {'p75':>5} {'p95':>5} {'max':>6}")
    logging.info(f"  {sep}")
    for label, s in [
        ("Summaries (sent as-is)",        ss),
        ("Raw ≤700 chars",                srs),
        ("Raw >700 chars (no summary)",   srl),
        ("ALL EFFECTIVE",                 sa),
    ]:
        logging.info(
            f"  {label:<34} {s['count']:>8,} {s['avg']:>6.1f} "
            f"{s['p50']:>5} {s['p75']:>5} {s['p95']:>5} {s['max']:>6}"
        )

    logging.info(f"\n  BRACKET DISTRIBUTION (effective texts)")
    logging.info(f"  {sep}")
    for bracket, count in sa.get("bracket_distribution", {}).items():
        pct = count / max(total, 1) * 100
        bar = "█" * int(pct / 2)
        logging.info(f"  {bracket:<14} {count:>9,} {pct:>6.1f}%  {bar}")

    logging.info(sep)
    logging.info(f"  Truncation limit      : {TRUNCATION_LIMIT} chars")
    logging.info(f"  Reviews to truncate   : {n_trunc:,} ({pct_trunc:.1f}%)")
    logging.info(f"  Avg chars sent/review : {avg_sent:.1f}")
    logging.info(f"  Est. total cost       : ${est_cost:,.2f}")
    logging.info(sep)

    report = {
        "total_reviews"     : total,
        "truncation_limit"  : TRUNCATION_LIMIT,
        "n_would_truncate"  : n_trunc,
        "pct_would_truncate": round(pct_trunc, 2),
        "summarised"        : ss,
        "raw_short"         : srs,
        "raw_long_no_summary": {**srl, "review_ids_sample": missing_ids[:50]},
        "all_effective"     : sa,
        "cost_estimate"     : {
            "model"                 : MODEL,
            "batch_size"            : 1,
            "truncation_limit"      : TRUNCATION_LIMIT,
            "avg_sent_chars_review" : round(avg_sent, 1),
            "est_input_tok_per_call": round(avg_in),
            "est_output_tok_per_call": avg_out,
            "est_total_calls"       : round(n_calls),
            "est_total_cost_usd"    : round(est_cost, 4),
        },
        "generated_at": datetime.now().isoformat(),
    }
    rp = output_dir / "length_analysis.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"  📄 Full report → {rp}\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# 14. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="classifier_v6",
        description=(
            "Pool review racial classifier.\n\n"
            "MODES\n"
            "  lengths   Scan dataset, report char-length stats + cost estimate.\n"
            "  classify  Run full classification pipeline.\n"
            "  both      Run length analysis then classify.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["lengths", "classify", "both"])
    parser.add_argument("--input",      default=None,
                        help="Path to input parquet. Default: /home/ethel/Desktop/swimming_pool_project/spp_15052026/data/social_reviews.parquet")
    parser.add_argument("--base",       default=DEFAULT_BASE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-col",    default="trans_review",
                        help="Column name for review text. Default: trans_review")
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--chunk-size", type=int,   default=10000)
    parser.add_argument("--budget",     type=float, default=800.0)
    parser.add_argument(
        "--keys", default=None,
        help="Comma-separated OpenRouter API keys (overrides env vars).",
    )
    return parser.parse_args()


def _load_api_keys(args: argparse.Namespace) -> list[str]:
    """
    Priority:
      1. --keys CLI argument
      2. OPENROUTER_API_KEYS env var (comma-separated)
      3. OPENROUTER_API_KEY env var (single key)
    
    Filters out any keys containing non-ASCII characters (corrupted keys).
    """
    load_dotenv()

    def _filter_keys(raw_keys: list[str]) -> list[str]:
        """Remove corrupted keys with non-ASCII characters."""
        valid = []
        invalid = []
        for k in raw_keys:
            k = k.strip()
            if not k:
                continue
            if k.isascii():
                valid.append(k)
            else:
                invalid.append(k)
        if invalid:
            logging.warning(
                f"Filtered out {len(invalid)} corrupted API key(s) "
                f"with non-ASCII characters"
            )
        return valid

    if args.keys:
        raw_keys = [k for k in args.keys.split(",")]
        keys = _filter_keys(raw_keys)
        if keys:
            logging.info(f"Using {len(keys)} API key(s) from --keys flag")
            return keys

    multi = os.getenv("OPENROUTER_API_KEYS", "")
    if multi:
        raw_keys = [k for k in multi.split(",")]
        keys = _filter_keys(raw_keys)
        if keys:
            logging.info(f"Using {len(keys)} API key(s) from OPENROUTER_API_KEYS env var")
            return keys

    single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if single:
        if single.isascii():
            logging.info("Using 1 API key from OPENROUTER_API_KEY env var")
            return [single]
        else:
            logging.warning("OPENROUTER_API_KEY contains non-ASCII characters — skipping")

    raise SystemExit(
        "❌  No valid (ASCII-only) API keys found. "
        "Check OPENROUTER_API_KEYS or OPENROUTER_API_KEY for corruption. "
        "Supply via --keys, OPENROUTER_API_KEYS, or OPENROUTER_API_KEY."
    )



def main() -> None:
    # ── Parse args FIRST ────────────────────────────────────────────────────
    args = parse_args()

    base       = Path(args.base)
    input_file = Path(args.input) if args.input else Path(DEFAULT_INPUT)
    output_dir = Path(args.output_dir) if args.output_dir else base / "classification_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_file = base / "processed_success.csv"
    failed_file    = base / "processed_failed.csv"

    # ── Logging (must come before _load_api_keys so key-count message is logged) ──
    setup_logging(output_dir)
    logging.info(
        f"🚀 classifier_v6 | mode={args.mode} | input={input_file} | "
        f"truncation_limit={TRUNCATION_LIMIT}"
    )

    # ── API keys ─────────────────────────────────────────────────────────────
    api_keys = _load_api_keys(args)

    # ── Load data ────────────────────────────────────────────────────────────
    logging.info(f"Loading {input_file} …")
    all_df = pd.read_parquet(input_file)
    all_df["review_id"] = all_df["review_id"].astype(str)
    logging.info(f"{len(all_df):,} reviews loaded from input file")

    try:
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
                api_keys       = api_keys,
            )
    finally:
        stop_logging()


if __name__ == "__main__":
    main()