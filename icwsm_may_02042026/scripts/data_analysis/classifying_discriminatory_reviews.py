#!/usr/bin/env python3
"""
Discriminatory Review Sub-Classifier  (v2 — batched)
======================================================
"I am not racist, but..." — Swimming pool reviews, Germany

Changes from v1:
  - Reviews sent in batches (default 10) per API call → ~10× prompt-token savings
  - Prompt tightened (~40% shorter) without loss of taxonomic precision
  - max_tokens scales automatically with batch size
  - JSON array response, keyed by review_id for safe alignment
  - All other logic (checkpoint, plots, kappa, exemplars) unchanged

Usage:
    python classifying_discriminatory_reviews.py --data path/to/data.csv
    python classifying_discriminatory_reviews.py --data path/to/data.csv --batch_size 10
    python classifying_discriminatory_reviews.py --data path/to/data.csv --max_reviews 100
    python classifying_discriminatory_reviews.py --data path/to/data.csv --skip_classify
    python classifying_discriminatory_reviews.py --data path/to/data.csv 
        --manual_labels annotation_template.csv

Requirements:
    pip install requests pandas tqdm scikit-learn matplotlib seaborn python-dotenv
"""

import os
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import random
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from sklearn.metrics import cohen_kappa_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not set.\n"
        "Add it to your .env file: OPENROUTER_API_KEY=your_key_here"
    )

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          =  "anthropic/claude-3.7-sonnet" #model naming could change on openrouter old one I used was 3.5-sonnet
print(f"\n  Using model: {MODEL}")

##quick check for available models in bash
# curl https://openrouter.ai/api/v1/models \
#   -H "Authorization: Bearer $OPENROUTER_API_KEY" \
#   | python3 -m json.tool | grep '"id"' | grep anthropic

TARGET_COL       = "connotation_type"
TEXT_EN_COL      = "trans_review"
TEXT_DE_COL      = "org_review"
REVIEW_ID_COL    = "review_id"
PLACE_ID_COL     = "place_id"
CONSTITUENCY_COL = "constituency"
STATE_COL        = "state"

OUTPUT_DIR = "/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/data/analysis/discriminatory_subclasses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["experienced", "perpetuated", "observed"]

EAST_STATES = {
    "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen",
    "Sachsen-Anhalt", "Thüringen", "Berlin"
}

# Output tokens per review (headroom for JSON).
# Total max_tokens = TOKENS_PER_REVIEW × batch_size, capped at 8000.
TOKENS_PER_REVIEW = 350


# =============================================================================
# PROMPT — built once at startup, reused for every batch
# =============================================================================

def build_prompt() -> str:
    """
    Tightened classification prompt. ~40% shorter than v1; full taxonomy preserved.
    Called ONCE at startup; prepended to every batch request.
    """
    return """You are a researcher coding racialised discourse in German swimming-pool reviews (English translations).

=== CATEGORIES (assign ALL that apply; multi-label is allowed) ===

experienced  — reviewer recounts discrimination directed AT them or their companions.
               Extra flag experiencer_is_target:
                 true  = reviewer is from a minority group and was personally targeted
                 false = companion was targeted, or reviewer's group membership is unclear

perpetuated  — reviewer's OWN language reproduces racial discrimination: coded attribution
               ("certain groups"), disclaimer framing ("I'm not racist, but…"), culturalisation
               ("their hygiene culture"), threat framing ("felt unsafe since they started coming"),
               nostalgic displacement ("used to be a proper family place"), othering positives
               ("finally no foreign groups").

observed     — reviewer describes discrimination directed at a THIRD PARTY they witnessed.

Rules:
- Assign [] if none fit.
- experiencer_is_target = false unless "experienced" is assigned AND reviewer is clearly
  a minority group member describing personal targeting.
- A review can be experienced+perpetuated (describes own discrimination but also uses
  othering language) or observed+perpetuated (describes third-party discrimination but
  frames it with othering language).

=== OUTPUT FORMAT ===

Return ONLY a JSON array — one object per review, in input order.
No markdown fences, no preamble.

[
  {
    "review_id": "<id>",
    "categories": [],
    "experiencer_is_target": false,
    "confidence": "high|medium|low",
    "reasoning": "1-2 sentences",
    "key_evidence": []
  },
  ...
]

=== EXAMPLES ===

Input:
[
  {"review_id": "r01", "text": "As a Turkish family we were asked by the lifeguard to leave even though we had paid. Other families were not asked. Pure discrimination."},
  {"review_id": "r02", "text": "Pool is fine but the atmosphere has changed. Certain groups of young men don't know how to behave — you can imagine who I mean. Won't bring my kids anymore."},
  {"review_id": "r03", "text": "I watched foreign teenagers get followed by security the whole time while other kids were left alone. Clearly discriminatory."},
  {"review_id": "r04", "text": "Affordable and clean. Finally few seniors and no foreign groups. A comfortable family place."},
  {"review_id": "r05", "text": "As a Black woman I was told I needed a swim cap; white women next to me were not. Also the pool gets crowded with people who never follow lane rules — always the same crowd."},
  {"review_id": "r06", "text": "Went last Sunday. Pool was okay. Changing rooms could be cleaner. Staff unhelpful about the schedule. Three stars."}
]

Output:
[
  {"review_id": "r01", "categories": ["experienced"], "experiencer_is_target": true, "confidence": "high", "reasoning": "Turkish family personally targeted by staff; other families unaffected.", "key_evidence": ["asked to leave", "other families not asked", "paid entry"]},
  {"review_id": "r02", "categories": ["perpetuated"], "experiencer_is_target": false, "confidence": "high", "reasoning": "Coded attribution ('certain groups', 'you can imagine who I mean') and threat/nostalgic framing; reviewer is the source, not a target.", "key_evidence": ["certain groups", "you can imagine who I mean", "atmosphere changed"]},
  {"review_id": "r03", "categories": ["observed"], "experiencer_is_target": false, "confidence": "high", "reasoning": "Reviewer is a bystander; discrimination was directed at third-party teenagers.", "key_evidence": ["followed by security", "other kids left alone", "clearly discriminatory"]},
  {"review_id": "r04", "categories": ["perpetuated"], "experiencer_is_target": false, "confidence": "high", "reasoning": "Frames absence of 'foreign groups' as a positive — othering through coded language.", "key_evidence": ["no foreign groups", "finally", "pleasant clientele"]},
  {"review_id": "r05", "categories": ["experienced", "perpetuated"], "experiencer_is_target": true, "confidence": "high", "reasoning": "First part: Black woman directly targeted by differential rule enforcement (experienced, reviewer is target). Second part: 'always the same crowd' is coded othering (perpetuated).", "key_evidence": ["told I needed a swim cap", "white women not", "always the same crowd"]},
  {"review_id": "r06", "categories": [], "experiencer_is_target": false, "confidence": "high", "reasoning": "Generic facility complaint; no racial or discriminatory content.", "key_evidence": []}
]

=== REVIEWS TO CLASSIFY ===

"""


# =============================================================================
# OPENROUTER API CALL  (batch-aware)
# =============================================================================

def call_openrouter_batch(prompt: str,
                          batch: list[dict],
                          model: str,
                          max_tokens: int,
                          max_retries: int = 3) -> list[dict]:
    """
    Send a batch of reviews to OpenRouter in a single API call.

    batch  : list of {"review_id": str, "text": str}
    Returns: list of parsed result dicts (same length and order as batch),
             or a list of empty results on unrecoverable failure.
    """
    referers = [
        "http://localhost:3000", "http://localhost:8080", "https://www.google.com",
        "https://github.com", "https://example.com", "http://127.0.0.1:3000"
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  random.choice(referers),
        "X-Title":       "Swimming Pool Discrimination Study",
    }

    batch_json = json.dumps(batch, ensure_ascii=False)
    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt + batch_json}
        ],
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,          # longer timeout for batches
            )

            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n  Rate limit (429) — waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"\n  HTTP {response.status_code}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return [_empty_result(r["review_id"], f"http_{response.status_code}")
                        for r in batch]

            data = response.json()
            raw  = data["choices"][0]["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()

            parsed_list = json.loads(raw)

            if not isinstance(parsed_list, list):
                raise ValueError("Response is not a JSON array")

            # Build a lookup by review_id so order mismatches are safe
            result_map = {str(item.get("review_id", "")): item
                          for item in parsed_list}

            results = []
            for item in batch:
                rid    = str(item["review_id"])
                result = result_map.get(rid, {})
                results.append(_normalise_result(rid, result))

            return results

        except (json.JSONDecodeError, ValueError, KeyError):
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                # Fall back to individual calls for this batch
                print(f"\n  Batch parse failed — falling back to per-review calls...")
                return _fallback_individual(prompt, batch, model, max_tokens)

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return [_empty_result(r["review_id"], "timeout") for r in batch]

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return [_empty_result(r["review_id"], f"request_error: {str(e)[:50]}")
                        for r in batch]

    return [_empty_result(r["review_id"], "max_retries_exceeded") for r in batch]


def _fallback_individual(prompt: str,
                         batch: list[dict],
                         model: str,
                         max_tokens: int) -> list[dict]:
    """
    Called only when batch JSON parsing fails after retries.
    Sends each review in the failed batch individually (wraps as a 1-item array).
    """
    results = []
    for item in batch:
        single = [item]
        res    = call_openrouter_batch(
            prompt, single, model,
            max_tokens=TOKENS_PER_REVIEW,
            max_retries=2,
        )
        results.extend(res)
        time.sleep(0.5)
    return results


def _normalise_result(review_id: str, raw: dict) -> dict:
    """Validate and normalise a single result dict from the model."""
    cats = [c for c in raw.get("categories", []) if c in CATEGORIES]
    if "experienced" not in cats:
        eit = False
    else:
        eit = bool(raw.get("experiencer_is_target", False))

    return {
        "review_id":             review_id,
        "categories":            cats,
        "experiencer_is_target": eit,
        "confidence":            raw.get("confidence", "low"),
        "reasoning":             raw.get("reasoning", ""),
        "key_evidence":          raw.get("key_evidence", []),
        "parse_status":          "ok",
    }


def _empty_result(review_id: str, reason: str = "") -> dict:
    return {
        "review_id":             review_id,
        "categories":            [],
        "experiencer_is_target": False,
        "confidence":            "low",
        "reasoning":             "",
        "key_evidence":          [],
        "parse_status":          reason,
    }


# =============================================================================
# CLASSIFICATION ENGINE
# =============================================================================

class DiscriminatoryClassifier:
    def __init__(self, model: str = MODEL, reviews_per_batch: int = 10):
        self.model             = model
        self.reviews_per_batch = reviews_per_batch
        self.max_tokens        = min(TOKENS_PER_REVIEW * reviews_per_batch, 8000)

        print(f"  Model            : {self.model}")
        print(f"  Reviews per batch: {self.reviews_per_batch}")
        print(f"  max_tokens       : {self.max_tokens}")
        print(f"  Loading prompt...")
        self.prompt = build_prompt()
        print(f"  Prompt loaded    : {len(self.prompt):,} chars")
        print(f"  Ready to classify.")

    def classify_batch(self, items: list[dict]) -> list[dict]:
        """
        items: list of {"review_id": str, "text": str}
        Returns list of result dicts, one per item.
        """
        # Drop items with empty text; preserve placeholders for alignment
        valid, empty_results = [], []
        for item in items:
            text = str(item.get("text", "")).strip()
            if len(text) < 10:
                empty_results.append(
                    (item["review_id"], _empty_result(item["review_id"], "too_short"))
                )
            else:
                valid.append(item)

        if not valid:
            return [r for _, r in empty_results]

        api_results = call_openrouter_batch(
            self.prompt, valid, self.model, self.max_tokens
        )

        # Merge back in original order
        result_map = {r["review_id"]: r for r in api_results}
        for rid, r in empty_results:
            result_map[rid] = r

        return [result_map[item["review_id"]] for item in items]


# =============================================================================
# BATCH CLASSIFICATION WITH CHECKPOINT
# =============================================================================

def classify_reviews(df: pd.DataFrame,
                     classifier: DiscriminatoryClassifier,
                     checkpoint_every: int = 5,
                     pause_between_batches: float = 1.0) -> pd.DataFrame:
    """
    Classify all reviews in df using batched API calls.
    Checkpoints every `checkpoint_every` batches.
    """
    bsz   = classifier.reviews_per_batch
    total = len(df)

    print(f"\n  Reviews to classify     : {total:,}")
    print(f"  Batch size              : {bsz}")
    print(f"  API calls required      : ~{int(np.ceil(total / bsz)):,}")

    # Rough cost estimate: ~$3/M input tokens, prompt ≈ 900 tokens
    prompt_tokens   = len(classifier.prompt) // 4          # rough char→token
    avg_review_tok  = 120
    input_per_call  = prompt_tokens + avg_review_tok * bsz
    total_calls     = int(np.ceil(total / bsz))
    est_cost        = (input_per_call * total_calls / 1_000_000) * 3
    print(f"  Est. input-token cost   : ~${est_cost:.2f}")

    checkpoint_path = os.path.join(OUTPUT_DIR, "_checkpoint.json")

    # Resume from checkpoint
    results_map = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                saved = json.load(f)
            results_map = {r["review_id"]: r for r in saved.get("results", [])}
            print(f"  Resuming: {len(results_map):,} reviews already done")
        except Exception:
            results_map = {}

    # Build list of items still needing classification
    all_items = [
        {"review_id": str(row[REVIEW_ID_COL]),
         "text":      str(row.get(TEXT_EN_COL, ""))}
        for _, row in df.iterrows()
    ]
    pending = [it for it in all_items if it["review_id"] not in results_map]

    # Split into batches
    batches = [pending[i:i + bsz] for i in range(0, len(pending), bsz)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="  Classifying batches")):
        batch_results = classifier.classify_batch(batch)
        for r in batch_results:
            results_map[r["review_id"]] = r

        if (batch_idx + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, results_map)

        time.sleep(pause_between_batches)

    _save_checkpoint(checkpoint_path, results_map)

    # ---- Unpack into DataFrame ----
    results = [results_map.get(str(row[REVIEW_ID_COL]), _empty_result(str(row[REVIEW_ID_COL]), "missing"))
               for _, row in df.iterrows()]

    out_df = df.copy()
    out_df["sub_categories"]        = [r["categories"]            for r in results]
    out_df["experiencer_is_target"] = [r["experiencer_is_target"] for r in results]
    out_df["llm_confidence"]        = [r["confidence"]            for r in results]
    out_df["llm_reasoning"]         = [r["reasoning"]             for r in results]
    out_df["llm_key_evidence"]      = [r["key_evidence"]          for r in results]
    out_df["parse_status"]          = [r["parse_status"]          for r in results]

    # One-hot encode
    for cat in CATEGORIES:
        out_df[f"is_{cat}"] = out_df["sub_categories"].apply(
            lambda cats: 1 if cat in cats else 0
        )
    out_df["multi_label"] = out_df["sub_categories"].apply(
        lambda cats: 1 if len(cats) > 1 else 0
    )

    # Verify round-trip
    missing = sum(1 for r in results if r["parse_status"] == "missing")
    if missing:
        print(f"\n  WARNING: {missing} reviews missing from results map")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return out_df


def _save_checkpoint(path: str, results_map: dict):
    with open(path, "w") as f:
        json.dump({"results": list(results_map.values())}, f)


# =============================================================================
# ANALYSIS & VISUALISATION  (unchanged from v1)
# =============================================================================

def summarise_results(tagged_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}\n  CLASSIFICATION RESULTS SUMMARY\n{'='*60}")

    total = len(tagged_df)
    print(f"\n  Total reviews classified : {total:,}")

    print(f"\n  Sub-category distribution:")
    for cat in CATEGORIES:
        count = tagged_df[f"is_{cat}"].sum()
        print(f"    {cat:<15} : {count:>5,}  ({count/total*100:.1f}%)")

    ml = tagged_df["multi_label"].sum()
    print(f"\n  Multi-label (2+)         : {ml:,} ({ml/total*100:.1f}%)")

    exp_df = tagged_df[tagged_df["is_experienced"] == 1]
    if len(exp_df) > 0:
        n_target = exp_df["experiencer_is_target"].sum()
        print(f"\n  Within experienced reviews:")
        print(f"    Reviewer is target    : {n_target:,} ({n_target/len(exp_df)*100:.1f}%)")
        print(f"    Reviewer not target   : {len(exp_df)-n_target:,} "
              f"({(len(exp_df)-n_target)/len(exp_df)*100:.1f}%)")

    print(f"\n  Category combinations:")
    combos = (
        tagged_df["sub_categories"]
        .apply(lambda x: "+".join(sorted(x)) if x else "none")
        .value_counts()
    )
    for combo, count in combos.items():
        print(f"    {combo:<35} : {count:>5,}  ({count/total*100:.1f}%)")

    failed = (tagged_df["parse_status"] != "ok").sum()
    if failed:
        print(f"\n  Parse failures           : {failed} ({failed/total*100:.1f}%)")

    print(f"\n  Confidence distribution:")
    print(tagged_df["llm_confidence"].value_counts().to_string())

    summary_rows = [
        {"category": cat,
         "count":    int(tagged_df[f"is_{cat}"].sum()),
         "pct":      round(tagged_df[f"is_{cat}"].mean() * 100, 1)}
        for cat in CATEGORIES
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "subclass_summary.csv"), index=False)
    print(f"\n  Saved: subclass_summary.csv")
    return summary_df


def plot_results(tagged_df: pd.DataFrame):
    # 1. Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    total  = len(tagged_df)
    counts = [tagged_df[f"is_{c}"].sum() for c in CATEGORIES]
    pcts   = [c / total * 100 for c in counts]
    colors = ["#1D9E75", "#7F77DD", "#D85A30"]

    bars = ax.bar(CATEGORIES, pcts, color=colors, alpha=0.85, width=0.5)
    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of discriminatory reviews")
    ax.set_title("Sub-category distribution — discriminatory reviews", fontsize=11)
    ax.set_ylim(0, max(pcts) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "subclass_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: subclass_distribution.png")

    # 2. Co-occurrence heatmap
    cat_cols = [f"is_{c}" for c in CATEGORIES]
    cooccur  = tagged_df[cat_cols].T.dot(tagged_df[cat_cols])
    cooccur.index = cooccur.columns = CATEGORIES

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cooccur, ax=ax, annot=True, fmt="d",
                cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "review count"})
    ax.set_title("Sub-category co-occurrence", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "subclass_cooccurrence.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: subclass_cooccurrence.png")

    # 3. East vs West
    if "east_germany" in tagged_df.columns:
        region_rows = []
        for label, mask in [("East", tagged_df["east_germany"] == True),
                             ("West", tagged_df["east_germany"] == False)]:
            sub = tagged_df[mask]
            if len(sub) == 0:
                continue
            for cat in CATEGORIES:
                region_rows.append({
                    "region": label, "category": cat,
                    "pct": sub[f"is_{cat}"].mean() * 100,
                })
        if region_rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=pd.DataFrame(region_rows),
                        x="category", y="pct", hue="region",
                        palette={"East": "#D85A30", "West": "#1D9E75"},
                        ax=ax, alpha=0.85)
            ax.set_ylabel("% of discriminatory reviews")
            ax.set_title("Sub-category: East vs West Germany", fontsize=11)
            ax.legend(title="region")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "subclass_east_west.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: subclass_east_west.png")

    # 4. Experiencer split
    exp_df = tagged_df[tagged_df["is_experienced"] == 1]
    if len(exp_df) > 0:
        n_target    = exp_df["experiencer_is_target"].sum()
        n_nontarget = len(exp_df) - n_target
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["reviewer is target", "companion / unclear"],
               [n_target / len(exp_df) * 100, n_nontarget / len(exp_df) * 100],
               color=["#D85A30", "#B4B2A9"], alpha=0.85, width=0.4)
        ax.set_ylabel("% of 'experienced' reviews")
        ax.set_title("Who was targeted — within experienced reviews", fontsize=11)
        ax.set_ylim(0, 110)
        for i, val in enumerate([n_target, n_nontarget]):
            ax.text(i, val / len(exp_df) * 100 + 1.5,
                    f"{val:,} ({val/len(exp_df)*100:.1f}%)",
                    ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "experienced_target_split.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: experienced_target_split.png")


def extract_exemplars(tagged_df: pd.DataFrame, n_per_category: int = 5) -> pd.DataFrame:
    print(f"\n{'='*60}\n  EXEMPLAR EXTRACTION\n{'='*60}")

    rows = []
    for cat in CATEGORIES:
        sub = tagged_df[
            (tagged_df[f"is_{cat}"] == 1) &
            (tagged_df["llm_confidence"] == "high")
        ].copy()
        if len(sub) == 0:
            sub = tagged_df[tagged_df[f"is_{cat}"] == 1].copy()

        sub["n_cats"]     = sub["sub_categories"].apply(len)
        sub["review_len"] = sub[TEXT_EN_COL].str.len().fillna(0)
        sub = sub.sort_values(["n_cats", "review_len"], ascending=[True, False])

        print(f"\n  --- {cat.upper()} (showing top {min(n_per_category, len(sub))}) ---")

        for _, row in sub.head(n_per_category).iterrows():
            en = str(row.get(TEXT_EN_COL, ""))
            de = str(row.get(TEXT_DE_COL, ""))
            print(f"\n  review_id : {row.get(REVIEW_ID_COL, 'n/a')}")
            print(f"  EN: {en[:220]}{'...' if len(en) > 220 else ''}")
            print(f"  DE: {de[:220]}{'...' if len(de) > 220 else ''}")
            print(f"  categories        : {row['sub_categories']}")
            print(f"  experiencer_target: {row['experiencer_is_target']}")
            print(f"  confidence        : {row['llm_confidence']}")
            print(f"  evidence          : {row.get('llm_key_evidence', [])}")

            rows.append({
                "review_id":             row.get(REVIEW_ID_COL, ""),
                "category":              cat,
                "review_en":             en,
                "review_de":             de,
                "all_categories":        str(row["sub_categories"]),
                "experiencer_is_target": row["experiencer_is_target"],
                "confidence":            row["llm_confidence"],
                "key_evidence":          str(row.get("llm_key_evidence", [])),
                "state":                 row.get(STATE_COL, ""),
                "region":                row.get("region", ""),
            })

    exemplar_df = pd.DataFrame(rows)
    exemplar_df.to_csv(os.path.join(OUTPUT_DIR, "exemplars_by_subclass.csv"), index=False)
    print(f"\n  Saved: exemplars_by_subclass.csv")
    print("  NOTE: Always quote org_review (German original) in your paper.")
    return exemplar_df


# =============================================================================
# VALIDATION — Cohen's Kappa  (unchanged from v1)
# =============================================================================

def validate_with_kappa(tagged_df: pd.DataFrame, manual_labels_path: str):
    print(f"\n{'='*60}\n  VALIDATION — Cohen's Kappa\n{'='*60}")

    if not SKLEARN_AVAILABLE:
        print("  scikit-learn not available: pip install scikit-learn")
        return

    if not os.path.exists(manual_labels_path):
        print(f"  File not found: {manual_labels_path}")
        print("  Generating annotation template instead...")
        _generate_annotation_template(tagged_df)
        return

    manual_df = pd.read_csv(manual_labels_path)
    required  = {"review_id", "category", "manual_label"}
    if not required.issubset(manual_df.columns):
        print(f"  Manual labels CSV must have columns: {required}")
        return

    kappa_rows = []
    for cat in CATEGORIES:
        sub = manual_df[manual_df["category"] == cat].dropna(subset=["manual_label"])
        if len(sub) < 10:
            print(f"  {cat}: too few labels ({len(sub)}) — need ≥10")
            continue

        merged = sub.merge(
            tagged_df[[REVIEW_ID_COL, f"is_{cat}"]],
            on=REVIEW_ID_COL, how="inner"
        )
        if len(merged) < 10:
            print(f"  {cat}: fewer than 10 rows after merge")
            continue

        kappa = cohen_kappa_score(
            merged["manual_label"].astype(int),
            merged[f"is_{cat}"].astype(int),
        )
        acc = (merged["manual_label"].astype(int) == merged[f"is_{cat}"].astype(int)).mean()
        kappa_rows.append({
            "category": cat, "kappa": round(kappa, 3),
            "accuracy": round(acc, 3), "n": len(merged),
        })
        print(f"  {cat:<15}: kappa={kappa:.3f}  accuracy={acc:.3f}  n={len(merged)}")

    if kappa_rows:
        kappa_df = pd.DataFrame(kappa_rows)
        mean_k   = kappa_df["kappa"].mean()
        print(f"\n  Mean kappa: {mean_k:.3f}")
        if   mean_k >= 0.8: print("  STRONG agreement — report with confidence.")
        elif mean_k >= 0.7: print("  SUBSTANTIAL agreement — suitable for ICWSM.")
        elif mean_k >= 0.6: print("  MODERATE agreement — acceptable with caveats.")
        else:
            print("  WEAK agreement — revise prompt examples.")
            print("  Tip: add more examples for low-kappa categories in build_prompt()")

        kappa_df.to_csv(os.path.join(OUTPUT_DIR, "kappa_validation.csv"), index=False)
        print("  Saved: kappa_validation.csv")
        return kappa_df

    return None


def _generate_annotation_template(tagged_df: pd.DataFrame):
    rows        = []
    sampled_ids = set()

    for cat in CATEGORIES:
        sub    = tagged_df[tagged_df[f"is_{cat}"] == 1]
        sample = sub.sample(min(30, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            sampled_ids.add(row[REVIEW_ID_COL])

    for rid in sampled_ids:
        row = tagged_df[tagged_df[REVIEW_ID_COL] == rid].iloc[0]
        for cat in CATEGORIES:
            rows.append({
                "review_id":       rid,
                "review_en":       str(row.get(TEXT_EN_COL, ""))[:400],
                "review_de":       str(row.get(TEXT_DE_COL, ""))[:400],
                "category":        cat,
                "llm_label":       row.get(f"is_{cat}", 0),
                "llm_reasoning":   str(row.get("llm_reasoning", ""))[:300],
                "manual_label":    "",
                "annotator_notes": "",
            })

    template_path = os.path.join(OUTPUT_DIR, "annotation_template.csv")
    pd.DataFrame(rows).to_csv(template_path, index=False)

    print(f"\n  Annotation template saved: annotation_template.csv")
    print(f"  {len(sampled_ids)} reviews × {len(CATEGORIES)} categories = {len(rows)} rows")
    print("\n  Instructions:")
    print("    1. Open annotation_template.csv")
    print("    2. Set manual_label = 1 (yes) or 0 (no) for each row")
    print("    3. Save and re-run: --manual_labels annotation_template.csv")
    print("    4. Target kappa ≥ 0.70 for ICWSM")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sub-classify discriminatory pool reviews via OpenRouter (batched)"
    )
    parser.add_argument("--data",          type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--batch_size",    type=int, default=10,
                        help="Reviews per API call (default: 10)")
    parser.add_argument("--max_reviews",   type=int, default=None,
                        help="Cap for dev/cost control (default: all)")
    parser.add_argument("--manual_labels", type=str, default=None,
                        help="Path to manual annotation CSV for kappa validation")
    parser.add_argument("--skip_classify", action="store_true",
                        help="Load existing results, skip API calls")
    parser.add_argument("--model",         type=str, default=MODEL,
                        help=f"OpenRouter model string (default: {MODEL})")
    args = parser.parse_args()

    # ---- Load data ----
    print(f"\n{'='*60}\nLOADING DATA\n{'='*60}")
    df = pd.read_csv(args.data, low_memory=False)
    print(f"  Total rows     : {len(df):,}")

    for col in [TARGET_COL, TEXT_EN_COL, REVIEW_ID_COL]:
        if col not in df.columns:
            print(f"  [ERROR] Required column '{col}' not found.")
            print(f"  Available: {list(df.columns)}")
            return

    if STATE_COL in df.columns:
        df["east_germany"] = df[STATE_COL].isin(EAST_STATES)
        df["region"]       = df["east_germany"].map({True: "East", False: "West"})

    discrim_df = df[
        df[TARGET_COL].str.lower().str.strip() == "discriminatory"
    ].copy().reset_index(drop=True)
    print(f"  Discriminatory : {len(discrim_df):,}")

    if args.max_reviews:
        discrim_df = discrim_df.head(args.max_reviews)
        print(f"  Capped at      : {args.max_reviews:,}")

    # ---- Classify or load ----
    results_path = os.path.join(OUTPUT_DIR, "discriminatory_subclassified.csv")

    if args.skip_classify and os.path.exists(results_path):
        print(f"\n  Loading existing results: {results_path}")
        tagged_df = pd.read_csv(results_path)
        for col in ["sub_categories", "llm_key_evidence"]:
            if col in tagged_df.columns:
                tagged_df[col] = tagged_df[col].apply(
                    lambda x: json.loads(x)
                    if isinstance(x, str) and x.startswith("[") else []
                )
    else:
        classifier = DiscriminatoryClassifier(
            model=args.model,
            reviews_per_batch=args.batch_size,
        )
        tagged_df = classify_reviews(discrim_df, classifier)
        tagged_df.to_csv(results_path, index=False)
        print(f"\n  Results saved: discriminatory_subclassified.csv")

    # ---- Summarise, plot, exemplars ----
    summarise_results(tagged_df)
    plot_results(tagged_df)
    extract_exemplars(tagged_df)

    # ---- Validation ----
    if args.manual_labels:
        validate_with_kappa(tagged_df, args.manual_labels)
    else:
        print("\n  Generating annotation template for future validation...")
        _generate_annotation_template(tagged_df)

    print(f"\n{'='*60}")
    print(f"  Complete. Outputs in: {OUTPUT_DIR}/")
    print('='*60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:<50} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()