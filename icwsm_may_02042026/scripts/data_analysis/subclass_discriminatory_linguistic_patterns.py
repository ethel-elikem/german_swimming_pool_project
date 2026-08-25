"""
RQ2 Part B — Linguistic Patterns  (v2 — batched OpenRouter)
=============================================================
Refactored to match the batched API pattern from the discriminatory
sub-classifier script.  Key changes:
  - Single prompt built once at startup (build_cda_prompt())
  - Reviews sent in configurable batches (default 10) per API call
  - JSON array response keyed by review_id for safe alignment
  - Checkpoint / resume logic identical to the classifier
  - Expanded KWIC anchor list
  - B1 tags both perpetuated AND ambiguous corpora
  - B3 TF-IDF contrast uses all three corpora (perpetuated / ambiguous /
    experienced) where available in the combined dataset

Dataset assumptions (from df.info() provided):
  Core text columns : trans_review, org_review, review_id
  Region columns   : state, region (already present), east_germany
  Subclass columns : sub_categories, is_experienced, is_perpetuated,
                     is_observed (may be NaN for ambiguous rows)
  connotation_type : "discriminatory" | "ambiguous"

Usage:
    python rq2_partB_linguistic_v2.py --data path/to/combined.csv
    python rq2_partB_linguistic_v2.py --data path/to/combined.csv \\
        --batch_size 10 --max_tag 300 --part B1 B2 B3 B4 B5
    python rq2_partB_linguistic_v2.py --data path/to/combined.csv \\
        --skip_tag --manual_labels path/to/annotation_template.csv

Requirements:
    pip install requests pandas tqdm scikit-learn matplotlib seaborn
                nltk wordcloud python-dotenv
"""

import os
import re
import json
import time
import random
import argparse
import warnings
import requests
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    nltk.download(resource, quiet=True)

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "anthropic/claude-3.7-sonnet"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TARGET_COL        = "connotation_type"
TEXT_EN_COL       = "trans_review"
TEXT_DE_COL       = "org_review"
REVIEW_ID_COL     = "review_id"
STATE_COL         = "state"
SUBCLASS_COL      = "sub_categories"
CONSTITUENCY_COL  = "constituency"

OUTPUT_DIR = (
    "/home/ethel/Desktop/swimming_pool_project/"
    "icwsm_may_02042026/output_files/rq2_v2/partB_linguistic"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EAST_STATES = {
    "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen",
    "Sachsen-Anhalt", "Thüringen", "Berlin"
}

CORPUS_COLORS = {
    "perpetuated": "#7F77DD",
    "ambiguous":   "#888780",
    "experienced": "#1D9E75",
}

# Tokens per review for max_tokens budget (same logic as classifier)
TOKENS_PER_REVIEW = 400   # slightly higher than classifier — CDA has more fields

# ---------------------------------------------------------------------------
# SIX CDA STRATEGIES
# ---------------------------------------------------------------------------

CDA_STRATEGIES = {
    "disclaimer": (
        "Preemptive denial of prejudice before a racialised claim. "
        "E.g. 'I have nothing against foreigners, but…', "
        "'I'm not racist, but…', 'I don't want to be xenophobic, however…'"
    ),
    "othering": (
        "Us/them construction via pronouns, spatial or possessive language. "
        "E.g. 'our regulars', 'they take over', 'us locals', 'these people', "
        "'people like that', 'them'."
    ),
    "coded_attribution": (
        "Negative traits attributed indirectly without naming a group. "
        "E.g. 'certain groups', 'a particular clientele', "
        "'you can imagine who I mean', 'people who can't behave'."
    ),
    "culturalisation": (
        "Behaviour explained as an innate or essential cultural trait. "
        "E.g. 'their culture around hygiene', 'that's how they are raised', "
        "'different cultural norms', 'their mentality'."
    ),
    "threat_safety": (
        "Others' presence framed as danger, threat, or discomfort. "
        "E.g. 'felt unsafe', 'women were stared at', 'I was scared', "
        "'aggressive atmosphere', 'no longer feel comfortable'."
    ),
    "nostalgia": (
        "The past coded as a racially pure or ordered ideal. "
        "E.g. 'used to be great', 'changed since the refugee wave', "
        "'it wasn't like this before', 'the old crowd', "
        "'since they started coming here'."
    ),
}

# ---------------------------------------------------------------------------
# EXPANDED KWIC ANCHOR LIST
# ---------------------------------------------------------------------------

KWIC_ANCHORS = [
    # Pronouns / deixis
    r"\bthey\b",
    r"\bthem\b",
    r"\bthose people\b",
    r"\bthese people\b",
    r"\bpeople like (that|them|us|you)\b",
    r"\bpeople like this\b",
    # Possessives / spatial othering
    r"\bour (regulars?|guests?|pool|people|locals?|culture|country|values?)\b",
    r"\bus (locals?|Germans?|Europeans?)\b",
    r"\btheir (culture|hygiene|mentality|customs?|behavior|children|women|men)\b",
    # Coded group references
    r"\bcertain (groups?|people|clientele|visitors?|individuals?|types?)\b",
    r"\bsome (groups?|people|cultures?|men|families|visitors?)\b",
    r"\ba certain (type|kind|sort|group|clientele)\b",
    r"\byou can (imagine|guess|think)\b",
    r"\bif you know what I mean\b",
    r"\b(you|we) know (who|what) I mean\b",
    r"\bthat sort (of person|of people|of thing)\b",
    r"\bthe usual (crowd|suspects?|types?)\b",
    # Spatial take-over / dominance
    r"\btak(e|ing|en) over\b",
    r"\boverrun\b",
    r"\bovercrowded with\b",
    r"\bflooded (with|by)\b",
    r"\binvad(e|ing|ed)\b",
    r"\bbesieg(e|ing|ed)\b",
    r"\boccup(y|ied|ying)\b",
    # Safety / threat framing
    r"\b(felt?|feel(ing)?|no longer feel(ing)?)\s+(unsafe|uncomfortable|uneasy|unwelcome|scared|threatened)\b",
    r"\bstared? at\b",
    r"\bharass(ed|ing|ment)\b",
    r"\baggress?iv(e|ely|eness)\b",
    r"\bno longer safe\b",
    r"\bnot safe\b",
    r"\bfear for\b",
    # Nostalgia / temporal displacement
    r"\bused to (be|have|come)\b",
    r"\bit (was|used to be|has) (better|nice|great|fine|okay) (before|back then|years ago)\b",
    r"\bno longer (what it|the same|as nice|as good)\b",
    r"\bsince (the|they|it) (refugee|migrant|asylum|arrived|came|started)\b",
    r"\bchanged (since|after|when)\b",
    r"\bused to enjoy\b",
    r"\bwon.?t (come|go|return) back\b",
    r"\bnever (again|coming back)\b",
    # Disclaimer / preemptive denial
    r"\b(i.?m|i am|i.?ve) not (a |being )?(racist|xenophob|prejudiced)\b",
    r"\bnothing against (foreigners?|immigrants?|Muslims?|refugees?|them)\b",
    r"\bjust saying\b",
    r"\bdon.?t get me wrong\b",
    r"\bwithout wanting to sound\b",
    r"\bi (have|had) no problem with\b",
    r"\bi.?m not (trying to|being)\b",
    # Hygiene / culturalisation
    r"\bhygien(e|ic|ically)\b",
    r"\bunhygien(ic|ically)\b",
    r"\bdirty\b",
    r"\bdon.?t shower\b",
    r"\bclothes? in the water\b",
    r"\bstreet clothes?\b",
    r"\bbedding rules?\b",
    # Integration / adaptation framing
    r"\bintegrat(e|ion|ed|ing)\b",
    r"\badapt(ation|ed|ing)?\b",
    r"\blearn (German|our rules|our customs)\b",
    r"\bfollow (the )?rules?\b",
    r"\brespect (our|the) (culture|values?|rules?|customs?)\b",
    # Migration / refugee lexis
    r"\b(refugee|migrants?|asylum[- ]seekers?|newcomers?|skilled workers?)\b",
    r"\b(foreign|migrant) background\b",
    r"\bpeople (with|of) (a )?migra(tion|nt)\b",
    r"\bnew (residents?|citizens?|Germans?|arrivals?)\b",
    r"\bMerkel.?s (guests?|people)\b",
    # Burkini / religious dress (context-specific to dataset)
    r"\bburkini\b",
    r"\bheadscarf\b",
    r"\bfull.?body (suit|swimsuit|clothing|cover(ing)?)\b",
    r"\bfully (clothed|covered|veiled|dressed)\b",
    # Othering positives (framing absence as ideal)
    r"\bno (foreign(ers?)?|immigrants?|migrants?|refugees?|Arabs?|Turks?)\b",
    r"\bfinally (no|without|free from)\b",
    r"\bat least (no|without)\b",
    r"\bhardly any (foreigners?|immigrants?|migrants?)\b",
    # Quantifying dominance
    r"\b\d{2,3}\s*%\s*(of|are|were)\s*(foreigners?|migrants?|immigrants?|Arabs?|Turks?)\b",
    r"\b(most|majority|almost all|nearly all|80|90|95)\s*(of\s*)?(the\s*)?(visitors?|guests?|bathers?|people)\b",
    r"\bhardly (any|a single) (German|local)\b",
    r"\bno (more |longer )?(any )?Germans?\b",
]

# ---------------------------------------------------------------------------
# EXAMPLE REVIEWS FOR FEW-SHOT PROMPT
# ---------------------------------------------------------------------------

# Drawn from the dataset extract provided — one example per strategy
FEW_SHOT_EXAMPLES = [
    {
        "review_id": "ex_01",
        "text": (
            "Actually quite nice. Everything new. But it's pretty full and you "
            "just have to read the bathing rules and then you can imagine the "
            "audience. At least there is external security on some days. "
            "Strangely enough, the lifeguards only restrict German-speaking visitors..."
        ),
        "expected": {
            "strategies": ["coded_attribution", "othering"],
            "implicit": True,
            "east_west_marker": False,
            "confidence": "high",
            "key_phrases": ["you can imagine the audience", "German-speaking visitors"]
        }
    },
    {
        "review_id": "ex_02",
        "text": (
            "So don't get me wrong, but you just can't go in there anymore. "
            "I'm really open to everything and have nothing against "
            "multiculturalism, but what's going on here is too multicultural. "
            "Apart from the lifeguards, there are no Germans left in this pool."
        ),
        "expected": {
            "strategies": ["disclaimer", "nostalgia", "othering"],
            "implicit": True,
            "east_west_marker": False,
            "confidence": "high",
            "key_phrases": [
                "don't get me wrong", "nothing against multiculturalism",
                "no Germans left"
            ]
        }
    },
    {
        "review_id": "ex_03",
        "text": (
            "Affordable, clean, everything you need, and a pleasant clientele. "
            "Family-friendly, with finally few seniors and no foreign groups. "
            "A comfortable place to eat with the family."
        ),
        "expected": {
            "strategies": ["othering", "coded_attribution"],
            "implicit": True,
            "east_west_marker": False,
            "confidence": "high",
            "key_phrases": ["finally", "no foreign groups", "pleasant clientele"]
        }
    },
    {
        "review_id": "ex_04",
        "text": (
            "I've been going to the indoor pool since I was a child, and I "
            "don't want to criticize the pool itself. Since Schauer took over "
            "as a sub, it's been a complete mess. It's just a shame that "
            "Muslim women aren't allowed to swim. The atmosphere has changed "
            "since the refugee wave."
        ),
        "expected": {
            "strategies": ["nostalgia"],
            "implicit": True,
            "east_west_marker": False,
            "confidence": "medium",
            "key_phrases": ["atmosphere has changed", "since the refugee wave"]
        }
    },
    {
        "review_id": "ex_05",
        "text": (
            "It's really fun, especially the two slides. Unfortunately, what's "
            "not so nice anymore is that immigrants often stare at you and you "
            "feel uncomfortable as a woman."
        ),
        "expected": {
            "strategies": ["threat_safety"],
            "implicit": False,
            "east_west_marker": False,
            "confidence": "high",
            "key_phrases": ["stare at you", "feel uncomfortable"]
        }
    },
    {
        "review_id": "ex_06",
        "text": (
            "The visit was disappointing. Many people don't shower, and some "
            "enter the water in street clothes or without appropriate swimwear. "
            "People wearing hijabs, in particular, don't wear special swimwear. "
            "Unfortunately, this isn't monitored in any way – very problematic "
            "from a hygiene perspective."
        ),
        "expected": {
            "strategies": ["culturalisation", "coded_attribution"],
            "implicit": True,
            "east_west_marker": False,
            "confidence": "high",
            "key_phrases": [
                "their culture around hygiene", "people wearing hijabs",
                "hygiene perspective"
            ]
        }
    },
]


# ---------------------------------------------------------------------------
# PROMPT — built once at startup
# ---------------------------------------------------------------------------

def build_cda_prompt() -> str:
    """
    Tightened CDA tagging prompt.  ~40 % shorter than the v1 single-review
    prompt; full taxonomy and all six strategies preserved.  Structure mirrors
    the discriminatory classifier: examples → format → batch input.
    """
    strategy_block = "\n".join(
        f"  {i+1}. {name:<22} — {desc}"
        for i, (name, desc) in enumerate(CDA_STRATEGIES.items())
    )

    examples_json = json.dumps(
        [
            {
                "review_id": ex["review_id"],
                "text": ex["text"],
            }
            for ex in FEW_SHOT_EXAMPLES
        ],
        ensure_ascii=False,
        indent=2,
    )
    examples_output = json.dumps(
        [ex["expected"] | {"review_id": ex["review_id"]} for ex in FEW_SHOT_EXAMPLES],
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are a researcher in Critical Discourse Analysis (CDA) studying
implicit and subtle racialised language in online reviews of public swimming
pools in Germany (English translations of German originals).

You will tag each review with the CDA discourse strategies it employs.
A review may use MULTIPLE strategies or NONE.

These reviews come from two corpora:
  perpetuated — the reviewer's OWN language reproduces racial discrimination
                (coded, indirect, distancing, nostalgic).
  ambiguous   — racial discourse where intent or direction is unclear.

Look carefully for SUBTLE and IMPLICIT patterns, not only overt racism.

=== SIX STRATEGIES ===

{strategy_block}

=== EXTRA FLAGS ===

implicit         : true if the strategy is subtle/indirect rather than overtly racist.
east_west_marker : true if the review contains language tied to East/West German
                   identity or discourse (Ossi/Wessi, Wende, Heimat, "since 2015",
                   Plattenbau, GDR references, AfD references, "wave of refugees 2015").

=== OUTPUT FORMAT ===

Return ONLY a JSON array — one object per review, in input order.
No markdown fences, no preamble, no trailing text.

[
  {{
    "review_id": "<id>",
    "strategies": ["strategy_name", ...],
    "implicit": true/false,
    "east_west_marker": true/false,
    "confidence": "high"|"medium"|"low",
    "key_phrases": ["phrase1", "phrase2"]
  }},
  ...
]

=== EXAMPLES ===

Input:
{examples_json}

Output:
{examples_output}

=== REVIEWS TO TAG ===

"""


# ---------------------------------------------------------------------------
# OPENROUTER BATCH CALL  (mirrors classifier exactly)
# ---------------------------------------------------------------------------

def _empty_result(review_id: str, reason: str = "") -> dict:
    return {
        "review_id":        review_id,
        "strategies":       [],
        "implicit":         False,
        "east_west_marker": False,
        "confidence":       "low",
        "key_phrases":      [],
        "parse_status":     reason,
    }


def _normalise_result(review_id: str, raw: dict) -> dict:
    valid_strats = set(CDA_STRATEGIES.keys())
    strategies = [s for s in raw.get("strategies", []) if s in valid_strats]
    return {
        "review_id":        review_id,
        "strategies":       strategies,
        "implicit":         bool(raw.get("implicit", False)),
        "east_west_marker": bool(raw.get("east_west_marker", False)),
        "confidence":       raw.get("confidence", "low"),
        "key_phrases":      raw.get("key_phrases", []),
        "parse_status":     "ok",
    }


def _fallback_individual(
    prompt: str, batch: list[dict], model: str, max_tokens: int
) -> list[dict]:
    """Called only when batch JSON parsing fails after retries."""
    results = []
    for item in batch:
        res = call_openrouter_batch(
            prompt, [item], model, max_tokens=TOKENS_PER_REVIEW, max_retries=2
        )
        results.extend(res)
        time.sleep(0.5)
    return results


def call_openrouter_batch(
    prompt: str,
    batch: list[dict],
    model: str,
    max_tokens: int,
    max_retries: int = 3,
) -> list[dict]:
    """
    Send a batch of reviews to OpenRouter in a single API call.

    batch   : list of {"review_id": str, "text": str}
    Returns : list of normalised result dicts (same length and order as batch).
    """
    referers = [
        "http://localhost:3000", "http://localhost:8080",
        "https://www.google.com", "https://github.com",
        "https://example.com", "http://127.0.0.1:3000",
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
        "messages": [{"role": "user", "content": prompt + batch_json}],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=90
            )
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n  Rate limit (429) — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"\n  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return [
                    _empty_result(r["review_id"], f"http_{resp.status_code}")
                    for r in batch
                ]

            raw_text = resp.json()["choices"][0]["message"]["content"].strip()
            raw_text = re.sub(r"```json|```", "", raw_text).strip()
            parsed_list = json.loads(raw_text)

            if not isinstance(parsed_list, list):
                raise ValueError("Response is not a JSON array")

            result_map = {
                str(item.get("review_id", "")): item for item in parsed_list
            }
            return [
                _normalise_result(str(it["review_id"]),
                                   result_map.get(str(it["review_id"]), {}))
                for it in batch
            ]

        except (json.JSONDecodeError, ValueError, KeyError):
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"\n  Batch parse failed — falling back to per-review calls...")
                return _fallback_individual(prompt, batch, model, max_tokens)

        except Exception as e:   # requests exceptions
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return [
                    _empty_result(r["review_id"], f"error: {str(e)[:60]}")
                    for r in batch
                ]

    return [_empty_result(r["review_id"], "max_retries_exceeded") for r in batch]


# ---------------------------------------------------------------------------
# B1 — CDA STRATEGY TAGGING  (batched)
# ---------------------------------------------------------------------------

class CDABatchTagger:
    """Mirrors DiscriminatoryClassifier from the sub-classifier script."""

    def __init__(self, model: str = MODEL, reviews_per_batch: int = 10):
        self.model             = model
        self.reviews_per_batch = reviews_per_batch
        self.max_tokens        = min(TOKENS_PER_REVIEW * reviews_per_batch, 8000)

        print(f"\n  Model            : {self.model}")
        print(f"  Reviews per batch: {self.reviews_per_batch}")
        print(f"  max_tokens       : {self.max_tokens}")
        print(f"  Loading prompt...")
        self.prompt = build_cda_prompt()
        print(f"  Prompt loaded    : {len(self.prompt):,} chars")

    def tag_batch(self, items: list[dict]) -> list[dict]:
        valid, skipped = [], []
        for item in items:
            text = str(item.get("text", "")).strip()
            if len(text) < 10:
                skipped.append(
                    (item["review_id"], _empty_result(item["review_id"], "too_short"))
                )
            else:
                valid.append(item)

        if not valid:
            return [r for _, r in skipped]

        api_results = call_openrouter_batch(
            self.prompt, valid, self.model, self.max_tokens
        )
        result_map = {r["review_id"]: r for r in api_results}
        for rid, r in skipped:
            result_map[rid] = r
        return [result_map[item["review_id"]] for item in items]


def tag_cda_strategies(
    corpus_df: pd.DataFrame,
    corpus_name: str,
    tagger: CDABatchTagger,
    checkpoint_every: int = 5,
    pause_between_batches: float = 1.0,
) -> pd.DataFrame:
    """
    Tag a corpus with CDA strategies using batched OpenRouter calls.
    Supports checkpoint / resume (identical to discriminatory classifier).
    """
    print(f"\n{'='*60}")
    print(f"  B1 — CDA Tagging: {corpus_name}  ({len(corpus_df):,} reviews)")
    print(f"{'='*60}")

    bsz   = tagger.reviews_per_batch
    total = len(corpus_df)

    prompt_tokens  = len(tagger.prompt) // 4
    avg_rev_tok    = 150
    total_calls    = int(np.ceil(total / bsz))
    est_cost       = (prompt_tokens + avg_rev_tok * bsz) * total_calls / 1_000_000 * 3
    print(f"  API calls required : ~{total_calls:,}")
    print(f"  Est. input cost    : ~${est_cost:.2f}")

    checkpoint_path = os.path.join(
        OUTPUT_DIR, f"_cda_checkpoint_{corpus_name}.json"
    )
    results_map: dict = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                saved = json.load(f)
            results_map = {r["review_id"]: r for r in saved.get("results", [])}
            print(f"  Resuming from checkpoint: {len(results_map):,} done")
        except Exception:
            results_map = {}

    all_items = [
        {
            "review_id": str(row[REVIEW_ID_COL]),
            "text":      str(row.get(TEXT_EN_COL, ""))[:800],
        }
        for _, row in corpus_df.iterrows()
    ]
    pending  = [it for it in all_items if it["review_id"] not in results_map]
    batches  = [pending[i: i + bsz] for i in range(0, len(pending), bsz)]

    for batch_idx, batch in enumerate(
        tqdm(batches, desc=f"  Tagging {corpus_name}")
    ):
        batch_results = tagger.tag_batch(batch)
        for r in batch_results:
            results_map[r["review_id"]] = r

        if (batch_idx + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, results_map)

        time.sleep(pause_between_batches)

    _save_checkpoint(checkpoint_path, results_map)

    # ---- Unpack into DataFrame ----
    results = [
        results_map.get(
            str(row[REVIEW_ID_COL]),
            _empty_result(str(row[REVIEW_ID_COL]), "missing"),
        )
        for _, row in corpus_df.iterrows()
    ]

    out_df = corpus_df.copy()
    out_df["cda_strategies"]     = [r["strategies"]       for r in results]
    out_df["cda_implicit"]       = [r["implicit"]         for r in results]
    out_df["cda_east_west"]      = [r["east_west_marker"] for r in results]
    out_df["cda_confidence"]     = [r["confidence"]       for r in results]
    out_df["cda_key_phrases"]    = [r["key_phrases"]      for r in results]
    out_df["cda_parse_status"]   = [r["parse_status"]     for r in results]

    for strategy in CDA_STRATEGIES:
        out_df[f"strat_{strategy}"] = out_df["cda_strategies"].apply(
            lambda s: 1 if strategy in s else 0
        )
    out_df["n_strategies"] = out_df["cda_strategies"].apply(len)
    out_df["implicit_only"] = (
        out_df["cda_implicit"].astype(int) &
        (out_df["n_strategies"] > 0).astype(int)
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"cda_tagged_{corpus_name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: cda_tagged_{corpus_name}.csv")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    _plot_strategy_distribution(out_df, corpus_name)
    _plot_implicit_by_strategy(out_df, corpus_name)
    if "region" in out_df.columns:
        _plot_strategy_by_region(out_df, corpus_name)

    return out_df


def _save_checkpoint(path: str, results_map: dict):
    with open(path, "w") as f:
        json.dump({"results": list(results_map.values())}, f)


# ---------------------------------------------------------------------------
# B1 PLOT HELPERS
# ---------------------------------------------------------------------------

def _plot_strategy_distribution(tag_df: pd.DataFrame, corpus_name: str):
    strat_cols   = [c for c in tag_df.columns if c.startswith("strat_")]
    if not strat_cols:
        return
    strat_counts = tag_df[strat_cols].sum().sort_values(ascending=True)
    strat_labels = [c.replace("strat_", "").replace("_", " ") for c in strat_counts.index]
    strat_pct    = strat_counts / len(tag_df) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strat_counts)))
    bars   = ax.barh(strat_labels, strat_pct, color=colors, alpha=0.85)
    ax.set_xlabel("% of reviews using this strategy")
    ax.set_title(f"CDA strategies — {corpus_name}", fontsize=11)
    for bar, val in zip(bars, strat_pct):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, strat_pct.max() * 1.25)
    _save_fig(f"cda_strategies_{corpus_name}.png")


def _plot_implicit_by_strategy(tag_df: pd.DataFrame, corpus_name: str):
    strat_cols = [c for c in tag_df.columns if c.startswith("strat_")]
    rows = []
    for col in strat_cols:
        sub = tag_df[tag_df[col] == 1]
        if len(sub) == 0:
            continue
        rows.append({
            "strategy":     col.replace("strat_", "").replace("_", " "),
            "implicit_pct": sub["cda_implicit"].mean() * 100,
            "total":        len(sub),
        })
    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values("implicit_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["strategy"], plot_df["implicit_pct"],
            color=CORPUS_COLORS.get(corpus_name, "#7F77DD"), alpha=0.8)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("% of uses flagged as implicit / subtle")
    ax.set_title(f"Implicitness by strategy — {corpus_name}", fontsize=11)
    ax.set_xlim(0, 100)
    _save_fig(f"cda_implicit_by_strategy_{corpus_name}.png")


def _plot_strategy_by_region(tag_df: pd.DataFrame, corpus_name: str):
    strat_cols = [c for c in tag_df.columns if c.startswith("strat_")]
    pivot = tag_df.groupby("region")[strat_cols].mean() * 100
    pivot.columns = [c.replace("strat_", "").replace("_", " ") for c in pivot.columns]
    if pivot.empty or pivot.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.3, cbar_kws={"label": "% of reviews"})
    ax.set_title(f"CDA strategies by region — {corpus_name}", fontsize=11)
    _save_fig(f"cda_strategy_by_region_{corpus_name}.png")


def compare_strategy_profiles(
    tagged_perp: pd.DataFrame, tagged_ambi: pd.DataFrame
):
    """Side-by-side grouped bar chart: perpetuated vs ambiguous."""
    strat_cols = [c for c in tagged_perp.columns if c.startswith("strat_")]
    rows = []
    for col in strat_cols:
        label = col.replace("strat_", "").replace("_", " ")
        rows.append({
            "strategy":    label,
            "perpetuated": tagged_perp[col].mean() * 100,
            "ambiguous":   tagged_ambi[col].mean() * 100
                           if col in tagged_ambi.columns else 0.0,
        })

    plot_df = pd.DataFrame(rows).sort_values("perpetuated", ascending=True)
    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(x - width / 2, plot_df["perpetuated"], width,
            label="perpetuated", color=CORPUS_COLORS["perpetuated"], alpha=0.85)
    ax.barh(x + width / 2, plot_df["ambiguous"],   width,
            label="ambiguous",   color=CORPUS_COLORS["ambiguous"],   alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["strategy"], fontsize=9)
    ax.set_xlabel("% of reviews using this strategy")
    ax.set_title("CDA strategy comparison: perpetuated vs ambiguous", fontsize=11)
    ax.legend()
    _save_fig("cda_strategy_comparison.png")

    # Also save as CSV for paper table
    plot_df.to_csv(
        os.path.join(OUTPUT_DIR, "cda_strategy_comparison.csv"), index=False
    )
    print("  Saved: cda_strategy_comparison.csv + cda_strategy_comparison.png")


# ---------------------------------------------------------------------------
# B2 — KWIC CONCORDANCE
# ---------------------------------------------------------------------------

def kwic_analysis(
    corpora: dict,
    window: int = 8,
    max_per_anchor: int = 40,
) -> pd.DataFrame:
    """
    KWIC concordance on perpetuated + ambiguous corpora.
    Expanded anchor list; returns full concordance DataFrame.
    """
    print(f"\n{'='*60}\n  B2 — KWIC Concordance Analysis\n{'='*60}")

    all_rows = []
    for corpus_name in ["perpetuated", "ambiguous"]:
        corp = corpora.get(corpus_name, pd.DataFrame())
        if len(corp) == 0:
            print(f"  Skipping {corpus_name} — empty.")
            continue
        corp = corp.copy()
        corp["text_lower"] = corp[TEXT_EN_COL].fillna("").str.lower()

        for pattern in KWIC_ANCHORS:
            anchor_label = re.sub(r"[\\b()?\[\]^$]", "", pattern).strip()
            count = 0
            for _, row in corp.iterrows():
                text  = row["text_lower"]
                words = text.split()
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    char_pos  = m.start()
                    word_pos  = len(text[:char_pos].split())
                    left_ctx  = words[max(0, word_pos - window): word_pos]
                    right_ctx = words[word_pos: word_pos + window + 1]
                    all_rows.append({
                        "corpus":        corpus_name,
                        "anchor_pattern":anchor_label,
                        "keyword":       m.group(),
                        "left_context":  " ".join(left_ctx),
                        "right_context": " ".join(right_ctx),
                        "concordance_line": (
                            "... " + " ".join(left_ctx)
                            + f"  [{m.group()}]  "
                            + " ".join(right_ctx) + " ..."
                        ),
                        "review_en":     row.get(TEXT_EN_COL, ""),
                        "review_de":     row.get(TEXT_DE_COL, ""),
                        "region":        row.get("region", ""),
                        "state":         row.get(STATE_COL, ""),
                        REVIEW_ID_COL:   row.get(REVIEW_ID_COL, ""),
                    })
                    count += 1
                    if count >= max_per_anchor:
                        break
                if count >= max_per_anchor:
                    break

    if not all_rows:
        print("  No KWIC matches found.")
        return pd.DataFrame()

    kwic_df = pd.DataFrame(all_rows)
    print(f"  Total concordance lines : {len(kwic_df):,}")
    print(f"  Unique anchors matched  : {kwic_df['anchor_pattern'].nunique()}")

    kwic_df.to_csv(os.path.join(OUTPUT_DIR, "kwic_concordances.csv"), index=False)

    # Frequency summary: anchor × corpus
    summary = (
        kwic_df.groupby(["anchor_pattern", "corpus"])
        .size()
        .unstack(fill_value=0)
        .sort_values(
            "perpetuated" if "perpetuated" in kwic_df["corpus"].values else
            kwic_df["corpus"].iloc[0],
            ascending=False,
        )
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "kwic_summary.csv"))

    print("\n  Top 25 anchors (perpetuated | ambiguous):")
    print(summary.head(25).to_string())

    _plot_kwic_comparison(kwic_df)
    return kwic_df


def _plot_kwic_comparison(kwic_df: pd.DataFrame):
    # Only plot top-20 anchors by total frequency
    top_anchors = (
        kwic_df.groupby("anchor_pattern").size()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    sub = kwic_df[kwic_df["anchor_pattern"].isin(top_anchors)]
    summary = sub.groupby(["anchor_pattern", "corpus"]).size().reset_index(name="count")
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(11, max(6, len(top_anchors) * 0.45)))
    palette = {k: v for k, v in CORPUS_COLORS.items()
               if k in summary["corpus"].values}
    sns.barplot(data=summary, y="anchor_pattern", x="count", hue="corpus",
                palette=palette, ax=ax, alpha=0.85)
    ax.set_title("KWIC top-20 anchor frequency: perpetuated vs ambiguous", fontsize=11)
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("")
    _save_fig("kwic_comparison.png")


# ---------------------------------------------------------------------------
# B3 — TF-IDF LEXICAL CONTRAST
# ---------------------------------------------------------------------------

def tfidf_contrast(corpora: dict, top_n: int = 30) -> dict:
    """
    Three-way TF-IDF contrast: perpetuated vs ambiguous vs experienced.
    Vocabulary gradient: coded/vague → hedged → direct/evidential.
    """
    print(f"\n{'='*60}\n  B3 — TF-IDF Lexical Contrast\n{'='*60}")

    sw = set(stopwords.words("english"))
    pool_stops = {
        "pool", "swim", "swimming", "water", "bath", "indoor",
        "outdoor", "facility", "staff", "visit", "visited", "place",
        "area", "good", "nice", "great", "bad", "one", "also", "would",
        "get", "go", "went", "come", "can", "like", "time", "really",
        "very", "quite", "well", "even", "always", "often", "ever",
        "never", "unfortunately", "unfortunately",
    }
    sw.update(pool_stops)
    lemmatizer = WordNetLemmatizer()

    def clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(
            lemmatizer.lemmatize(w) for w in text.split()
            if w not in sw and len(w) > 2
        )

    groups: dict = {}
    for label in ["perpetuated", "ambiguous", "experienced"]:
        corp = corpora.get(label, pd.DataFrame())
        if len(corp) == 0:
            continue
        joined = " ".join(corp[TEXT_EN_COL].apply(clean).tolist())
        if joined.strip():
            groups[label] = joined

    if len(groups) < 2:
        print("  Need at least 2 non-empty corpora. Skipping TF-IDF.")
        return {}

    labels = list(groups.keys())
    docs   = list(groups.values())

    vectorizer   = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(docs)
    vocab        = vectorizer.get_feature_names_out()

    results: dict = {}
    for i, label in enumerate(labels):
        scores  = tfidf_matrix[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        results[label] = [(vocab[j], float(scores[j])) for j in top_idx]

    print("\n  Top 15 terms per corpus:")
    for label, terms in results.items():
        print(f"\n  {label.upper()}:")
        print("    " + ", ".join(t for t, _ in terms[:15]))

    rows = []
    for label, terms in results.items():
        for rank, (term, score) in enumerate(terms):
            rows.append({
                "corpus": label, "rank": rank + 1,
                "term": term, "tfidf_score": round(score, 5),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "tfidf_contrast.csv"), index=False
    )
    print("  Saved: tfidf_contrast.csv")

    _plot_tfidf_contrast(results, top_n=20)
    if WORDCLOUD_AVAILABLE:
        _plot_wordclouds(results)

    return results


def _plot_tfidf_contrast(results: dict, top_n: int = 20):
    labels = list(results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 5, 7))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        terms  = results[label][:top_n]
        words  = [t for t, _ in terms]
        scores = [s for _, s in terms]
        ax.barh(words, scores,
                color=CORPUS_COLORS.get(label, "#888780"), alpha=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        ax.set_xlabel("TF-IDF score", fontsize=8)
    fig.suptitle(
        "Distinctive vocabulary: perpetuated / ambiguous / experienced",
        fontsize=12, y=1.01,
    )
    _save_fig("tfidf_contrast.png")


def _plot_wordclouds(results: dict):
    cmap_map = {
        "perpetuated": "Purples",
        "ambiguous":   "Greys",
        "experienced": "Greens",
    }
    for label, terms in results.items():
        freq = {t: s for t, s in terms}
        wc   = WordCloud(
            width=700, height=350, background_color="white",
            colormap=cmap_map.get(label, "Blues"), max_words=60,
        ).generate_from_frequencies(freq)
        plt.figure(figsize=(9, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Vocabulary — {label}", fontsize=11)
        _save_fig(f"wordcloud_{label}.png")


# ---------------------------------------------------------------------------
# B4 — EXEMPLAR EXTRACTION
# ---------------------------------------------------------------------------

def extract_exemplars(
    corpora: dict,
    tagged_perp: pd.DataFrame = None,
    tagged_ambi: pd.DataFrame = None,
    n_per_strategy: int = 3,
) -> pd.DataFrame:
    """
    Extract exemplars per CDA strategy and per corpus × region.
    Always retains German original for paper quotation.
    """
    print(f"\n{'='*60}\n  B4 — Exemplar Extraction\n{'='*60}")
    rows = []

    def _add_row(record: dict):
        rows.append(record)

    # --- Strategy exemplars from tagged perpetuated ---
    for tagged_df, corpus_name in [
        (tagged_perp, "perpetuated"),
        (tagged_ambi, "ambiguous"),
    ]:
        if tagged_df is None or len(tagged_df) == 0:
            continue
        for strategy in CDA_STRATEGIES:
            strat_col = f"strat_{strategy}"
            if strat_col not in tagged_df.columns:
                continue
            sub = tagged_df[tagged_df[strat_col] == 1].copy()
            if len(sub) == 0:
                continue

            sub["review_len"] = sub[TEXT_EN_COL].str.len().fillna(0)
            sub = sub.sort_values(
                ["cda_implicit", "cda_confidence", "n_strategies", "review_len"],
                ascending=[False, True, True, False],
            )

            for region in ["East", "West", "all"]:
                if region == "all":
                    pool = sub
                elif "region" in sub.columns:
                    pool = sub[sub["region"] == region]
                else:
                    continue

                for _, row in pool.head(n_per_strategy).iterrows():
                    _add_row({
                        "type":        "strategy",
                        "strategy":    strategy,
                        "corpus":      corpus_name,
                        "region":      row.get("region", ""),
                        "review_en":   row.get(TEXT_EN_COL, ""),
                        "review_de":   row.get(TEXT_DE_COL, ""),
                        "state":       row.get(STATE_COL, ""),
                        "key_phrases": str(row.get("cda_key_phrases", [])),
                        "implicit":    row.get("cda_implicit", ""),
                        "confidence":  row.get("cda_confidence", ""),
                        "n_strategies": row.get("n_strategies", ""),
                        REVIEW_ID_COL: row.get(REVIEW_ID_COL, ""),
                    })

    # --- Corpus × region exemplars (raw, no tagging required) ---
    for corpus_name in ["perpetuated", "ambiguous"]:
        corp = corpora.get(corpus_name, pd.DataFrame())
        if len(corp) == 0:
            continue
        corp = corp.copy()
        corp["review_len"] = corp[TEXT_EN_COL].str.len().fillna(0)
        for region in ["East", "West"]:
            if "region" in corp.columns:
                pool = corp[corp["region"] == region]
            else:
                pool = corp
            pool = pool[pool["review_len"] > 80].sort_values(
                "review_len", ascending=False
            )
            for _, row in pool.head(n_per_strategy).iterrows():
                _add_row({
                    "type":        "corpus_region",
                    "strategy":    f"{corpus_name}_{region}",
                    "corpus":      corpus_name,
                    "region":      region,
                    "review_en":   row.get(TEXT_EN_COL, ""),
                    "review_de":   row.get(TEXT_DE_COL, ""),
                    "state":       row.get(STATE_COL, ""),
                    "key_phrases": "",
                    "implicit":    "",
                    "confidence":  "n/a",
                    "n_strategies": "",
                    REVIEW_ID_COL: row.get(REVIEW_ID_COL, ""),
                })

    exemplar_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "exemplars_rq2.csv")
    exemplar_df.to_csv(out_path, index=False)
    print(f"  Saved: exemplars_rq2.csv  ({len(exemplar_df):,} exemplars)")
    print("  NOTE: Always quote org_review (German original) in the paper.")
    return exemplar_df


# ---------------------------------------------------------------------------
# B5 — VALIDATION (Cohen's Kappa)
# ---------------------------------------------------------------------------

def compute_kappa(
    tagged_df: pd.DataFrame,
    manual_labels_path: str = None,
    corpus_name: str = "perpetuated",
):
    """
    Kappa validation for CDA strategy tagging.
    If no manual labels are supplied, generates an annotation template.
    """
    print(f"\n{'='*60}\n  B5 — Validation: Cohen's Kappa ({corpus_name})\n{'='*60}")

    if manual_labels_path is None or not os.path.exists(str(manual_labels_path)):
        print("  No manual labels file — generating annotation template.")
        _generate_cda_template(tagged_df, corpus_name)
        return None

    manual_df = pd.read_csv(manual_labels_path)
    required  = {"review_id", "strategy", "manual_label"}
    if not required.issubset(manual_df.columns):
        print(f"  Manual CSV must contain: {required}")
        return None

    kappa_rows = []
    for strategy in CDA_STRATEGIES:
        sub = manual_df[manual_df["strategy"] == strategy].dropna(
            subset=["manual_label"]
        )
        if len(sub) < 10:
            print(f"  {strategy}: too few labels ({len(sub)}) — need ≥10, skipping")
            continue
        strat_col = f"strat_{strategy}"
        if strat_col not in tagged_df.columns:
            continue
        merged = sub.merge(
            tagged_df[[REVIEW_ID_COL, strat_col]], on=REVIEW_ID_COL, how="inner"
        )
        if len(merged) < 10:
            print(f"  {strategy}: fewer than 10 rows after merge — skipping")
            continue
        kappa = cohen_kappa_score(
            merged["manual_label"].astype(int),
            merged[strat_col].astype(int),
        )
        acc = (
            merged["manual_label"].astype(int) == merged[strat_col].astype(int)
        ).mean()
        kappa_rows.append({
            "strategy": strategy,
            "kappa":    round(kappa, 3),
            "accuracy": round(acc, 3),
            "n":        len(merged),
        })
        print(f"  {strategy:<22}: κ={kappa:.3f}  acc={acc:.3f}  n={len(merged)}")

    if kappa_rows:
        kappa_df = pd.DataFrame(kappa_rows)
        mean_k   = kappa_df["kappa"].mean()
        print(f"\n  Mean κ: {mean_k:.3f}")
        verdict = (
            "STRONG — report with confidence"    if mean_k >= 0.80 else
            "SUBSTANTIAL — suitable for ICWSM"  if mean_k >= 0.70 else
            "MODERATE — acceptable with caveats" if mean_k >= 0.60 else
            "WEAK — revise prompt examples"
        )
        print(f"  Agreement: {verdict}")
        kappa_df.to_csv(os.path.join(OUTPUT_DIR, f"kappa_{corpus_name}.csv"),
                        index=False)
        return kappa_df
    return None


def _generate_cda_template(tagged_df: pd.DataFrame, corpus_name: str):
    rows = []
    for strategy in CDA_STRATEGIES:
        strat_col = f"strat_{strategy}"
        if strat_col not in tagged_df.columns:
            continue
        sub_pos = tagged_df[tagged_df[strat_col] == 1]
        sub_neg = tagged_df[tagged_df[strat_col] == 0]
        # balanced sample: equal pos/neg
        n_each  = min(15, len(sub_pos), len(sub_neg))
        sample  = pd.concat([
            sub_pos.sample(n_each, random_state=42),
            sub_neg.sample(n_each, random_state=42),
        ]).sample(frac=1, random_state=42)   # shuffle

        for _, row in sample.iterrows():
            rows.append({
                REVIEW_ID_COL:   row.get(REVIEW_ID_COL, ""),
                "review_en":     str(row.get(TEXT_EN_COL, ""))[:400],
                "review_de":     str(row.get(TEXT_DE_COL, ""))[:400],
                "strategy":      strategy,
                "llm_label":     row.get(strat_col, 0),
                "llm_key_phrases": str(row.get("cda_key_phrases", [])),
                "llm_confidence": row.get("cda_confidence", ""),
                "manual_label":  "",
                "annotator_notes": "",
            })

    path = os.path.join(OUTPUT_DIR, f"cda_annotation_template_{corpus_name}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Annotation template: cda_annotation_template_{corpus_name}.csv")
    print("  Fill 'manual_label' (0/1) then re-run with --manual_labels <path>")
    print("  Target κ ≥ 0.70 for ICWSM.")


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def _save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")


def load_and_split(data_path: str) -> dict:
    """
    Load the combined dataset (perpetuated + ambiguous rows) and split into
    named corpora.  Also reconstructs 'experienced' from is_experienced == 1
    if the column is present (for B3 TF-IDF contrast).
    """
    print(f"\n{'='*60}\nLOADING DATA\n{'='*60}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Total rows          : {len(df):,}")

    # Normalise connotation label
    df["connotation"] = df[TARGET_COL].str.lower().str.strip()

    # Region flag (may already be present; re-derive to be safe)
    if STATE_COL in df.columns:
        df["east_germany"] = df[STATE_COL].isin(EAST_STATES)
        df["region"]       = df["east_germany"].map(
            {True: "East", False: "West"}
        ).fillna("Unknown")

    import ast

    # Parse sub_categories if stored as string
    if SUBCLASS_COL in df.columns:
        def safe_parse_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                x = x.strip()
                if not x:
                    return []
                try:
                    # Parse Python literal (list, tuple, etc.)
                    parsed = ast.literal_eval(x)
                    # Ensure result is a list
                    return parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse {repr(x)[:100]} -> {e}")
                    return []
            return []

        df[SUBCLASS_COL] = df[SUBCLASS_COL].apply(safe_parse_list)
        
    # Build corpora
    perpetuated_df = df[
        (df["connotation"] == "discriminatory") &
        (df.get("is_perpetuated", pd.Series(0, index=df.index)).fillna(0) == 1)
    ].copy()

    if len(perpetuated_df) == 0:
        # Fallback: all discriminatory rows if sub-classification absent
        print("  WARNING: no is_perpetuated == 1 rows found.")
        print("  Using all discriminatory rows as perpetuated proxy.")
        perpetuated_df = df[df["connotation"] == "discriminatory"].copy()

    ambiguous_df = df[df["connotation"] == "ambiguous"].copy()

    experienced_df = pd.DataFrame()
    if "is_experienced" in df.columns:
        experienced_df = df[
            (df["connotation"] == "discriminatory") &
            (df["is_experienced"].fillna(0) == 1)
        ].copy()

    corpora = {
        "perpetuated": perpetuated_df,
        "ambiguous":   ambiguous_df,
        "experienced": experienced_df,
    }
    print("\n  Corpus sizes:")
    for name, corp in corpora.items():
        print(f"    {name:<15}: {len(corp):,} reviews")

    return corpora


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    import requests  # local import to surface missing-package error clearly

    parser = argparse.ArgumentParser(
        description="RQ2 Part B — Linguistic Patterns (batched OpenRouter)"
    )
    parser.add_argument("--data",         type=str, required=True,
                        help="Path to combined CSV (perpetuated + ambiguous rows).")
    parser.add_argument("--part",         nargs="+",
                        default=["B1", "B2", "B3", "B4", "B5"],
                        choices=["B1", "B2", "B3", "B4", "B5"],
                        help="Which sub-parts to run (default: all).")
    parser.add_argument("--batch_size",   type=int, default=10,
                        help="Reviews per API call for B1 tagging (default 10).")
    parser.add_argument("--max_tag",      type=int, default=None,
                        help="Cap reviews for LLM tagging — cost control.")
    parser.add_argument("--skip_tag",     action="store_true",
                        help="Skip B1 API calls; load existing cda_tagged_*.csv.")
    parser.add_argument("--manual_labels",type=str, default=None,
                        help="Path to filled annotation template for B5 kappa.")
    parser.add_argument("--model",        type=str, default=MODEL,
                        help=f"OpenRouter model string (default: {MODEL}).")
    args = parser.parse_args()

    corpora = load_and_split(args.data)

    tagged_perp = None
    tagged_ambi = None

    # ---- B1: CDA Tagging ----
    if "B1" in args.part:
        if not OPENROUTER_API_KEY:
            print("\n  OPENROUTER_API_KEY not set — skipping B1.")
        elif args.skip_tag:
            # Attempt to load previously saved tagged files
            for corpus_name, var_name in [
                ("perpetuated", "tagged_perp"),
                ("ambiguous",   "tagged_ambi"),
            ]:
                path = os.path.join(OUTPUT_DIR, f"cda_tagged_{corpus_name}.csv")
                if os.path.exists(path):
                    df_tagged = pd.read_csv(path)
                    for col in ["cda_strategies", "cda_key_phrases"]:
                        if col in df_tagged.columns:
                            df_tagged[col] = df_tagged[col].apply(
                                lambda x: json.loads(x)
                                if isinstance(x, str) and x.startswith("[")
                                else []
                            )
                    if corpus_name == "perpetuated":
                        tagged_perp = df_tagged
                    else:
                        tagged_ambi = df_tagged
                    print(f"  Loaded existing: cda_tagged_{corpus_name}.csv "
                          f"({len(df_tagged):,} rows)")
                else:
                    print(f"  WARNING: {path} not found — run without --skip_tag.")
        else:
            tagger = CDABatchTagger(
                model=args.model, reviews_per_batch=args.batch_size
            )
            for corpus_name in ["perpetuated", "ambiguous"]:
                corp = corpora[corpus_name]
                if len(corp) == 0:
                    print(f"  Skipping B1 {corpus_name} — empty corpus.")
                    continue
                tag_corpus = corp.head(args.max_tag) if args.max_tag else corp
                result = tag_cda_strategies(
                    tag_corpus, corpus_name, tagger
                )
                if corpus_name == "perpetuated":
                    tagged_perp = result
                else:
                    tagged_ambi = result

            if tagged_perp is not None and tagged_ambi is not None:
                compare_strategy_profiles(tagged_perp, tagged_ambi)

    # ---- B2: KWIC ----
    if "B2" in args.part:
        kwic_analysis(corpora)

    # ---- B3: TF-IDF ----
    if "B3" in args.part:
        tfidf_contrast(corpora)

    # ---- B4: Exemplars ----
    if "B4" in args.part:
        extract_exemplars(corpora, tagged_perp, tagged_ambi)

    # ---- B5: Kappa ----
    if "B5" in args.part:
        if tagged_perp is not None:
            compute_kappa(tagged_perp, args.manual_labels, "perpetuated")
        if tagged_ambi is not None:
            compute_kappa(tagged_ambi, args.manual_labels, "ambiguous")

    print(f"\n{'='*60}")
    print(f"  Part B complete. Outputs: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:<55} {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()