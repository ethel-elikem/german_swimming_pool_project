"""
RQ2 Analysis Pipeline: "I am not racist, but..."
=================================================
Research Question 2:
    What social themes and linguistic patterns characterise
    racialised reviews of public swimming pools in Germany?

Two parts:
    Part A — Social themes  (BERTopic + LDA)
    Part B — Linguistic patterns (CDA strategy tagging, KWIC,
                                  TF-IDF contrast, exemplar extraction)

Usage:
    python rq2_pipeline.py --data path/to/data.csv --part all
    python rq2_pipeline.py --data path/to/data.csv --part A
    python rq2_pipeline.py --data path/to/data.csv --part B

Requirements:
    pip install bertopic sentence-transformers umap-learn hdbscan
                scikit-learn nltk matplotlib seaborn wordcloud pandas
                anthropic tqdm

Note on translations:
    Analysis runs on trans_review (English Google translation).
    For exemplars, org_review (original German) is always retrieved
    so you can quote the original in your paper.
"""

import os
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
from collections import Counter, defaultdict

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# BERTopic
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Note: BERTopic dependencies not installed.")
    print("  Run: pip install bertopic sentence-transformers umap-learn hdbscan")

# Anthropic API for LLM strategy tagging
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: anthropic package not installed — LLM tagging will be skipped.")
    print("  Run: pip install anthropic")

# WordCloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Download NLTK resources (run once)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_COL       = "connotation_type"    # discriminatory / ambiguous / pleasant
TEXT_EN_COL      = "trans_review"        # English translation
TEXT_DE_COL      = "org_review"          # Original German
PLACE_ID_COL     = "place_id"
CONSTITUENCY_COL = "constituency"
STATE_COL        = "state"

OUTPUT_DIR = "/mnt/user-data/outputs/rq2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# East German states (for regional comparison)
EAST_STATES = {
    "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen",
    "Sachsen-Anhalt", "Thüringen", "Berlin"
}

# CDA discourse strategies (van Dijk) — used for LLM tagging and KWIC
CDA_STRATEGIES = {
    "disclaimer":      "Preemptive denial of prejudice before making a racialised claim "
                       "(e.g. 'I have nothing against foreigners, but...')",
    "othering":        "Us/them construction through pronouns or spatial language "
                       "(e.g. 'our regulars', 'they take over', 'us locals')",
    "coded_attribution": "Negative traits attributed indirectly via vague language "
                          "(e.g. 'certain groups', 'you can imagine what kind of clientele')",
    "culturalisation": "Behaviour explained as innate cultural trait "
                       "(e.g. 'their culture around hygiene', 'they haven't learned our customs')",
    "threat_safety":   "Presence of others constructed as danger or discomfort "
                       "(e.g. 'felt unsafe', 'women were being stared at', 'overwhelming')",
    "nostalgia":       "The past as a racially coded ideal — how it used to be "
                       "(e.g. 'used to be wonderful', 'changed since the refugee wave')",
}

# KWIC anchor terms — English translations of key markers
KWIC_ANCHORS = [
    # Othering pronouns / deixis
    r"\bthey\b", r"\bthem\b", r"\bthese people\b", r"\bthose people\b",
    r"\bour\b", r"\bus locals\b", r"\bregulars\b", r"\bstrangers\b",
    # Coded attribution
    r"\bcertain (groups?|people|clientele|visitors?)\b",
    r"\bsome (groups?|people|cultures?)\b",
    r"\byou can (imagine|guess)\b",
    # Culturalisation
    r"\btheir culture\b", r"\bcultural(ly)?\b", r"\bbackground\b",
    # Threat / safety
    r"\bunsafe\b", r"\buncomfortable\b", r"\boverwhelm\b",
    r"\binvad\b", r"\btak(e|ing) over\b",
    # Nostalgia / temporal displacement
    r"\bused to\b", r"\bbefore\b", r"\bchanged\b", r"\bno longer\b",
    r"\brefugee\b", r"\bmigrant\b",
    # Disclaimer
    r"\bnot racist\b", r"\bnot against\b", r"\bnothing against\b",
    r"\bjust saying\b", r"\bto be honest\b",
]


# =============================================================================
# HELPERS
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nLOADING DATA\n{'='*60}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Total rows            : {len(df):,}")

    # Keep only racial reviews
    if "racial_connotations" in df.columns:
        df = df[df["racial_connotations"].str.lower().str.strip() == "yes"].copy()
        print(f"  After racial filter   : {len(df):,}")

    # East/West flag
    if STATE_COL in df.columns:
        df["east_germany"] = df[STATE_COL].isin(EAST_STATES)
        df["region"] = df["east_germany"].map({True: "East", False: "West"})

    # Normalise target
    df["connotation"] = df[TARGET_COL].str.lower().str.strip()

    print(f"  Connotation breakdown :")
    print(df["connotation"].value_counts().to_string())

    return df


def clean_text(text: str, extra_stopwords: list = None) -> str:
    """Clean English review text for topic modelling."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sw = set(stopwords.words("english"))
    # Add domain-specific stopwords (pool / generic review words)
    pool_stops = {
        "pool", "swim", "swimming", "water", "baths", "bath",
        "indoor", "outdoor", "facility", "facilities", "staff",
        "visit", "visited", "place", "area", "good", "nice", "great",
        "bad", "one", "also", "would", "get", "go", "went", "come",
        "can", "us", "like", "time", "really", "very", "quite",
    }
    sw.update(pool_stops)
    if extra_stopwords:
        sw.update(extra_stopwords)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in sw and len(w) > 2]
    return " ".join(tokens)


def print_section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")
    return path


# =============================================================================
# PART A — SOCIAL THEMES
# =============================================================================

# --- A1: BERTopic ---

def run_bertopic(df: pd.DataFrame, subset: str = "discriminatory",
                 n_topics: int = 15) -> tuple:
    """
    Run BERTopic on a subset of reviews.
    subset: 'discriminatory', 'ambiguous', 'all_racial'
    Returns (topic_model, topics, topic_info)
    """
    print_section(f"Part A — BERTopic ({subset} reviews)")

    if not BERTOPIC_AVAILABLE:
        print("  BERTopic not available — skipping.")
        return None, None, None

    if subset == "all_racial":
        texts_df = df.copy()
    else:
        texts_df = df[df["connotation"] == subset].copy()

    print(f"  Reviews for modelling : {len(texts_df):,}")

    # Clean texts
    texts_df["clean_text"] = texts_df[TEXT_EN_COL].apply(clean_text)
    texts_df = texts_df[texts_df["clean_text"].str.len() > 20].copy()
    docs = texts_df["clean_text"].tolist()

    print(f"  After cleaning        : {len(docs):,} non-empty docs")

    if len(docs) < 50:
        print("  Too few documents for reliable topic modelling (need ≥50).")
        return None, None, None

    # Embedding model — multilingual to handle any residual German
    print("  Loading sentence transformer (this downloads ~500MB on first run)...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Dimensionality reduction — conservative settings for stability
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Clustering — min_cluster_size scales with corpus size
    min_cluster = max(10, len(docs) // 100)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectoriser — controls the words that represent each topic
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=3,
        ngram_range=(1, 2),  # unigrams and bigrams
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    print("  Fitting BERTopic...")
    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    print(f"\n  Topics found          : {len(topic_info) - 1} (excl. outlier topic -1)")
    print(f"\n  Topic overview:")
    print(topic_info[topic_info["Topic"] != -1][["Topic", "Count", "Name"]].to_string(index=False))

    # Assign topics back to dataframe
    texts_df["bertopic_topic"] = topics
    texts_df["bertopic_prob"]  = [p.max() if hasattr(p, "max") else p for p in probs]

    # Save topic-document assignments
    out_path = os.path.join(OUTPUT_DIR, f"bertopic_assignments_{subset}.csv")
    texts_df[[TEXT_EN_COL, TEXT_DE_COL, "connotation", "bertopic_topic",
              STATE_COL, CONSTITUENCY_COL]].to_csv(out_path, index=False)
    print(f"  Saved: bertopic_assignments_{subset}.csv")

    # Visualise
    _plot_bertopic_barchart(topic_model, subset)
    _plot_topic_by_region(texts_df, topic_info, subset)

    return topic_model, texts_df, topic_info


def _plot_bertopic_barchart(topic_model, subset: str):
    """Top words per topic — horizontal bar chart."""
    topic_info = topic_model.get_topic_info()
    topics_to_plot = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()[:12]

    n_cols = 3
    n_rows = (len(topics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
    axes = axes.flatten()

    palette = plt.cm.Set2.colors

    for i, topic_id in enumerate(topics_to_plot):
        words = topic_model.get_topic(topic_id)
        if not words:
            continue
        terms, scores = zip(*words[:8])
        ax = axes[i]
        ax.barh(terms, scores, color=palette[i % len(palette)], alpha=0.8)
        ax.set_title(f"Topic {topic_id}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Score", fontsize=7)
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"BERTopic — top terms per topic ({subset} reviews)", fontsize=11)
    save_fig(f"bertopic_topics_{subset}.png")


def _plot_topic_by_region(texts_df: pd.DataFrame, topic_info: pd.DataFrame, subset: str):
    """Heatmap: topic distribution by East vs West Germany."""
    if "region" not in texts_df.columns:
        return
    if "bertopic_topic" not in texts_df.columns:
        return

    valid = texts_df[texts_df["bertopic_topic"] != -1].copy()
    if len(valid) < 20:
        return

    # Map topic ID to short name
    topic_names = dict(zip(topic_info["Topic"], topic_info["Name"]))
    valid["topic_name"] = valid["bertopic_topic"].map(topic_names)

    pivot = (
        valid.groupby(["region", "topic_name"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise rows to proportions
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    if pivot_norm.empty or pivot_norm.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(max(10, pivot_norm.shape[1] * 0.9), 4))
    sns.heatmap(
        pivot_norm, ax=ax, cmap="YlOrRd", annot=True,
        fmt=".2f", linewidths=0.3, cbar_kws={"label": "proportion"},
    )
    ax.set_title(f"Topic distribution by region ({subset} reviews)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    save_fig(f"bertopic_region_heatmap_{subset}.png")


# --- A2: LDA (comparison / interpretable alternative) ---

def run_lda(df: pd.DataFrame, n_topics: int = 10, subset: str = "discriminatory"):
    """
    LDA topic model — more interpretable for paper tables.
    Produces a clean topic × top-words table.
    """
    print_section(f"Part A — LDA ({subset} reviews, {n_topics} topics)")

    if subset == "all_racial":
        texts_df = df.copy()
    else:
        texts_df = df[df["connotation"] == subset].copy()

    texts_df["clean_text"] = texts_df[TEXT_EN_COL].apply(clean_text)
    texts_df = texts_df[texts_df["clean_text"].str.len() > 10]

    if len(texts_df) < 30:
        print("  Too few documents for LDA.")
        return None, None

    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=3,
        max_features=3000,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(texts_df["clean_text"])
    vocab = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        learning_method="online",
        random_state=42,
        n_jobs=-1,
    )
    lda.fit(dtm)

    # Build topic table
    topic_rows = []
    for i, comp in enumerate(lda.components_):
        top_idx  = comp.argsort()[::-1][:12]
        top_words = [vocab[j] for j in top_idx]
        topic_rows.append({
            "topic":     i + 1,
            "top_words": ", ".join(top_words[:8]),
        })

    topic_df = pd.DataFrame(topic_rows)
    print(f"\n  LDA topic summary:\n")
    print(topic_df.to_string(index=False))

    # Assign dominant topic per document
    doc_topics = lda.transform(dtm)
    texts_df["lda_topic"] = doc_topics.argmax(axis=1) + 1
    texts_df["lda_confidence"] = doc_topics.max(axis=1)

    # Save
    topic_df.to_csv(os.path.join(OUTPUT_DIR, f"lda_topics_{subset}.csv"), index=False)
    texts_df[[TEXT_EN_COL, TEXT_DE_COL, "connotation", "lda_topic",
              STATE_COL, CONSTITUENCY_COL]].to_csv(
        os.path.join(OUTPUT_DIR, f"lda_assignments_{subset}.csv"), index=False
    )
    print(f"  Saved: lda_topics_{subset}.csv")

    _plot_lda_topics(lda, vocab, n_topics, subset)
    _plot_topic_connotation_heatmap(texts_df, "lda_topic", subset)

    return lda, texts_df


def _plot_lda_topics(lda, vocab, n_topics: int, subset: str):
    """Grid of horizontal bar charts — top words per LDA topic."""
    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
    axes = axes.flatten() if n_topics > 1 else [axes]
    colors = plt.cm.Paired.colors

    for i, comp in enumerate(lda.components_):
        top_idx   = comp.argsort()[::-1][:8]
        top_words = [vocab[j] for j in top_idx]
        scores    = comp[top_idx]
        axes[i].barh(top_words, scores, color=colors[i % len(colors)], alpha=0.8)
        axes[i].set_title(f"Topic {i+1}", fontsize=9, fontweight="bold")
        axes[i].tick_params(labelsize=7)
        axes[i].invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"LDA — top terms per topic ({subset})", fontsize=11)
    save_fig(f"lda_topics_{subset}.png")


def _plot_topic_connotation_heatmap(texts_df: pd.DataFrame,
                                     topic_col: str, suffix: str):
    """Heatmap: topic × connotation type (how topics distribute across labels)."""
    if "connotation" not in texts_df.columns:
        return

    pivot = (
        texts_df.groupby(["connotation", topic_col])
        .size()
        .unstack(fill_value=0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    if pivot_norm.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot_norm.shape[1] * 0.6), 4))
    sns.heatmap(
        pivot_norm, ax=ax, cmap="Blues", annot=True,
        fmt=".2f", linewidths=0.3,
    )
    ax.set_title(f"Topic distribution by connotation type ({suffix})", fontsize=11)
    save_fig(f"topic_connotation_heatmap_{suffix}.png")


# =============================================================================
# PART B — LINGUISTIC PATTERNS
# =============================================================================

# --- B1: CDA Strategy Tagging via LLM ---

STRATEGY_TAG_PROMPT = """You are a researcher in Critical Discourse Analysis (CDA) studying 
racialised language in online reviews of public swimming pools in Germany.

You will be given a review (English translation). Your task is to tag it with the 
discourse strategies it uses. A review can use MULTIPLE strategies or NONE.

The six strategies are:
1. disclaimer      — Preemptive denial of prejudice before a racialised claim ("I have nothing against X, but...")
2. othering        — Us/them construction via pronouns or spatial language ("our regulars", "they take over")
3. coded_attribution — Negative traits attributed indirectly ("certain groups", "you can imagine the clientele")
4. culturalisation — Behaviour explained as innate cultural trait ("their culture around hygiene")
5. threat_safety   — Others' presence framed as danger ("felt unsafe", "women were stared at")
6. nostalgia       — Past as racially coded ideal ("used to be great", "changed since the refugee wave")

Respond ONLY with a valid JSON object. No explanation, no preamble, no markdown.
Format:
{
  "strategies": ["strategy_name", ...],
  "implicit": true/false,
  "confidence": "high"/"medium"/"low",
  "key_phrases": ["phrase1", "phrase2"]
}

If no strategy applies, return: {"strategies": [], "implicit": false, "confidence": "high", "key_phrases": []}

Review to analyse:
"""


def tag_with_llm(df: pd.DataFrame, batch_size: int = 50,
                 max_reviews: int = None) -> pd.DataFrame:
    """
    Tag each review with CDA discourse strategies using Claude API.
    
    IMPORTANT: Requires ANTHROPIC_API_KEY environment variable.
    Set it with: export ANTHROPIC_API_KEY=your_key_here
    
    For large datasets, set max_reviews to limit API calls during development.
    For the paper, run on all discriminatory + ambiguous reviews.
    """
    print_section("Part B — LLM CDA Strategy Tagging")

    if not ANTHROPIC_AVAILABLE:
        print("  anthropic package not installed. Run: pip install anthropic")
        return df

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set. Skipping LLM tagging.")
        print("  Set it with: export ANTHROPIC_API_KEY=your_key_here")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    # Tag discriminatory and ambiguous reviews (skip pleasant for efficiency)
    tag_df = df[df["connotation"].isin(["discriminatory", "ambiguous"])].copy()

    if max_reviews:
        tag_df = tag_df.head(max_reviews)

    print(f"  Reviews to tag        : {len(tag_df):,}")
    print(f"  Estimated API calls   : {len(tag_df):,}")
    print(f"  (at ~$0.003/review this costs ~${len(tag_df)*0.003:.2f})")

    results = []
    failed  = 0

    for i, row in tqdm(tag_df.iterrows(), total=len(tag_df), desc="  Tagging"):
        review_text = str(row.get(TEXT_EN_COL, ""))
        if len(review_text.strip()) < 10:
            results.append({"strategies": [], "implicit": False,
                             "confidence": "low", "key_phrases": []})
            continue

        # Truncate very long reviews
        review_text = review_text[:800]

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": STRATEGY_TAG_PROMPT + review_text
                }]
            )
            raw = response.content[0].text.strip()

            # Strip any accidental markdown fences
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            results.append(parsed)

        except json.JSONDecodeError:
            results.append({"strategies": [], "implicit": False,
                             "confidence": "low", "key_phrases": []})
            failed += 1
        except Exception as e:
            results.append({"strategies": [], "implicit": False,
                             "confidence": "low", "key_phrases": []})
            failed += 1
            # Brief backoff on rate limit
            if "rate" in str(e).lower():
                time.sleep(2)

        # Polite pause every batch
        if (i + 1) % batch_size == 0:
            time.sleep(1)

    print(f"  Tagging complete. Failed parses: {failed}/{len(tag_df)}")

    # Unpack results into columns
    tag_df["cda_strategies"]  = [r.get("strategies", [])    for r in results]
    tag_df["cda_implicit"]    = [r.get("implicit", False)   for r in results]
    tag_df["cda_confidence"]  = [r.get("confidence", "low") for r in results]
    tag_df["cda_key_phrases"] = [r.get("key_phrases", [])   for r in results]

    # One-hot encode each strategy for easy analysis
    for strategy in CDA_STRATEGIES.keys():
        tag_df[f"strat_{strategy}"] = tag_df["cda_strategies"].apply(
            lambda s: 1 if strategy in s else 0
        )

    # Save tagged data
    save_path = os.path.join(OUTPUT_DIR, "reviews_cda_tagged.csv")
    tag_df.to_csv(save_path, index=False)
    print(f"  Saved: reviews_cda_tagged.csv")

    _plot_strategy_distribution(tag_df)
    _plot_strategy_by_connotation(tag_df)
    _plot_strategy_by_region(tag_df)

    return tag_df


def _plot_strategy_distribution(tag_df: pd.DataFrame):
    """Bar chart: how often each CDA strategy appears overall."""
    strat_cols  = [c for c in tag_df.columns if c.startswith("strat_")]
    strat_counts = tag_df[strat_cols].sum().sort_values(ascending=True)
    strat_labels = [c.replace("strat_", "").replace("_", " ") for c in strat_counts.index]
    strat_pct    = strat_counts / len(tag_df) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strat_counts)))
    bars = ax.barh(strat_labels, strat_pct, color=colors, alpha=0.85)
    ax.set_xlabel("% of reviews using this strategy")
    ax.set_title("CDA discourse strategies — frequency across racial reviews", fontsize=11)
    for bar, val in zip(bars, strat_pct):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, strat_pct.max() * 1.2)
    save_fig("cda_strategy_distribution.png")


def _plot_strategy_by_connotation(tag_df: pd.DataFrame):
    """Grouped bar chart: strategy usage split by discriminatory vs ambiguous."""
    strat_cols = [c for c in tag_df.columns if c.startswith("strat_")]
    rows = []
    for connotation in ["discriminatory", "ambiguous"]:
        sub = tag_df[tag_df["connotation"] == connotation]
        if len(sub) == 0:
            continue
        for col in strat_cols:
            rows.append({
                "strategy":    col.replace("strat_", "").replace("_", " "),
                "connotation": connotation,
                "pct":         sub[col].mean() * 100,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df, x="pct", y="strategy", hue="connotation",
        palette={"discriminatory": "#D85A30", "ambiguous": "#888780"},
        ax=ax, alpha=0.85,
    )
    ax.set_xlabel("% of reviews using this strategy")
    ax.set_title("Strategy usage: discriminatory vs ambiguous reviews", fontsize=11)
    ax.legend(title="connotation type")
    save_fig("cda_strategy_by_connotation.png")


def _plot_strategy_by_region(tag_df: pd.DataFrame):
    """Heatmap: strategy × East/West region."""
    if "region" not in tag_df.columns:
        return

    strat_cols = [c for c in tag_df.columns if c.startswith("strat_")]
    pivot = tag_df.groupby("region")[strat_cols].mean() * 100
    pivot.columns = [c.replace("strat_", "").replace("_", " ") for c in pivot.columns]

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True,
        fmt=".1f", linewidths=0.3,
        cbar_kws={"label": "% of reviews"},
    )
    ax.set_title("CDA strategy usage by region (East vs West Germany)", fontsize=11)
    save_fig("cda_strategy_by_region.png")


# --- B2: KWIC — Keywords in Context ---

def kwic_analysis(df: pd.DataFrame, window: int = 8,
                  max_per_anchor: int = 30) -> pd.DataFrame:
    """
    Keyword In Context (concordance) analysis.
    For each anchor term, extract surrounding words (window tokens each side).
    Returns a dataframe of concordance lines for close reading.
    """
    print_section("Part B — KWIC (Keywords in Context)")

    # Run on discriminatory + ambiguous
    texts_df = df[df["connotation"].isin(["discriminatory", "ambiguous"])].copy()
    texts_df["text_lower"] = texts_df[TEXT_EN_COL].str.lower().fillna("")

    all_concordances = []

    for pattern in KWIC_ANCHORS:
        anchor_label = pattern.replace(r"\b", "").replace("\\", "").strip("()?")
        matches_found = 0

        for _, row in texts_df.iterrows():
            text  = row["text_lower"]
            words = text.split()

            for m in re.finditer(pattern, text):
                # Find word position of match
                char_pos   = m.start()
                word_pos   = len(text[:char_pos].split())
                left_ctx   = words[max(0, word_pos - window): word_pos]
                right_ctx  = words[word_pos: word_pos + window + 1]
                matched    = m.group()

                all_concordances.append({
                    "anchor":         anchor_label,
                    "left_context":   " ".join(left_ctx),
                    "keyword":        matched,
                    "right_context":  " ".join(right_ctx),
                    "full_review_en": row.get(TEXT_EN_COL, ""),
                    "full_review_de": row.get(TEXT_DE_COL, ""),
                    "connotation":    row["connotation"],
                    "state":          row.get(STATE_COL, ""),
                    "region":         row.get("region", ""),
                    "place_id":       row.get(PLACE_ID_COL, ""),
                })
                matches_found += 1
                if matches_found >= max_per_anchor:
                    break

            if matches_found >= max_per_anchor:
                break

        if matches_found > 0:
            print(f"  [{anchor_label:<25}] {matches_found:>4} occurrences")

    kwic_df = pd.DataFrame(all_concordances)
    print(f"\n  Total concordance lines : {len(kwic_df):,}")

    if not kwic_df.empty:
        save_path = os.path.join(OUTPUT_DIR, "kwic_concordances.csv")
        kwic_df.to_csv(save_path, index=False)
        print(f"  Saved: kwic_concordances.csv")

        # Summary: anchor frequency by connotation type
        summary = (
            kwic_df.groupby(["anchor", "connotation"])
            .size()
            .unstack(fill_value=0)
            .sort_values("discriminatory" if "discriminatory" in kwic_df["connotation"].values
                         else kwic_df["connotation"].iloc[0], ascending=False)
        )
        print(f"\n  KWIC anchor frequency by connotation:\n")
        print(summary.to_string())

        summary_path = os.path.join(OUTPUT_DIR, "kwic_summary.csv")
        summary.to_csv(summary_path)

        _plot_kwic_frequency(kwic_df)

    return kwic_df


def _plot_kwic_frequency(kwic_df: pd.DataFrame):
    """Dot plot: anchor term frequency split by connotation."""
    summary = (
        kwic_df.groupby(["anchor", "connotation"])
        .size()
        .reset_index(name="count")
    )
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(5, summary["anchor"].nunique() * 0.4)))
    palette = {"discriminatory": "#D85A30", "ambiguous": "#888780", "pleasant": "#1D9E75"}
    sns.barplot(
        data=summary, y="anchor", x="count", hue="connotation",
        palette=palette, ax=ax, alpha=0.85,
    )
    ax.set_title("KWIC anchor term frequency by connotation type", fontsize=11)
    ax.set_xlabel("Number of occurrences")
    ax.legend(title="connotation type")
    save_fig("kwic_anchor_frequency.png")


# --- B3: TF-IDF Lexical Contrast ---

def tfidf_contrast(df: pd.DataFrame, top_n: int = 30):
    """
    Compute TF-IDF scores per connotation class.
    Shows the vocabulary most distinctive of discriminatory vs ambiguous vs pleasant reviews.
    This is your 'vocabulary of implicit racism' analysis.
    """
    print_section("Part B — TF-IDF lexical contrast")

    df["clean_text"] = df[TEXT_EN_COL].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].copy()

    # Group texts by connotation
    groups = {}
    for label in ["discriminatory", "ambiguous", "pleasant"]:
        sub = df[df["connotation"] == label]["clean_text"].tolist()
        if sub:
            groups[label] = " ".join(sub)   # One mega-document per class

    if len(groups) < 2:
        print("  Need at least 2 connotation classes. Skipping.")
        return None

    labels = list(groups.keys())
    docs   = list(groups.values())

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    results = {}
    for i, label in enumerate(labels):
        scores = tfidf_matrix[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        results[label] = [(vocab[j], scores[j]) for j in top_idx]

    # Print top terms per class
    print(f"\n  Top {top_n} distinctive terms per connotation class:\n")
    for label, terms in results.items():
        term_str = ", ".join([t for t, s in terms[:15]])
        print(f"  {label.upper():<20}: {term_str}")

    # Save full results
    rows = []
    for label, terms in results.items():
        for rank, (term, score) in enumerate(terms):
            rows.append({"connotation": label, "rank": rank+1,
                         "term": term, "tfidf_score": round(score, 5)})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "tfidf_contrast.csv"), index=False
    )
    print("  Saved: tfidf_contrast.csv")

    _plot_tfidf_contrast(results, top_n=20)

    # Word clouds
    if WORDCLOUD_AVAILABLE:
        _plot_wordclouds(results)

    return results


def _plot_tfidf_contrast(results: dict, top_n: int = 20):
    """Side-by-side horizontal bar charts — top TF-IDF terms per class."""
    labels = list(results.keys())
    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 7))
    if n_cols == 1:
        axes = [axes]

    colors = {"discriminatory": "#D85A30", "ambiguous": "#888780", "pleasant": "#1D9E75"}

    for ax, label in zip(axes, labels):
        terms  = results[label][:top_n]
        words  = [t for t, s in terms]
        scores = [s for t, s in terms]
        ax.barh(words, scores, color=colors.get(label, "#888780"), alpha=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        ax.set_xlabel("TF-IDF score", fontsize=8)

    fig.suptitle("Most distinctive vocabulary per connotation type", fontsize=12, y=1.01)
    save_fig("tfidf_contrast.png")


def _plot_wordclouds(results: dict):
    """Word cloud per connotation class."""
    colors_map = {
        "discriminatory": "Reds",
        "ambiguous":      "Greys",
        "pleasant":       "Greens",
    }
    for label, terms in results.items():
        freq_dict = {t: s for t, s in terms}
        wc = WordCloud(
            width=600, height=300,
            background_color="white",
            colormap=colors_map.get(label, "Blues"),
            max_words=60,
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Most distinctive vocabulary — {label} reviews", fontsize=11)
        save_fig(f"wordcloud_{label}.png")


# --- B4: Exemplar Extraction ---

def extract_exemplars(df: pd.DataFrame,
                      tagged_df: pd.DataFrame = None,
                      n_per_strategy: int = 3) -> pd.DataFrame:
    """
    Extract the most representative review exemplars per:
      - CDA strategy (if tagged)
      - Connotation type
      - Region (East vs West)
    
    These go directly into the paper as qualitative evidence.
    Always retrieves org_review (German original) alongside translation.
    """
    print_section("Part B — Exemplar extraction")

    exemplars = []

    # --- Strategy exemplars (if LLM tagging was run) ---
    if tagged_df is not None and "cda_strategies" in tagged_df.columns:
        print("  Extracting strategy exemplars from tagged reviews...")
        for strategy in CDA_STRATEGIES.keys():
            strat_col = f"strat_{strategy}"
            if strat_col not in tagged_df.columns:
                continue

            sub = tagged_df[
                (tagged_df[strat_col] == 1) &
                (tagged_df["connotation"].isin(["discriminatory", "ambiguous"]))
            ].copy()

            if len(sub) == 0:
                continue

            # Prefer high-confidence tags, discriminatory first, then ambiguous
            sub = sub.sort_values(
                ["cda_confidence", "connotation"],
                ascending=[True, True]
            )

            for region in ["East", "West", "all"]:
                if region == "all":
                    pool = sub
                elif "region" in sub.columns:
                    pool = sub[sub["region"] == region]
                else:
                    continue

                for _, row in pool.head(n_per_strategy).iterrows():
                    exemplars.append({
                        "type":        "strategy",
                        "category":    strategy,
                        "region":      row.get("region", ""),
                        "connotation": row.get("connotation", ""),
                        "review_en":   row.get(TEXT_EN_COL, ""),
                        "review_de":   row.get(TEXT_DE_COL, ""),
                        "state":       row.get(STATE_COL, ""),
                        "key_phrases": str(row.get("cda_key_phrases", [])),
                        "confidence":  row.get("cda_confidence", ""),
                    })

    # --- Connotation × region exemplars (no tagging needed) ---
    print("  Extracting connotation × region exemplars...")
    for connotation in ["discriminatory", "ambiguous"]:
        for region_label, mask in [
            ("East", df.get("region", pd.Series(dtype=str)) == "East" if "region" in df.columns else pd.Series([True]*len(df))),
            ("West", df.get("region", pd.Series(dtype=str)) == "West" if "region" in df.columns else pd.Series([False]*len(df))),
        ]:
            sub = df[(df["connotation"] == connotation) & mask].copy()

            # Sort by review length — longer reviews tend to be more substantive
            sub["review_length"] = sub[TEXT_EN_COL].str.len().fillna(0)
            sub = sub[sub["review_length"] > 80].sort_values("review_length", ascending=False)

            for _, row in sub.head(n_per_strategy).iterrows():
                exemplars.append({
                    "type":        "connotation_region",
                    "category":    f"{connotation}_{region_label}",
                    "region":      region_label,
                    "connotation": connotation,
                    "review_en":   row.get(TEXT_EN_COL, ""),
                    "review_de":   row.get(TEXT_DE_COL, ""),
                    "state":       row.get(STATE_COL, ""),
                    "key_phrases": "",
                    "confidence":  "n/a",
                })

    exemplar_df = pd.DataFrame(exemplars)
    print(f"  Total exemplars extracted : {len(exemplar_df):,}")

    save_path = os.path.join(OUTPUT_DIR, "exemplars.csv")
    exemplar_df.to_csv(save_path, index=False)
    print(f"  Saved: exemplars.csv")
    print("\n  NOTE: Always cite org_review (German original) in the paper.")
    print("        The review_de column contains the original text.")

    return exemplar_df


# --- B5: Validation — Cohen's Kappa for LLM tagging ---

def compute_validation_kappa(tagged_df: pd.DataFrame,
                              manual_labels_path: str = None):
    """
    Compute Cohen's kappa between LLM strategy tags and manual labels.
    For ICWSM you need this to validate the LLM tagging approach.
    
    If manual_labels_path provided: expects a CSV with columns:
        review_id, strategy, manual_label (0/1)
    Otherwise: prints instructions for creating the manual annotation file.
    """
    print_section("Part B — Validation: Cohen's Kappa")

    from sklearn.metrics import cohen_kappa_score

    if manual_labels_path is None or not os.path.exists(str(manual_labels_path)):
        print("  No manual labels file found.")
        print("\n  To validate LLM tagging for your paper:")
        print("  1. Sample ~100 reviews from reviews_cda_tagged.csv")
        print("  2. Manually label each with the 6 strategies (0/1 per strategy)")
        print("  3. Save as: manual_labels.csv with columns:")
        print("       review_index, strategy, manual_label")
        print("  4. Re-run with: --manual_labels manual_labels.csv")
        print("\n  Target kappa ≥ 0.7 for ICWSM reviewers.")

        # Generate a sample annotation file to fill in
        if tagged_df is not None and len(tagged_df) > 0:
            sample = tagged_df.sample(min(100, len(tagged_df)), random_state=42)
            sample_rows = []
            for idx, row in sample.iterrows():
                for strategy in CDA_STRATEGIES.keys():
                    sample_rows.append({
                        "review_index": idx,
                        "review_en":    str(row.get(TEXT_EN_COL, ""))[:300],
                        "strategy":     strategy,
                        "llm_label":    row.get(f"strat_{strategy}", 0),
                        "manual_label": "",   # <- fill this in
                    })
            sample_df = pd.DataFrame(sample_rows)
            annotation_path = os.path.join(OUTPUT_DIR, "annotation_template.csv")
            sample_df.to_csv(annotation_path, index=False)
            print(f"\n  Annotation template saved: annotation_template.csv")
            print("  Fill in the 'manual_label' column (0 or 1) and re-run.")
        return None

    print(f"  Loading manual labels from {manual_labels_path}...")
    manual_df = pd.read_csv(manual_labels_path)

    kappa_results = []
    for strategy in CDA_STRATEGIES.keys():
        sub = manual_df[manual_df["strategy"] == strategy].dropna(subset=["manual_label"])
        if len(sub) < 10:
            continue

        # Merge with LLM labels
        strat_col = f"strat_{strategy}"
        if strat_col not in tagged_df.columns:
            continue

        merged = sub.merge(
            tagged_df[[strat_col]].reset_index().rename(columns={"index": "review_index"}),
            on="review_index", how="inner"
        )
        if len(merged) < 10:
            continue

        kappa = cohen_kappa_score(
            merged["manual_label"].astype(int),
            merged[strat_col].astype(int)
        )
        kappa_results.append({"strategy": strategy, "kappa": round(kappa, 3),
                               "n": len(merged)})

    if kappa_results:
        kappa_df = pd.DataFrame(kappa_results)
        print("\n  Cohen's Kappa — LLM vs manual labels:")
        print(kappa_df.to_string(index=False))
        mean_k = kappa_df["kappa"].mean()
        print(f"\n  Mean kappa: {mean_k:.3f}")
        if mean_k >= 0.7:
            print("  Agreement is SUBSTANTIAL — suitable to report in paper.")
        elif mean_k >= 0.6:
            print("  Agreement is MODERATE — acceptable with caveats in paper.")
        else:
            print("  Agreement is LOW — reconsider LLM prompting strategy.")

        kappa_df.to_csv(os.path.join(OUTPUT_DIR, "kappa_validation.csv"), index=False)
        return kappa_df

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 pipeline — social themes and linguistic patterns"
    )
    parser.add_argument("--data",   type=str, required=True, help="Path to CSV data")
    parser.add_argument("--part",   type=str, default="all",
                        choices=["all", "A", "B"],
                        help="Which part to run: A (themes), B (linguistics), or all")
    parser.add_argument("--subset", type=str, default="discriminatory",
                        choices=["discriminatory", "ambiguous", "all_racial"],
                        help="Review subset for topic modelling")
    parser.add_argument("--n_topics", type=int, default=10,
                        help="Number of topics for LDA (default: 10)")
    parser.add_argument("--max_tag",  type=int, default=None,
                        help="Max reviews to send to LLM for tagging (cost control)")
    parser.add_argument("--manual_labels", type=str, default=None,
                        help="Path to manual annotation CSV for kappa validation")
    args = parser.parse_args()

    df = load_data(args.data)

    tagged_df = None

    # ---- PART A: SOCIAL THEMES ----
    if args.part in ("all", "A"):
        print("\n" + "#"*60)
        print("# PART A — SOCIAL THEMES")
        print("#"*60)

        # BERTopic on selected subset
        topic_model, bertopic_df, topic_info = run_bertopic(
            df, subset=args.subset, n_topics=args.n_topics + 5
        )

        # LDA on same subset (comparison)
        lda_model, lda_df = run_lda(df, n_topics=args.n_topics, subset=args.subset)

        # Also run on all racial reviews for overview
        if args.subset != "all_racial":
            _, _ = run_lda(df, n_topics=args.n_topics, subset="all_racial")

    # ---- PART B: LINGUISTIC PATTERNS ----
    if args.part in ("all", "B"):
        print("\n" + "#"*60)
        print("# PART B — LINGUISTIC PATTERNS")
        print("#"*60)

        # B1: LLM strategy tagging
        tagged_df = tag_with_llm(df, max_reviews=args.max_tag)

        # B2: KWIC concordance
        kwic_df = kwic_analysis(df)

        # B3: TF-IDF lexical contrast
        tfidf_results = tfidf_contrast(df)

        # B4: Exemplar extraction
        exemplar_df = extract_exemplars(df, tagged_df=tagged_df)

        # B5: Validation (kappa)
        compute_validation_kappa(tagged_df, manual_labels_path=args.manual_labels)

    print("\n" + "="*60)
    print(f"  RQ2 pipeline complete.")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)
    print("\n  Output files summary:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:<45} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()