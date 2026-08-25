"""Extended theme insights: n-grams, log-odds, exemplar reviews, and word clouds.

Produces per-theme CSVs with top terms ranked by log-odds z-score and saves example
reviews and optional word cloud PNGs into the label_analysis folder.
"""

from __future__ import annotations

from collections import Counter
from math import log, sqrt
from pathlib import Path
import sys

import pandas as pd

from label_text_analysis_utils import (
    DATA_DIR,
    REPORT_DIR,
    tokenize,
    build_ngrams,
    prepare_class_frames,
    representative_reviews,
    format_combo,
)


ALPHA = 0.01  # Dirichlet prior strength
TOP_K = 50
MAX_N = 3


def compute_counts(class_frames: dict[tuple[str, ...], pd.DataFrame], max_n: int = 3) -> dict[tuple[str, ...], Counter]:
    class_counts: dict[tuple[str, ...], Counter] = {}
    total_counts = Counter()
    for combo, frame in class_frames.items():
        c = Counter()
        for text in frame["_text"]:
            tokens = tokenize(text)
            ngrams = build_ngrams(tokens, max_n=max_n)
            c.update(ngrams)
        class_counts[combo] = c
        total_counts.update(c)
    return class_counts, total_counts


def compute_log_odds(class_counts: dict[tuple[str, ...], Counter], background_counts: Counter, alpha: float = ALPHA) -> dict[tuple[str, ...], dict[str, float]]:
    # Precompute total tokens per class and background
    totals = {combo: sum(counter.values()) for combo, counter in class_counts.items()}
    background_total = sum(background_counts.values())

    results: dict[tuple[str, ...], dict[str, float]] = {}

    for combo, counter in class_counts.items():
        other_counts = Counter(background_counts)
        # subtract this class counts to get rest-of-corpus counts
        for term, ct in counter.items():
            other_counts[term] -= ct
            if other_counts[term] <= 0:
                del other_counts[term]

        other_total = background_total - totals[combo]
        combo_scores: dict[str, float] = {}
        for term in set(list(counter.keys()) + list(other_counts.keys())):
            count_c = counter.get(term, 0)
            count_not = other_counts.get(term, 0)

            p_c = (count_c + alpha * (background_counts.get(term, 0) / background_total)) / (totals[combo] + alpha)
            p_not = (count_not + alpha * (background_counts.get(term, 0) / background_total)) / (other_total + alpha)

            # logistic transform
            # avoid zero or one
            eps = 1e-12
            p_c = min(max(p_c, eps), 1 - eps)
            p_not = min(max(p_not, eps), 1 - eps)

            logodds = log(p_c / (1 - p_c)) - log(p_not / (1 - p_not))

            # variance approximation
            var = 1.0 / (count_c + alpha * (background_counts.get(term, 0) / background_total) + eps) + 1.0 / (count_not + alpha * (background_counts.get(term, 0) / background_total) + eps)
            z = logodds / sqrt(var)
            combo_scores[term] = z

        results[combo] = combo_scores

    return results


def write_per_theme_outputs(dimension: str, class_frames: dict[tuple[str, ...], pd.DataFrame], logodds: dict[tuple[str, ...], dict[str, float]], top_k: int = TOP_K):
    out_dir = REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for combo, scores in logodds.items():
        if len(combo) != 1:
            continue
        label = combo[0]
        rows = []
        for term, z in sorted(scores.items(), key=lambda item: -abs(item[1]))[:top_k]:
            rows.append({"term": term, "z_score": z, "count": class_frames[combo].apply(lambda r: tokenize(r['_text']).count(term.split()[0]) if term.split() else 0, axis=1).sum()})

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"{dimension}_{label}_top_terms.csv"
        df.to_csv(csv_path, index=False)

        # exemplar reviews (use existing helper)
        exemplar = representative_reviews(class_frames[combo], logodds[combo], max_examples=5)
        exemplar_path = out_dir / f"{dimension}_{label}_exemplar.csv"
        exemplar.to_csv(exemplar_path, index=False)


def try_make_wordclouds(class_frames: dict[tuple[str, ...], pd.DataFrame], logodds: dict[tuple[str, ...], dict[str, float]], max_words: int = 100):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except Exception:
        print("wordcloud or matplotlib not available; skipping word cloud generation")
        return

    out_dir = REPORT_DIR
    for combo, scores in logodds.items():
        if len(combo) != 1:
            continue
        label = combo[0]
        freqs = {term: abs(score) for term, score in scores.items()}
        if not freqs:
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freqs)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"theme_wordcloud_{label}.png")
        plt.close()


def main():
    parquet = DATA_DIR / "reclassed_racialised_theme.parquet"
    frame = pd.read_parquet(parquet, columns=["review_id", "org_review", "trans_review", "theme"]) 
    class_frames = prepare_class_frames(frame, "theme")
    class_counts, background_counts = compute_counts(class_frames, max_n=MAX_N)
    logodds = compute_log_odds(class_counts, background_counts, alpha=ALPHA)
    write_per_theme_outputs("themes", class_frames, logodds, top_k=TOP_K)
    try_make_wordclouds(class_frames, logodds)
    print("Theme insights written to:", REPORT_DIR)


if __name__ == "__main__":
    main()
