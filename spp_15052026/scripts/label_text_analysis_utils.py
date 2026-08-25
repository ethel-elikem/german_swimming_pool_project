"""Shared helpers for analyzing label-specific language in classifier outputs.

The utilities in this module build contrastive term profiles for each label or
label-combination class and select representative reviews for manual inspection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "data_annotation_validation" / "label_analysis"

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'\-]*")

STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "nope",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "also",
    "das",
    "der",
    "die",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    "und",
    "oder",
    "aber",
    "ist",
    "sind",
    "war",
    "waren",
    "sein",
    "im",
    "in",
    "am",
    "an",
    "auf",
    "zu",
    "von",
    "für",
    "mit",
    "ohne",
    "nicht",
    "kein",
    "keine",
    "als",
    "auch",
    "noch",
    "mehr",
    "sehr",
    "schon",
    "dass",
    "da",
    "dort",
    "hier",
    "wie",
    "wer",
    "was",
    "wo",
    "wann",
    "warum",
    "weil",
    "pool",
    "swimming",
    "swim",
    "bath",
    "baths",
    "sauna",
    "water",
    "place",
    "review",
    "reviews",
}


def parse_label_value(value: object) -> tuple[str, ...]:
    """Parse the stored label field into a stable tuple of labels."""
    if value is None or pd.isna(value):
        return tuple()

    if isinstance(value, (list, tuple, set)):
        labels = [str(item).strip().strip("'\"") for item in value if str(item).strip().strip("'\"")]
    else:
        text = str(value).strip()
        if not text:
            return tuple()

        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        labels = [left or right for left, right in quoted if (left or right)]

        if not labels:
            stripped = text.strip("[]")
            if stripped and ("," in stripped or " " in stripped):
                parts = re.split(r"[\s,]+", stripped)
                labels = [part.strip().strip("'\"") for part in parts if part.strip().strip("'\"")]
            else:
                labels = [text.strip("'\"")]

    labels = [label for label in labels if label]
    labels = sorted(dict.fromkeys(labels), key=str.casefold)
    return tuple(labels)


def format_combo(combo: tuple[str, ...]) -> str:
    return " + ".join(combo)


def normalize_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    return str(text).strip()


def select_review_text(row: pd.Series) -> str:
    """Prefer translated text, then original text."""
    trans_review = normalize_text(row.get("trans_review"))
    if trans_review:
        return trans_review
    return normalize_text(row.get("org_review"))


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text.lower())]
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def build_ngrams(tokens: list[str], max_n: int = 3) -> list[str]:
    ngrams: list[str] = []
    for size in range(1, max_n + 1):
        if len(tokens) < size:
            continue
        for index in range(len(tokens) - size + 1):
            ngrams.append(" ".join(tokens[index : index + size]))
    return ngrams


def load_dimension_data(parquet_path: Path, label_column: str) -> pd.DataFrame:
    columns = ["review_id", "org_review", "trans_review", label_column]
    frame = pd.read_parquet(parquet_path, columns=columns)
    if label_column not in frame.columns:
        raise KeyError(f"{label_column!r} not found in {parquet_path}")
    return frame


def prepare_class_frames(df: pd.DataFrame, label_column: str) -> dict[tuple[str, ...], pd.DataFrame]:
    working = df.copy()
    working["_text"] = working.apply(select_review_text, axis=1)
    working["_combo"] = working[label_column].map(parse_label_value)
    working = working[working["_combo"].map(bool)]
    grouped = {combo: group.copy() for combo, group in working.groupby("_combo", sort=True)}
    return grouped


def compute_class_term_scores(class_frames: dict[tuple[str, ...], pd.DataFrame], max_n: int = 3) -> dict[tuple[str, ...], dict[str, float]]:
    class_term_counts: dict[tuple[str, ...], Counter] = {}
    class_vocab: dict[tuple[str, ...], set[str]] = {}

    for combo, frame in class_frames.items():
        counter: Counter[str] = Counter()
        vocab: set[str] = set()
        for text in frame["_text"]:
            tokens = tokenize(text)
            ngrams = build_ngrams(tokens, max_n=max_n)
            counter.update(ngrams)
            vocab.update(ngrams)
        class_term_counts[combo] = counter
        class_vocab[combo] = vocab

    term_class_df: Counter[str] = Counter()
    for vocab in class_vocab.values():
        for term in vocab:
            term_class_df[term] += 1

    n_classes = len(class_frames)
    class_term_scores: dict[tuple[str, ...], dict[str, float]] = {}
    for combo, counter in class_term_counts.items():
        total_terms = sum(counter.values()) or 1
        scores: dict[str, float] = {}
        for term, count in counter.items():
            class_df = term_class_df[term]
            idf = math.log((n_classes + 1) / (class_df + 1)) + 1.0
            scores[term] = (count / total_terms) * idf
        class_term_scores[combo] = scores

    return class_term_scores


def top_terms_for_class(scores: dict[str, float], top_n: int = 15) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def contrast_terms_for_combo(
    combo: tuple[str, ...],
    class_scores: dict[tuple[str, ...], dict[str, float]],
    top_n: int = 15,
) -> list[tuple[str, float]]:
    if len(combo) <= 1:
        return top_terms_for_class(class_scores[combo], top_n=top_n)

    constituent_single_scores = [class_scores[tuple([label])] for label in combo if tuple([label]) in class_scores]
    if not constituent_single_scores:
        return top_terms_for_class(class_scores[combo], top_n=top_n)

    contrasted: dict[str, float] = {}
    combo_scores = class_scores[combo]
    for term, score in combo_scores.items():
        baseline = max(single_scores.get(term, 0.0) for single_scores in constituent_single_scores)
        delta = score - baseline
        if delta > 0:
            contrasted[term] = delta
    return sorted(contrasted.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def representative_reviews(
    frame: pd.DataFrame,
    class_scores: dict[str, float],
    max_examples: int = 5,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["review_id", "label_combo", "score", "matched_terms", "text"])

    top_terms = {term for term, _ in top_terms_for_class(class_scores, top_n=25)}
    ranked_rows = []
    for _, row in frame.iterrows():
        text = row["_text"]
        tokens = tokenize(text)
        ngrams = set(build_ngrams(tokens, max_n=3))
        matched = sorted(term for term in ngrams if term in top_terms)
        score = sum(class_scores.get(term, 0.0) for term in matched)
        ranked_rows.append(
            {
                "review_id": row["review_id"],
                "label_combo": format_combo(row["_combo"]),
                "score": score,
                "matched_terms": "; ".join(matched),
                "text": text,
            }
        )

    result = pd.DataFrame(ranked_rows)
    result = result.sort_values(["score", "review_id"], ascending=[False, True]).head(max_examples)
    return result.reset_index(drop=True)


def build_analysis_rows(
    class_frames: dict[tuple[str, ...], pd.DataFrame],
    class_scores: dict[tuple[str, ...], dict[str, float]],
    top_terms: int = 15,
    contrast_top_terms: int = 15,
    example_reviews: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    example_rows = []

    for combo in sorted(class_frames.keys(), key=lambda item: (len(item), item)):
        frame = class_frames[combo]
        scores = class_scores[combo]
        summary_rows.append(
            {
                "label_combo": format_combo(combo),
                "labels": "|".join(combo),
                "count": len(frame),
                "top_terms": "; ".join(f"{term}:{score:.4f}" for term, score in top_terms_for_class(scores, top_n=top_terms)),
                "contrast_terms": "; ".join(f"{term}:{score:.4f}" for term, score in contrast_terms_for_combo(combo, class_scores, top_n=contrast_top_terms)),
            }
        )

        examples = representative_reviews(frame, scores, max_examples=example_reviews)
        for example_index, example_row in examples.iterrows():
            example_rows.append(
                {
                    "label_combo": format_combo(combo),
                    "example_rank": example_index + 1,
                    "review_id": example_row["review_id"],
                    "score": example_row["score"],
                    "matched_terms": example_row["matched_terms"],
                    "text": example_row["text"],
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(example_rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe to a markdown table string.

    Keeps implementation dependency-free so scripts can run in minimal venvs.
    """
    if df is None or df.empty:
        return "| column | value |\n| --- | --- |"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_analysis_report(
    dimension: str,
    source_file: Path,
    label_column: str,
    top_terms: int = 15,
    contrast_top_terms: int = 15,
    example_reviews: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source_file, columns=["review_id", "org_review", "trans_review", label_column])
    class_frames = prepare_class_frames(frame, label_column)
    class_scores = compute_class_term_scores(class_frames)
    summary_df, example_df = build_analysis_rows(
        class_frames,
        class_scores,
        top_terms=top_terms,
        contrast_top_terms=contrast_top_terms,
        example_reviews=example_reviews,
    )

    summary_csv = REPORT_DIR / f"{dimension}_label_terms.csv"
    examples_csv = REPORT_DIR / f"{dimension}_representative_reviews.csv"
    summary_md = REPORT_DIR / f"{dimension}_summary.md"

    summary_df.to_csv(summary_csv, index=False)
    example_df.to_csv(examples_csv, index=False)

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write(f"# {dimension.title()} label analysis\n\n")
        handle.write(f"Source file: `{source_file.name}`\n\n")
        handle.write(f"Total label combinations: {len(summary_df)}\n\n")
        handle.write("## Top label combinations by size\n\n")
        preview = summary_df.sort_values(["count", "label_combo"], ascending=[False, True]).head(20)
        handle.write(dataframe_to_markdown(preview[["label_combo", "count", "top_terms", "contrast_terms"]]))
        handle.write("\n\n## Where to inspect examples\n\n")
        handle.write(f"See `{examples_csv.name}` for representative reviews per label combination.\n")

    return summary_df, example_df
