"""
review_analysis.py
------------------
1. Loads the output parquet files from pool_pipeline.py:
     - places_filtered.parquet
     - reviews_filtered.parquet

2. Filters reviews to only those with non-empty, non-whitespace org_review text.
   All subsequent steps operate on this text-only subset.

3. Checks review coverage per place:
     - Coverage = number of text reviews present / total_reviews in places table
     - Flags places where coverage < 50%
     - Prints a coverage report

4. Checks the distribution of review lengths using the trans_review column.

5. Adds a 'needs_summary' flag (True if len(trans_review) > 100).

6. Saves the text-only subset with the 'needs_summary' flag as the output parquet.

7. Passes the flagged DataFrame to summarize_reviews() from summarize_reviews.py.

Usage:
    python review_analysis.py \
        --places places_filtered.parquet \
        --reviews reviews_filtered.parquet \
        --output_dir ./data
"""

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COVERAGE_THRESHOLD = 0.50       # 50% minimum
SUMMARY_CHAR_THRESHOLD = 700    # reviews longer than this get flagged changed to 700 chars to better balance the needs_summary flag (was 100 chars, but many reviews are just over that)

DB_USER = "ethel"
DB_PASSWORD = "pa55w0rd13"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "swimming_pools_db"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(places_path: str, reviews_path: str):
    print(f"Loading places  : {places_path}")
    places = pd.read_parquet(places_path)

    print(f"Loading reviews : {reviews_path}")
    reviews = pd.read_parquet(reviews_path)

    print(f"  Places  rows : {len(places):,}")
    print(f"  Reviews rows : {len(reviews):,}\n")
    return places, reviews


# ---------------------------------------------------------------------------
# 2. Filter to text-only reviews
# ---------------------------------------------------------------------------

def filter_text_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where org_review contains actual text —
    i.e. not null and not blank/whitespace-only.
    """
    if "org_review" not in reviews.columns:
        raise ValueError("'org_review' column not found in reviews DataFrame.")

    before = len(reviews)
    mask = reviews["org_review"].notna() & reviews["org_review"].str.strip().ne("")
    text_reviews = reviews[mask].copy()
    after = len(text_reviews)

    print(f"Filtering to text reviews (org_review non-empty):")
    print(f"  Before : {before:,}")
    print(f"  After  : {after:,}")
    print(f"  Dropped: {before - after:,}\n")

    return text_reviews


def fill_missing_trans_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty or whitespace-only trans_review values from org_review.

    This keeps English org_review text available to downstream steps when the
    translation field was never populated.
    """
    if "trans_review" not in reviews.columns:
        raise ValueError("'trans_review' column not found in reviews DataFrame.")
    if "org_review" not in reviews.columns:
        raise ValueError("'org_review' column not found in reviews DataFrame.")

    reviews = reviews.copy()
    mask = reviews["trans_review"].isna() | reviews["trans_review"].astype(str).str.strip().eq("")
    filled = int(mask.sum())
    if filled:
        reviews.loc[mask, "trans_review"] = reviews.loc[mask, "org_review"]
        print(f"Filled trans_review from org_review for {filled:,} rows\n")
    return reviews


# ---------------------------------------------------------------------------
# 3. Coverage check
# ---------------------------------------------------------------------------

def check_coverage(places: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    For each place, compare the number of reviews present in the reviews
    DataFrame against the total_reviews column in the places DataFrame.

    Returns a coverage DataFrame with one row per place, including a
    'below_threshold' flag for places under 50% coverage.
    """
    if "unique_identifier" in places.columns:
        place_id_col = "unique_identifier"
    elif "id" in places.columns:
        place_id_col = "id"
    else:
        raise ValueError(
            "Could not find a place ID column ('unique_identifier' or 'id') "
            "in the places DataFrame."
        )

    if "total_reviews" not in places.columns:
        raise ValueError("'total_reviews' column not found in places DataFrame.")

    # Count reviews per place from the reviews table
    review_counts = (
        reviews.groupby("unique_identifier")
        .size()
        .reset_index(name="reviews_present")
    )

    # Merge with places
    coverage = places[[place_id_col, "total_reviews"]].merge(
        review_counts,
        left_on=place_id_col,
        right_on="unique_identifier",
        how="left",
    )

    # Always expose a clearly named 'place_id' column regardless of source name
    if place_id_col != "place_id":
        coverage.insert(0, "place_id", coverage[place_id_col])

    coverage["reviews_present"] = coverage["reviews_present"].fillna(0).astype(int)
    coverage["total_reviews"] = pd.to_numeric(coverage["total_reviews"], errors="coerce").fillna(0)

    # Avoid division by zero
    coverage["coverage_pct"] = coverage.apply(
        lambda row: (row["reviews_present"] / row["total_reviews"])
        if row["total_reviews"] > 0
        else None,
        axis=1,
    )

    coverage["below_threshold"] = coverage["coverage_pct"].apply(
        lambda x: (x < COVERAGE_THRESHOLD) if x is not None else None
    )

    return coverage


def _compute_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expected_jsons and collected_jsons columns to a coverage DataFrame.

    expected_jsons : ceil(total_reviews / 10)
    collected_jsons: ceil(reviews_present / 10), but if that value is
                     between 1 and 9 inclusive it is clamped to 1.
                     Zero stays zero (no reviews collected at all).
    """
    import math

    def expected(total):
        total = int(total) if not pd.isna(total) else 0
        return math.ceil(total / 10) if total > 0 else 0

    def collected(present):
        present = int(present) if not pd.isna(present) else 0
        if present == 0:
            return 0
        val = math.ceil(present / 10)
        return 1 if 0 < val < 10 else val

    df = df.copy()
    df["expected_jsons"]  = df["total_reviews"].apply(expected)
    df["collected_jsons"] = df["reviews_present"].apply(collected)
    return df


def build_below_threshold_csv(coverage: pd.DataFrame) -> pd.DataFrame:
    """
    Build the below-threshold flagged places CSV.

    Columns:
        place_id        : place identifier
        unique_identifier: original identifier column (same value, kept for traceability)
        expected_jsons  : ceil(total_reviews / 10)
        collected_jsons : ceil(reviews_present / 10), clamped to 1 when between 1-9
        total_reviews   : raw total from places table
        reviews_present : reviews found in reviews table
        coverage_pct    : fraction covered
    """
    flagged = coverage[coverage["below_threshold"] == True].copy()
    if flagged.empty:
        return flagged

    flagged = _compute_json_columns(flagged)

    # Determine unique_identifier column name (may be same as place_id)
    uid_col = "unique_identifier" if "unique_identifier" in flagged.columns else "place_id"

    out = flagged[[
        "place_id",
        uid_col,
        "expected_jsons",
        "collected_jsons",
        "total_reviews",
        "reviews_present",
        "coverage_pct",
    ]].copy()

    if uid_col != "unique_identifier":
        out = out.rename(columns={uid_col: "unique_identifier"})

    return out.sort_values("coverage_pct")


def fetch_latest_next_keys(place_ids: pd.Series) -> pd.DataFrame:
    """Fetch the most recent non-null next_key for each place_id."""
    ids = pd.Series(place_ids).dropna().astype(str).unique().tolist()
    if not ids:
        return pd.DataFrame(columns=["place_id", "next_key"])

    query = text(
        """
        WITH ranked_next_keys AS (
            SELECT
                unique_identifier AS place_id,
                next_key,
                ROW_NUMBER() OVER (
                    PARTITION BY unique_identifier
                    ORDER BY retrieved_time DESC NULLS LAST
                ) AS rn
            FROM reviews_details
            WHERE unique_identifier = ANY(:place_ids)
              AND next_key IS NOT NULL
        )
        SELECT place_id, next_key
        FROM ranked_next_keys
        WHERE rn = 1
        """
    )

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"place_ids": ids})


def add_latest_next_key(below_df: pd.DataFrame) -> pd.DataFrame:
    """Add the latest next_key to the below-threshold output table."""
    if below_df.empty:
        return below_df

    next_keys = fetch_latest_next_keys(below_df["place_id"])
    return below_df.merge(next_keys, on="place_id", how="left")


def report_coverage(coverage: pd.DataFrame):
    """Print a summary coverage report to stdout."""
    total_places = len(coverage)
    no_total      = coverage["total_reviews"].eq(0).sum()
    evaluated     = coverage[coverage["total_reviews"] > 0]
    below         = evaluated["below_threshold"].sum()
    above         = len(evaluated) - below

    print("=" * 55)
    print("COVERAGE REPORT")
    print("=" * 55)
    print(f"  Total places           : {total_places:,}")
    print(f"  Places with no total   : {no_total:,}  (coverage undefined)")
    print(f"  Places evaluated       : {len(evaluated):,}")
    print(f"  Above threshold (>=50%): {above:,}")
    print(f"  Below threshold (<50%) : {int(below):,}")

    if len(evaluated) > 0:
        avg = evaluated["coverage_pct"].mean()
        med = evaluated["coverage_pct"].median()
        print(f"  Average coverage       : {avg:.1%}")
        print(f"  Median  coverage       : {med:.1%}")

    print()

    flagged = coverage[coverage["below_threshold"] == True].copy()
    if not flagged.empty:
        # print(f"Flagged places (below {COVERAGE_THRESHOLD:.0%} coverage):")
        # print(
        #     flagged[["place_id", "total_reviews", "reviews_present", "coverage_pct"]]
        #     .sort_values("coverage_pct")
        #     .to_string(index=False)
        # )
         print(f"{len(flagged):,} places flagged for below-threshold coverage. See below_threshold_places.csv for details.")
    else:
        print("All evaluated places meet the 50% coverage threshold.")
    print()


# ---------------------------------------------------------------------------
# 3. Review length distribution
# ---------------------------------------------------------------------------

def analyse_length_distribution(reviews: pd.DataFrame) -> tuple:
    """
    Compute character length of trans_review, print a distribution summary,
    and return (lengths Series, stats DataFrame) for saving to disk.

    The stats DataFrame has one row per metric, with columns:
        metric : descriptive name
        value  : numeric value
    Covers descriptive statistics, bucket counts/percentages, and the
    needs_summary threshold count and percentage.
    """
    if "trans_review" not in reviews.columns:
        raise ValueError("'trans_review' column not found in reviews DataFrame.")

    lengths = reviews["trans_review"].dropna().str.len()
    n       = len(lengths)

    print("=" * 55)
    print("REVIEW LENGTH DISTRIBUTION  (trans_review, chars)")
    print("=" * 55)
    print(lengths.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).to_string())
    print()

    # Bucket breakdown
    buckets = {
        "0-50 chars"   : (lengths <= 50).sum(),
        "51-100 chars" : ((lengths > 50)  & (lengths <= 100)).sum(),
        "101-250 chars": ((lengths > 100) & (lengths <= 250)).sum(),
        "251-500 chars": ((lengths > 250) & (lengths <= 500)).sum(),
        "500+ chars"   : (lengths > 500).sum(),
    }
    print("Bucket breakdown:")
    for label, count in buckets.items():
        pct = count / n * 100 if n else 0
        print(f"  {label:<16} : {count:>6,}  ({pct:5.1f}%)")

    long_count = (lengths > SUMMARY_CHAR_THRESHOLD).sum()
    print(f"\nReviews > {SUMMARY_CHAR_THRESHOLD} chars (will be summarized): "
          f"{long_count:,} / {n:,} "
          f"({long_count / n * 100:.1f}%)\n")

    # Build a saveable stats DataFrame (section = "review_length")
    desc      = lengths.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
    stat_rows = [{"section": "review_length", "metric": k, "value": round(float(v), 4)} for k, v in desc.items()]

    for label, count in buckets.items():
        pct = count / n * 100 if n else 0
        stat_rows.append({"section": "review_length", "metric": f"bucket_{label}_count", "value": int(count)})
        stat_rows.append({"section": "review_length", "metric": f"bucket_{label}_pct",   "value": round(pct, 2)})

    stat_rows.append({
        "section": "review_length",
        "metric" : f"needs_summary_count (>{SUMMARY_CHAR_THRESHOLD} chars)",
        "value"  : int(long_count),
    })
    stat_rows.append({
        "section": "review_length",
        "metric" : "needs_summary_pct",
        "value"  : round(long_count / n * 100, 2) if n else 0,
    })

    stats_df = pd.DataFrame(stat_rows)

    return lengths, stats_df


# ---------------------------------------------------------------------------
# 3b. Coverage distribution stats
# ---------------------------------------------------------------------------

def coverage_distribution_stats(coverage: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution statistics over the coverage_pct column and
    return them as a DataFrame with section="coverage_distribution",
    ready to be appended to the length_distribution CSV.
    """
    evaluated = coverage[coverage["total_reviews"] > 0].copy()
    pct       = evaluated["coverage_pct"].dropna()
    n         = len(pct)

    if n == 0:
        return pd.DataFrame(columns=["section", "metric", "value"])

    desc = pct.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
    rows = [{"section": "coverage_distribution", "metric": k, "value": round(float(v), 4)}
            for k, v in desc.items()]

    # Bucket breakdown by coverage bands
    cov_buckets = {
        "0-25%"  : ((pct >= 0.00) & (pct <  0.25)).sum(),
        "25-50%"  : ((pct >= 0.25) & (pct <  0.50)).sum(),
        "50-75%"  : ((pct >= 0.50) & (pct <  0.75)).sum(),
        "75-100%" : ((pct >= 0.75) & (pct <= 1.00)).sum(),
        ">100%"   :  (pct >  1.00).sum(),
    }
    for label, count in cov_buckets.items():
        band_pct = count / n * 100 if n else 0
        rows.append({"section": "coverage_distribution", "metric": f"bucket_{label}_count", "value": int(count)})
        rows.append({"section": "coverage_distribution", "metric": f"bucket_{label}_pct",   "value": round(band_pct, 2)})

    # Below / above threshold counts
    below = (pct < COVERAGE_THRESHOLD).sum()
    above = (pct >= COVERAGE_THRESHOLD).sum()
    rows.append({"section": "coverage_distribution", "metric": "below_threshold_count", "value": int(below)})
    rows.append({"section": "coverage_distribution", "metric": "below_threshold_pct",   "value": round(below / n * 100, 2)})
    rows.append({"section": "coverage_distribution", "metric": "above_threshold_count", "value": int(above)})
    rows.append({"section": "coverage_distribution", "metric": "above_threshold_pct",   "value": round(above / n * 100, 2)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Flag reviews for summarization
# ---------------------------------------------------------------------------

def flag_for_summarization(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean 'needs_summary' column.
    True  → trans_review length > SUMMARY_CHAR_THRESHOLD
    False → short review or empty/null
    """
    reviews = reviews.copy()
    reviews["needs_summary"] = (
        reviews["trans_review"]
        .fillna("")
        .str.len()
        .gt(SUMMARY_CHAR_THRESHOLD)
    )
    return reviews


# ---------------------------------------------------------------------------
# 5. Hand off to summarize_reviews.py
# ---------------------------------------------------------------------------

def call_summarizer(reviews: pd.DataFrame, output_dir: str):
    """
    Import and call summarize_reviews() from summarize_reviews.py,
    passing all reviews (flagged) and the output directory.
    """
    try:
        from summarize_reviews import summarize_reviews  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Could not import 'summarize_reviews' from summarize_reviews.py. "
            "Make sure the file exists in the same directory."
        ) from exc

    print("Handing off to summarize_reviews.py …")
    summarize_reviews(reviews, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Review coverage & length analysis pipeline.")
    parser.add_argument("--places",     required=True, help="Path to places_filtered.parquet")
    parser.add_argument("--reviews",    required=True, help="Path to reviews_filtered.parquet")
    parser.add_argument("--output_dir", default="./data", help="Directory for output files")
    parser.add_argument("--skip_summarization", action="store_true",
                        help="If set, skip the summarization step and stop after saving reviews_flagged.parquet")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    places, reviews = load_data(args.places, args.reviews)

    # 2. Coverage check on the full reviews file (all reviews, not just text ones)
    #    total_reviews in the places table counts all reviews regardless of content.
    coverage = check_coverage(places, reviews)
    report_coverage(coverage)

    coverage_path = os.path.join(args.output_dir, "coverage_report.parquet")
    coverage.to_parquet(coverage_path, index=False)
    print(f"Coverage report saved -> {coverage_path}")

    # Save below-threshold places with expected/collected json columns
    below_df = build_below_threshold_csv(coverage)
    below_path = os.path.join(args.output_dir, "below_threshold_places.csv")
    if not below_df.empty:
        below_df = add_latest_next_key(below_df)
        below_df.to_csv(below_path, index=False)
        print(f"Below-threshold places saved -> {below_path}  ({len(below_df):,} places)")
    else:
        print("No below-threshold places — CSV not written.")
    print()

    # 3. Filter to text-only reviews (non-empty, non-whitespace org_review)
    #    All subsequent steps operate on this subset only.
    reviews = filter_text_reviews(reviews)
    reviews = fill_missing_trans_reviews(reviews)

    # 4. Length distribution (based on text reviews only)
    _, length_stats = analyse_length_distribution(reviews)

    # Append coverage distribution stats to the same CSV under a separate section
    cov_stats    = coverage_distribution_stats(coverage)
    combined_stats = pd.concat([length_stats, cov_stats], ignore_index=True)

    length_stats_path = os.path.join(args.output_dir, "length_distribution.csv")
    combined_stats.to_csv(length_stats_path, index=False)
    print(f"Length + coverage distribution saved -> {length_stats_path}\n")

    # 5. Flag reviews for summarization and save output parquet
    #    Output is the text-only subset with 'needs_summary' column added.
    reviews_flagged = flag_for_summarization(reviews)
    reviews_flagged = reviews_flagged.drop_duplicates(subset=["review_id"], keep="first")

    flagged_path = os.path.join(args.output_dir, "text_reviews_flagged.parquet")
    reviews_flagged.to_parquet(flagged_path, index=False)
    print(f"Flagged reviews saved → {flagged_path}\n")

    # 6. Call summarizer (skipped if --skip_summarization is set)
    if args.skip_summarization:
        print("Summarization skipped (--skip_summarization flag set).")
    else:
        call_summarizer(reviews_flagged, output_dir=args.output_dir)

    print("\nDone.")
    print(f"  Text reviews : {len(reviews_flagged):,} rows  →  {flagged_path}")


if __name__ == "__main__":
    main()