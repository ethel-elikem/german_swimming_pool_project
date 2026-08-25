"""
test_sample_50.py
-----------------
Sample 50 reviews from reclassed_racialised_approach.parquet that are not
already in approach_validation_recalss3.csv, run the approach classifier on
them, and compare predicted labels to the existing parquet labels.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import random

import pandas as pd
from dotenv import load_dotenv

# ── Load API key from the project .env ────────────────────────────────────────
_env_path = Path(__file__).resolve().parents[3] / "notebooks" / ".env"
load_dotenv(_env_path)

# ── Import classifier helpers from sibling script ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from approach_classifier import (
    classify_batch,
    CostTracker,
    ApiKeyTracker,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).resolve().parents[3] / "data"
PARQUET_PATH  = DATA_DIR / "reclassed_racialised_approach.parquet"
CSV_PATH      = DATA_DIR / "approach_validation_recalss3.csv"
OUTPUT_PATH   = Path(__file__).parent / "test_sample_50_results.csv"

SAMPLE_N      = 50
RANDOM_SEED   = 42
BATCH_SIZE    = 5   # small batches for a quick test run


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    parquet = pd.read_parquet(PARQUET_PATH)
    parquet["review_id"] = parquet["review_id"].astype(str)

    csv_df = pd.read_csv(CSV_PATH)
    csv_ids = set(csv_df["review_id"].astype(str))

    # ── Exclude reviews already in the validation CSV ─────────────────────────
    pool = parquet[~parquet["review_id"].isin(csv_ids)].copy()
    print(f"Pool after exclusion: {len(pool):,} reviews")

    # ── Sample 50 reviews with text ───────────────────────────────────────────
    pool_with_text = pool[pool["trans_review"].notna() & (pool["trans_review"].str.strip() != "")]
    sample = pool_with_text.sample(n=SAMPLE_N, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled {len(sample)} reviews for classification")

    # ── Set up tracker and API key ────────────────────────────────────────────
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("No OPENROUTER_API_KEY found. Check your .env file.")

    tracker     = CostTracker(budget_limit=5.0)
    key_tracker = ApiKeyTracker([api_key])

    # ── Run classifier in batches ─────────────────────────────────────────────
    batch_data  = list(zip(sample["review_id"], sample["trans_review"]))
    all_success = []
    all_failed  = []

    batches = [batch_data[i : i + BATCH_SIZE] for i in range(0, len(batch_data), BATCH_SIZE)]
    for idx, batch in enumerate(batches):
        print(f"  Batch {idx + 1}/{len(batches)} ({len(batch)} reviews)…", flush=True)
        result = classify_batch(
            batch_data   = batch,
            batch_id     = f"test_b{idx}",
            api_key      = api_key,
            tracker      = tracker,
            key_tracker  = key_tracker,
        )
        all_success.extend(result["success"])
        all_failed.extend(result["failed"])

    print(f"\nClassified: {len(all_success)} success, {len(all_failed)} failed")

    # ── Build results dataframe ───────────────────────────────────────────────
    pred_map = {r["review_id"]: r["approach"] for r in all_success}

    results = sample[["review_id", "org_review", "trans_review", "approach"]].copy()
    results = results.rename(columns={"approach": "approach_existing"})
    results["approach_predicted"] = results["review_id"].map(pred_map)
    results["match"] = results.apply(
        lambda row: (
            sorted(row["approach_existing"]) == sorted(row["approach_predicted"])
            if isinstance(row["approach_existing"], list)
               and isinstance(row["approach_predicted"], list)
            else str(row["approach_existing"]) == str(row["approach_predicted"])
        ),
        axis=1,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    classified = results["approach_predicted"].notna()
    match_rate = results.loc[classified, "match"].mean()
    print(f"\nMatch rate (classified reviews): {match_rate:.1%}")
    print(f"Cost so far: ${tracker.summary()['total_cost']:.4f}")

    print("\n── Label distribution (predicted) ──")
    flat_pred = [lbl for preds in results["approach_predicted"].dropna() for lbl in preds]
    print(pd.Series(flat_pred).value_counts().to_string())

    print("\n── Mismatches ──")
    mismatches = results[classified & ~results["match"]][
        ["review_id", "approach_existing", "approach_predicted", "trans_review"]
    ]
    print(f"{len(mismatches)} mismatches out of {classified.sum()} classified")
    for _, row in mismatches.iterrows():
        print(f"\n  review_id : {row['review_id']}")
        print(f"  existing  : {row['approach_existing']}")
        print(f"  predicted : {row['approach_predicted']}")
        print(f"  text      : {str(row['trans_review'])[:200]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
