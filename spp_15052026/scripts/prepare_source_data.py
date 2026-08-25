"""
prepare_source_data.py
---------------------
Preprocessing step for the classification pipeline.

Ensures that the source data (data/social_reviews.parquet) is clean and ready
for downstream classifiers and summarizers. Specifically:
  - Fills empty trans_review fields from org_review where available
  - Validates that all rows have at least one non-empty review text
  - Logs statistics for audit

This script should be run ONCE before any classification pipeline starts.
It mutates data/social_reviews.parquet in place to ensure all downstream
processes consume clean data with no empty trans_review fields.

Usage:
    python scripts/prepare_source_data.py
    # or
    python scripts/prepare_source_data.py --input data/social_reviews.parquet --output data/social_reviews.parquet
"""

import argparse
import os
import sys

import pandas as pd


def prepare_source_data(
    input_path: str = "data/social_reviews.parquet",
    output_path: str = None,
) -> dict:
    """
    Load source reviews parquet, fill empty trans_review from org_review, validate, and save.

    Parameters
    ----------
    input_path : str
        Path to source reviews parquet (default: data/social_reviews.parquet)
    output_path : str, optional
        Path to save cleaned parquet (default: same as input_path, mutating in place)

    Returns
    -------
    dict
        Summary statistics: {
            'total_rows': int,
            'filled_rows': int,
            'rows_with_no_text': int,
            'sample_filled_ids': list,
        }
    """
    if output_path is None:
        output_path = input_path

    print("=" * 70)
    print("PREPARE SOURCE DATA")
    print("=" * 70)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load source data
    print("  Loading source data...")
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    print(f"    Total rows: {total_rows:,}")

    # Validate required columns
    required_cols = ["review_id", "org_review", "trans_review"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Count empty trans_review rows (org_review present but trans_review empty/null)
    mask_trans_empty = (
        (df["org_review"].notna()) & 
        (df["org_review"].astype(str).str.strip().ne("")) &
        ((df["trans_review"].isna()) | (df["trans_review"].astype(str).str.strip().eq("")))
    )
    n_empty_trans = mask_trans_empty.sum()

    if n_empty_trans == 0:
        print(f"  ✓ No empty trans_review fields found.")
        print()
        return {
            'total_rows': total_rows,
            'filled_rows': 0,
            'rows_with_no_text': 0,
            'sample_filled_ids': [],
        }

    # Fill empty trans_review from org_review
    print(f"  Filling empty trans_review fields...")
    print(f"    Rows to fill: {n_empty_trans:,}")
    
    filled_ids = df[mask_trans_empty]["review_id"].tolist()
    sample_ids = filled_ids[:5]
    
    df.loc[mask_trans_empty, "trans_review"] = df.loc[mask_trans_empty, "org_review"]
    print(f"    ✓ Filled {n_empty_trans:,} rows")
    print(f"    Sample IDs filled: {sample_ids[:3]}")

    # Validate: ensure no remaining rows with both org_review and trans_review empty
    mask_both_empty = (
        ((df["org_review"].isna()) | (df["org_review"].astype(str).str.strip().eq(""))) &
        ((df["trans_review"].isna()) | (df["trans_review"].astype(str).str.strip().eq("")))
    )
    n_both_empty = mask_both_empty.sum()

    if n_both_empty > 0:
        print(f"  [WARN] {n_both_empty:,} rows have both org_review and trans_review empty")
        print(f"         These rows will be skipped by classifiers and should be investigated.")
    else:
        print(f"  ✓ All rows have at least one non-empty review text field")

    # Save cleaned data
    print(f"  Saving cleaned data...")
    df.to_parquet(output_path, index=False)
    print(f"    Saved to: {output_path}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total rows processed    : {total_rows:,}")
    print(f"  Rows with trans filled  : {n_empty_trans:,}")
    print(f"  Rows with no text       : {n_both_empty:,}")
    print(f"  Status                  : {'✓ CLEAN' if n_both_empty == 0 else '⚠ WARNINGS'}")
    print()

    return {
        'total_rows': total_rows,
        'filled_rows': n_empty_trans,
        'rows_with_no_text': n_both_empty,
        'sample_filled_ids': sample_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess source data for classification pipeline."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/social_reviews.parquet",
        help="Path to source reviews parquet (default: data/social_reviews.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save cleaned parquet (default: same as input, mutating in place)",
    )
    args = parser.parse_args()

    try:
        result = prepare_source_data(input_path=args.input, output_path=args.output)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
