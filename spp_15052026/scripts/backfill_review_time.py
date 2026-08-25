"""Backfill missing review_time values from the available review parquet sources.

This script does two things in order:
1. Updates the Postgres `reviews` table from the review parquet sources.
2. Rewrites review-level parquet files in `spp_15052026/data` so their
   `review_time` columns are populated where possible.

The lookup key is `review_id`.
"""

from __future__ import annotations

import argparse
import io
import glob
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg2


DEFAULT_SOURCE = Path(
    "/home/ethel/Desktop/swimming_pool_project/german_pools_data_09042026/germany/germany_reviews.parquet"
)
DEFAULT_EXTRA_SOURCE_DIR = Path(
    "/home/ethel/Desktop/swimming_pool_project/more_reviews_json"
)
DEFAULT_DATA_DIR = Path(
    "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data"
)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "swimming_pools_db",
    "user": "ethel",
    "password": "pa55w0rd13",
}


def load_lookup(source_path: Path) -> pd.DataFrame:
    """Load review_id -> review_time pairs, keeping only populated values."""
    frame = pd.read_parquet(source_path, columns=["review_id", "review_time"])
    return frame


def iter_source_paths(source_path: Path, extra_source_dir: Path | None = None) -> list[Path]:
    """Collect parquet sources that may contain usable review_time values."""
    paths = [source_path]
    if extra_source_dir is not None and extra_source_dir.exists():
        paths.extend(sorted(extra_source_dir.glob("*_reviews.parquet")))
    return paths


def build_lookup(source_paths: list[Path]) -> pd.DataFrame:
    """Combine review_id -> review_time rows from multiple parquet sources."""
    frames = []
    for path in source_paths:
        if not path.exists():
            continue
        try:
            frame = load_lookup(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if "review_id" not in frame.columns or "review_time" not in frame.columns:
            continue
        frame = frame.dropna(subset=["review_id"])
        frame = frame[frame["review_time"].notna()].copy()
        if frame.empty:
            continue
        frame["review_id"] = frame["review_id"].astype(str)
        frame["review_time"] = frame["review_time"].astype(str)
        frames.append(frame[["review_id", "review_time"]])

    if not frames:
        return pd.DataFrame(columns=["review_id", "review_time"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["review_id"], keep="first")
    return combined


def update_postgres_review_time(lookup: pd.DataFrame, dry_run: bool = False) -> int:
    """Backfill the Postgres reviews table from the lookup frame."""
    if lookup.empty:
        return 0

    if dry_run:
        return len(lookup)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TEMP TABLE review_time_lookup (
                        review_id TEXT PRIMARY KEY,
                        review_time TEXT
                    ) ON COMMIT DROP
                    """
                )

                buffer = io.StringIO()
                lookup.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                cur.copy_expert(
                    "COPY review_time_lookup (review_id, review_time) FROM STDIN WITH (FORMAT CSV)",
                    buffer,
                )

                cur.execute(
                    """
                    UPDATE reviews AS r
                    SET review_time = l.review_time
                    FROM review_time_lookup AS l
                    WHERE r.review_id = l.review_id
                      AND (r.review_time IS NULL OR btrim(r.review_time) = '')
                      AND l.review_time IS NOT NULL
                      AND btrim(l.review_time) <> ''
                    """
                )
                return cur.rowcount
    finally:
        conn.close()


def review_parquet_files(data_dir: Path) -> Iterable[Path]:
    """Yield review-level parquet files in the target data directory."""
    for path in sorted(data_dir.glob("*.parquet")):
        if path.name == "places_filtered.parquet":
            continue
        yield path


def backfill_parquet_file(path: Path, lookup: pd.Series, dry_run: bool = False) -> dict:
    """Add or fill review_time in one parquet file using review_id."""
    frame = pd.read_parquet(path)
    if "review_id" not in frame.columns:
        return {"file": path.name, "skipped": True, "reason": "no review_id"}

    if "review_time" in frame.columns:
        existing_non_null = int(frame["review_time"].notna().sum())
        missing_mask = frame["review_time"].isna() | frame["review_time"].astype(str).str.strip().eq("")
    else:
        existing_non_null = 0
        missing_mask = pd.Series(True, index=frame.index)
        frame["review_time"] = pd.NA

    filled_values = frame["review_id"].astype(str).map(lookup)
    fill_mask = missing_mask & filled_values.notna()
    filled_rows = int(fill_mask.sum())

    if filled_rows and not dry_run:
        frame.loc[fill_mask, "review_time"] = filled_values.loc[fill_mask]
        frame.to_parquet(path, index=False)

    return {
        "file": path.name,
        "skipped": False,
        "rows": len(frame),
        "existing_non_null": existing_non_null,
        "filled_rows": filled_rows,
        "now_non_null": int((~(frame["review_time"].isna() | frame["review_time"].astype(str).str.strip().eq(""))).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill review_time in Postgres and parquet files.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Master reviews parquet")
    parser.add_argument("--extra-source-dir", type=Path, default=DEFAULT_EXTRA_SOURCE_DIR, help="Directory containing additional review parquet sources")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing review parquets")
    parser.add_argument("--skip-db", action="store_true", help="Skip Postgres backfill")
    parser.add_argument("--skip-files", action="store_true", help="Skip parquet file backfill")
    parser.add_argument("--dry-run", action="store_true", help="Report intended changes without writing")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source parquet not found: {args.source}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    source_paths = iter_source_paths(args.source, args.extra_source_dir)
    print(f"Loading lookup from {len(source_paths)} source files")
    lookup_df = build_lookup(source_paths)
    lookup_series = lookup_df.set_index("review_id")["review_time"]
    print(f"Lookup rows: {len(lookup_df):,}")

    if not args.skip_db:
        print("Backfilling Postgres reviews table...")
        updated = update_postgres_review_time(lookup_df, dry_run=args.dry_run)
        print(f"Postgres rows updated: {updated:,}")

    if not args.skip_files:
        print(f"Backfilling parquet files in {args.data_dir}...")
        total_filled = 0
        touched = 0
        for path in review_parquet_files(args.data_dir):
            result = backfill_parquet_file(path, lookup_series, dry_run=args.dry_run)
            if result.get("skipped"):
                continue
            touched += 1
            total_filled += result["filled_rows"]
            print(f"{result['file']}: filled {result['filled_rows']:,} rows (now non-null {result['now_non_null']:,} / {result['rows']:,})")
        print(f"Parquet files touched: {touched:,}")
        print(f"Total parquet rows filled: {total_filled:,}")


if __name__ == "__main__":
    main()