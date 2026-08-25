"""
parse_timestamps.py  — v3
─────────────────────────
Converts Google Maps relative timestamps into approximate calendar years.

KEY CHANGES FROM v2:
  - FALLBACK_DATE = datetime(2025, 10, 21)  (earliest known scrape date)
  - Handles "Edited X years ago" / "Edited a year ago" etc.
  - Missing retrieved_time (2,444 rows) → uses fallback, flagged with retrieved_is_fallback=True
  - Missing review_time (35 rows) → labelled period="unknown", kept in output
  - Updated period buckets to match Syrian refugee influx narrative

PERIOD BUCKETS:
    pre-2015    — before the main refugee influx
    2015–2017   — refugee influx and immediate aftermath
    2017–2019   — post-influx settlement period
    2019–2021   — COVID and lockdown period
    2021–2023   — post-COVID, removed restrictions
    2023+       — recent

OUTPUT columns:
    review_id, review_time, retrieved_time,
    retrieved_date_used,      ← actual date used (real or fallback)
    retrieved_is_fallback,    ← True if fallback was applied
    approx_date, approx_year, approx_month,
    period, days_before_retrieval, time_confidence
"""

import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE   = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/openday_2026/racialised_reviews_cols.csv"   # ← adjust
OUTPUT_FILE  = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/openday_2026/reviews_with_years.csv"

# Earliest known scrape date — used as fallback for missing retrieved_time
FALLBACK_DATE = datetime(2025, 10, 21)

# ── PARSING ───────────────────────────────────────────────────────────────
def parse_relative_time(time_str, reference_date):
    """
    Parse a Google Maps relative time string into an approximate datetime.
    Handles English, German, and "Edited X ago" variants.
    Returns (approx_date, days_before_retrieval, confidence)
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return None, None, "unknown"

    s = time_str.strip().lower()

    # Strip "edited " prefix — treat the same as a normal timestamp
    s = re.sub(r'^edited\s+', '', s)

    patterns = [
        # ── English years ──────────────────────────────────────────────────
        (r'^(\d+)\s+year[s]?\s+ago$',    lambda m: int(m.group(1)) * 365,  "medium"),
        (r'^(a|an|one)\s+year\s+ago$',   lambda m: 365,                     "medium"),
        # ── English months ─────────────────────────────────────────────────
        (r'^(\d+)\s+month[s]?\s+ago$',   lambda m: int(m.group(1)) * 30,   "medium"),
        (r'^(a|an|one)\s+month\s+ago$',  lambda m: 30,                      "medium"),
        # ── English weeks ──────────────────────────────────────────────────
        (r'^(\d+)\s+week[s]?\s+ago$',    lambda m: int(m.group(1)) * 7,    "high"),
        (r'^(a|an|one)\s+week\s+ago$',   lambda m: 7,                       "high"),
        # ── English days ───────────────────────────────────────────────────
        (r'^(\d+)\s+day[s]?\s+ago$',     lambda m: int(m.group(1)),         "high"),
        (r'^(a|an|one)\s+day\s+ago$',    lambda m: 1,                       "high"),
        (r'^yesterday$',                  lambda m: 1,                       "high"),
        # ── English hours / now ────────────────────────────────────────────
        (r'^(\d+)\s+hour[s]?\s+ago$',    lambda m: 0,                       "high"),
        (r'^(a|an|one)\s+hour\s+ago$',   lambda m: 0,                       "high"),
        (r'^(just\s+now|moments?\s+ago)$', lambda m: 0,                     "high"),
        # ── German years ───────────────────────────────────────────────────
        (r'^vor\s+(\d+)\s+jahr[en]*$',   lambda m: int(m.group(1)) * 365,  "medium"),
        (r'^vor\s+einem?\s+jahr$',        lambda m: 365,                     "medium"),
        # ── German months ──────────────────────────────────────────────────
        (r'^vor\s+(\d+)\s+monat[en]*$',  lambda m: int(m.group(1)) * 30,   "medium"),
        (r'^vor\s+einem?\s+monat$',       lambda m: 30,                      "medium"),
        # ── German weeks ───────────────────────────────────────────────────
        (r'^vor\s+(\d+)\s+woch[en]*$',   lambda m: int(m.group(1)) * 7,    "high"),
        (r'^vor\s+einer?\s+woche$',       lambda m: 7,                       "high"),
        # ── German days ────────────────────────────────────────────────────
        (r'^vor\s+(\d+)\s+tag[en]*$',    lambda m: int(m.group(1)),         "high"),
        (r'^vor\s+einem?\s+tag$',         lambda m: 1,                       "high"),
        (r'^gestern$',                    lambda m: 1,                       "high"),
    ]

    for pattern, days_fn, confidence in patterns:
        m = re.match(pattern, s)
        if m:
            days = days_fn(m)
            approx = reference_date - timedelta(days=days)
            return approx, days, confidence

    return None, None, "low"


def assign_period(year):
    """Assign a meaningful period bucket."""
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return "unknown"
    year = int(year)
    if year < 2015:    return "pre-2015"
    elif year <= 2017: return "2015–2017"
    elif year <= 2019: return "2017–2019"
    elif year <= 2021: return "2019–2021"
    elif year <= 2023: return "2021–2023"
    else:              return "2023+"


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────
def parse_reviews_timestamps(df, fallback_date=FALLBACK_DATE):
    """
    df must have columns: review_id, review_time, retrieved_time
    Returns enriched dataframe.
    """
    df = df.copy()

    # Parse retrieved_time
    df['retrieved_date'] = pd.to_datetime(df['retrieved_time'], errors='coerce')

    # Flag and fill missing retrieved_time with fallback
    df['retrieved_is_fallback'] = df['retrieved_date'].isna()
    df['retrieved_date_used'] = df['retrieved_date'].fillna(
        pd.Timestamp(fallback_date)
    )

    n_fallback = df['retrieved_is_fallback'].sum()
    if n_fallback > 0:
        print(f"  Applied fallback date ({fallback_date.date()}) "
              f"to {n_fallback:,} rows with missing retrieved_time")

    # Parse review_time using retrieved_date_used as reference
    results = df.apply(
        lambda row: parse_relative_time(
            row['review_time'],
            row['retrieved_date_used'].to_pydatetime()
        ),
        axis=1
    )

    df['approx_date']           = results.apply(lambda x: x[0])
    df['days_before_retrieval'] = results.apply(lambda x: x[1])
    df['time_confidence']       = results.apply(lambda x: x[2])

    df['approx_year']  = df['approx_date'].apply(lambda d: d.year  if d else None)
    df['approx_month'] = df['approx_date'].apply(lambda d: d.month if d else None)
    df['period']       = df['approx_year'].apply(assign_period)

    # Rows where both are missing are least reliable
    both_missing = df['retrieved_is_fallback'] & df['review_time'].isna()
    df.loc[both_missing, 'time_confidence'] = 'unknown'

    return df


# ── SUMMARY ────────────────────────────────────────────────────────────────
def summarise(df):
    total = len(df)
    print(f"\n{'─'*52}")
    print(f"Total reviews:              {total:,}")
    print(f"review_time missing:        {df['review_time'].isna().sum():,}")
    print(f"retrieved_time (fallback):  {df['retrieved_is_fallback'].sum():,}")
    print(f"Successfully parsed:        "
          f"{df['approx_year'].notna().sum():,} "
          f"({df['approx_year'].notna().mean()*100:.1f}%)")

    print(f"\nYear distribution:")
    for year, count in df['approx_year'].value_counts().sort_index().items():
        bar = "█" * int(count / total * 40)
        print(f"  {int(year):4d}: {count:4,}  {bar}")

    print(f"\nPeriod distribution:")
    order = ["pre-2015","2015–2017","2017–2019","2019–2021","2021–2023","2023+","unknown"]
    for period in order:
        count = (df['period'] == period).sum()
        if count > 0:
            print(f"  {period:<12}: {count:,} ({count/total*100:.1f}%)")

    print(f"\nConfidence breakdown:")
    for conf, count in df['time_confidence'].value_counts().items():
        print(f"  {conf}: {count:,}")

    # Show impact of fallback on period distribution
    fb  = df[df['retrieved_is_fallback']]
    nfb = df[~df['retrieved_is_fallback']]
    if len(fb) > 0:
        print(f"\nPeriod breakdown — real retrieved_time ({len(nfb):,} reviews):")
        for p in order:
            c = (nfb['period']==p).sum()
            if c: print(f"  {p}: {c:,} ({c/len(nfb)*100:.1f}%)")
        print(f"\nPeriod breakdown — fallback retrieved_time ({len(fb):,} reviews):")
        for p in order:
            c = (fb['period']==p).sum()
            if c: print(f"  {p}: {c:,} ({c/len(fb)*100:.1f}%)")

    unparsed = df[df['time_confidence']=='low']['review_time'].dropna().unique()[:10]
    if len(unparsed):
        print(f"\nUnparsed review_time examples (check for new patterns):")
        for s in unparsed:
            print(f"  '{s}'")


# ── RUN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    required = ['review_id', 'review_time', 'retrieved_time']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}. Available: {df.columns.tolist()}"
        )

    print(f"Loaded: {len(df):,} reviews")
    print(f"review_time non-null:    {df['review_time'].notna().sum():,}")
    print(f"retrieved_time non-null: {df['retrieved_time'].notna().sum():,}")
    print(f"\nTop review_time values:")
    print(df['review_time'].value_counts().head(12).to_string())
    print(f"\nretrieved_time range: "
          f"{df['retrieved_time'].dropna().min()} → "
          f"{df['retrieved_time'].dropna().max()}")
    print(f"\nFallback date: {FALLBACK_DATE.date()}")

    print("\nParsing timestamps...")
    df_out = parse_reviews_timestamps(df)
    summarise(df_out)

    out_cols = [
        'review_id', 'review_time', 'retrieved_time',
        'retrieved_date_used', 'retrieved_is_fallback',
        'approx_date', 'approx_year', 'approx_month',
        'period', 'days_before_retrieval', 'time_confidence'
    ]
    df_out[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved: {OUTPUT_FILE}")
    print("Upload this file and I will add the timeline to poolwatch v5.")