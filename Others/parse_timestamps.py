"""
parse_timestamps.py
───────────────────
Converts Google Maps relative timestamps ("3 years ago", "a month ago" etc.)
into approximate calendar years and period buckets, using the download date
as the reference point.

INPUT CSV columns required:
    review_id       — unique review identifier
    review_time     — relative timestamp string from Google Maps
    retrieved_time  — date the review was downloaded (any parseable date format)

OUTPUT columns:
    review_id
    review_time         — original string (kept for reference)
    retrieved_time      — original download date
    retrieved_date      — parsed download date
    approx_date         — estimated review date (datetime)
    approx_year         — estimated year (int)
    approx_month        — estimated month (int, may be approximate)
    period              — broad period bucket (str)
    days_before_retrieval — estimated days between review and download (int)
    time_confidence     — how precise the estimate is (high/medium/low)

PERIOD BUCKETS (relative to ~2015 Syrian refugee influx as reference):
    pre-2015    — before the main refugee influx period
    2015-2017   — refugee influx and immediate aftermath
    2018-2020   — post-influx settlement period
    2021-2022   — COVID and recovery period
    2023+       — recent

USAGE:
    Adjust INPUT_FILE and OUTPUT_FILE paths, then run:
    python parse_timestamps.py

    Or call parse_reviews_timestamps(df) directly with your dataframe.
"""

import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────
INPUT_FILE  = "your_reviews_with_timestamps.csv"   # ← adjust
OUTPUT_FILE = "reviews_with_years.csv"

# ── PARSING ───────────────────────────────────────────────────────────────
def parse_relative_time(time_str, reference_date):
    """
    Parse a Google Maps relative time string into an approximate datetime.
    
    Handles patterns like:
        "a year ago", "2 years ago", "3 years ago"
        "a month ago", "6 months ago"
        "a week ago", "3 weeks ago"
        "a day ago", "5 days ago"
        "just now", "an hour ago", "5 hours ago"
        "yesterday"
    
    Returns (approx_date, days_before_retrieval, confidence)
    """
    if not isinstance(time_str, str):
        return None, None, "low"
    
    s = time_str.strip().lower()
    
    # ── Years ──────────────────────────────────────────────────────────────
    m = re.match(r'^(\d+)\s+year[s]?\s+ago$', s)
    if m:
        years = int(m.group(1))
        days = years * 365
        return reference_date - timedelta(days=days), days, "medium"
    
    m = re.match(r'^(a|an|one)\s+year\s+ago$', s)
    if m:
        return reference_date - timedelta(days=365), 365, "medium"
    
    # ── Months ─────────────────────────────────────────────────────────────
    m = re.match(r'^(\d+)\s+month[s]?\s+ago$', s)
    if m:
        months = int(m.group(1))
        days = months * 30
        return reference_date - timedelta(days=days), days, "medium"
    
    m = re.match(r'^(a|an|one)\s+month\s+ago$', s)
    if m:
        return reference_date - timedelta(days=30), 30, "medium"
    
    # ── Weeks ──────────────────────────────────────────────────────────────
    m = re.match(r'^(\d+)\s+week[s]?\s+ago$', s)
    if m:
        weeks = int(m.group(1))
        days = weeks * 7
        return reference_date - timedelta(days=days), days, "high"
    
    m = re.match(r'^(a|an|one)\s+week\s+ago$', s)
    if m:
        return reference_date - timedelta(days=7), 7, "high"
    
    # ── Days ───────────────────────────────────────────────────────────────
    m = re.match(r'^(\d+)\s+day[s]?\s+ago$', s)
    if m:
        days = int(m.group(1))
        return reference_date - timedelta(days=days), days, "high"
    
    m = re.match(r'^(a|an|one)\s+day\s+ago$', s)
    if m:
        return reference_date - timedelta(days=1), 1, "high"
    
    if s == 'yesterday':
        return reference_date - timedelta(days=1), 1, "high"
    
    # ── Hours / Just now ───────────────────────────────────────────────────
    m = re.match(r'^(\d+)\s+hour[s]?\s+ago$', s)
    if m:
        return reference_date, 0, "high"
    
    m = re.match(r'^(a|an|one)\s+hour\s+ago$', s)
    if m:
        return reference_date, 0, "high"
    
    if s in ('just now', 'moments ago', 'a moment ago'):
        return reference_date, 0, "high"
    
    # ── German language variants (common in this dataset) ──────────────────
    # "vor X Jahren", "vor einem Jahr", "vor X Monaten", etc.
    m = re.match(r'^vor\s+(\d+)\s+jahr[en]*$', s)
    if m:
        years = int(m.group(1))
        days = years * 365
        return reference_date - timedelta(days=days), days, "medium"
    
    m = re.match(r'^vor\s+einem?\s+jahr$', s)
    if m:
        return reference_date - timedelta(days=365), 365, "medium"
    
    m = re.match(r'^vor\s+(\d+)\s+monat[en]*$', s)
    if m:
        months = int(m.group(1))
        days = months * 30
        return reference_date - timedelta(days=days), days, "medium"
    
    m = re.match(r'^vor\s+einem?\s+monat$', s)
    if m:
        return reference_date - timedelta(days=30), 30, "medium"
    
    m = re.match(r'^vor\s+(\d+)\s+woch[en]*$', s)
    if m:
        weeks = int(m.group(1))
        return reference_date - timedelta(days=weeks*7), weeks*7, "high"
    
    m = re.match(r'^vor\s+einer?\s+woche$', s)
    if m:
        return reference_date - timedelta(days=7), 7, "high"
    
    # ── Fallback ───────────────────────────────────────────────────────────
    return None, None, "low"


def assign_period(year):
    """Assign a meaningful period bucket based on year."""
    if year is None or np.isnan(year):
        return "unknown"
    year = int(year)
    if year < 2015:
        return "pre-2015"
    elif year <= 2017:
        return "2015–2017"
    elif year <= 2019:
        return "2017–2019"
    elif year <= 2022:
        return "2019–2022"
    else:
        return "2022+"


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────
def parse_reviews_timestamps(df):
    """
    df must have columns: review_id, review_time, retrieved_time
    Returns enriched dataframe.
    """
    df = df.copy()
    
    # Parse retrieved_time to datetime
    df['retrieved_date'] = pd.to_datetime(df['retrieved_time'], errors='coerce')
    
    # Apply parsing
    results = df.apply(
        lambda row: parse_relative_time(
            row['review_time'],
            row['retrieved_date'] if pd.notna(row['retrieved_date']) else datetime.now()
        ),
        axis=1
    )
    
    df['approx_date']            = results.apply(lambda x: x[0])
    df['days_before_retrieval']  = results.apply(lambda x: x[1])
    df['time_confidence']        = results.apply(lambda x: x[2])
    
    df['approx_year']  = df['approx_date'].apply(lambda d: d.year  if d is not None else None)
    df['approx_month'] = df['approx_date'].apply(lambda d: d.month if d is not None else None)
    df['period']       = df['approx_year'].apply(assign_period)
    
    return df


# ── SUMMARY STATS ─────────────────────────────────────────────────────────
def summarise(df):
    print(f"\nTotal reviews: {len(df):,}")
    print(f"Successfully parsed: {df['approx_year'].notna().sum():,} ({df['approx_year'].notna().mean()*100:.1f}%)")
    print(f"Unparsed: {df['approx_year'].isna().sum():,}")
    
    print("\nYear distribution:")
    yr = df['approx_year'].value_counts().sort_index()
    for year, count in yr.items():
        print(f"  {int(year)}: {count:,}")
    
    print("\nPeriod distribution:")
    per = df['period'].value_counts()
    for period, count in per.items():
        print(f"  {period}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\nConfidence breakdown:")
    print(df['time_confidence'].value_counts().to_string())
    
    print("\nUnparsed examples (first 10):")
    unparsed = df[df['approx_year'].isna()]['review_time'].dropna().unique()[:10]
    for s in unparsed:
        print(f"  '{s}'")


# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Verify required columns
    required = ['review_id', 'review_time', 'retrieved_time']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
    
    print(f"Loaded {len(df):,} reviews")
    print(f"Sample review_time values: {df['review_time'].dropna().unique()[:8].tolist()}")
    print(f"retrieved_time sample: {df['retrieved_time'].iloc[0]}")
    
    print("\nParsing timestamps...")
    df_out = parse_reviews_timestamps(df)
    
    summarise(df_out)
    
    # Save
    out_cols = ['review_id', 'review_time', 'retrieved_time', 
                'retrieved_date', 'approx_date', 'approx_year', 
                'approx_month', 'period', 'days_before_retrieval', 'time_confidence']
    df_out[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print("Upload this file and I will embed the timeline into the interactive.")