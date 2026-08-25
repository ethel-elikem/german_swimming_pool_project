#!/usr/bin/env python3
"""
Script to extract unique identifiers from Google Maps URLs.
Searches for hex patterns like 0x[hex]:0x[hex] in the URL column
and saves both UID-only and full updated CSV outputs.
"""

import pandas as pd
import re

# Define input and output files
INPUT_FILE = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/data/pool_selection/all_manual_search_baederleben_pools_24042026.csv'
OUTPUT_UID_FILE = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/data/pool_selection/all_manual_search_baederleben_pools_24042026.csv_uids.csv'
OUTPUT_FULL_FILE = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/data/pool_selection/all_manual_search_baederleben_pools_24042026.csv'

# Define regex patterns
# Strict pattern excludes 0x0:0x
strict_pattern = r'(?!0x0:0x)0x[0-9a-f]+:0x[0-9a-f]+'
# Relaxed pattern includes all
relaxed_pattern = r'0x[0-9a-f]+:0x[0-9a-f]+'

def extract_uid(url, use_strict=False):
    """
    Extract unique identifier from Google Maps URL.
    
    Args:
        url: The URL string to search
        use_strict: If True, use strict pattern (excludes 0x0:0x)
    
    Returns:
        The extracted UID or None if not found
    """
    if pd.isna(url):
        return None
    
    pattern = strict_pattern if use_strict else relaxed_pattern
    match = re.search(pattern, str(url))
    
    if match:
        return match.group(0)
    return None


def main():
    """Main function to process the CSV file."""
    # Read the input CSV
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Display available columns
    print(f"Columns in file: {list(df.columns)}")
    
    # Check if 'url' column exists (case-insensitive)
    url_column = None
    for col in df.columns:
        if col.lower() == 'url':
            url_column = col
            break
    
    if url_column is None:
        print("ERROR: No 'url' column found in the CSV file.")
        print("Please specify the correct column name.")
        return
    
    print(f"\nUsing column: '{url_column}'")
    print(f"Total rows: {len(df)}")
    
    # Try strict pattern first
    print("\nExtracting UIDs using strict pattern (excludes 0x0:0x)...")
    df['unique_identifier'] = df[url_column].apply(lambda x: extract_uid(x, use_strict=True))
    
    # For rows where strict pattern didn't find anything, try relaxed pattern
    mask = df['unique_identifier'].isna()
    print(f"Strict pattern found {(~mask).sum()} UIDs")
    
    print("Applying relaxed pattern for remaining rows...")
    df.loc[mask, 'unique_identifier'] = df.loc[mask, url_column].apply(lambda x: extract_uid(x, use_strict=False))
    
    # Count results
    total_found = df['unique_identifier'].notna().sum()
    total_missing = df['unique_identifier'].isna().sum()
    
    print(f"\nResults:")
    print(f"  UIDs extracted: {total_found}")
    print(f"  URLs without UID: {total_missing}")
    
    # Save UID-only output
    print(f"\nSaving UID-only results to {OUTPUT_UID_FILE}...")
    df[['unique_identifier']].to_csv(OUTPUT_UID_FILE, index=False)

    # Save full input data with populated unique_identifier column
    print(f"Saving full updated file to {OUTPUT_FULL_FILE}...")
    df.to_csv(OUTPUT_FULL_FILE, index=False)
    print("Done!")
    
    # Display sample results
    print("\nSample of extracted UIDs:")
    print(df[df['unique_identifier'].notna()][['unique_identifier']].head(10))


if __name__ == '__main__':
    main()
