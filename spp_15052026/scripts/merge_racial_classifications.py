"""
Merge social_reviews with racial classifications to create:
1. racialised_classified_text_reviews.parquet - all reviews with racial classification data
2. racialised_reviews.parquet - only reviews classified as racialised='yes'

Note: Over 100k social reviews are not yet classified, so the merged file will be smaller.
"""

import pandas as pd
import os
import glob

# Configuration
SOCIAL_REVIEWS_FILE = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data/social_reviews.parquet"
CLASSIFICATION_CHUNKS_DIR = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data_annotation/stage_2_racial/classification_chunks"
OUTPUT_DIR = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data"

def main():
    """Merge and create racialised datasets."""
    print("Loading social_reviews.parquet...")
    social_df = pd.read_parquet(SOCIAL_REVIEWS_FILE)
    print(f"Social reviews loaded: {len(social_df)} rows, {len(social_df.columns)} columns")
    
    # Load all classification chunks
    print(f"\nLoading classification chunks from {CLASSIFICATION_CHUNKS_DIR}...")
    chunk_files = sorted(glob.glob(os.path.join(CLASSIFICATION_CHUNKS_DIR, "classified_chunk_*.parquet")))
    print(f"Found {len(chunk_files)} chunk files")
    
    # Concatenate all chunks
    print("Concatenating chunks...")
    dfs = []
    for chunk_file in chunk_files:
        chunk_df = pd.read_parquet(chunk_file)
        dfs.append(chunk_df)
        if len(dfs) % 10 == 0:
            print(f"  Loaded {len(dfs)} chunks...")
    
    classifications_df = pd.concat(dfs, ignore_index=True)
    print(f"Total classified reviews: {len(classifications_df)} rows")
    print(f"Columns: {classifications_df.columns.tolist()}")
    print(f"Unique racialised values: {classifications_df['racialised'].unique().tolist()}")
    
    # Merge on review_id
    print("\nMerging social_reviews with classifications...")
    merged_df = social_df.merge(
        classifications_df,
        on='review_id',
        how='inner'  # Inner join to keep only reviews with classifications
    )
    print(f"Merged dataframe: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Save racialised_classified_text_reviews.parquet
    print("\nSaving racialised_classified_text_reviews.parquet...")
    output1 = os.path.join(OUTPUT_DIR, "racialised_classified_text_reviews.parquet")
    merged_df.to_parquet(output1, index=False)
    print(f"Saved to: {output1}")
    
    # Filter for racialised='yes'
    print("\nFiltering for racialised='yes'...")
    racialised_yes_df = merged_df[merged_df['racialised'] == 'yes']
    print(f"Reviews with racialised='yes': {len(racialised_yes_df)} rows")
    
    # Save racialised_reviews.parquet
    print("Saving racialised_reviews.parquet...")
    output2 = os.path.join(OUTPUT_DIR, "racialised_reviews.parquet")
    racialised_yes_df.to_parquet(output2, index=False)
    print(f"Saved to: {output2}")
    
    print("\nDone!")
    print(f"\nSummary:")
    print(f"  - racialised_classified_text_reviews.parquet: {len(merged_df)} reviews")
    print(f"  - racialised_reviews.parquet: {len(racialised_yes_df)} reviews")

if __name__ == "__main__":
    main()
