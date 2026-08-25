"""
analyze_social_classifications.py
----------------------------------
Analyze all reviews to identify likely misclassifications
using heuristic text analysis based on the social classifier system prompt.

Does NOT change classifications — only flags suspicious ones for manual review.

Output:
  - data/misclassification_analysis.parquet/.csv
    Columns: review_id, trans_review, social, confidence_score, 
             likely_misclassified, reason, text_length, word_count, 
             has_people_words, used_org_review
"""

import re
import pandas as pd
import numpy as np

# System prompt definition (from social_classifier_parallelization.py)
SOCIAL_DEFINITION = """
SOCIAL ENCOUNTER = references other people (users/staff): their presence, 
behaviour, appearance, language, or the atmosphere they create.

NON-SOCIAL = only physical environment, facilities, pricing, or reviewer's 
solitary experience.
"""

# Keywords that strongly indicate SOCIAL content
SOCIAL_KEYWORDS = {
    # Staff/service mentions
    "staff": 2.0,
    "employee": 2.0,
    "attendant": 2.0,
    "worker": 2.0,
    "service": 1.8,
    "friendly": 1.8,
    "rude": 1.8,
    "helpful": 1.8,
    "polite": 1.8,
    "professional": 1.5,
    "discourteous": 1.8,
    "courteous": 1.8,
    "kind": 1.5,
    "mean": 1.5,
    
    # People/crowd mentions
    "people": 1.5,
    "crowd": 1.5,
    "crowded": 1.5,
    "busy": 1.2,
    "packed": 1.5,
    "guests": 1.5,
    "children": 1.3,
    "families": 1.3,
    "families": 1.3,
    "kids": 1.3,
    "men": 1.2,
    "women": 1.2,
    "couples": 1.3,
    "atmosphere": 1.5,
    "vibe": 1.5,
    "energy": 1.3,
    
    # Interaction/behavior
    "rude": 1.8,
    "friendly": 1.8,
    "welcoming": 1.8,
    "nice": 1.2,
    "slow": 1.1,
    "fast": 1.0,
    "chatty": 1.5,
    "quiet": 1.0,
    "laughed": 1.3,
    "fun": 1.2,
    "social": 2.0,
}

# Keywords that indicate NON-SOCIAL (physical/facility focus)
NON_SOCIAL_KEYWORDS = {
    "pool": 0.3,
    "water": 0.3,
    "temperature": 0.3,
    "clean": 0.4,
    "dirty": 0.4,
    "facilities": 0.3,
    "size": 0.3,
    "large": 0.3,
    "small": 0.3,
    "deep": 0.3,
    "shallow": 0.3,
    "chlorine": 0.4,
    "chemicals": 0.4,
    "price": 0.4,
    "cost": 0.4,
    "expensive": 0.4,
    "cheap": 0.4,
    "parking": 0.3,
    "location": 0.3,
    "hours": 0.3,
    "open": 0.3,
    "closed": 0.3,
    "maintenance": 0.3,
    "broken": 0.3,
    "working": 0.3,
}

# Generic short reviews often non-social
GENERIC_WORDS = {"good", "bad", "great", "ok", "fine", "nice", "top", "perfect", "awful", "terrible"}


def calculate_social_confidence_score(text: str) -> float:
    """
    Calculate a 0-1 confidence score that this review is truly SOCIAL.
    
    Logic:
    - High score (>0.7): Strong social indicators
    - Medium score (0.4-0.7): Mixed or weak indicators
    - Low score (<0.4): Mostly physical/facility/generic indicators
    
    Returns (score, reason).
    """
    if not text or pd.isna(text):
        return 0.0, "empty_text"
    
    text_lower = text.lower().strip()
    text_len = len(text)
    word_count = len(text.split())
    
    # Feature extraction
    social_score = 0.0
    non_social_score = 0.0
    matches = []
    
    # Check social keywords
    for keyword, weight in SOCIAL_KEYWORDS.items():
        if keyword in text_lower:
            social_score += weight
            matches.append(f"social:{keyword}")
    
    # Check non-social keywords
    for keyword, weight in NON_SOCIAL_KEYWORDS.items():
        if keyword in text_lower:
            non_social_score += weight
            matches.append(f"non_social:{keyword}")
    
    # Check for generic short reviews (often misclassified as social)
    n_generic = sum(1 for g in GENERIC_WORDS if g in text_lower)
    if n_generic > 0 and word_count < 5:
        non_social_score += 1.5
        matches.append(f"generic_short:{n_generic}_words")
    
    # Length feature: very short reviews are rarely social
    if word_count <= 2:
        non_social_score += 1.0
        matches.append("too_short")
    elif word_count <= 5:
        non_social_score += 0.5
    elif word_count >= 15:
        social_score += 0.3  # longer reviews more likely social
        matches.append("longer_review")
    
    # Normalization and final confidence
    total_score = social_score + non_social_score
    if total_score == 0:
        # No strong indicators either way
        # Use length heuristic: short = likely non-social
        if word_count <= 3:
            confidence = 0.2
            reason = "generic_short_review"
        elif word_count <= 10:
            confidence = 0.4
            reason = "short_generic"
        else:
            confidence = 0.5
            reason = "neutral_no_indicators"
    else:
        # Normalize: social_score relative to total
        confidence = social_score / (total_score + 1e-6)
        if confidence > 0.7:
            reason = "strong_social"
        elif confidence > 0.5:
            reason = "mixed_indicators"
        else:
            reason = "non_social_dominant"
    
    return confidence, reason


def analyze_filled_reviews(
    source_parquet: str = "data/social_reviews.parquet",
    classified_parquet: str = "data/social_classified_text_reviews.parquet",
    output_path: str = "data/misclassification_analysis",
) -> dict:
    """
    Load all source reviews (with trans_review filled), get their classifications, and score them.
    
    Focus analysis on identifying which reviews are likely misclassified despite having
    trans_review filled from org_review.
    
    Returns:
    - Parquet and CSV with analysis
    - Summary stats dict
    """
    print("=" * 75)
    print("MISCLASSIFICATION ANALYSIS")
    print("Analyzing all reviews to identify likely misclassifications")
    print("=" * 75)
    print()
    
    # Load source reviews with filled trans_review
    print("Loading source reviews (with trans_review filled)...")
    source = pd.read_parquet(source_parquet, columns=["review_id", "trans_review", "org_review"])
    print(f"  Loaded {len(source):,} rows")
    
    # Load classifications (get unique per review_id, keep latest)
    print("Loading social classifications...")
    classified = pd.read_parquet(
        classified_parquet,
        columns=["review_id", "social"]
    )
    # Keep only unique review_ids (drop_duplicates keeps first, so if there are dupes, we keep one)
    classified = classified.drop_duplicates(subset=["review_id"], keep="first")
    print(f"  Loaded {len(classified):,} unique classified rows")
    
    # Merge on review_id
    print("Merging on review_id...")
    merged = source.merge(
        classified,
        on="review_id",
        how="inner"
    )
    print(f"  Merged: {len(merged):,} rows with both source text and classification")
    print()
    
    # Analyze each review
    print("Analyzing reviews for misclassification patterns...")
    scores = []
    reasons = []
    text_lens = []
    word_counts = []
    has_people = []
    used_org_review = []
    
    for idx, row in merged.iterrows():
        # Prefer trans_review, fall back to org_review
        text = row.get("trans_review") or row.get("org_review") or ""
        used_org = not (row.get("trans_review") and str(row.get("trans_review")).strip())
        
        score, reason = calculate_social_confidence_score(text)
        
        scores.append(score)
        reasons.append(reason)
        text_lens.append(len(text))
        word_counts.append(len(text.split()))
        has_people.append(
            any(w in text.lower() for w in ["staff", "people", "crowd", "service", "friendly"])
        )
        used_org_review.append(used_org)
    
    merged["confidence_score"] = scores
    merged["reason"] = reasons
    merged["text_length"] = text_lens
    merged["word_count"] = word_counts
    merged["has_people_words"] = has_people
    merged["used_org_review"] = used_org_review
    
    # Flag likely misclassifications
    # Hypothesis: reviews marked "no" but with high social confidence are likely misclassified
    merged["likely_misclassified"] = (
        (merged["social"] == "no") & (merged["confidence_score"] > 0.65)
    ) | (
        (merged["social"] == "yes") & (merged["confidence_score"] < 0.25)
    )
    
    print("  Done")
    print()
    
    # Summary stats
    print("=" * 75)
    print("SUMMARY STATISTICS")
    print("=" * 75)
    
    total_rows = len(merged)
    n_social_yes = (merged["social"] == "yes").sum()
    n_social_no = (merged["social"] == "no").sum()
    n_likely_misclass = merged["likely_misclassified"].sum()
    
    print(f"  Total rows analyzed: {total_rows:,}")
    print(f"  Classified as social=YES: {n_social_yes:,} ({100*n_social_yes/total_rows:.1f}%)")
    print(f"  Classified as social=NO: {n_social_no:,} ({100*n_social_no/total_rows:.1f}%)")
    print()
    print(f"  Likely misclassified (flagged): {n_likely_misclass:,} ({100*n_likely_misclass/total_rows:.1f}%)")
    print()
    
    # Breakdown by current classification
    print("MISCLASSIFICATIONS BY CURRENT LABEL:")
    misclass_yes = (
        (merged["social"] == "yes") & (merged["confidence_score"] < 0.25)
    ).sum()
    misclass_no = (
        (merged["social"] == "no") & (merged["confidence_score"] > 0.65)
    ).sum()
    print(f"  Marked YES but likely NO: {misclass_yes:,} ({100*misclass_yes/total_rows:.2f}%)")
    print(f"  Marked NO but likely YES: {misclass_no:,} ({100*misclass_no/total_rows:.2f}%) ← These are your ~40% suspicion")
    print()
    
    # Confidence distribution
    print("CONFIDENCE SCORE DISTRIBUTION:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = merged["confidence_score"].quantile(p / 100)
        print(f"  {p}th percentile: {val:.3f}")
    print()
    
    # Text length stats
    print("TEXT LENGTH DISTRIBUTION:")
    print(f"  Avg characters: {merged['text_length'].mean():.0f}")
    print(f"  Avg words: {merged['word_count'].mean():.0f}")
    print(f"  With 'people' keywords: {merged['has_people_words'].sum():,} ({100*merged['has_people_words'].sum()/total_rows:.1f}%)")
    print()
    
    # Save outputs
    print("=" * 75)
    print("SAVING OUTPUTS")
    print("=" * 75)
    
    # Sort by confidence score and flagged status
    output_df = merged.sort_values(
        by=["likely_misclassified", "confidence_score"],
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Select key columns for output
    output_cols = [
        "review_id",
        "trans_review",
        "social",
        "confidence_score",
        "likely_misclassified",
        "reason",
        "text_length",
        "word_count",
        "has_people_words",
        "used_org_review",
    ]
    output_df_subset = output_df[output_cols]
    
    # Save parquet
    parquet_path = f"{output_path}.parquet"
    output_df_subset.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet: {parquet_path}")
    
    # Save CSV for easy viewing
    csv_path = f"{output_path}.csv"
    output_df_subset.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Save flagged subset (just the likely misclassified)
    flagged = output_df_subset[output_df_subset["likely_misclassified"]]
    flagged_path = f"{output_path}_flagged_for_review"
    flagged.to_parquet(f"{flagged_path}.parquet", index=False)
    flagged.to_csv(f"{flagged_path}.csv", index=False)
    print(f"  Saved flagged subset: {flagged_path}.(parquet|csv)")
    print(f"    Flagged rows: {len(flagged):,}")
    print()
    
    return {
        "total_rows": total_rows,
        "social_yes": n_social_yes,
        "social_no": n_social_no,
        "likely_misclassified": n_likely_misclass,
        "misclass_marked_yes": misclass_yes,
        "misclass_marked_no": misclass_no,
        "output_parquet": parquet_path,
        "output_csv": csv_path,
        "flagged_parquet": f"{flagged_path}.parquet",
        "flagged_csv": f"{flagged_path}.csv",
    }


if __name__ == "__main__":
    result = analyze_filled_reviews()
    
    print("=" * 75)
    print("DONE")
    print("=" * 75)
    print()
    print("Next steps:")
    print("1. Review data/misclassification_analysis_flagged_for_review.csv")
    print("2. Manually validate a sample of the flagged rows")
    print("3. If ~40% of the 'NO' reviews are actually 'YES', you have options:")
    print("   a) Re-classify just the flagged subset (~$5-10)")
    print("   b) Manually correct the highest-confidence misclassified rows")
    print("   c) Use the flagged data as-is with confidence scores in downstream analysis")
    print()
