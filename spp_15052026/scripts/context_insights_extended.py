"""Extended context insights: n-grams, log-odds, exemplar reviews, and optional word clouds."""

from pathlib import Path
import pandas as pd

from label_text_analysis_utils import (
    DATA_DIR,
    REPORT_DIR,
    tokenize,
    build_ngrams,
    prepare_class_frames,
    representative_reviews,
)

from theme_insights_extended import compute_counts, compute_log_odds, write_per_theme_outputs, try_make_wordclouds


def main():
    parquet = DATA_DIR / "reclassed_racialised_context.parquet"
    frame = pd.read_parquet(parquet, columns=["review_id", "org_review", "trans_review", "context"]) 
    class_frames = prepare_class_frames(frame, "context")
    class_counts, background_counts = compute_counts(class_frames, max_n=3)
    logodds = compute_log_odds(class_counts, background_counts, alpha=0.01)
    # write outputs using reused writer (it expects single-label combos names but will write csvs for combos)
    write_per_theme_outputs("context", class_frames, logodds, top_k=50)
    try_make_wordclouds(class_frames, logodds)
    print("Context insights written to:", REPORT_DIR)


if __name__ == "__main__":
    main()
