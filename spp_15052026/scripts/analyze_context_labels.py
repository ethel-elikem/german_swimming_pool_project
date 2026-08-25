"""Analyze characteristic language for racialised context labels."""

from label_text_analysis_utils import DATA_DIR, write_analysis_report


def main() -> None:
    source_file = DATA_DIR / "reclassed_racialised_context.parquet"
    summary_df, example_df = write_analysis_report(
        dimension="context",
        source_file=source_file,
        label_column="context",
        top_terms=15,
        contrast_top_terms=15,
        example_reviews=5,
    )
    print(f"Wrote {len(summary_df)} context label rows and {len(example_df)} representative examples")


if __name__ == "__main__":
    main()