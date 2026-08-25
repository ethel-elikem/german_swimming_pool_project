"""Analyze characteristic language for racialised approach labels."""

from label_text_analysis_utils import DATA_DIR, write_analysis_report


def main() -> None:
    source_file = DATA_DIR / "reclassed_racialised_approach.parquet"
    summary_df, example_df = write_analysis_report(
        dimension="approach",
        source_file=source_file,
        label_column="approach",
        top_terms=15,
        contrast_top_terms=15,
        example_reviews=5,
    )
    print(f"Wrote {len(summary_df)} approach label rows and {len(example_df)} representative examples")


if __name__ == "__main__":
    main()