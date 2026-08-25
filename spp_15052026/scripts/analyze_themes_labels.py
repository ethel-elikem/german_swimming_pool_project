"""Analyze characteristic language for racialised theme labels."""

from pathlib import Path

from label_text_analysis_utils import DATA_DIR, write_analysis_report


def main() -> None:
    source_file = DATA_DIR / "reclassed_racialised_theme.parquet"
    summary_df, example_df = write_analysis_report(
        dimension="themes",
        source_file=source_file,
        label_column="theme",
        top_terms=15,
        contrast_top_terms=15,
        example_reviews=5,
    )
    print(f"Wrote {len(summary_df)} theme label rows and {len(example_df)} representative examples")


if __name__ == "__main__":
    main()