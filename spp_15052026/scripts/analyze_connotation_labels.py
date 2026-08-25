"""Analyze characteristic language for racialised connotation labels."""

from label_text_analysis_utils import DATA_DIR, write_analysis_report


def main() -> None:
    source_file = DATA_DIR / "reclassed_racialised_connotation.parquet"
    summary_df, example_df = write_analysis_report(
        dimension="connotation",
        source_file=source_file,
        label_column="connotation",
        top_terms=15,
        contrast_top_terms=15,
        example_reviews=5,
    )
    print(f"Wrote {len(summary_df)} connotation label rows and {len(example_df)} representative examples")


if __name__ == "__main__":
    main()