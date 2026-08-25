"""Generate reproducible validation samples for social and racialised review checks.

This script also builds the stage-3 racialised validation sources by merging each
classifier's ``processed_success.csv`` with ``data/racialised_reviews.parquet``.
The merged files are saved in ``spp_15052026/data`` and then used to generate one
validation CSV per classifier.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/home/ethel/Desktop/swimming_pool_project/spp_15052026")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_annotation_validation"
STAGE3_SUBS_DIR = BASE_DIR / "data_annotation/stage_3_subs"

SOCIAL_DATA_FILE = DATA_DIR / "social_classified_text_reviews.parquet"
RACIALISED_RECALL_DATA_FILE = DATA_DIR / "racialised_classified_text_reviews.parquet"
RACIALISED_PRECISION_DATA_FILE = DATA_DIR / "racialised_reviews.parquet"

RACIALISED_SOURCE_FILE = DATA_DIR / "racialised_reviews.parquet"

SOCIAL_RANDOM_SEED = 42
RACIALISED_RANDOM_SEED = 42
RACIALISED_DIMENSION_RANDOM_SEED = 43

MERGED_CLASSIFICATION_SPECS = [
    {
        "name": "approach",
        "source_file": STAGE3_SUBS_DIR / "approach/processed_success.csv",
        "classification_column": "approach",
        "output_file": DATA_DIR / "racialised_approach_reviews.parquet",
        "sample_output": "approach_validation_sample.csv",
        "single_target": 50,
        "multi_target": 50,
        "seed": 1101,
    },
    {
        "name": "context",
        "source_file": STAGE3_SUBS_DIR / "context/processed_success.csv",
        "classification_column": "context",
        "output_file": DATA_DIR / "racialised_context_reviews.parquet",
        "sample_output": "context_validation_sample.csv",
        "single_target": 50,
        "multi_target": 50,
        "seed": 1102,
    },
    {
        "name": "connotation",
        "source_file": STAGE3_SUBS_DIR / "connotation/processed_success.csv",
        "classification_column": "connotation",
        "output_file": DATA_DIR / "racialised_connotation_reviews.parquet",
        "sample_output": "connotation_validation_sample.csv",
        "single_target": 50,
        "multi_target": 50,
        "seed": 1103,
    },
    {
        "name": "themes",
        "source_file": STAGE3_SUBS_DIR / "themes/processed_success.csv",
        "classification_column": "theme",
        "output_file": DATA_DIR / "racialised_themes_reviews.parquet",
        "sample_output": "themes_validation_sample.csv",
        "single_target": 50,
        "multi_target": 20,
        "seed": 1104,
    },
]


def save_sample(df: pd.DataFrame, sample_size: int, seed: int, output_name: str) -> None:
    sample = df.sample(n=sample_size, random_state=seed)
    output_path = OUTPUT_DIR / output_name
    sample.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Shape: {sample.shape}")


def extract_label_combo(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()

    if isinstance(value, (list, tuple, set)):
        labels = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).strip()
        quoted_labels = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        labels = [left or right for left, right in quoted_labels if (left or right)]

        if not labels and text:
            stripped = text.strip("[]")
            if stripped:
                labels = [part.strip().strip("'\"") for part in stripped.split(",") if part.strip().strip("'\"")]

    return tuple(sorted(dict.fromkeys(labels), key=str.casefold))


def merge_racialised_reviews(source_file: Path, classification_file: Path, classification_column: str, output_file: Path) -> pd.DataFrame:
    source_df = pd.read_parquet(source_file)
    classifier_df = pd.read_csv(classification_file)

    merged_df = classifier_df.merge(
        source_df[["review_id", "org_review", "trans_review"]],
        on="review_id",
        how="inner",
        validate="one_to_one",
    )

    merged_df = merged_df[["review_id", "org_review", "trans_review", classification_column]].copy()
    merged_df.to_parquet(output_file, index=False)
    print(f"Saved merged parquet: {output_file} ({len(merged_df)} rows)")
    return merged_df


def build_merged_racialised_sources() -> dict[str, pd.DataFrame]:
    merged_sources: dict[str, pd.DataFrame] = {}
    for spec in MERGED_CLASSIFICATION_SPECS:
        merged_sources[spec["name"]] = merge_racialised_reviews(
            RACIALISED_SOURCE_FILE,
            spec["source_file"],
            spec["classification_column"],
            spec["output_file"],
        )
    return merged_sources


def sample_by_label_groups(
    df: pd.DataFrame,
    classification_column: str,
    seed: int,
    single_target: int,
    multi_target: int,
) -> pd.DataFrame:
    working_df = df.copy()
    working_df["_label_combo"] = working_df[classification_column].map(extract_label_combo)
    working_df = working_df[working_df["_label_combo"].map(bool)]

    sampled_parts = []
    grouped = working_df.groupby("_label_combo", sort=True)
    for offset, (_, group) in enumerate(grouped):
        label_combo = group.iloc[0]["_label_combo"]
        target_size = single_target if len(label_combo) == 1 else multi_target
        sample_size = min(len(group), target_size)
        sampled_parts.append(group.sample(n=sample_size, random_state=seed + offset))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled_df.drop(columns=["_label_combo"])


def save_validation_sample(
    merged_df: pd.DataFrame,
    classification_column: str,
    output_name: str,
    seed: int,
    single_target: int,
    multi_target: int,
) -> None:
    sample_df = sample_by_label_groups(
        merged_df,
        classification_column=classification_column,
        seed=seed,
        single_target=single_target,
        multi_target=multi_target,
    )
    output_path = OUTPUT_DIR / output_name
    sample_df.to_csv(output_path, index=False)
    print(f"Saved validation sample: {output_path} ({len(sample_df)} rows)")


def main() -> None:
    """Generate validation samples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading social data...")
    social_df = pd.read_parquet(SOCIAL_DATA_FILE)
    print(f"Total social reviews loaded: {len(social_df)}")
    print(f"Columns: {social_df.columns.tolist()}")

    print("\nGenerating social_recall.csv (100 random reviews)...")
    save_sample(social_df, 100, SOCIAL_RANDOM_SEED, "social_recall.csv")

    print("\nGenerating social_precision.csv (50 reviews where social='yes')...")
    social_yes = social_df[social_df["social"] == "yes"]
    print(f"Reviews with social='yes': {len(social_yes)}")
    save_sample(social_yes, 50, SOCIAL_RANDOM_SEED, "social_precision.csv")

    print("\nLoading racialised recall data...")
    racialised_recall_df = pd.read_parquet(RACIALISED_RECALL_DATA_FILE)
    print(f"Total racialised recall reviews loaded: {len(racialised_recall_df)}")
    print(f"Columns: {racialised_recall_df.columns.tolist()}")

    print("\nGenerating racialised_recall.csv (100 random reviews)...")
    save_sample(racialised_recall_df, 100, RACIALISED_RANDOM_SEED, "racialised_recall.csv")

    print("\nLoading racialised precision data...")
    racialised_precision_df = pd.read_parquet(RACIALISED_PRECISION_DATA_FILE)
    print(f"Total racialised precision reviews loaded: {len(racialised_precision_df)}")
    print(f"Columns: {racialised_precision_df.columns.tolist()}")

    print("\nGenerating racialised_precision.csv (50 random reviews)...")
    save_sample(racialised_precision_df, 50, RACIALISED_RANDOM_SEED, "racialised_precision.csv")

    print("Generating a sample of 200 racialised reviews for dimension annotation...")
    save_sample(racialised_precision_df, 200, RACIALISED_DIMENSION_RANDOM_SEED, "racialised_dimension.csv")

    print("\nBuilding merged stage-3 racialised validation sources...")
    merged_sources = build_merged_racialised_sources()

    print("\nGenerating classifier validation samples...")
    for spec in MERGED_CLASSIFICATION_SPECS:
        merged_df = merged_sources[spec["name"]]
        print(f"Sampling {spec['name']} from {spec['output_file'].name} ({len(merged_df)} rows)...")
        save_validation_sample(
            merged_df,
            classification_column=spec["classification_column"],
            output_name=spec["sample_output"],
            seed=spec["seed"],
            single_target=spec["single_target"],
            multi_target=spec["multi_target"],
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
