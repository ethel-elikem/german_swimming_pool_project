"""Build the directionality pilot sample (attribution / reporting / both).

Draws 100 reviews from the existing 200-row racialised dimension pool
(``data_annotation_validation/reclassed_racialised_dimension.csv``, already
used for approach/context/connotation/themes) so directionality can be
checked against the four existing dimensions and predicted ethnicity without
a fresh sampling round. Writes a BLIND coding sheet (50 reviews, no other
labels) for hand-coding, and a KEY file (all 100, with existing dimension
labels + predicted ethnicity group) held back for later comparison.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path("/home/ethel/Desktop/swimming_pool_project/spp_15052026")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_annotation_validation" / "directionality"
POOL_FILE = BASE_DIR / "data_annotation_validation" / "reclassed_racialised_dimension.csv"
LOOKUP_FILE = BASE_DIR / "analysis_10072026/gmaps_data/reviewer_ethnicity_lookup.parquet"

SUB_DIMENSIONS = {"approach": "approach", "context": "context", "connotation": "connotation", "themes": "theme"}

PILOT_SEED = 2601
CODE_SAMPLE_SEED = 2602

ARAB_LEAGUE_VARIANTS = {
    "Algeria": ["Algeria", "People's Democratic Republic of Algeria"], "Bahrain": ["Bahrain", "Kingdom of Bahrain"],
    "Comoros": ["Comoros", "Union of the Comoros"], "Djibouti": ["Djibouti", "Republic of Djibouti"],
    "Egypt": ["Egypt", "Egypt, Arab Republic of"], "Iraq": ["Iraq", "Republic of Iraq"],
    "Jordan": ["Jordan", "Hashemite Kingdom of Jordan"], "Kuwait": ["Kuwait", "State of Kuwait"],
    "Lebanon": ["Lebanon", "Lebanese Republic"], "Libya": ["Libya", "Libyan Arab Jamahiriya", "State of Libya"],
    "Mauritania": ["Mauritania", "Islamic Republic of Mauritania"], "Morocco": ["Morocco", "Kingdom of Morocco"],
    "Oman": ["Oman", "Sultanate of Oman"],
    "Palestine": ["Palestine, State of", "State of Palestine", "Palestinian Territory, Occupied"],
    "Qatar": ["Qatar", "State of Qatar"], "Saudi Arabia": ["Saudi Arabia", "Kingdom of Saudi Arabia"],
    "Somalia": ["Somalia", "Federal Republic of Somalia"], "Sudan": ["Sudan", "Republic of the Sudan"],
    "Syria": ["Syrian Arab Republic", "Syria"], "Tunisia": ["Tunisia", "Tunisian Republic"],
    "United Arab Emirates": ["United Arab Emirates"], "Yemen": ["Yemen", "Republic of Yemen"],
}


def build_predicted_group5(country_series: pd.Series) -> pd.Series:
    present_values = set(country_series.dropna().unique())
    arab_value_map = {v: c for c, variants in ARAB_LEAGUE_VARIANTS.items() for v in variants if v in present_values}
    arab_mask = country_series.isin(arab_value_map.keys())
    non_arab_mask = ~arab_mask & country_series.notna()
    top_countries = country_series.loc[non_arab_mask].value_counts().head(10).index
    final = pd.Series("Other", index=country_series.index)
    final = final.mask(country_series.isin(top_countries), country_series)
    final = final.mask(arab_mask, "Arab")
    final = final.mask(country_series.isna(), "Unknown")
    return final.map(lambda v: v if v in ("Germany", "Arab", "Türkiye", "Unknown") else "Other")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(POOL_FILE)
    print(f"Pool loaded: {pool.shape}")

    for name, col in SUB_DIMENSIONS.items():
        labels = pd.read_parquet(DATA_DIR / f"reclassed_racialised_{name}.parquet")[["review_id", col]]
        pool = pool.merge(labels, on="review_id", how="left")

    lookup = pd.read_parquet(LOOKUP_FILE)
    lookup["predicted_group5"] = build_predicted_group5(lookup["ethnicity_country_name"])
    pool["reviewer_id"] = pool["reviewer_id"].astype(str)
    lookup["reviewer_id"] = lookup["reviewer_id"].astype(str)
    pool = pool.merge(lookup[["reviewer_id", "predicted_group5"]], on="reviewer_id", how="left")

    pilot_100 = pool.sample(n=100, random_state=PILOT_SEED).reset_index(drop=True)

    key_cols = ["review_id", "org_review", "trans_review"] + list(SUB_DIMENSIONS.values()) + ["predicted_group5"]
    key = pilot_100[key_cols].copy()
    key_path = OUTPUT_DIR / "directionality_pilot_100_key.csv"
    key.to_csv(key_path, index=False)

    code_50 = pilot_100.sample(n=50, random_state=CODE_SAMPLE_SEED)
    blind = code_50[["review_id", "org_review", "trans_review"]].copy()
    blind["directionality"] = ""  # attribution / reporting / both
    blind["notes"] = ""
    blind = blind.sample(frac=1, random_state=CODE_SAMPLE_SEED).reset_index(drop=True)
    blind_path = OUTPUT_DIR / "directionality_hand_code_50.csv"
    blind.to_csv(blind_path, index=False)

    print(f"key_path: {key_path} ({len(key)} rows)")
    print(f"blind_path: {blind_path} ({len(blind)} rows)")
    print(f"remaining {len(pilot_100) - len(code_50)} of the 100 held out (no LLM run yet)")


if __name__ == "__main__":
    main()
