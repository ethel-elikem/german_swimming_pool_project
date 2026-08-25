"""
Build the canonical analysis dataset for the swimming-pool review project.

Single script covering the full transformation chain (previously four
separate scripts — build_analysis_dataset.py, update_master_with_place_data.py,
update_master_with_gmaps_poi_db.py, merge_name_ethnicity_predictions.py —
consolidated into one on 2026-07-13 so every change to the corpus is
documented and reproducible from one place):

    places_filtered.parquet          verified POIs in Germany (key: unique_identifier)
      └─ social_classified_text_reviews.parquet   full corpus, social yes/no
           └─ racial classification (model) + keyword flags
                └─ stage-2 racial RE-classification (corrected yes/no)
                     └─ dimensions on racialised+Germany set (7,768):
                        approach / theme / connotation / context
                     └─ place metadata merge          (places_filtered.parquet, key: unique_identifier)
                     └─ gmaps_poi Postgres timestamp merge  (local DB `gmaps_poi`.`reviews`, key: review_id)
                     └─ name-to-country ethnicity prediction merge  (Prediction Result.zip, key: reviewer_id)
                        + drop of low-use metadata columns

Every input is read from its authoritative source; all cross-checks are
asserted so a silent upstream change fails the build instead of producing a
quietly wrong dataset. No intermediate parquet files are written — each
stage runs in memory and only the final outputs below are written to disk.
Re-run after any future reclassification, place-data refresh, or DB update.

Outputs (written next to this script):
    master_corpus_final.parquet          1,265,806 rows — canonical corpus, all layers
    racialised_reviews_analysis.parquet  7,768 rows — dimension-scope subset, same columns
    reviewer_ethnicity_lookup.parquet    reviewer-id lookup used for the ethnicity merge
    build_report.json                    counts + verification results of this build

Usage:  python build_analysis_dataset.py
"""

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pycountry

ROOT = Path("/home/ethel/Desktop/swimming_pool_project/spp_15052026")
DATA = ROOT / "data"
DIMS = ROOT / "data_annotation" / "stage_3_subs"
OUT = Path(__file__).resolve().parent

SRC = {
    "places": DATA / "places_filtered.parquet",
    "corpus": DATA / "social_classified_text_reviews.parquet",
    # complete model racial pass over all 541,248 social-yes reviews (its 'yes'
    # rows ARE the 11,471 reclassification pool)
    "racial_model": ROOT / "data_annotation" / "stage_2_racial" / "processed_success.csv",
    "reclass": DATA / "racialised_reclassified_text_reviews_11471.parquet",  # corrected yes/no
    "scope": DATA / "racialised_reclassed_reviews.parquet",  # racialised-yes ∩ Germany = dimension scope
    "approach": DIMS / "approach" / "reclass_approach" / "processed_success.csv",
    "theme": DIMS / "themes" / "processed_success.csv",
    "connotation": DIMS / "connotation" / "processed_success.csv",
    "context": DIMS / "context" / "processed_success.csv",
    "prediction_zip": OUT / "Prediction Result.zip",
}

BOOKKEEPING_COLS = [
    "classification_status", "classification_error",
    "failure_batch_id", "failure_root_cause", "raw_response_content",
]

# ── stage: place metadata merge ────────────────────────────────────────────
PLACE_COLUMNS = {
    "retrieved_time": "place_retrieved_time",
    "place_name": "place_name",
    "place_rating": "place_rating",
    "total_reviews": "place_total_reviews",
    "primary_place_type": "primary_place_type",
    "place_subtype": "place_subtype",
    "latitude": "latitude",
    "longitude": "longitude",
    "opening_hours": "opening_hours",
}

# ── stage: gmaps_poi Postgres timestamp merge ──────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "gmaps_poi",
    "user": "ethel",
    "password": "pa55w0rd13",
}

DB_COLUMNS = {
    "review_timestamp": "review_timestamp",
    "review_edited_timestamp": "review_edited_timestamp",
    "reply_time_text": "reply_time_text",
}
# Not merged: retrieved_time/review_time_text from the DB looked like
# duplicates of the corpus's own retrieved_time/review_time at first glance,
# but on 2026-07-13 they matched only ~95-97% of rows (the rest differ by
# days/hours, not just a timezone offset) — a later/different gmaps_poi
# retrieval snapshot, not the same value restated. Dropped rather than kept
# as a second, drifting copy. reply_timestamp/reply_edited_timestamp and
# gmaps_review_created_at/updated_at are also not merged — not used in
# analysis.

# ── stage: name-to-country ethnicity prediction merge ──────────────────────
PREDICTION_MEMBER = "Prediction Result.csv"
RAW_PREDICTION_COLUMN = "demographic"
DROP_OUTPUT_COLUMNS = (
    "ethnicity_prediction_raw",
    "ethnicity_age_prediction",
    "ethnicity_gender_prediction",
)
DROP_CORPUS_COLUMNS = (
    "needs_summary",
    "racialised_source",
    "in_dimension_scope",
    "contributor_link",
    "profile_pic",
    "uploaded_pics_links",
)
COUNTRY_OVERRIDES = {
    "AN": "Netherlands Antilles",
    "XK": "Kosovo",
}


def parse_labels(val):
    """Normalise a label cell to a sorted list of strings.

    Handles the three storage formats found in the processed_success CSVs:
    JSON  '["Explicit", "Coded"]',  numpy repr  "['Crowd' 'Etiquette']",
    and bare strings  'Explicit'.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.startswith("["):
        tokens = re.findall(r"""'([^']+)'|"([^"]+)\"""", s)
        labels = [a or b for a, b in tokens]
        if labels:
            return labels
        inner = s.strip("[]").strip()
        return [t for t in re.split(r"[,\s]+", inner) if t] or None
    return [s]


def load_dimension(name: str, path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "classification_status" in df.columns:
        df = df[df["classification_status"].astype(str).str.upper() == "SUCCESS"]
    df = df.drop_duplicates(subset=["review_id"], keep="last")
    return pd.Series(
        df[name].map(parse_labels).values,
        index=df["review_id"].astype(str),
        name=name,
    )


def build_base_corpus(places: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Stage 1: verified-Germany corpus with social/racialised/dimension layers."""
    uids = set(places["unique_identifier"].dropna().astype(str))
    report["verified_pois"] = len(uids)

    corpus = pd.read_parquet(SRC["corpus"])
    assert corpus["review_id"].duplicated().sum() == 0, "duplicate review_ids in corpus"
    assert corpus["unique_identifier"].astype(str).isin(uids).all(), \
        "corpus contains reviews from POIs outside places_filtered"
    corpus = corpus.drop(columns=[c for c in BOOKKEEPING_COLS if c in corpus.columns])
    report["corpus_rows"] = len(corpus)

    # racialised layer: model pass over ALL social-yes reviews overlaid with the
    # stage-2 corrected pass (authoritative for the 11,471 reclass pool)
    model_df = pd.read_csv(SRC["racial_model"])
    if "classification_status" in model_df.columns:
        model_df = model_df[model_df["classification_status"].astype(str).str.upper() == "SUCCESS"]
    model_df = model_df.drop_duplicates(subset=["review_id"], keep="last")
    model = pd.Series(model_df["racialised"].values, index=model_df["review_id"].astype(str))

    reclass = pd.read_parquet(SRC["reclass"], columns=["review_id", "racialised"])
    reclass = reclass.set_index(reclass["review_id"].astype(str))["racialised"]
    model_yes = set(model[model == "yes"].index)
    assert set(reclass.index) == model_yes, (
        f"reclass pool ({len(reclass)}) != model-yes ({len(model_yes)}); "
        f"diff={len(set(reclass.index) ^ model_yes)}"
    )
    report["racial_model_rows"] = len(model)
    report["reclass_rows"] = len(reclass)

    rid = corpus["review_id"].astype(str)
    racialised = rid.map(model)
    corrected = rid.map(reclass)
    corpus["racialised"] = corrected.fillna(racialised)

    social_yes_mask = corpus["social"].astype(str).str.lower() == "yes"
    unclassified = int((social_yes_mask & corpus["racialised"].isna()).sum())
    assert unclassified == 0, f"{unclassified} social-yes reviews have no racialised label"

    source = pd.Series(pd.NA, index=corpus.index, dtype="object")
    source[racialised.notna()] = "model"
    source[racialised.notna() & corrected.notna()] = "model+reclassified"
    corpus["racialised_source"] = source

    # dimension layers, defined on racialised-yes ∩ Germany (n=7,768)
    scope = pd.read_parquet(SRC["scope"], columns=["review_id"])
    scope_ids = set(scope["review_id"].astype(str))
    report["dimension_scope_rows"] = len(scope_ids)

    in_corpus_yes = set(rid[corpus["racialised"] == "yes"])
    assert in_corpus_yes == scope_ids, (
        f"racialised-yes in corpus ({len(in_corpus_yes)}) != dimension scope ({len(scope_ids)}); "
        f"diff={len(in_corpus_yes ^ scope_ids)}"
    )
    corpus["in_dimension_scope"] = rid.isin(scope_ids)

    for dim in ("approach", "theme", "connotation", "context"):
        labels = load_dimension(dim, SRC[dim])
        covered = len(scope_ids & set(labels.index))
        assert covered == len(scope_ids), f"{dim}: only {covered}/{len(scope_ids)} in-scope reviews labelled"
        mapped = rid.map(labels)
        corpus[dim] = mapped.where(corpus["in_dimension_scope"], None)
        report[f"{dim}_labelled"] = covered

    def counts(s: pd.Series) -> dict:
        return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}

    report["social_counts"] = counts(corpus["social"])
    report["racialised_counts"] = counts(corpus["racialised"])
    report["racialised_source_counts"] = counts(corpus["racialised_source"])
    return corpus


def merge_place_data(corpus: pd.DataFrame, places: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Stage 2: attach place-level metadata (key: unique_identifier)."""
    if corpus["unique_identifier"].isna().any():
        raise ValueError("corpus has missing unique_identifier values")
    if places["unique_identifier"].isna().any():
        raise ValueError("places data has missing unique_identifier values")
    if places["unique_identifier"].duplicated().any():
        duplicates = places.loc[
            places["unique_identifier"].duplicated(keep=False), "unique_identifier"
        ].head(10)
        raise ValueError(
            "places data has duplicate unique_identifier values: "
            + ", ".join(duplicates.astype(str))
        )

    place_columns = ["unique_identifier", *PLACE_COLUMNS]
    missing_columns = [name for name in place_columns if name not in places.columns]
    if missing_columns:
        raise ValueError(f"places data is missing columns: {missing_columns}")

    place_lookup = places[place_columns].rename(columns=PLACE_COLUMNS)
    merged = corpus.merge(place_lookup, on="unique_identifier", how="left", validate="many_to_one")

    unmatched_rows = int(merged["place_retrieved_time"].isna().sum())
    if unmatched_rows:
        raise ValueError(f"{unmatched_rows} review rows did not match place data")

    report["place_rows_not_in_corpus"] = int(
        len(set(places["unique_identifier"]) - set(corpus["unique_identifier"]))
    )
    report["unmatched_review_rows"] = unmatched_rows
    report["place_columns_added"] = list(PLACE_COLUMNS.values())
    report["note"] = (
        "place_id was not used because places_filtered.parquet has duplicate and blank place_id values."
    )
    return merged


def fetch_db_timestamps(review_ids: pd.Series) -> pd.DataFrame:
    unique_review_ids = review_ids.dropna().astype(str).drop_duplicates()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TEMP TABLE master_review_ids (
                        review_id TEXT PRIMARY KEY
                    ) ON COMMIT DROP
                    """
                )
                buffer = io.StringIO()
                unique_review_ids.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                cur.copy_expert(
                    "COPY master_review_ids (review_id) FROM STDIN WITH (FORMAT CSV)",
                    buffer,
                )
                select_columns = ", ".join(
                    f"r.{source} AS {target}" for source, target in DB_COLUMNS.items()
                )
                cur.execute(
                    f"""
                    SELECT r.review_id, {select_columns}
                    FROM reviews AS r
                    JOIN master_review_ids AS m
                      ON r.review_id = m.review_id
                    """
                )
                columns = [description[0] for description in cur.description]
                rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame.from_records(rows, columns=columns)


def merge_gmaps_poi_timestamps(corpus: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Stage 3: attach parsed review/reply timestamps from Postgres `gmaps_poi.reviews` (key: review_id)."""
    if "review_id" not in corpus.columns:
        raise ValueError("corpus has no review_id column")
    if corpus["review_id"].isna().any():
        raise ValueError("corpus has missing review_id values")

    db_timestamps = fetch_db_timestamps(corpus["review_id"])
    if db_timestamps["review_id"].duplicated().any():
        raise ValueError("gmaps_poi returned duplicate review_id rows")

    merged = corpus.merge(db_timestamps, on="review_id", how="left", validate="one_to_one")

    report["db_rows_matched"] = int(len(db_timestamps))
    report["unmatched_db_rows"] = int(len(corpus) - len(db_timestamps))
    report["missing_review_timestamp_rows"] = int(merged["review_timestamp"].isna().sum())
    report["gmaps_poi_columns_added"] = list(DB_COLUMNS.values())
    report["gmaps_poi_non_null_counts"] = {
        column: int(merged[column].notna().sum()) for column in DB_COLUMNS.values()
    }
    return merged


def country_name_from_code(code: "str | None") -> "str | None":
    if code is None or pd.isna(code):
        return None

    normalized = str(code).strip().upper()
    if not normalized:
        return None
    if normalized in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[normalized]

    country = pycountry.countries.get(alpha_2=normalized)
    if country is not None:
        return country.name

    historic = getattr(pycountry, "historic_countries", None)
    if historic is not None:
        historic_country = historic.get(alpha_2=normalized)
        if historic_country is not None:
            return historic_country.name

    return None


def load_predictions(prediction_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(prediction_zip) as archive:
        with archive.open(PREDICTION_MEMBER) as source:
            predictions = pd.read_csv(source)

    predictions = predictions.rename(columns={"name": "reviewer_name"})
    predictions = predictions[
        ~(
            predictions["reviewer_name"].eq("reviewer_name")
            & predictions[RAW_PREDICTION_COLUMN].eq("IN__female")
        )
    ].copy()

    parts = predictions[RAW_PREDICTION_COLUMN].str.split("_", expand=True)
    predictions["predicted_country_code"] = parts[0].str.strip().str.upper()
    predictions.loc[predictions[RAW_PREDICTION_COLUMN].isna(), "predicted_country_code"] = pd.NA
    predictions["predicted_country_name"] = predictions["predicted_country_code"].map(
        country_name_from_code
    )

    return predictions[
        ["reviewer_name", RAW_PREDICTION_COLUMN, "predicted_country_code", "predicted_country_name"]
    ].rename(columns={RAW_PREDICTION_COLUMN: "ethnicity_prediction_raw"})


def most_frequent_name_per_reviewer(master_names: pd.DataFrame) -> pd.DataFrame:
    name_counts = (
        master_names.dropna(subset=["reviewer_id", "reviewer_name"])
        .groupby(["reviewer_id", "reviewer_name"], as_index=False)
        .size()
        .sort_values(["reviewer_id", "size", "reviewer_name"], ascending=[True, False, True])
    )
    return name_counts.drop_duplicates("reviewer_id", keep="first").drop(columns="size")


def merge_ethnicity_predictions(corpus: pd.DataFrame, report: dict) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Stage 4: attach name-to-country ethnicity predictions (key: reviewer_id).

    The prediction file is name-level (`name`, `demographic`), not
    reviewer-id-level. We build a reviewer-id lookup using each reviewer's
    most frequent observed name, attach the prediction to that name, then
    merge the lookup back onto the corpus by reviewer_id. Age and gender
    predictions are intentionally excluded.
    """
    master_names = corpus[["reviewer_id", "reviewer_name"]]
    predictions = load_predictions(SRC["prediction_zip"])

    reviewer_lookup = most_frequent_name_per_reviewer(master_names)
    name_to_reviewer_count = (
        master_names.dropna(subset=["reviewer_id", "reviewer_name"])
        .drop_duplicates(["reviewer_name", "reviewer_id"])
        .groupby("reviewer_name")["reviewer_id"]
        .size()
    )
    reviewer_name_count = (
        master_names.dropna(subset=["reviewer_id", "reviewer_name"])
        .drop_duplicates(["reviewer_id", "reviewer_name"])
        .groupby("reviewer_id")["reviewer_name"]
        .size()
    )

    reviewer_lookup["reviewer_name_prediction_id_count"] = reviewer_lookup["reviewer_name"].map(
        name_to_reviewer_count
    )
    reviewer_lookup["reviewer_id_name_count"] = reviewer_lookup["reviewer_id"].map(reviewer_name_count)

    reviewer_lookup = reviewer_lookup.merge(
        predictions, on="reviewer_name", how="left", validate="many_to_one"
    )
    # reviewer_ethnicity_lookup.parquet keeps the diagnostic name/count columns;
    # the merge onto the main corpus below only carries the two prediction
    # columns, dropped of the "predicted_" prefix.
    reviewer_lookup = reviewer_lookup.rename(
        columns={
            "reviewer_name": "reviewer_name_for_ethnicity_prediction",
            "predicted_country_code": "ethnicity_country_code",
            "predicted_country_name": "ethnicity_country_name",
        }
    )

    output_columns = ["reviewer_id", "ethnicity_country_code", "ethnicity_country_name"]
    merged = corpus.merge(reviewer_lookup[output_columns], on="reviewer_id", how="left", validate="many_to_one")

    unresolved_codes = sorted(
        {
            code
            for code, country in zip(predictions["predicted_country_code"], predictions["predicted_country_name"])
            if pd.notna(code) and pd.isna(country)
        }
    )
    report["join_strategy"] = (
        "Predictions are name-level. A reviewer-id lookup is built using each "
        "reviewer's most frequent observed name, then merged to the corpus on reviewer_id."
    )
    report["prediction_rows"] = int(len(predictions))
    report["prediction_rows_with_country_code"] = int(predictions["predicted_country_code"].notna().sum())
    report["prediction_rows_with_country_name"] = int(predictions["predicted_country_name"].notna().sum())
    report["review_rows_with_country_prediction"] = int(
        merged["ethnicity_country_name"].notna().sum()
    )
    report["reviewer_ids_with_country_prediction"] = int(
        reviewer_lookup["ethnicity_country_name"].notna().sum()
    )
    report["reviewer_names_shared_by_multiple_ids"] = int((name_to_reviewer_count > 1).sum())
    report["review_rows_using_shared_reviewer_name_prediction"] = int(
        master_names["reviewer_name"].map(name_to_reviewer_count).gt(1).sum()
    )
    report["reviewer_ids_with_multiple_observed_names"] = int((reviewer_name_count > 1).sum())
    report["unresolved_country_codes"] = unresolved_codes
    report["excluded_prediction_columns"] = list(DROP_OUTPUT_COLUMNS)
    return merged, reviewer_lookup


def main() -> None:
    report = {"built_at": datetime.now().isoformat(), "sources": {k: str(v) for k, v in SRC.items()}}

    places = pd.read_parquet(SRC["places"])

    report["base"] = {}
    corpus = build_base_corpus(places, report["base"])

    report["place_data"] = {}
    corpus = merge_place_data(corpus, places, report["place_data"])

    report["gmaps_poi"] = {}
    corpus = merge_gmaps_poi_timestamps(corpus, report["gmaps_poi"])

    report["ethnicity"] = {}
    corpus, reviewer_lookup = merge_ethnicity_predictions(corpus, report["ethnicity"])

    # capture dimension scope before dropping in_dimension_scope, then drop
    # the same low-use metadata columns from both outputs for consistency
    scope_mask = corpus["in_dimension_scope"]
    missing_drop_columns = [c for c in DROP_CORPUS_COLUMNS if c not in corpus.columns]
    if missing_drop_columns:
        raise ValueError(f"Corpus is missing columns expected for dropping: {missing_drop_columns}")
    final_corpus = corpus.drop(columns=list(DROP_CORPUS_COLUMNS))
    racialised = final_corpus.loc[scope_mask].reset_index(drop=True)

    master_path = OUT / "master_corpus_final.parquet"
    final_corpus.to_parquet(master_path, index=False)

    rac_path = OUT / "racialised_reviews_analysis.parquet"
    racialised.to_parquet(rac_path, index=False)

    lookup_path = OUT / "reviewer_ethnicity_lookup.parquet"
    reviewer_lookup.to_parquet(lookup_path, index=False)

    report["dropped_corpus_columns"] = list(DROP_CORPUS_COLUMNS)
    report["master_rows"] = len(final_corpus)
    report["master_columns"] = len(final_corpus.columns)
    report["racialised_analysis_rows"] = len(racialised)
    with open(OUT / "build_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"master_corpus_final.parquet:          {len(final_corpus):,} rows")
    print(f"racialised_reviews_analysis.parquet:  {len(racialised):,} rows")
    print(f"reviewer_ethnicity_lookup.parquet:    {len(reviewer_lookup):,} rows")
    print("build_report.json written — all assertions passed")


if __name__ == "__main__":
    main()
