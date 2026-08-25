"""
Task D: build ethnicity-validation sampling deliverables.

Reads reviewer_ethnicity_lookup.parquet (read-only), replicates the
build_ethnicity_final / to_group5 classification used in
supervisor_run_2026-08-16.ipynb cell 5 (and descriptive_analysis.ipynb) at
reviewer-id level, draws a precision sample (100 per predicted Germany/Arab/
Türkiye) and a recall sample (300 from the full reviewer pool), tags name
features, and writes three output files: a BLIND coding sheet, a KEY, and a
FEATURES file. Writes nothing back into the project's data files.
"""
import unicodedata  # imported per task spec; classification below uses codepoint ranges directly
import pandas as pd

LOOKUP = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/analysis_10072026/gmaps_data/reviewer_ethnicity_lookup.parquet"
OUT_DIR = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/analysis_10072026/supervisor_run_2026-08-16/outputs/ethnicity_validation"
RUN_DATE = "20260819"

# ---------------------------------------------------------------------------
# Classification logic, copied verbatim from supervisor_run_2026-08-16.ipynb
# cell 5 (jq -r '.cells[5].source|join("")' <notebook>), identical in
# descriptive_analysis.ipynb.
# ---------------------------------------------------------------------------
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


def build_ethnicity_final(country_series, n_top=10):
    present_values = set(country_series.dropna().unique())
    arab_value_map = {v: c for c, variants in ARAB_LEAGUE_VARIANTS.items() for v in variants if v in present_values}
    arab_mask = country_series.isin(arab_value_map.keys())
    non_arab_mask = ~arab_mask & country_series.notna()
    top_countries = country_series.loc[non_arab_mask].value_counts().head(n_top).index
    final = pd.Series("Other", index=country_series.index)
    final = final.mask(country_series.isin(top_countries), country_series)
    final = final.mask(arab_mask, "Arab")
    final = final.mask(country_series.isna(), "Unknown")
    return final


def to_group5(v):
    return v if v in ("Germany", "Arab", "Türkiye", "Unknown") else "Other"


# ---------------------------------------------------------------------------
# D3 name-feature tags
# ---------------------------------------------------------------------------
EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),  # misc symbols/pictographs, emoticons, transport, supplemental symbols, extended-A
    (0x1F1E6, 0x1F1FF),  # regional indicator letters (flag emoji)
    (0x2600, 0x27BF),    # misc symbols & dingbats -- widely rendered as emoji
]
BUSINESS_MARKERS = ["gmbh", "e.v.", "praxis", "studio", "shop", "restaurant"]


def classify_script(name):
    counts = {"Arabic": 0, "Cyrillic": 0, "Latin": 0, "Other": 0}
    for ch in name:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF:
            counts["Arabic"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["Cyrillic"] += 1
        elif cp < 0x2000:
            # ASCII + Latin-1 Supplement + Latin Extended A/B/etc. cover
            # German/Turkish/French/... accented letters; anything alphabetic
            # below U+2000 that isn't Arabic/Cyrillic is treated as Latin.
            counts["Latin"] += 1
        else:
            counts["Other"] += 1
    if sum(counts.values()) == 0:
        return "Other"
    return max(counts, key=counts.get)


def has_emoji(name):
    return any(any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES) for ch in name)


def is_initials_token(tok):
    core = tok.replace(".", "")
    return bool(core) and len(core) <= 2 and core.isupper()


def tag_features(names: pd.Series) -> pd.DataFrame:
    tokens = names.str.split()
    n_tokens = tokens.map(len)
    return pd.DataFrame({
        "n_tokens": n_tokens,
        "script": names.map(classify_script),
        "has_digits": names.map(lambda s: any(ch.isdigit() for ch in s)),
        "has_emoji": names.map(has_emoji),
        "looks_business": names.str.lower().map(lambda s: any(m in s for m in BUSINESS_MARKERS)),
        "initials_only": tokens.map(lambda toks: bool(toks) and all(is_initials_token(t) for t in toks)),
        "single_token": n_tokens == 1,
    }, index=names.index)


def _self_check():
    assert classify_script("Petra Wachter") == "Latin"
    assert classify_script("محمد أحمد") == "Arabic"
    assert classify_script("Иван Петров") == "Cyrillic"
    assert has_emoji("Petra 😀") is True
    assert has_emoji("Petra Wachter") is False
    assert is_initials_token("J.") is True
    assert is_initials_token("J") is True
    assert is_initials_token("Petra") is False
    feats = tag_features(pd.Series(["J. K.", "GmbH Studio", "Anna"]))
    assert feats.loc[0, "initials_only"] and feats.loc[0, "n_tokens"] == 2
    assert feats.loc[1, "looks_business"]
    assert feats.loc[2, "single_token"]
    assert to_group5("Germany") == "Germany" and to_group5("France") == "Other" and to_group5(None) != "Germany"


def main():
    _self_check()
    lookup = pd.read_parquet(LOOKUP)
    lookup = lookup.rename(columns={"reviewer_name_for_ethnicity_prediction": "reviewer_name"})
    lookup = lookup[lookup["reviewer_name"].notna()].copy()

    lookup["predicted_group5"] = build_ethnicity_final(lookup["ethnicity_country_name"]).map(to_group5)

    # --- precision sample: 100 per predicted Germany/Arab/Türkiye ----------
    precision_parts = []
    for grp in ("Germany", "Arab", "Türkiye"):
        pool = lookup[lookup["predicted_group5"] == grp]
        precision_parts.append(pool.sample(n=100, random_state=42))
    precision = pd.concat(precision_parts)
    precision_ids = set(precision["reviewer_id"])

    # --- recall sample: 300 from the full reviewer pool ---------------------
    recall = lookup.sample(n=300, random_state=42)
    recall_ids = set(recall["reviewer_id"])

    overlap_ids = precision_ids & recall_ids

    def source_for(rid):
        in_p, in_r = rid in precision_ids, rid in recall_ids
        if in_p and in_r:
            return "both"
        return "precision" if in_p else "recall"

    combined_ids = precision_ids | recall_ids
    combined = lookup[lookup["reviewer_id"].isin(combined_ids)].drop_duplicates("reviewer_id").copy()
    combined["sample_source"] = combined["reviewer_id"].map(source_for)

    features = tag_features(combined["reviewer_name"])
    combined_features = pd.concat([combined[["reviewer_id", "predicted_group5", "sample_source"]], features], axis=1)

    # --- KEY file -------------------------------------------------------
    key = combined[["reviewer_id", "reviewer_name", "predicted_group5", "sample_source"]].copy()
    # No probability/confidence column: the prediction pipeline (Prediction
    # Result.zip -> "Prediction Result.csv", columns [name, demographic] where
    # demographic is "COUNTRYCODE_AGERANGE_GENDER", e.g. "DE_55-64_female")
    # carries no numeric confidence score anywhere -- confirmed by inspecting
    # the zip contents directly and build_analysis_dataset.py's merge, which
    # only ever splits/renames that string. Omitted here for that reason.
    key_path = f"{OUT_DIR}/ethnicity_validation_key_{RUN_DATE}.csv"
    key.to_csv(key_path, index=False)

    # --- FEATURES file ----------------------------------------------------
    feat_path = f"{OUT_DIR}/ethnicity_validation_features_{RUN_DATE}.csv"
    combined_features.to_csv(feat_path, index=False)

    # --- BLIND file (no prediction/source/feature columns), shuffled -------
    blind = combined[["reviewer_id", "reviewer_name"]].rename(columns={"reviewer_name": "username"}).copy()
    blind["manual_label"] = ""
    blind["coder_confidence"] = ""
    blind["notes"] = ""
    blind = blind.sample(frac=1, random_state=42).reset_index(drop=True)

    blind_path = f"{OUT_DIR}/ethnicity_validation_blind_{RUN_DATE}.csv"
    import os
    if os.path.exists(blind_path):
        blind_path = f"{OUT_DIR}/ethnicity_validation_blind_{RUN_DATE}_v2.csv"
    blind.to_csv(blind_path, index=False)

    print("blind_path:", blind_path, len(blind))
    print("key_path:", key_path, len(key))
    print("feat_path:", feat_path, len(combined_features))
    print("precision_rows:", len(precision), "recall_rows:", len(recall))
    print("overlap (precision & recall) count:", len(overlap_ids))
    print("combined unique reviewers:", len(combined))
    print("sample_source counts:\n", combined["sample_source"].value_counts())
    print("blind columns:", list(blind.columns))


if __name__ == "__main__":
    main()
