"""
get_review_examples.py
──────────────────────
Extracts representative review examples for the poolwatch interactive,
stratified by state × theme × connotation × approach × context.

INPUT FILES  (set paths in the CONFIG block at the bottom)
  racialised_dimensions_spatial_exploded.csv
      review_id, theme, context, connotation, approach,
      state_name, constituency_name, east_west

  racialised_reviews_cols.csv
      review_id, org_review, trans_review, org_review_lang

  reviews_with_years.csv   (output of parse_timestamps.py)
      review_id, period

OUTPUT FILES
  examples_for_interactive.json   ← upload to Claude
  examples_by_state.csv
  examples_full.csv
"""

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
MAX_TEXT_LEN = 400
MIN_TEXT_LEN = 30
CONNOTATIONS = ["negative", "neutral", "positive"]
APPROACHES   = ["explicit", "implicit", "both"]
CONTEXTS     = ["generalised", "incident"]


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def truncate(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    """Trim to the last sentence boundary within max_len characters."""
    if not isinstance(text, str):
        return ""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= max_len:
        return text
    cut        = text[:max_len]
    last_break = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last_break > max_len * 0.5:
        return cut[:last_break + 1] + " […]"
    return cut + " […]"


def clean_text(value) -> str:
    return truncate(str(value or "").strip())


def build_period_lookup(reviews_with_time_df: pd.DataFrame) -> dict:
    """Return {review_id: period} for all rows that have a period value."""
    return (
        reviews_with_time_df
        .dropna(subset=["period"])
        .set_index("review_id")["period"]
        .to_dict()
    )


def pick_example(
    group: pd.DataFrame,
    review_lookup: pd.DataFrame,
    period_lookup: dict,
) -> dict | None:
    """
    Choose one review from *group*, preferring rows that have a translation.
    Returns a dict with org_text, en_text, language, period — or None if the
    texts are too short to be useful.
    """
    translated_ids = set(
        review_lookup.loc[review_lookup["trans_review"].notna(), "review_id"]
    )
    pool      = group[group["review_id"].isin(translated_ids)]
    pool      = pool if len(pool) > 0 else group
    chosen_id = pool.sample(1)["review_id"].iloc[0]

    row = review_lookup.loc[review_lookup["review_id"] == chosen_id]
    if row.empty:
        return None
    row = row.iloc[0]

    org = clean_text(row.get("org_review"))
    en  = clean_text(row.get("trans_review"))
    if len(org) < MIN_TEXT_LEN and len(en) < MIN_TEXT_LEN:
        return None

    return {
        "review_id": chosen_id,
        "org_text":  org,
        "en_text":   en,
        "language":  clean_text(row.get("org_review_lang")),
        "period":    period_lookup.get(chosen_id),
    }


def make_entry(example: dict, connotation: str, count: int = 0) -> dict:
    """Serialisable dict for one example in the JSON output."""
    return {
        "tone":   connotation,
        "en":     example["en_text"],
        "de":     example["org_text"],
        "lang":   example["language"],
        "count":  count,
        "period": example.get("period"),
    }


# ══════════════════════════════════════════════════════════════════════════
#  Per-state builders
# ══════════════════════════════════════════════════════════════════════════

def build_theme_examples(
    state_df: pd.DataFrame,
    review_lookup: pd.DataFrame,
    period_lookup: dict,
    theme_counts: pd.Series,
) -> tuple[dict, list, list]:
    """
    For every theme in this state, pick the best representative example
    (negative > neutral > positive).

    Returns
    -------
    theme_results : {theme: entry_dict}
    rows_state    : flat rows for examples_by_state.csv
    rows_full     : flat rows for examples_full.csv (one per tone)
    """
    theme_results = {}
    rows_state    = []
    rows_full     = []

    for theme, count in theme_counts.items():
        theme_df = (
            state_df[state_df["theme"] == theme]
            .drop_duplicates("review_id")
        )
        best = None

        for connotation in CONNOTATIONS:
            sub = theme_df[theme_df["connotation"] == connotation]
            if sub.empty:
                continue
            example = pick_example(sub, review_lookup, period_lookup)
            if not example:
                continue
            example.update({
                "connotation": connotation,
                "approach":    sub.sample(1)["approach"].iloc[0],
                "context":     sub.sample(1)["context"].iloc[0],
                "theme":       theme,
                "count":       int(count),
            })
            rows_full.append(example)
            if best is None:   # first found wins: negative → neutral → positive
                best = example

        if best:
            theme_results[theme] = make_entry(best, best["connotation"], int(count))
            rows_state.append({
                "theme":       theme,
                "connotation": best["connotation"],
                "org_text":    best["org_text"],
                "en_text":     best["en_text"],
                "language":    best["language"],
                "theme_count": int(count),
                "period":      best.get("period"),
            })

    return theme_results, rows_state, rows_full


def build_overview_examples(
    state_df: pd.DataFrame,
    review_lookup: pd.DataFrame,
    period_lookup: dict,
) -> dict:
    """One example per connotation for the state-level overview."""
    overview = {}
    for connotation in CONNOTATIONS:
        sub = (
            state_df[state_df["connotation"] == connotation]
            .drop_duplicates("review_id")
        )
        if sub.empty:
            continue
        example = pick_example(sub, review_lookup, period_lookup)
        if example:
            overview[connotation] = make_entry(example, connotation)
    return overview


def build_combo_examples(
    state_df: pd.DataFrame,
    review_lookup: pd.DataFrame,
    period_lookup: dict,
) -> dict:
    """One example per approach × context combination."""
    combos = {}
    for approach in APPROACHES:
        for context in CONTEXTS:
            sub = (
                state_df[
                    (state_df["approach"] == approach) &
                    (state_df["context"]  == context)
                ]
                .drop_duplicates("review_id")
            )
            if sub.empty:
                continue
            example = pick_example(sub, review_lookup, period_lookup)
            if example:
                combos[f"{approach}_{context}"] = {
                    **make_entry(example, sub.sample(1)["connotation"].iloc[0]),
                    "approach": approach,
                    "context":  context,
                }
    return combos


# ══════════════════════════════════════════════════════════════════════════
#  Main builder
# ══════════════════════════════════════════════════════════════════════════

def build_examples(
    df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    reviews_with_time_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    period_lookup = build_period_lookup(reviews_with_time_df)
    print(f"Period lookup: {len(period_lookup):,} entries")

    results        = {}
    all_rows_state = []
    all_rows_full  = []

    for state in sorted(df["state_name"].unique()):
        print(f"  {state}…")
        state_df = df[df["state_name"] == state]

        theme_counts = (
            state_df
            .drop_duplicates(subset=["review_id", "theme"])
            .groupby("theme")["review_id"]
            .count()
            .sort_values(ascending=False)
        )

        theme_results, rows_state, rows_full = build_theme_examples(
            state_df, reviews_df, period_lookup, theme_counts
        )

        results[state] = {
            **theme_results,
            "_overview": build_overview_examples(state_df, reviews_df, period_lookup),
            "_combos":   build_combo_examples(state_df, reviews_df, period_lookup),
        }

        for row in rows_state:
            row["state"] = state
        for row in rows_full:
            row["state"] = state

        all_rows_state.extend(rows_state)
        all_rows_full.extend(rows_full)

    # ── Write outputs ─────────────────────────────────────────────────────
    json_path = out_dir / "examples_for_interactive.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {json_path}")

    csv_state = out_dir / "examples_by_state.csv"
    pd.DataFrame(all_rows_state).to_csv(csv_state, index=False, encoding="utf-8")
    print(f"✓ {csv_state}")

    csv_full = out_dir / "examples_full.csv"
    pd.DataFrame(all_rows_full).to_csv(csv_full, index=False, encoding="utf-8")
    print(f"✓ {csv_full}")

    # ── Summary ───────────────────────────────────────────────────────────
    with_period = sum(1 for r in all_rows_state if r.get("period"))
    print(f"\nSummary")
    print(f"  States:                    {len(results)}")
    print(f"  State × theme examples:    {len(all_rows_state)}")
    print(f"  State × theme × tone:      {len(all_rows_full)}")
    print(f"  Examples with period:      {with_period}/{len(all_rows_state)}")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CONFIG ─────────────────────────────────────────────────────────────
    BASE = Path("/home/ethel/Desktop/swimming_pool_project/spp_15052026/openday_2026")

    EXPLODED_CSV   = BASE / "racialised_dimensions_spaitial_exploded.csv"
    REVIEWS_CSV    = BASE / "racialised_reviews_cols.csv"
    TIMESTAMPS_CSV = BASE / "reviews_with_years.csv"
    OUT_DIR        = BASE
    # ──────────────────────────────────────────────────────────────────────

    print("Loading data…")
    df                   = pd.read_csv(EXPLODED_CSV)
    reviews_df           = pd.read_csv(REVIEWS_CSV)
    reviews_with_time_df = pd.read_csv(TIMESTAMPS_CSV)

    for col in ("review_id", "period"):
        assert col in reviews_with_time_df.columns, (
            f"reviews_with_years.csv is missing column '{col}'"
        )

    for col in ("theme", "connotation", "approach", "context"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    print(f"Exploded df:  {len(df):,} rows, {df['state_name'].nunique()} states")
    print(f"Reviews df:   {len(reviews_df):,} rows")
    print(f"Timestamps:   {len(reviews_with_time_df):,} rows, "
          f"{reviews_with_time_df['period'].notna().sum():,} with period\n")

    build_examples(df, reviews_df, reviews_with_time_df, OUT_DIR)
    print("\nDone. Upload examples_for_interactive.json to Claude.")