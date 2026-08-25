"""
pool_pipeline.py
----------------
1. Queries the `place` table from the swimming_pools_db Postgres database.
2. Filters rows to public swimming pool venues using filter_pools().
3. Queries the `reviews` table for those places where org_review or
   trans_review is not empty.
4. Saves both DataFrames as Parquet files to the output directory.
"""

import ast
import os

import pandas as pd
from sqlalchemy import create_engine, text

# ------------------------------
# Postgres connection
# ------------------------------
user     = "ethel"
password = "pa55w0rd13"
host     = "localhost"
port     = "5432"
database = "swimming_pools_db"

engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)

# ------------------------------
# Output directory
# ------------------------------
OUTPUT_DIR = "/home/ethel/Desktop/swimming_pool_project/spp_15052026/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Place-type filter sets
# ---------------------------------------------------------------------------

STRICT_PLACE_TYPES = {
    'beach',
    'indoor swimming pool',
    'lake',
    'lake shore swimming area',
    'lido',
    'outdoor bath',
    'outdoor swimming pool',
    'public bath',
    'public bathroom',
    'public beach',
    'public swimming pool',
    'super public bath',
    'swimming facility',
    'swimming lake',
    'swimming pool',
    'thermal baths',
}

RELAXED_PLACE_TYPES = {
    'aquatic center',
    'leisure center',
    'recreation center',
    'sauna',
    'spa',
    'spa and health club',
    'sports complex',
    'swimming facility',
    'swimming lake',
    'swimming pool',
    'thermal baths',
    'water park',
}

ASSUMED_POOL_PRIMARIES = {
    'aquatic center',
    'sports complex',
    'recreation center',
    'water park',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_subtypes(val):
    """Parse a stringified list of subtypes into a list of lowercase strings."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        flattened = []
        for item in val:
            if isinstance(item, (list, tuple, set)):
                flattened.extend(_parse_subtypes(item))
                continue
            item_text = str(item).strip()
            if not item_text:
                continue
            if len(item_text) >= 2 and ((item_text[0] == item_text[-1] == '"') or (item_text[0] == item_text[-1] == "'")):
                item_text = item_text[1:-1].strip()
            if item_text.startswith('[') and item_text.endswith(']'):
                flattened.extend(_parse_subtypes(item_text))
                continue
            flattened.append(item_text.lower())
        return flattened
    if hasattr(val, "tolist") and not isinstance(val, str):
        return _parse_subtypes(val.tolist())
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass

    text = str(val).strip()
    if not text:
        return []
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, str):
        return _parse_subtypes(parsed)
    if isinstance(parsed, (list, tuple, set)):
        return _parse_subtypes(list(parsed))
    return []


def _is_pool(primary_tag, sub_tags_parsed):
    """
    Returns True if the place is a public swimming pool venue.

    Rules (evaluated in order):
      1. Primary tag is in STRICT_PLACE_TYPES → True
      2. Primary tag is in RELAXED_PLACE_TYPES and at least one subtype
         is in STRICT_PLACE_TYPES → True
      3. Primary tag is in ASSUMED_POOL_PRIMARIES and no subtypes → True
      4. Otherwise → False
    """
    if pd.isna(primary_tag):
        return False

    primary = primary_tag.strip().lower()

    if primary in STRICT_PLACE_TYPES:
        return True

    if primary in RELAXED_PLACE_TYPES:
        if any(s in STRICT_PLACE_TYPES for s in sub_tags_parsed):
            return True

    if primary in ASSUMED_POOL_PRIMARIES and not sub_tags_parsed:
        return True

    return False


def filter_pools(
    df,
    primary_col='primary_place_type',
    sub_col='place_subtype',
    keep_mask_col=False,
):
    """
    Filter a DataFrame to rows that are likely public swimming pool venues.
    """
    if primary_col not in df.columns:
        raise ValueError(f"Column '{primary_col}' not found in DataFrame.")
    if sub_col not in df.columns:
        raise ValueError(f"Column '{sub_col}' not found in DataFrame.")

    parsed_subs = df[sub_col].apply(_parse_subtypes)
    mask = [
        _is_pool(primary, subs)
        for primary, subs in zip(df[primary_col], parsed_subs)
    ]

    result = df[mask].copy()

    if keep_mask_col:
        result['is_pool'] = True

    return result


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

def query_places():
    """
    Pull all rows from the place table and filter to pool venues.
    Returns a filtered DataFrame.
    """
    print("Querying place table...")
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM places"), conn)

    print(f"  Total rows fetched: {len(df)}")
    filtered = filter_pools(df)
    print(f"  Rows after pool filter: {len(filtered)}")
    return filtered


def query_reviews(unique_identifiers):
    """
    Pull reviews for the given place IDs where org_review or trans_review
    is not empty/null.

    Parameters
    ----------
    unique_identifiers : list or array-like
        The place IDs to filter reviews by.

    Returns
    -------
    pd.DataFrame
    """
    if len(unique_identifiers) == 0:
        print("No place IDs provided — skipping reviews query.")
        return pd.DataFrame()

    print(f"Querying reviews for {len(unique_identifiers)} places...")
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT *
                FROM reviews
                WHERE unique_identifier = ANY(:ids)
            """),
            conn,
            params={"ids": list(unique_identifiers)},
        )

    print(f"  Reviews fetched: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------
 
def sanitise_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix mixed-type object columns that cause ArrowTypeError when writing
    to Parquet.  Columns whose values are a mix of strings and floats
    (e.g. NaN leaked into a text column) are cast to pure string, with
    actual NaN/None preserved as pd.NA so PyArrow writes them as null.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        # Check whether the column contains any non-string, non-null value
        sample = df[col].dropna()
        if sample.empty:
            continue
        has_non_str = sample.apply(lambda x: not isinstance(x, str)).any()
        if has_non_str:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    # 1. Fetch and filter places
    places_df = query_places()
 
    # 2. Save places parquet
    places_path = os.path.join(OUTPUT_DIR, "places_filtered.parquet")
    places_df = sanitise_for_parquet(places_df)
    places_df.to_parquet(places_path, index=False)
    print(f"Saved places → {places_path}")
 
    # 3. Fetch reviews for the filtered place IDs
    #    Assumes the place table has a column named 'unique_identifier' or 'id'.
    id_col = "unique_identifier" if "unique_identifier" in places_df.columns else "id"
    unique_identifiers = places_df[id_col].dropna().unique().tolist()
 
    reviews_df = query_reviews(unique_identifiers)
 
    # 4. Save reviews parquet
    reviews_path = os.path.join(OUTPUT_DIR, "reviews_filtered.parquet")
    reviews_df = sanitise_for_parquet(reviews_df)
    reviews_df.to_parquet(reviews_path, index=False)
    print(f"Saved reviews → {reviews_path}")
 
    print("\nDone.")
    print(f"  Places : {len(places_df):,} rows  →  {places_path}")
    print(f"  Reviews: {len(reviews_df):,} rows  →  {reviews_path}")
 
 
if __name__ == "__main__":
    main()