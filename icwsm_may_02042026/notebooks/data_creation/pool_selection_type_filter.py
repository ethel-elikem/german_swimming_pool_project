"""
pool_filter.py
--------------
Filters a DataFrame to identify public swimming pool venues based on
primary_place_type and sub_place_types columns.

Usage in notebook or script:
    from pool_filter import filter_pools

    filtered_df = filter_pools(df)
    # or with custom column names:
    filtered_df = filter_pools(df, primary_col='my_primary', sub_col='my_subtypes')
"""

import ast
import pandas as pd


# ---------------------------------------------------------------------------
# Place type lists
# ---------------------------------------------------------------------------

# Primary tag alone is sufficient to classify as a pool venue
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

# Primary tag qualifies only if at least one subtype is in STRICT_PLACE_TYPES
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

# Primary tag qualifies even with no subtypes (assumed to have a pool)
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
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [s.strip().lower() for s in val]
    try:
        parsed = ast.literal_eval(val)
        return [s.strip().lower() for s in parsed]
    except (ValueError, SyntaxError):
        return []


def _is_pool(primary_tag, sub_tags_parsed):
    """
    Returns True if the place is a public swimming pool venue.

    Rules (evaluated in order):
      1. Primary tag is in STRICT_PLACE_TYPES → True
      2. Primary tag is in RELAXED_PLACE_TYPES and at least one subtype
         is in STRICT_PLACE_TYPES → True
      4. Primary tag is in ASSUMED_POOL_PRIMARIES and no subtypes → True
      5. Otherwise → False
    """
    if pd.isna(primary_tag):
        return False

    primary = primary_tag.strip().lower()

    # Rule 1: strict primary match
    if primary in STRICT_PLACE_TYPES:
        return True

    # Rule 2: relaxed primary + at least one strict subtype
    if primary in RELAXED_PLACE_TYPES:
        if any(s in STRICT_PLACE_TYPES for s in sub_tags_parsed):
            return True

    # Rule 3: assumed pool primary with no subtypes
    if primary in ASSUMED_POOL_PRIMARIES and not sub_tags_parsed:
        return True

    return False


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def filter_pools(
    df,
    primary_col='primary_place_type',
    sub_col='sub_place_types',
    keep_mask_col=False,
):
    """
    Filter a DataFrame to rows that are likely public swimming pool venues.

    Parameters
    ----------
    df : pd.DataFrame
    primary_col : str
        Column name for the primary place type.
    sub_col : str
        Column name for the sub place types (stringified list or actual list).
    keep_mask_col : bool
        If True, adds a boolean column 'is_pool' to the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (and optionally with 'is_pool' column).
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