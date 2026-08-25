"""
Prediction Pipeline: "I am not racist, but..."
Swimming pool review discrimination analysis — Germany
=======================================================
Three prediction levels:
  L1 — Review-level binary classification (discriminatory vs not)
  L2 — Pool-level regression (share of discriminatory reviews per pool)
  L3 — Constituency-level regression (share per constituency)

Progression per level:
  Step A: Logistic/Linear regression  (interpretable baseline)
  Step B: XGBoost + SHAP              (nonlinear, feature importance)
  Step C: Multilevel / spatial        (accounts for nesting / geography)

Usage:
  python prediction_pipeline.py --data path/to/your_data.csv --level all
  python prediction_pipeline.py --data path/to/your_data.csv --level 1
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    mean_squared_error, r2_score, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import shap

# Optional — spatial regression (install: pip install pysal libpysal spreg esda)
try:
    import libpysal
    from spreg import OLS as SpatialOLS, ML_Lag
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("Note: pysal/spreg not installed — spatial regression (L3 Step C) will be skipped.")

# Optional — mixed effects (install: pip install statsmodels)
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Note: statsmodels not installed — multilevel model (L1 Step C) will be skipped.")


# =============================================================================
# CONFIGURATION
# Adjust column name constants here if your CSV uses different names.
# Do NOT add ID or raw-count columns to the feature lists below.
# =============================================================================

TARGET_COL       = "connotation_type"   # values: discriminatory / ambiguous / pleasant
REVIEW_TEXT_COL  = "trans_review"       # English translation of the review
PLACE_ID_COL     = "place_id"           # unique place identifier
CONSTITUENCY_COL = "constituency"       # constituency name
STATE_COL        = "state"              # federal state name

# Output directory — all plots and artefacts go here
OUTPUT_DIR = "/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/plots/predicting_discriminatory_reviews"

# ---------------------------------------------------------------------------
# ID columns — NEVER used as model features.
# split_features() enforces this via _NEVER_USE_AS_FEATURES below.
# ---------------------------------------------------------------------------
ID_COLS = [
    "review_id",         # str    — unique review identifier
    "reviewer_id",       # str    — unique reviewer identifier
    "place_id",          # str    — unique place identifier
    "unique_identifier", # str    — alternative place identifier
    "state_nr",          # str    — numeric code for federal state
    "constituency_nr",   # float  — numeric code for constituency
    "index_right",       # float  — spatial join artefact
    "FIRST_MIN_",        # str    — spatial join artefact
]

# ---------------------------------------------------------------------------
# Columns that must never enter a model regardless of how feature lists are
# edited. Includes IDs, the raw target, raw vote counts (shares are used
# instead), raw population counts (percentages are used instead), and
# electoral denominators.
# ---------------------------------------------------------------------------
_NEVER_USE_AS_FEATURES = set(ID_COLS) | {
    # target
    TARGET_COL, "target_binary", "racial_connotations",
    # electoral denominators
    "valid_votes", "eligible_voters", "voters", "invalid_votes",
    # raw party vote counts — shares are engineered in load_and_prepare()
    "afd", "bsw", "spd", "greens", "fdp", "the_left",
    "cdu", "volt", "mlpd", "alliance_germany", "thebasis",
    "animal_protection_party", "free_voters", "the_party",
    # raw population counts — percentages are used instead
    "total_population", "german_population", "foreign_population",
    "population_total_thousands", "population_german_thousands",
}

# ---------------------------------------------------------------------------
# Feature lists — only engineered / meaningful predictors.
# ---------------------------------------------------------------------------

REVIEWER_FEATURES = [
    "reviewer_rating",    # float — star rating given by this reviewer
    "other_reviews",      # float — total Google Maps reviews by this reviewer
    #"nationality",        # str   — LLM-classified reviewer nationality
    "org_review_lang",    # str   — original review language (matches df column name)
]

PLACE_FEATURES = [
    "primary_place_type", # str   — e.g. swimming_pool, outdoor_pool
    #"place_subtype",      # str   — finer place category
    "place_rating",       # float — overall Google Maps rating of the place
    "total_reviews",      # float — total reviews for the place (all types)
    "owner_present",      # int   — 1 if owner has responded to any review (engineered)
]

# All constituency features use shares / percentages, not raw counts.
# Shares are engineered in engineer_constituency_features().
CONSTITUENCY_FEATURES = [
    # population composition (shares)
    "population_foreign_percent",     # already a % in source data
    "population_german_percent",      # engineered: german_population / total_population * 100
    "population_density_per_km2",     # already a rate
    # migration
    "migration_balance_per_1000",     # renamed for formula safety (no special chars)
    # age structure (already percentages in source data)
    "age_under_18_pct",               # renamed for formula safety
    "age_18_24_pct",
    "age_60_74_pct",
    "age_75plus_pct",
    # party vote shares (engineered: party_votes / valid_votes)
    "afd_share",
    "bsw_share",
    "spd_share",
    "greens_share",
    "fdp_share",
    "the_left_share",
    # geographic / structural
    "east_germany",                   # engineered binary
]

ALL_FEATURES = REVIEWER_FEATURES + PLACE_FEATURES + CONSTITUENCY_FEATURES


# =============================================================================
# FEATURE ENGINEERING HELPERS
# =============================================================================

def engineer_constituency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all raw constituency counts into shares/rates and rename columns
    that contain special characters (spaces, parentheses, +/-) so they are
    safe to use in statsmodels Patsy formulas and as dict keys.

    Denominator choices
    -------------------
    Party shares  : valid_votes  — the standard electoral denominator;
                    excludes spoilt/blank ballots and non-voters, matching
                    how official results are reported.
    Pop. shares   : total_population — direct denominator for composition.

    Returns the dataframe with new columns added (originals kept for reference).
    """
    print("\n  [Engineer] Computing constituency shares and renaming columns...")

    # --- Party vote shares (denominator = valid_votes) ---
    parties = ["afd", "bsw", "spd", "greens", "fdp", "the_left"]
    for party in parties:
        raw_col = party if party != "the_left" else "the left"
        if raw_col in df.columns and "valid_votes" in df.columns:
            df[f"{party}_share"] = (
                df[raw_col] / df["valid_votes"].replace(0, np.nan)
            )
        else:
            missing = []
            if raw_col not in df.columns:
                missing.append(raw_col)
            if "valid_votes" not in df.columns:
                missing.append("valid_votes")
            print(f"    Warning: cannot compute {party}_share — missing: {missing}")

    # --- Population shares (denominator = total_population) ---
    for raw_col, new_col in [
        ("german_population", "population_german_percent"),
        ("foreign_population", "population_foreign_percent_calc"),
    ]:
        if raw_col in df.columns and "total_population" in df.columns:
            df[new_col] = (
                df[raw_col] / df["total_population"].replace(0, np.nan) * 100
            )
        # Note: population_foreign_percent already exists as a % in the source;
        # we keep that as the primary column since it is pre-verified.

    # --- Rename columns with special characters for formula safety ---
    rename_map = {
        "the left": "the_left",   # raw vote count (kept for reference only)
        # age columns — strip special chars
        "age from ... to ... years on 31.12.2023 - under 18 (%)":  "age_under_18_pct",
        "age from ... to ... years on 31.12.2023 - 18-24 (%)":     "age_18_24_pct",
        "age from ... to ... years on 31.12.2023 - 25-34 (%)":     "age_25_34_pct",
        "age from ... to ... years on 31.12.2023 - 35-59 (%)":     "age_35_59_pct",
        "age from ... to ... years on 31.12.2023 - 60-74 (%)":     "age_60_74_pct",
        "age from ... to ... years on 31.12.2023 - 75 and over (%)": "age_75plus_pct",
        # migration balance
        "population increase (+) or decrease (-) in 2022 - migration balance (per 1000 inhabitants)":
            "migration_balance_per_1000",
        "population increase (+) or decrease (-) in 2023 - birth balance (per 1000 inhabitants)":
            "birth_balance_per_1000",
        # other party columns with spaces
        "the party":              "the_party",
        "free voters":            "free_voters",
        "alliance germany":       "alliance_germany",
        "animal protection party": "animal_protection_party",
    }
    # Only rename columns that actually exist
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)
    if actual_rename:
        print(f"    Renamed {len(actual_rename)} columns for formula safety.")

    # Report what was engineered
    engineered = [f"{p}_share" for p in parties if f"{p}_share" in df.columns]
    print(f"    Party share columns created : {engineered}")
    print(f"    Age columns renamed         : {[v for k,v in actual_rename.items() if 'age' in k.lower()]}")

    return df


# =============================================================================
# CORE HELPERS
# =============================================================================

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load CSV, optionally filter to racial reviews, engineer all derived
    features, and create the binary classification target.
    """
    # --- Validate input file ---
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"\n{'='*60}")
    print("LOADING DATA")
    print('='*60)
    print(f"  Input file: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"  Total rows loaded          : {len(df):,}")
    print(f"  Total columns              : {len(df.columns):,}")

    # --- Filter to racial reviews (defensive — dataset may already be filtered) ---
    if "racial_connotations" in df.columns:
        df = df[df["racial_connotations"].str.lower().str.strip() == "yes"].copy()
        print(f"  After racial filter        : {len(df):,}")

    # --- Binary target: discriminatory=1 vs ambiguous+pleasant=0 ---
    df["target_binary"] = (
        df[TARGET_COL].str.lower().str.strip() == "discriminatory"
    ).astype(int)
    print(f"  Discriminatory reviews     : {df['target_binary'].sum():,} "
          f"({df['target_binary'].mean()*100:.1f}%)")
    print(f"  Non-discriminatory (racial): {(1-df['target_binary']).sum():,}")

    # --- Engineer all constituency-level features ---
    df = engineer_constituency_features(df)

    # --- East / West binary indicator ---
    east_states = {
        "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen",
        "Sachsen-Anhalt", "Thüringen", "Berlin"
    }
    if STATE_COL in df.columns:
        df["east_germany"] = df[STATE_COL].isin(east_states).astype(int)
        print(f"  East Germany reviews       : {df['east_germany'].sum():,}")

    # --- Owner response binary ---
    if "owner" in df.columns:
        df["owner_present"] = df["owner"].notna().astype(int)

    # --- Verify no ID columns accidentally entered feature lists ---
    leaked = [f for f in ALL_FEATURES if f in _NEVER_USE_AS_FEATURES]
    if leaked:
        raise ValueError(
            f"The following columns are in ALL_FEATURES but should never be "
            f"used as features: {leaked}\nRemove them from the feature lists."
        )

    print(f"\n  Feature engineering complete. DataFrame shape: {df.shape}")
    return df


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Sklearn ColumnTransformer: impute + scale numerics; impute + OHE categoricals."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def split_features(feature_list: list, df: pd.DataFrame):
    """
    From a feature list, return only columns that:
      (a) exist in the dataframe, AND
      (b) are not in _NEVER_USE_AS_FEATURES (ID / raw count guard).
    Then split into numeric and categorical based on dtype.
    """
    available = [
        f for f in feature_list
        if f in df.columns and f not in _NEVER_USE_AS_FEATURES
    ]
    # Warn about any requested features that are missing from the dataframe
    missing = [f for f in feature_list if f not in df.columns and f not in _NEVER_USE_AS_FEATURES]
    if missing:
        print(f"    Note: {len(missing)} requested feature(s) not found in df: {missing}")

    num_cols = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
    cat_cols = [f for f in available if not pd.api.types.is_numeric_dtype(df[f])]
    return num_cols, cat_cols, available


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def save_fig(filename: str):
    """Save current figure to OUTPUT_DIR, creating directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fpath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# =============================================================================
# LEVEL 1 — REVIEW-LEVEL BINARY CLASSIFICATION
# =============================================================================

def level1_logistic(df: pd.DataFrame, feature_blocks: dict) -> dict:
    """
    Step A: Logistic regression with odds ratios.
    Run once per feature block, then on all features combined.
    Returns a results dict for downstream summary reporting.
    """
    print_section("L1 — Step A: Logistic Regression (per block + combined)")

    y = df["target_binary"]
    results = {}

    blocks_to_run = {**feature_blocks, "combined": ALL_FEATURES}

    for block_name, features in blocks_to_run.items():
        num_cols, cat_cols, available = split_features(features, df)
        if not available:
            print(f"  [{block_name}] No features available — skipping")
            continue

        X = df[available]
        preprocessor = build_preprocessor(num_cols, cat_cols)

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        cw = dict(zip(classes, weights))

        pipe = Pipeline([
            ("prep",  preprocessor),
            ("model", LogisticRegression(
                class_weight=cw, max_iter=1000, random_state=42, C=1.0
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["roc_auc", "f1_weighted"],
            return_train_score=False
        )

        roc_mean = scores["test_roc_auc"].mean()
        roc_std  = scores["test_roc_auc"].std()
        f1_mean  = scores["test_f1_weighted"].mean()
        f1_std   = scores["test_f1_weighted"].std()

        print(f"\n  Block: {block_name.upper()}")
        print(f"    Features used  : {len(available)}")
        print(f"    ROC-AUC (5-CV) : {roc_mean:.3f} ± {roc_std:.3f}")
        print(f"    F1-weighted    : {f1_mean:.3f} ± {f1_std:.3f}")

        # Fit on full data for coefficient / odds ratio extraction
        pipe.fit(X, y)
        lr_model = pipe.named_steps["model"]
        prep     = pipe.named_steps["prep"]

        try:
            ohe_names  = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
            feat_names = num_cols + list(ohe_names)
        except Exception:
            feat_names = [f"feature_{i}" for i in range(len(lr_model.coef_[0]))]

        coefs = lr_model.coef_[0]
        odds  = np.exp(coefs)
        odds_df = (
            pd.DataFrame({"feature": feat_names, "coef": coefs, "odds_ratio": odds})
            .sort_values("odds_ratio", ascending=False)
        )

        print(f"\n    Top 10 features by odds ratio:")
        print(odds_df.head(10).to_string(index=False))

        results[block_name] = {
            "roc_auc": roc_mean,
            "roc_std": roc_std,
            "f1":      f1_mean,
            "f1_std":  f1_std,
            "n_features": len(available),
            "odds_df": odds_df,
            "pipe":    pipe,
        }

        if block_name == "combined":
            _plot_odds_ratios(odds_df, title="Logistic regression — odds ratios (combined block)")

    return results

def sanitise_feature_names(names: list) -> list:
    """
    Replace characters that XGBoost forbids in feature names: [ ] 
    Called after OHE feature name extraction, before any SHAP computation.
    """
    return [
        re.sub(r"[\[\]<>]", "", name).replace("'", "").replace(" ", "_").strip("_")
        for name in names
    ]

def level1_xgboost_shap(df: pd.DataFrame) -> tuple:
    """
    Step B: XGBoost + SHAP for L1.
    Returns (pipe, shap_values, feat_names) for downstream use.
    """
    print_section("L1 — Step B: XGBoost + SHAP values")

    y = df["target_binary"]
    num_cols, cat_cols, available = split_features(ALL_FEATURES, df)
    X = df[available]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", xgb.XGBClassifier(
            scale_pos_weight=spw,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["roc_auc", "f1_weighted"],
        return_train_score=False
    )

    roc_mean = scores["test_roc_auc"].mean()
    f1_mean  = scores["test_f1_weighted"].mean()
    print(f"  ROC-AUC (5-CV) : {roc_mean:.3f} ± {scores['test_roc_auc'].std():.3f}")
    print(f"  F1-weighted    : {f1_mean:.3f} ± {scores['test_f1_weighted'].std():.3f}")

    pipe.fit(X, y)
    # Patch feature names on the booster directly after fitting
    prep      = pipe.named_steps["prep"]
    xgb_model = pipe.named_steps["model"]
    X_trans   = prep.transform(X)

    try:
        if cat_cols:
            ohe_names  = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
            feat_names = sanitise_feature_names(num_cols + list(ohe_names))
        else:
            feat_names = sanitise_feature_names(num_cols)
    except Exception:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
    
    # Apply sanitised names to the booster so XGBoost accepts them
    xgb_model.get_booster().feature_names = feat_names
    X_trans_df = pd.DataFrame(X_trans, columns=feat_names)

    print("\n  Computing SHAP values (this may take a moment)...")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_trans_df)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_trans_df, max_display=20, show=False, plot_type="dot")
    plt.title("L1 — XGBoost SHAP values (review level)", fontsize=12)
    plt.tight_layout()
    save_fig("l1_shap_beeswarm.png")

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_trans_df, max_display=15, show=False, plot_type="bar")
    plt.title("L1 — Feature importance (mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    save_fig("l1_shap_bar.png")

    return pipe, shap_values, feat_names, {"roc_auc": roc_mean, "f1": f1_mean}


def level1_multilevel(df: pd.DataFrame):
    """
    Step C: Mixed-effects linear model approximation for binary outcome.

    NOTE: smf.mixedlm fits a *linear* mixed model. This is an approximation
    for the binary target_binary outcome. For publication-quality results,
    consider using pymer4 (Python wrapper for R lme4) for a proper GLMM.
    The linear approximation is acceptable for exploratory inference on the
    fixed effects when the outcome is not extremely skewed.

    Random intercept for pool (place_id) accounts for the non-independence
    of reviews from the same pool — reviewers at committee review presentations
    will ask about this.
    """
    print_section("L1 — Step C: Mixed-effects model (random intercept for pool)")

    if not STATSMODELS_AVAILABLE:
        print("  statsmodels not available — skipping.")
        return None

    formula_fixed = (
        "target_binary ~ "
        "reviewer_rating + other_reviews + "
        "afd_share + population_foreign_percent + "
        "population_density_per_km2 + east_germany"
    )

    # Parse variable names from formula
    rhs = formula_fixed.split("~")[1]
    formula_vars = [c.strip() for c in rhs.split("+")]
    available_vars = [c for c in formula_vars if c in df.columns]
    missing_vars   = [c for c in formula_vars if c not in df.columns]
    if missing_vars:
        print(f"  Warning: formula variables not in df: {missing_vars}")

    model_df = df[["target_binary", PLACE_ID_COL, CONSTITUENCY_COL] + available_vars].dropna()
    print(f"  Rows after dropping NaN    : {len(model_df):,}")
    print(f"  Unique pools               : {model_df[PLACE_ID_COL].nunique():,}")
    print(f"  Unique constituencies      : {model_df[CONSTITUENCY_COL].nunique():,}")

    # Rebuild formula with only available variables
    if missing_vars:
        available_rhs = " + ".join(available_vars)
        formula_fixed = f"target_binary ~ {available_rhs}"
        print(f"  Adjusted formula: {formula_fixed}")

    try:
        md = smf.mixedlm(formula_fixed, data=model_df, groups=model_df[PLACE_ID_COL])
        result = md.fit(method="lbfgs", maxiter=300)
        print("\n  Fixed effects summary:")
        print(result.summary().tables[1])
        return result
    except Exception as e:
        print(f"  Mixed-effects model failed: {e}")
        print("  Try reducing the formula or increasing maxiter.")
        return None


# =============================================================================
# LEVEL 2 — POOL-LEVEL REGRESSION
# =============================================================================

def aggregate_to_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review-level data to pool level."""
    print_section("L2 — Aggregating to pool level")

    constituency_carry = [
        "population_foreign_percent",
        "population_density_per_km2",
        "afd_share", "bsw_share", "spd_share", "greens_share",
        "east_germany", "migration_balance_per_1000",
        CONSTITUENCY_COL, STATE_COL,
    ]

    pool_agg = df.groupby(PLACE_ID_COL).agg(
        total_racial_reviews = ("target_binary", "count"),
        discriminatory_count = ("target_binary", "sum"),
        mean_reviewer_rating = ("reviewer_rating", "mean"),
        mean_place_rating    = ("place_rating",    "first"),
        total_reviews        = ("total_reviews",   "first"),
        primary_place_type   = ("primary_place_type",
                                lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
        **{feat: (feat, "first") for feat in constituency_carry if feat in df.columns}
    ).reset_index()

    pool_agg["discrim_share"]  = (
        pool_agg["discriminatory_count"] / pool_agg["total_racial_reviews"]
    )
    pool_agg["review_weight"] = np.log1p(pool_agg["total_racial_reviews"])

    print(f"  Pools in dataset             : {len(pool_agg):,}")
    print(f"  Mean discrim share per pool  : {pool_agg['discrim_share'].mean():.3f}")
    print(f"  Pools with ≥10 racial reviews: "
          f"{(pool_agg['total_racial_reviews'] >= 10).sum():,}")

    pool_agg = pool_agg[pool_agg["total_racial_reviews"] >= 3].copy()
    print(f"  After min-3 filter           : {len(pool_agg):,} pools")
    return pool_agg


def level2_regression(pool_df: pd.DataFrame) -> tuple:
    """
    Step A: Ridge regression on pool-level data (interpretable coefficients).
    Step B: XGBoost regressor + SHAP (nonlinear importance).
    Returns (ridge_pipe, xgb_pipe).
    """
    print_section("L2 — Pool-level regression")

    y = pool_df["discrim_share"].values

    pool_features = [
        "mean_reviewer_rating", "mean_place_rating", "total_reviews",
        "primary_place_type",
        "population_foreign_percent", "population_density_per_km2",
        "afd_share", "east_germany", "migration_balance_per_1000",
    ]
    num_cols, cat_cols, available = split_features(pool_features, pool_df)
    X = pool_df[available]
    preprocessor = build_preprocessor(num_cols, cat_cols)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Step A: Ridge ---
    print("\n  Step A: Ridge regression")
    ridge_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])
    scores = cross_validate(
        ridge_pipe, X, y, cv=cv,
        scoring=["r2", "neg_root_mean_squared_error"]
    )
    print(f"    R² (5-CV)  : {scores['test_r2'].mean():.3f} ± {scores['test_r2'].std():.3f}")
    print(f"    RMSE (5-CV): {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
    ridge_pipe.fit(X, y)

    # --- Step B: XGBoost ---
    print("\n  Step B: XGBoost regressor + SHAP")
    xgb_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42, verbosity=0
        )),
    ])
    scores_xgb = cross_validate(
        xgb_pipe, X, y, cv=cv,
        scoring=["r2", "neg_root_mean_squared_error"]
    )
    print(f"    R² (5-CV)  : {scores_xgb['test_r2'].mean():.3f} ± {scores_xgb['test_r2'].std():.3f}")
    print(f"    RMSE (5-CV): {-scores_xgb['test_neg_root_mean_squared_error'].mean():.4f}")
    xgb_pipe.fit(X, y)

    prep      = xgb_pipe.named_steps["prep"]
    xgb_model = xgb_pipe.named_steps["model"]
    X_trans   = prep.transform(X)
    try:
        if cat_cols:
            ohe_names  = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
            feat_names = sanitise_feature_names(num_cols + list(ohe_names))
        else:
            feat_names = sanitise_feature_names(num_cols)
    except Exception:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]

    X_trans_df  = pd.DataFrame(X_trans, columns=feat_names)
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_trans_df)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_trans_df, max_display=15, show=False)
    plt.title("L2 — Pool-level: SHAP values (discriminatory share)", fontsize=12)
    plt.tight_layout()
    save_fig("l2_shap.png")

    return ridge_pipe, xgb_pipe


# =============================================================================
# LEVEL 3 — CONSTITUENCY-LEVEL REGRESSION
# =============================================================================

def aggregate_to_constituency(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review-level data to constituency level."""
    print_section("L3 — Aggregating to constituency level")

    carry_features = [
        "population_foreign_percent",
        "population_density_per_km2",
        "afd_share", "bsw_share", "spd_share", "greens_share", "fdp_share",
        "east_germany",
        "age_under_18_pct", "age_18_24_pct", "age_60_74_pct", "age_75plus_pct",
        "migration_balance_per_1000",
    ]

    const_agg = df.groupby(CONSTITUENCY_COL).agg(
        total_racial_reviews = ("target_binary", "count"),
        discriminatory_count = ("target_binary", "sum"),
        n_pools              = (PLACE_ID_COL,    "nunique"),
        latitude             = ("latitude",      "mean"),
        longitude            = ("longitude",     "mean"),
        **{feat: (feat, "first") for feat in carry_features if feat in df.columns}
    ).reset_index()

    const_agg["discrim_share"] = (
        const_agg["discriminatory_count"] / const_agg["total_racial_reviews"]
    )
    const_agg = const_agg[const_agg["total_racial_reviews"] >= 5].copy()
    print(f"  Constituencies retained    : {len(const_agg):,}")
    print(f"  Mean discrim share         : {const_agg['discrim_share'].mean():.3f}")
    return const_agg


def level3_regression(const_df: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Step A: OLS via statsmodels (full inference: CIs, p-values, R²).
    Step B: XGBoost + SHAP.
    Step C: Spatial lag model with Moran's I test (requires pysal/esda).
    Also reports East vs West Germany split.
    """
    print_section("L3 — Constituency-level regression")

    y = const_df["discrim_share"].values

    const_features = [
        "population_foreign_percent",
        "population_density_per_km2",
        "afd_share", "bsw_share", "spd_share",
        "east_germany", "n_pools",
        "age_under_18_pct", "age_60_74_pct",
        "migration_balance_per_1000",
    ]
    num_cols, cat_cols, available = split_features(const_features, const_df)
    X = const_df[available]

    # --- Step A: OLS ---
    print("\n  Step A: OLS regression with confidence intervals")
    if STATSMODELS_AVAILABLE:
        X_sm = sm.add_constant(X[num_cols].fillna(X[num_cols].median()))
        ols_result = sm.OLS(y, X_sm).fit()
        print(ols_result.summary())
        print(f"\n  R²: {ols_result.rsquared:.3f}  Adj-R²: {ols_result.rsquared_adj:.3f}")
    else:
        lr = LinearRegression()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(lr, X[num_cols].fillna(0), y, cv=cv, scoring="r2")
        print(f"  R² (5-CV): {scores['test_r2'].mean():.3f}")

    # --- Step B: XGBoost ---
    print("\n  Step B: XGBoost regressor")
    preprocessor = build_preprocessor(num_cols, cat_cols)
    xgb_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            subsample=0.8, random_state=42, verbosity=0
        )),
    ])
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(xgb_pipe, X, y, cv=cv, scoring="r2", error_score=np.nan)
    print(f"  R² (3-CV): {scores['test_r2'].mean():.3f} ± {scores['test_r2'].std():.3f}")
    xgb_pipe.fit(X, y)

    prep      = xgb_pipe.named_steps["prep"]
    xgb_model = xgb_pipe.named_steps["model"]
    X_trans   = prep.transform(X)

    try:
        if cat_cols:
            ohe_names  = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
            feat_names = sanitise_feature_names(num_cols + list(ohe_names))
        else:
            feat_names = sanitise_feature_names(num_cols)
    except Exception:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]

    X_trans_df  = pd.DataFrame(X_trans, columns=feat_names)
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_trans_df)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_trans_df, max_display=15, show=False)
    plt.title("L3 — Constituency-level: SHAP values", fontsize=12)
    plt.tight_layout()
    save_fig("l3_shap.png")

    # --- East vs West split ---
    if "east_germany" in const_df.columns and "afd_share" in const_df.columns:
        print("\n  East vs West split:")
        for label, mask in [("East Germany", const_df["east_germany"] == 1),
                             ("West Germany", const_df["east_germany"] == 0)]:
            sub = const_df[mask]
            if len(sub) > 5:
                print(f"    {label}: n={len(sub)}, "
                      f"mean discrim share={sub['discrim_share'].mean():.3f}, "
                      f"mean AfD share={sub['afd_share'].mean():.3f}")

    # --- Step C: Spatial lag model ---
    if SPATIAL_AVAILABLE and "latitude" in const_df.columns:
        print("\n  Step C: Spatial lag model (Moran's I + ML_Lag)")
        try:
            coords = const_df[["longitude", "latitude"]].values
            w = libpysal.weights.KNN.from_array(coords, k=5)
            w.transform = "r"

            y_sp = const_df["discrim_share"].values
            X_sp = const_df[num_cols].fillna(const_df[num_cols].median()).values

            from esda.moran import Moran
            mi = Moran(y_sp, w)
            print(f"    Moran's I: {mi.I:.3f}  p-value: {mi.p_sim:.3f}")
            if mi.p_sim < 0.05:
                print("    Significant spatial autocorrelation — fitting spatial lag model")
                ml = ML_Lag(y_sp, X_sp, w=w, name_y="discrim_share", name_x=num_cols)
                print(ml.summary)
            else:
                print("    No significant spatial autocorrelation — OLS sufficient")
        except Exception as e:
            print(f"    Spatial model error: {e}")

    return xgb_pipe


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _plot_odds_ratios(odds_df: pd.DataFrame, title: str, top_n: int = 20):
    """Horizontal bar (forest-style) plot of odds ratios from logistic regression."""
    plot_df = odds_df.head(top_n).sort_values("odds_ratio")
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.35)))
    colors = ["#D85A30" if o > 1 else "#1D9E75" for o in plot_df["odds_ratio"]]
    ax.barh(plot_df["feature"], plot_df["odds_ratio"] - 1,
            left=1, color=colors, alpha=0.75, edgecolor="none")
    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    safe_name = (
        title.lower()
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("—", "").replace("/", "_")[:50]
    )
    save_fig(f"{safe_name}.png")


def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of connotation type distribution."""
    counts = df[TARGET_COL].str.lower().str.strip().value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    color_map = {"discriminatory": "#D85A30", "ambiguous": "#888780", "pleasant": "#1D9E75"}
    bars = ax.bar(
        counts.index, counts.values,
        color=[color_map.get(c, "#888780") for c in counts.index], alpha=0.85
    )
    ax.set_ylabel("Number of reviews")
    ax.set_title("Connotation type distribution (racial reviews only)", fontsize=11)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig("class_distribution.png")


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(l1_logistic_results: dict, l1_xgb_meta: dict):
    """Print a compact cross-level performance summary table."""
    print_section("RESULTS SUMMARY")

    print(f"\n  {'Block':<20} {'ROC-AUC':>10} {'F1-weighted':>12}")
    print(f"  {'-'*44}")
    for block, res in l1_logistic_results.items():
        print(f"  L1-LogReg {block:<12} {res['roc_auc']:>10.3f} {res['f1']:>12.3f}")
    print(f"  L1-XGBoost {'combined':<12} {l1_xgb_meta['roc_auc']:>10.3f} {l1_xgb_meta['f1']:>12.3f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Swimming pool discrimination prediction pipeline"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to input CSV file (e.g. /data/reviews.csv)"
    )
    parser.add_argument(
        "--level", type=str, default="all",
        choices=["all", "1", "2", "3"],
        help="Which prediction level to run (default: all)"
    )
    args = parser.parse_args()

    # load_and_prepare() validates file existence and raises FileNotFoundError if missing
    df = load_and_prepare(args.data)
    plot_class_distribution(df)

    feature_blocks = {
        "reviewer":     REVIEWER_FEATURES,
        "place":        PLACE_FEATURES,
        "constituency": CONSTITUENCY_FEATURES,
    }

    # Collect results for summary table
    l1_logistic_results = {}
    l1_xgb_meta         = {}

    if args.level in ("all", "1"):
        print("\n" + "#"*60)
        print("# LEVEL 1 — REVIEW-LEVEL CLASSIFICATION")
        print("#"*60)
        l1_logistic_results              = level1_logistic(df, feature_blocks)
        _, _, _, l1_xgb_meta             = level1_xgboost_shap(df)
        _                                = level1_multilevel(df)

    if args.level in ("all", "2"):
        print("\n" + "#"*60)
        print("# LEVEL 2 — POOL-LEVEL REGRESSION")
        print("#"*60)
        pool_df = aggregate_to_pool(df)
        level2_regression(pool_df)

    if args.level in ("all", "3"):
        print("\n" + "#"*60)
        print("# LEVEL 3 — CONSTITUENCY-LEVEL REGRESSION")
        print("#"*60)
        const_df = aggregate_to_constituency(df)
        level3_regression(const_df)

    # Print summary if L1 was run
    if l1_logistic_results and l1_xgb_meta:
        print_results_summary(l1_logistic_results, l1_xgb_meta)

    print("\n" + "="*60)
    print(f"  Pipeline complete. All outputs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()