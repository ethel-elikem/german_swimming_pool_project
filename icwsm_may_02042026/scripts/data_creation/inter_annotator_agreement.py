"""
Inter-Annotator Agreement Analysis
====================================
4 annotators, 3 hierarchical label categories: social, racial, type
- Calculates Krippendorff's Alpha per category
- Produces ground truth CSV with majority vote (ties flagged as 'tie')
- Produces one detailed publication-ready plot
- Produces agreement summary as CSV and TXT

NOTE: 'type' is hierarchical — it is only applicable when social='yes' or racial='yes'.
      Empty/NA values in 'type' when social=no and racial=no are NOT missing data,
      they are 'na' (not applicable) and are treated as a valid label.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from collections import Counter
import krippendorff          # pip install krippendorff
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ANNOTATOR_FILES = {
    'divya': '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/annotation/divya_100_swimreviews_labelled_12092025.csv',
    'theo':  '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/annotation/theo_100_swimreviews_labelled_12092025.csv',
    'ethel': '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/annotation/ethel_100_swimreviews_labelled_12092025.csv',
    'brahms':'/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/annotation/brahms_100_swimreviews_labelled_12092025.csv',
}

CATEGORIES = ['social', 'racial', 'type']

# Values that mean "not applicable" (hierarchical empty) — treated as valid label 'na'
NA_APPLICABLE_VALUES = {'na', 'n/a', 'none', 'null', '', 'nan', '-'}

# Values that are genuinely missing/erroneous — excluded from calculations
INVALID_VALUES = {'error', 'unknown'}

OUTPUT_LABELS_CSV    = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/output_files/annotations_with_majority_vote_and_ties_flagged.csv'
OUTPUT_AGREEMENT_CSV = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/output_files/agreement_summary.csv'
OUTPUT_PLOT          = '/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/plots/inter_annotator_agreement_figure.png'


# ─────────────────────────────────────────────
# 1. LOADING & CLEANING
# ─────────────────────────────────────────────

def clean_value(v):
    """Standardise a single label value."""
    v = str(v).strip().lower()
    # Fix common typos
    v = v.replace('ambigious', 'ambiguous').replace('ambigous', 'ambiguous')
    v = v.replace(' no', 'no').replace(' yes', 'yes')
    if v in NA_APPLICABLE_VALUES:
        return 'na'
    if v in INVALID_VALUES:
        return np.nan   # genuinely invalid → will be excluded
    return v


def load_and_clean(annotator_files: dict) -> pd.DataFrame:
    """
    Load all annotator CSVs, clean values, enforce hierarchy, merge on review_id.
    Returns wide dataframe: review_id | ann_social | ann_racial | ann_type  (per annotator)
    """
    dfs = []
    for name, path in annotator_files.items():
        df = pd.read_csv(path)[['review_id', 'social', 'racial', 'type']].copy()
        for col in CATEGORIES:
            df[col] = df[col].apply(clean_value)

        # Enforce hierarchy: if social=no AND racial=no, type must be 'na'
        mask_not_applicable = (df['social'].isin(['no', 'na'])) & (df['racial'].isin(['no', 'na']))
        df.loc[mask_not_applicable, 'type'] = 'na'

        df.rename(columns={c: f'{name}_{c}' for c in CATEGORIES}, inplace=True)
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='review_id', how='inner')

    return merged


# ─────────────────────────────────────────────
# 2. GROUND TRUTH VIA MAJORITY VOTE
# ─────────────────────────────────────────────

def majority_vote(values: list):
    """
    Given a list of labels from N annotators, return:
      - the majority label  (if >= ceil(N/2)+1 agree, i.e. clear majority)
      - 'tie'               (if no clear majority — e.g. 2-2 split with 4 annotators)
    NaN values (invalid) are excluded before counting.
    """
    valid = [v for v in values if pd.notna(v)]
    if not valid:
        return np.nan

    counts = Counter(valid)
    most_common_label, top_count = counts.most_common(1)[0]
    n = len(valid)

    # Strict majority: more than half
    if top_count > n / 2:
        return most_common_label
    else:
        return 'tie'


def create_ground_truth(merged_df: pd.DataFrame, annotators: list) -> pd.DataFrame:
    """
    Build the output CSV:
    review_id | ann_social … | ann_racial … | ann_type … | gt_social | gt_racial | gt_type
    'tie' where no majority exists.
    """
    df = merged_df.copy()

    for cat in CATEGORIES:
        cols = [f'{ann}_{cat}' for ann in annotators]
        df[f'gt_{cat}'] = df[cols].apply(
            lambda row: majority_vote(row.tolist()), axis=1
        )

    return df


# ─────────────────────────────────────────────
# 3. KRIPPENDORFF'S ALPHA
# ─────────────────────────────────────────────

def encode_labels_for_krippendorff(series: pd.Series) -> np.ndarray:
    """
    Convert a string label series to integer codes.
    NaN stays NaN (krippendorff library handles missing values natively).
    """
    valid_labels = sorted(series.dropna().unique())
    label_map = {label: i for i, label in enumerate(valid_labels)}
    return series.map(label_map).values.astype(float)   # NaN preserved as float NaN


def calculate_krippendorff(merged_df: pd.DataFrame, annotators: list) -> dict:
    """
    Calculate Krippendorff's Alpha for each category.
    Returns dict: {category: {'alpha': float, 'n_items': int, 'n_valid': int}}
    """
    results = {}

    for cat in CATEGORIES:
        cols = [f'{ann}_{cat}' for ann in annotators]

        # Build reliability matrix: rows = annotators, cols = items
        reliability_matrix = np.array([
            encode_labels_for_krippendorff(merged_df[col]) for col in cols
        ])

        # Count valid (non-NaN) items — items where at least 2 annotators labelled
        valid_items = np.sum(np.sum(~np.isnan(reliability_matrix), axis=0) >= 2)

        try:
            alpha = krippendorff.alpha(
                reliability_data=reliability_matrix,
                level_of_measurement='nominal'
            )
        except Exception as e:
            alpha = np.nan
            print(f"  Warning: could not compute alpha for '{cat}': {e}")

        results[cat] = {
            'alpha':    round(alpha, 4) if not np.isnan(alpha) else np.nan,
            'n_items':  len(merged_df),
            'n_valid':  int(valid_items)
        }

    return results


# ─────────────────────────────────────────────
# 4. AGREEMENT PATTERN COUNTS (for the plot)
# ─────────────────────────────────────────────

def get_agreement_patterns(merged_df: pd.DataFrame, annotators: list) -> dict:
    """
    Per category, count: unanimous | majority (3/4) | tie (2/2) | all different
    """
    patterns = {cat: {'Unanimous (4/4)': 0, 'Majority (3/4)': 0,
                       'Tie (2-2)': 0, 'All different': 0}
                for cat in CATEGORIES}

    for _, row in merged_df.iterrows():
        for cat in CATEGORIES:
            vals = [row[f'{ann}_{cat}'] for ann in annotators]
            vals = [v for v in vals if pd.notna(v)]
            if not vals:
                continue
            n_unique = len(set(vals))
            counts = Counter(vals)
            top = counts.most_common(1)[0][1]

            if n_unique == 1:
                patterns[cat]['Unanimous (4/4)'] += 1
            elif top == 3:
                patterns[cat]['Majority (3/4)'] += 1
            elif top == 2:
                patterns[cat]['Tie (2-2)'] += 1
            else:
                patterns[cat]['All different'] += 1

    return patterns


def get_label_distributions(merged_df: pd.DataFrame, annotators: list) -> dict:
    """Per category, per annotator: value_counts dict"""
    dist = {}
    for cat in CATEGORIES:
        dist[cat] = {}
        for ann in annotators:
            col = f'{ann}_{cat}'
            vc = merged_df[col].dropna().value_counts()
            dist[cat][ann] = vc.to_dict()
    return dist


def get_pairwise_kappa_matrix(merged_df: pd.DataFrame, annotators: list) -> dict:
    """
    Compute pairwise Cohen's Kappa between all annotator pairs per category.
    Returns dict: {category: pd.DataFrame (NxN matrix)}
    Note: included for the plot only. Primary metric remains Krippendorff's Alpha.
    """
    from sklearn.metrics import cohen_kappa_score

    matrices = {}
    n = len(annotators)

    for cat in CATEGORIES:
        mat = pd.DataFrame(np.nan, index=annotators, columns=annotators)
        for i in range(n):
            mat.iloc[i, i] = 1.0
            for j in range(i+1, n):
                col_i = f'{annotators[i]}_{cat}'
                col_j = f'{annotators[j]}_{cat}'
                # Only rows where both have valid labels
                valid = merged_df[[col_i, col_j]].dropna()
                if len(valid) < 2:
                    continue
                try:
                    k = cohen_kappa_score(valid[col_i], valid[col_j])
                    mat.iloc[i, j] = round(k, 3)
                    mat.iloc[j, i] = round(k, 3)
                except Exception:
                    pass
        matrices[cat] = mat

    return matrices


# ─────────────────────────────────────────────
# 5. THE DETAILED PLOT
# ─────────────────────────────────────────────

def make_publication_figure(
        agreement_results: dict,
        patterns: dict,
        pairwise_matrices: dict,
        label_distributions: dict,
        annotators: list,
        output_path: str
):
    """
    Single publication-ready figure with 4 panels per category (12 panels total):
      Row A — Krippendorff's Alpha summary bar (one bar per category)
      Row B — Agreement pattern stacked bars (per category)
      Row C — Pairwise kappa heatmap (per category)
      Row D — Label distribution per annotator (per category, small multiples)

    Layout: 4 rows × 3 cols
    """

    # ── palette & style ─────────────────────────────────────────────────────
    BG      = '#0f1117'
    PANEL   = '#1a1d27'
    ACCENT  = '#e8d5b7'        # warm parchment
    C1      = '#6fc3df'        # blue
    C2      = '#f4a261'        # amber
    C3      = '#8ecb8e'        # sage
    C4      = '#e06c75'        # rose
    GREY    = '#3a3d4d'
    WHITE   = '#e8e8e8'

    PATTERN_COLORS = [C3, C1, C2, C4]   # unanimous → tie → all different
    ANN_COLORS     = [C1, C2, C3, C4]

    plt.rcParams.update({
        'font.family':       'serif',
        'axes.facecolor':    PANEL,
        'figure.facecolor':  BG,
        'text.color':        WHITE,
        'axes.labelcolor':   WHITE,
        'xtick.color':       WHITE,
        'ytick.color':       WHITE,
        'axes.edgecolor':    GREY,
        'grid.color':        GREY,
        'grid.alpha':        0.3,
    })

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(BG)

    # ── title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, 'Inter-Annotator Agreement Analysis',
             ha='center', va='top', fontsize=20, fontweight='bold',
             color=ACCENT, fontfamily='serif')
    fig.text(0.5, 0.960,
             f'N=100 swimming pool reviews · 4 annotators · 3 label categories',
             ha='center', va='top', fontsize=11, color=WHITE, alpha=0.7)

    # ── alpha interpretation band helper ────────────────────────────────────
    ALPHA_BANDS = [
        (0.8,  1.0,  '#264d26', 'Excellent (≥0.80)'),
        (0.67, 0.8,  '#3d4d1a', 'Good (0.67–0.80)'),
        (0.0,  0.67, '#4d2a1a', 'Tentative (<0.67)'),
    ]

    # ────────────────────────────────────────────────────────────────────────
    # ROW 0 — Krippendorff's Alpha summary (spanning full width)
    # ────────────────────────────────────────────────────────────────────────
    ax_alpha = fig.add_axes([0.08, 0.875, 0.84, 0.07])
    ax_alpha.set_facecolor(PANEL)

    cat_labels   = [c.capitalize() for c in CATEGORIES]
    alpha_values = [agreement_results[c]['alpha'] for c in CATEGORIES]
    bar_colors   = []
    for a in alpha_values:
        if np.isnan(a):          bar_colors.append(GREY)
        elif a >= 0.80:          bar_colors.append(C3)
        elif a >= 0.67:          bar_colors.append(C1)
        else:                    bar_colors.append(C4)

    bars = ax_alpha.barh(cat_labels, alpha_values, color=bar_colors,
                         height=0.5, zorder=3)

    # interpretation bands
    for lo, hi, col, _ in ALPHA_BANDS:
        ax_alpha.axvspan(lo, hi, color=col, alpha=0.25, zorder=1)

    ax_alpha.axvline(0.8,  color=C3, lw=1, ls='--', alpha=0.6, zorder=2)
    ax_alpha.axvline(0.67, color=C1, lw=1, ls='--', alpha=0.6, zorder=2)
    ax_alpha.set_xlim(-0.1, 1.05)
    ax_alpha.set_xlabel("Krippendorff's α  (nominal)", fontsize=10)
    ax_alpha.set_title("Krippendorff's Alpha per Category", fontsize=12,
                       color=ACCENT, pad=8)
    ax_alpha.grid(axis='x', zorder=0)

    for bar, val in zip(bars, alpha_values):
        label = f'{val:.3f}' if not np.isnan(val) else 'N/A'
        ax_alpha.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                      label, va='center', fontsize=11, fontweight='bold', color=WHITE)

    # legend for bands
    legend_patches = [mpatches.Patch(color=col, alpha=0.5, label=lbl)
                      for _, _, col, lbl in ALPHA_BANDS]
    ax_alpha.legend(handles=legend_patches, loc='lower right',
                    fontsize=8, framealpha=0.3, labelcolor=WHITE)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 1 — Agreement patterns  (3 cols)
    # ────────────────────────────────────────────────────────────────────────
    pattern_keys = ['Unanimous (4/4)', 'Majority (3/4)', 'Tie (2-2)', 'All different']

    for col_idx, cat in enumerate(CATEGORIES):
        left = 0.08 + col_idx * 0.30
        ax = fig.add_axes([left, 0.700, 0.25, 0.14])
        ax.set_facecolor(PANEL)

        vals  = [patterns[cat][k] for k in pattern_keys]
        total = sum(vals) or 1
        pcts  = [v / total * 100 for v in vals]

        cum = 0
        for k_idx, (k, v, p) in enumerate(zip(pattern_keys, vals, pcts)):
            bar = ax.barh(0, p, left=cum, color=PATTERN_COLORS[k_idx],
                          height=0.5, label=k, zorder=3)
            if p > 5:
                ax.text(cum + p/2, 0, f'{v}\n({p:.0f}%)',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            cum += p

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('% of reviews', fontsize=9)
        ax.set_title(f'{cat.capitalize()}', fontsize=12, color=ACCENT)
        ax.grid(axis='x', alpha=0.3, zorder=0)

        if col_idx == 0:
            ax.set_ylabel('Agreement\nPatterns', fontsize=9)

    # pattern legend (once, outside)
    legend_patches2 = [mpatches.Patch(color=PATTERN_COLORS[i], label=k)
                       for i, k in enumerate(pattern_keys)]
    fig.legend(handles=legend_patches2, loc='upper right',
               bbox_to_anchor=(0.98, 0.85), fontsize=8,
               framealpha=0.25, labelcolor=WHITE, title='Agreement',
               title_fontsize=8)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 2 — Pairwise kappa heatmaps  (3 cols)
    # ────────────────────────────────────────────────────────────────────────
    pairwise = get_pairwise_kappa_matrix
    # already passed in as pairwise_matrices

    for col_idx, cat in enumerate(CATEGORIES):
        left = 0.08 + col_idx * 0.30
        ax = fig.add_axes([left, 0.490, 0.24, 0.185])
        ax.set_facecolor(PANEL)

        mat = pairwise_matrices[cat]
        n   = len(annotators)
        data = mat.values.astype(float)

        im = ax.imshow(data, cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                       aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([a.capitalize() for a in annotators],
                           rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels([a.capitalize() for a in annotators], fontsize=8)

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                txt = f'{val:.2f}' if not np.isnan(val) else '—'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='black' if (not np.isnan(val) and val > 0.3) else WHITE)

        ax.set_title(f'{cat.capitalize()} — pairwise κ', fontsize=11, color=ACCENT)
        if col_idx == 0:
            ax.set_ylabel("Cohen's κ\n(pairwise, supplementary)", fontsize=8)

    # ────────────────────────────────────────────────────────────────────────
    # ROW 3 — Label distribution per annotator  (3 cols × 4 small bars)
    # ────────────────────────────────────────────────────────────────────────
    for col_idx, cat in enumerate(CATEGORIES):
        left_outer = 0.08 + col_idx * 0.30

        # Collect all unique labels across all annotators for this category
        all_labels = sorted(set(
            lbl for ann in annotators
            for lbl in label_distributions[cat][ann].keys()
        ))

        sub_w = 0.055
        sub_h = 0.13
        gap   = 0.005

        for ann_idx, ann in enumerate(annotators):
            sub_left = left_outer + ann_idx * (sub_w + gap)
            ax = fig.add_axes([sub_left, 0.295, sub_w, sub_h])
            ax.set_facecolor(PANEL)

            counts = [label_distributions[cat][ann].get(lbl, 0) for lbl in all_labels]
            ys     = range(len(all_labels))

            ax.barh(list(ys), counts, color=ANN_COLORS[ann_idx],
                    height=0.6, zorder=3)
            ax.set_yticks(list(ys))
            ax.set_yticklabels(all_labels, fontsize=7)
            ax.set_xticks([])
            ax.set_title(ann.capitalize(), fontsize=8, color=ANN_COLORS[ann_idx], pad=3)
            ax.grid(axis='x', alpha=0.2)

            for y, c in zip(ys, counts):
                if c > 0:
                    ax.text(c + 0.3, y, str(c), va='center', fontsize=6, color=WHITE)

            if ann_idx == 0 and col_idx == 0:
                ax.set_ylabel('Label', fontsize=7)

        # category sub-title
        fig.text(left_outer + 0.12, 0.435,
                 f'{cat.capitalize()} — label distributions per annotator',
                 ha='center', fontsize=9, color=ACCENT)

    # ────────────────────────────────────────────────────────────────────────
    # ROW LABELS (left margin)
    # ────────────────────────────────────────────────────────────────────────
    row_labels = [
        (0.735, "Agreement\npatterns"),
        (0.575, "Pairwise\nCohen's κ"),
        (0.360, "Label\ndistributions"),
    ]
    for y, label in row_labels:
        fig.text(0.01, y, label, va='center', ha='left', fontsize=9,
                 color=GREY, rotation=90, style='italic')

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  ✓ Plot saved → {output_path}")


# ─────────────────────────────────────────────
# 6. SAVE AGREEMENT SUMMARY
# ─────────────────────────────────────────────

def save_agreement_summary(agreement_results: dict,
                            csv_path: str):
    rows = []
    for cat, res in agreement_results.items():
        rows.append({
            'category':          cat,
            'krippendorff_alpha': res['alpha'],
            'n_items_total':     res['n_items'],
            'n_items_valid':     res['n_valid'],
            'interpretation':    (
                'Excellent' if res['alpha'] >= 0.80 else
                'Good'      if res['alpha'] >= 0.67 else
                'Tentative' if not np.isnan(res['alpha']) else 'N/A'
            )
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Agreement summary → {csv_path} ")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    annotators = list(ANNOTATOR_FILES.keys())

    print("Loading and cleaning data …")
    merged_df = load_and_clean(ANNOTATOR_FILES)
    print(f"  ✓ {len(merged_df)} reviews loaded for {len(annotators)} annotators")

    print("\nCalculating Krippendorff's Alpha …")
    agreement_results = calculate_krippendorff(merged_df, annotators)
    for cat, res in agreement_results.items():
        print(f"  {cat.upper()}: α = {res['alpha']:.4f}  "
              f"(n_valid = {res['n_valid']})")

    print("\nBuilding ground truth via majority vote …")
    output_df = create_ground_truth(merged_df, annotators)
    tie_counts = {cat: (output_df[f'gt_{cat}'] == 'tie').sum()
                  for cat in CATEGORIES}
    for cat, n in tie_counts.items():
        print(f"  {cat.upper()}: {n} ties")
    output_df.to_csv(OUTPUT_LABELS_CSV, index=False)
    print(f"  ✓ Saved → {OUTPUT_LABELS_CSV}")

    print("\nSaving agreement summary …")
    save_agreement_summary(agreement_results, OUTPUT_AGREEMENT_CSV)

    print("\nGenerating publication figure …")
    patterns   = get_agreement_patterns(merged_df, annotators)
    label_dist = get_label_distributions(merged_df, annotators)
    pw_kappas  = get_pairwise_kappa_matrix(merged_df, annotators)
    make_publication_figure(
        agreement_results, patterns, pw_kappas, label_dist,
        annotators, OUTPUT_PLOT
    )

    print("\nDone.")
    print(f"  Output files:")
    print(f"    {OUTPUT_LABELS_CSV}    ← reviews + annotator labels + ground truth")
    print(f"    {OUTPUT_AGREEMENT_CSV} ← Krippendorff alpha per category")
    print(f"    {OUTPUT_PLOT}          ← publication figure")


if __name__ == '__main__':
    main()