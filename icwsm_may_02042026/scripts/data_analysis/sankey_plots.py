import pandas as pd
import plotly.graph_objects as go

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_parquet(
    "/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/data/analysis/merged_700k_classified_with_ethnicity_14012026.parquet"
)

# ── 2. Count flows ───────────────────────────────────────────────────────────
total = len(df)
social_yes = (df["social_encounter"] == "yes").sum()
social_no = (df["social_encounter"] == "no").sum()
racial_yes = (df["racial_connotations"] == "yes").sum()
racial_no = (df["racial_connotations"] == "no").sum()
discrim = (df["connotation_type"] == "discriminatory").sum()
pleasant = (df["connotation_type"] == "pleasant").sum()
ambiguous = (df["connotation_type"] == "ambiguous").sum()

# ── 3. Node labels with percentages ──────────────────────────────────────────
def label(title, n, denom):
    pct = 100 * n / denom
    return f"<b>{title}</b><br>{n:,}<br>({pct:.1f}%)"

node_labels = [
    label("All Reviews", total, total),                                    # 0
    label("Social Encounter", social_yes, total),                          # 1
    label("Racial Connotation", racial_yes, social_yes),                   # 2
    label("No Racial Connotation", racial_no, social_yes),                 # 3
    label("Discriminatory", discrim, racial_yes),                          # 4
    label("Pleasant", pleasant, racial_yes),                               # 5
    label("Ambiguous", ambiguous, racial_yes),                             # 6
    label("No Social Encounter", social_no, total),                        # 7 (separate branch)
]

# ── 4. Colors (more distinct, better contrast) ───────────────────────────────
node_colors = [
    "#2C3E50",   # 0 All Reviews (dark slate)
    "#3498DB",   # 1 Social Encounter (blue)
    "#E67E22",   # 2 Racial (orange)
    "#95A5A6",   # 3 Not Racial (gray)
    "#C0392B",   # 4 Discriminatory (deep red)
    "#27AE60",   # 5 Pleasant (green)
    "#8E44AD",   # 6 Ambiguous (purple)
    "#7F8C8D",   # 7 No Social Encounter (muted gray)
]

def rgba_from_hex(h, alpha=0.35):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── 5. Links — restructured for better visual spread ─────────────────────────
link_source = [0, 0,   1, 1,   2, 2, 2]
link_target = [1, 7,   2, 3,   4, 5, 6]
link_value  = [social_yes, social_no, racial_yes, racial_no,
               discrim, pleasant, ambiguous]
link_colors = [rgba_from_hex(node_colors[t], alpha=0.4) for t in link_target]

# ── 6. Figure with optimized layout ──────────────────────────────────────────
fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=40,                 # Increased padding between nodes (only once!)
        thickness=28,           # Thicker bands
        line=dict(color="white", width=1.2),
        label=node_labels,
        color=node_colors,
        x=[0.05, 0.35, 0.65, 0.65, 0.92, 0.92, 0.92, 0.35],  # Manual x positions
        y=[0.5,  0.3,  0.2,  0.6,  0.1,  0.4,  0.7,  0.8],   # Manual y spread
        hovertemplate="%{label}<extra></extra>",
    ),
    link=dict(
        source=link_source,
        target=link_target,
        value=link_value,
        color=link_colors,
        hovertemplate="%{source.label} → %{target.label}<br>n = %{value:,}<extra></extra>",
    ),
))

fig.update_layout(
    title=dict(
        text="Hierarchical Classification of Online Reviews",
        font=dict(size=30, family="Times New Roman, serif", color="#1a1a2e"),
        x=0.5,
        xanchor="center",
        y=0.96,
    ),
    font=dict(size=30, family="Times New Roman, serif", color="#2C3E50"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    width=1200,      # Wider canvas
    height=650,      # Taller to spread vertical bands
    margin=dict(l=50, r=50, t=70, b=30),
)

# ── 7. Save ───────────────────────────────────────────────────────────────────
fig.write_image(
    "/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/plots/sankey_classification_reviews.png",
    scale=3,
    width=1200,
    height=650,
)
print("PNG saved (improved layout).")

# # Optional HTML backup
# fig.write_html(
#     "/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/plots/sankey_classification_reviews.html"
# )
# print("HTML saved.")

# fig.show()

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='white')

# # Level 1: Social encounter
# axes[0].barh([0], [social_no], color='#7F8C8D', label=f'No Social: {social_no:,} ({social_no/total*100:.1f}%)')
# axes[0].barh([0], [social_yes], left=[social_no], color='#3498DB', label=f'Social: {social_yes:,} ({social_yes/total*100:.1f}%)')
# axes[0].set_title('Level 1: Social Encounter', fontsize=12, weight='bold')
# axes[0].set_xlim(0, total)
# axes[0].set_xlabel('Number of Reviews')
# axes[0].legend(loc='upper right', fontsize=9)

# # Level 2: Racial connotations (among social=yes)
# axes[1].barh([0], [racial_no], color='#95A5A6', label=f'Not Racial: {racial_no:,} ({racial_no/social_yes*100:.1f}%)')
# axes[1].barh([0], [racial_yes], left=[racial_no], color='#E67E22', label=f'Racial: {racial_yes:,} ({racial_yes/social_yes*100:.1f}%)')
# axes[1].set_title('Level 2: Racial Connotations\n(among Social Encounter)', fontsize=12, weight='bold')
# axes[1].set_xlim(0, social_yes)
# axes[1].set_xlabel('Number of Reviews')
# axes[1].legend(loc='upper right', fontsize=9)

# # Level 3: Connotation types (among racial=yes)
# types = ['Discriminatory', 'Pleasant', 'Ambiguous']
# counts = [discrim, pleasant, ambiguous]
# colors = ['#C0392B', '#27AE60', '#8E44AD']
# bars = axes[2].bar(types, counts, color=colors, edgecolor='white', linewidth=1.5)
# axes[2].set_title('Level 3: Connotation Type\n(among Racial Connotations)', fontsize=12, weight='bold')
# axes[2].set_ylabel('Number of Reviews')
# axes[2].set_yscale('log')  # Log scale to show small counts clearly

# # Add value labels on bars
# for bar, val in zip(bars, counts):
#     axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
#                 f'{val:,}\n({val/racial_yes*100:.1f}%)',
#                 ha='center', va='bottom', fontsize=9, weight='bold')

# plt.tight_layout()
# plt.savefig('/home/ethel/Desktop/swimming_pool_project/icwsm_may_02042026/plots/stacked_hierarchy.png',
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()