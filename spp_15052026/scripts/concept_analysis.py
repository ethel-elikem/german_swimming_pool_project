"""
concept_analysis_v2.py
────────────────────────────────────────────────────────────────────────────────
Concept-level visualisation for pre-labelled review data (parquet).

Labels
  • theme       : crowd | etiquette | belonging | hygiene | clothing | staff |
                  language | access | safety | other | security  (single per row)
  • context     : generalised | incident
  • connotation : negative | neutral | positive
  • approach    : implicit | explicit | both

Text column : trans_review

Outputs (all saved to ./outputs/)
  • Individual interactive HTML charts
  • report.html         — single self-contained page with ALL charts + styled nav
  • outputs/png/        — static PNG of every chart (use --export-png flag)

Usage:
    python concept_analysis_v2.py data.parquet
    python concept_analysis_v2.py data.parquet --export-png

Install dependencies (once):
    pip install pandas pyarrow sentence-transformers umap-learn bertopic
                keybert plotly kaleido
"""

import os, sys, warnings, textwrap
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── constants ─────────────────────────────────────────────────────────────────
TEXT_COL        = "trans_review"
THEME_COL       = "theme"
CONTEXT_COL     = "context"
CONNOTATION_COL = "connotation"
APPROACH_COL    = "approach"
LABEL_COLS      = [THEME_COL, CONTEXT_COL, CONNOTATION_COL, APPROACH_COL]

OUTPUT_DIR      = "./concept_analysis_outputs"
PNG_DIR         = "./concept_analysis_outputs/png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPORT_PNG      = "--export-png" in sys.argv

PALETTE = px.colors.qualitative.Bold   # consistent colour set across charts

# ── global figure registry (for report assembly) ─────────────────────────────
FIGURES: list[dict] = []   # {"section": str, "title": str, "fig": go.Figure}

def register(section: str, title: str, fig: go.Figure, filename: str):
    """Save individual HTML, optionally PNG, and register for report."""
    html_path = f"{OUTPUT_DIR}/{filename}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    print(f"   Saved → {html_path}")

    if EXPORT_PNG:
        os.makedirs(PNG_DIR, exist_ok=True)
        png_path = f"{PNG_DIR}/{filename}.png"
        try:
            fig.write_image(png_path, width=1400, height=700, scale=2)
            print(f"   Saved → {png_path}")
        except Exception as e:
            print(f"   PNG export failed for '{filename}': {e}")

    FIGURES.append({"section": section, "title": title, "fig": fig})


# ══════════════════════════════════════════════════════════════════════════════
# 0.  LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Loading  {path}")
    df = pd.read_parquet(path)
    df = df.dropna(subset=[TEXT_COL])
    df = df.dropna(subset=LABEL_COLS, how="all")

    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), other=None)

    df_themed = df.dropna(subset=[THEME_COL]).copy()
    for col in LABEL_COLS:
        if col in df_themed.columns:
            df_themed[col] = df_themed[col].str.strip().str.lower()

    print(f"   Rows after cleaning : {len(df_themed):,}")
    print(f"   Theme classes       : {sorted(df_themed[THEME_COL].unique())}")
    return df_themed


# ══════════════════════════════════════════════════════════════════════════════
# 1.  UMAP
# ══════════════════════════════════════════════════════════════════════════════

def run_umap(df: pd.DataFrame):
    print("\n[2/5] UMAP embedding clusters …")
    from sentence_transformers import SentenceTransformer
    import umap

    model     = SentenceTransformer("all-MiniLM-L6-v2")
    texts     = df[TEXT_COL].tolist()
    print("   Encoding …")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print("   UMAP reduction …")
    coords = umap.UMAP(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.1,
                       metric="cosine").fit_transform(embeddings)

    df_plot          = df[[TEXT_COL, THEME_COL, CONTEXT_COL,
                           CONNOTATION_COL, APPROACH_COL]].copy()
    df_plot["x"]     = coords[:, 0]
    df_plot["y"]     = coords[:, 1]
    df_plot["hover"] = df_plot[TEXT_COL].str[:120] + "…"

    fig = px.scatter(
        df_plot, x="x", y="y",
        color=THEME_COL,
        hover_data={"hover": True, CONNOTATION_COL: True,
                    APPROACH_COL: True, "x": False, "y": False},
        title="UMAP — Review Embeddings coloured by Theme",
        template="plotly_dark",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(legend_title_text="Theme", height=750,
                      font=dict(size=14))
    register("Embedding Space", "UMAP — Theme Clusters", fig, "umap_embeddings")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BERTOPIC PER THEME
# ══════════════════════════════════════════════════════════════════════════════

def run_bertopic(df: pd.DataFrame):
    print("\n[3/5] BERTopic per theme …")
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    model  = SentenceTransformer("all-MiniLM-L6-v2")
    themes = sorted(df[THEME_COL].dropna().unique())

    for theme in themes:
        docs = df[df[THEME_COL] == theme][TEXT_COL].dropna().tolist()
        if len(docs) < 10:
            print(f"   Skipping '{theme}' — {len(docs)} docs (need ≥10)")
            continue

        print(f"   BERTopic '{theme}' ({len(docs)} docs) …")
        try:
            tm = BERTopic(embedding_model=model, nr_topics="auto",
                          verbose=False, calculate_probabilities=False)
            tm.fit_transform(docs)

            fig = tm.visualize_barchart(
                top_n_topics=min(8, len(tm.get_topics())),
                title=f"BERTopic — Theme: {theme.title()}",
                n_words=8,
            )
            fig.update_layout(template="plotly_dark", height=620,
                               font=dict(size=13))
            register("BERTopic per Theme",
                     f"BERTopic — {theme.title()}",
                     fig, f"bertopic_{theme}")
        except Exception as e:
            print(f"   Failed for '{theme}': {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SANKEY + HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def run_sankey(df: pd.DataFrame):
    print("\n[4/5] Sankey + heatmap …")
    df_s = df.dropna(subset=[THEME_COL, CONNOTATION_COL, APPROACH_COL, CONTEXT_COL]).copy()

    themes       = sorted(df_s[THEME_COL].unique())
    connotations = sorted(df_s[CONNOTATION_COL].unique())
    approaches   = sorted(df_s[APPROACH_COL].unique())
    contexts     = sorted(df_s[CONTEXT_COL].unique())

    nodes    = themes + connotations + approaches + contexts
    node_idx = {n: i for i, n in enumerate(nodes)}
    colors   = (["#4E79A7"] * len(themes) +
                ["#F28E2B"] * len(connotations) +
                ["#59A14F"] * len(approaches) +
                ["#E15759"] * len(contexts))

    src, tgt, val = [], [], []
    for _, r in df_s.groupby([THEME_COL, CONNOTATION_COL]).size() \
                    .reset_index(name="n").iterrows():
        src.append(node_idx[r[THEME_COL]])
        tgt.append(node_idx[r[CONNOTATION_COL]])
        val.append(r["n"])
    for _, r in df_s.groupby([CONNOTATION_COL, APPROACH_COL]).size() \
                    .reset_index(name="n").iterrows():
        src.append(node_idx[r[CONNOTATION_COL]])
        tgt.append(node_idx[r[APPROACH_COL]])
        val.append(r["n"])
    for _, r in df_s.groupby([APPROACH_COL, CONTEXT_COL]).size() \
                    .reset_index(name="n").iterrows():
        src.append(node_idx[r[APPROACH_COL]])
        tgt.append(node_idx[r[CONTEXT_COL]])
        val.append(r["n"])
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=colors, pad=22, thickness=22,
                  line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
        link=dict(source=src, target=tgt, value=val,
                  color="rgba(200,200,200,0.25)"),
    ))
    fig.update_layout(title_text="Review Flow: Theme → Connotation → Approach → Context",
                      font=dict(size=14), template="plotly_dark", height=720)
    register("Flow & Co-occurrence", "Sankey — Theme → Connotation → Approach → Context",
             fig, "sankey_theme_connotation_context")

    # heatmap
    pivot = df_s.groupby([THEME_COL, CONNOTATION_COL]).size().unstack(fill_value=0)
    fig2  = px.imshow(pivot, text_auto=True,
                      color_continuous_scale="Blues",
                      title="Heatmap — Theme × Connotation counts",
                      template="plotly_dark", aspect="auto")
    fig2.update_layout(height=560, font=dict(size=14),
                       coloraxis_colorbar=dict(title="Count"))
    register("Flow & Co-occurrence", "Heatmap — Theme × Connotation",
             fig2, "cooccurrence_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  KEYBERT
# ══════════════════════════════════════════════════════════════════════════════

def run_keybert(df: pd.DataFrame):
    print("\n[5/5] KeyBERT keyphrases …")
    from keybert import KeyBERT

    kb     = KeyBERT(model="all-MiniLM-L6-v2")
    themes = sorted(df[THEME_COL].dropna().unique())

    all_results = {}
    for theme in themes:
        docs = df[df[THEME_COL] == theme][TEXT_COL].dropna().tolist()
        if len(docs) < 5:
            continue
        kws = kb.extract_keywords(
            " ".join(docs[:200]),
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=15,
            use_mmr=True,
            diversity=0.5,
        )
        all_results[theme] = kws
        print(f"   '{theme}': {[k for k, _ in kws[:4]]}")

    n    = len(all_results)
    cols = 2
    rows = (n + 1) // cols
    fig  = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[t.title() for t in all_results],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )
    for idx, (theme, kws) in enumerate(all_results.items()):
        r, c = divmod(idx, cols)
        labels = [k for k, _ in reversed(kws)]
        scores = [s for _, s in reversed(kws)]
        fig.add_trace(
            go.Bar(x=scores, y=labels, orientation="h",
                   marker_color=PALETTE[idx % len(PALETTE)],
                   showlegend=False, name=theme),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(title_text="KeyBERT — Top Keyphrases per Theme",
                      template="plotly_dark",
                      height=320 * rows,
                      font=dict(size=13))
    register("Keyphrases", "KeyBERT — Top Keyphrases per Theme",
             fig, "keyphrases_themes")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SINGLE-PAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════

REPORT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0d0f14;
  --surface:   #161a23;
  --border:    #252b38;
  --accent:    #4e9fff;
  --accent2:   #f28e2b;
  --text:      #e8eaf0;
  --muted:     #7a8299;
  --radius:    12px;
  --nav-w:     240px;
}

html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-weight: 300;
  min-height: 100vh;
  display: flex;
}

/* ── sidebar nav ── */
#nav {
  position: fixed; top: 0; left: 0;
  width: var(--nav-w); height: 100vh;
  background: var(--surface);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 32px 0 24px;
  z-index: 100;
  display: flex; flex-direction: column; gap: 0;
}
#nav .logo {
  font-family: 'DM Serif Display', serif;
  font-size: 1.15rem;
  color: var(--accent);
  padding: 0 24px 28px;
  border-bottom: 1px solid var(--border);
  line-height: 1.3;
}
#nav .logo span { display: block; font-family: 'DM Sans', sans-serif;
  font-size: .72rem; color: var(--muted); font-weight: 400; margin-top: 4px; }
#nav .section-label {
  font-size: .68rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted);
  padding: 22px 24px 6px;
}
#nav a {
  display: block; padding: 8px 24px;
  color: var(--text); text-decoration: none;
  font-size: .82rem; font-weight: 400;
  border-left: 3px solid transparent;
  transition: all .18s ease;
  line-height: 1.4;
}
#nav a:hover, #nav a.active {
  color: var(--accent);
  border-left-color: var(--accent);
  background: rgba(78,159,255,.07);
}

/* ── main ── */
#main {
  margin-left: var(--nav-w);
  flex: 1;
  padding: 48px 48px 80px;
  max-width: calc(100vw - var(--nav-w));
}

/* hero */
.hero {
  padding: 56px 0 48px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 64px;
}
.hero h1 {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1.15;
  color: var(--text);
}
.hero h1 em { color: var(--accent); font-style: normal; }
.hero p {
  margin-top: 14px;
  color: var(--muted);
  font-size: .95rem;
  max-width: 620px;
  line-height: 1.7;
}
.pills {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-top: 22px;
}
.pill {
  background: rgba(78,159,255,.12);
  color: var(--accent);
  border: 1px solid rgba(78,159,255,.3);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: .78rem;
  font-weight: 500;
}

/* section */
.section-block {
  margin-bottom: 80px;
}
.section-block h2 {
  font-family: 'DM Serif Display', serif;
  font-size: 1.6rem;
  color: var(--text);
  padding-bottom: 10px;
  border-bottom: 2px solid var(--accent2);
  margin-bottom: 36px;
  display: inline-block;
}

/* chart card */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 32px;
  box-shadow: 0 4px 24px rgba(0,0,0,.35);
  transition: box-shadow .2s;
}
.chart-card:hover {
  box-shadow: 0 8px 40px rgba(78,159,255,.12);
}
.chart-header {
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
}
.chart-header h3 {
  font-size: .95rem;
  font-weight: 500;
  color: var(--text);
}
.chart-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
}
.chart-body { padding: 8px 4px 4px; }
.chart-body iframe {
  width: 100%; border: none;
  min-height: 520px;
  display: block;
}

/* footer */
footer {
  margin-top: 80px;
  padding-top: 32px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: .8rem;
}
</pre>
"""

def build_report():
    print("\n[6/6] Building single-page report …")

    # Group figures by section
    sections: dict[str, list[dict]] = {}
    for entry in FIGURES:
        sections.setdefault(entry["section"], []).append(entry)

    def slug(text: str) -> str:
        return text.lower().replace(" ", "-").replace("/", "-") \
                   .replace("→", "").replace("  ", "-")

    # ── nav HTML ─────────────────────────────────────────────────────────────
    nav_html = ['<div class="logo">Review Concept Analysis<span>Label insight report</span></div>']
    for sec, items in sections.items():
        nav_html.append(f'<div class="section-label">{sec}</div>')
        for item in items:
            sid = slug(item["title"])
            nav_html.append(f'<a href="#{sid}">{item["title"]}</a>')

    # ── pills (theme list) ────────────────────────────────────────────────────
    themes = ["crowd","etiquette","belonging","hygiene","clothing",
              "staff","language","access","safety","other","security"]
    pills_html = "".join(f'<span class="pill">{t}</span>' for t in themes)

    # ── chart sections ────────────────────────────────────────────────────────
    body_html = []
    for sec, items in sections.items():
        sec_slug = slug(sec)
        body_html.append(f'<div class="section-block" id="{sec_slug}">')
        body_html.append(f'<h2>{sec}</h2>')
        for item in items:
            sid      = slug(item["title"])
            # embed figure as inline iframe via srcdoc
            inner    = item["fig"].to_html(include_plotlyjs="cdn",
                                           full_html=True,
                                           config={"responsive": True})
            # escape for srcdoc attribute
            inner_esc = inner.replace("&", "&amp;").replace('"', "&quot;")
            body_html.append(f"""
<div class="chart-card" id="{sid}">
  <div class="chart-header">
    <div class="chart-dot"></div>
    <h3>{item['title']}</h3>
  </div>
  <div class="chart-body">
    <iframe srcdoc="{inner_esc}" loading="lazy"
            style="min-height:620px"></iframe>
  </div>
</div>""")
        body_html.append("</div>")

    # ── assemble ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Review Concept Analysis Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>

<nav id="nav">
  {''.join(nav_html)}
</nav>

<main id="main">
  <div class="hero">
    <h1>Review<br/><em>Concept Analysis</em></h1>
    <p>Interactive visualisations of labelled review data — explore semantic
       clusters, sub-topic structures, label flows and keyphrase patterns.</p>
    <div class="pills">{pills_html}</div>
  </div>

  {''.join(body_html)}

  <footer>
    Generated by concept_analysis_v2.py &nbsp;·&nbsp;
    Themes: {len(themes)} classes &nbsp;·&nbsp;
    All charts interactive — hover, zoom, click legends to filter.
  </footer>
</main>

<script>
// Highlight active nav link on scroll
const links = document.querySelectorAll('#nav a');
const observer = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      links.forEach(l => l.classList.remove('active'));
      const match = document.querySelector(`#nav a[href="#${{e.target.id}}"]`);
      if (match) match.classList.add('active');
    }}
  }});
}}, {{ threshold: 0.3 }});
document.querySelectorAll('[id]').forEach(el => observer.observe(el));
</script>
</body>
</html>"""

    out = f"{OUTPUT_DIR}/report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   Saved → {out}")
    print(f"   Open in any browser — no server needed.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python concept_analysis.py <data.parquet> [--export-png]")
        sys.exit(1)

    df_clean = load_data(args[0])

    run_umap(df_clean)
    run_bertopic(df_clean)
    run_sankey(df_clean)
    run_keybert(df_clean)
    build_report()

    print(f"\n✅  Done!")
    print(f"   Report  → {os.path.abspath(OUTPUT_DIR)}/report.html")
    if EXPORT_PNG:
        print(f"   PNGs    → {os.path.abspath(PNG_DIR)}/")