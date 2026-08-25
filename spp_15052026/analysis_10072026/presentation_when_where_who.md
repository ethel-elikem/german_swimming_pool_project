# Racialised Discourse in German Swimming Pool Reviews — WHEN / WHERE / WHO

*Descriptive findings only. No socioeconomic or political data used anywhere in this document — that layer comes later.*

---

## Preamble: the data we're working with

**Corpus.** 1,265,806 text reviews across 5,529 verified German swimming pools. Classified hierarchically by a single LLM classifier: **504,072 (39.8%) are `social`** (about people/staff/atmosphere, not just facilities); of those, **7,768 (1.54%) are `racialised`** (attribute something to an ethnic/national/cultural/religious group).

**Racialised rate — the one number used throughout this whole document, defined once, here.** `racialised rate = racialised reviews / social reviews` (not `/ total reviews`) for whatever unit is being measured (corpus-wide, per year, per constituency, per group). Not `/ total`, because ~60% of reviews are about facilities/pricing/hours and were never in scope for racialised-discourse classification at all — dividing by total would dilute the rate by how chatty/social a unit's reviewers happen to be, which is a different thing from how racialised their social commentary is. Every "rate" figure below — WHEN's yearly trend, WHERE's constituency/state numbers, WHO's group comparisons — uses this same definition; not restated per-section.

**Review-timestamp coverage — how we got to a trustworthy WHEN.**
Every WHEN claim below depends on `review_timestamp`, which was not fully populated to start. Two separate fixes got us to (effectively) complete:

1. **Upstream DB refresh** (not something we computed — a fix to the source `gmaps_poi` database): lifted timestamp coverage from **69.5% → 99.5%** corpus-wide, before we touched anything.
2. **Our own recovery step**: for the rows still missing a timestamp after the refresh, we parsed the relative-time text Google shows (e.g. *"2 years ago"*) and anchored it to the pool's scrape date. This is an estimate (day-level precision), not a re-parse of an exact timestamp — flagged as `timestamp_is_estimated` everywhere it's used, and dropped back to actual-only timestamps for the finer month-level analysis (see WHEN below).

| Grain | n | still missing after recovery |
|---|---|---|
| Entire corpus | 1,265,806 | 5,762 (0.46%) | 
| `social == yes` | 504,072 | 2,910 (0.58%) | 
| `racialised == yes` | 7,768 | 138 (1.78%) | 



---

## WHEN — is racialised discourse increasing over time?

**Figure 1. Racialised rate per year (top) vs. review volume, actual + estimated (bottom)**
`descriptive_analysis_figures/racialised_rate_yearly_timestamped_social_reviews.png`

![Racialised rate over time](descriptive_analysis_figures/racialised_rate_yearly_timestamped_social_reviews.png)

**Figures 2–3. Indoor vs. outdoor, and Summer vs. other months, rate by year** (grey band = year where that season gap is statistically significant)

<table>
<tr>
<td align="center" width="50%">
<img src="descriptive_analysis_figures/indoor_outdoor_rate_by_year.png" width="380"><br>
<sub>Fig. 2 — indoor vs. outdoor, by year</sub>
</td>
<td align="center" width="50%">
<img src="descriptive_analysis_figures/summer_other_rate_by_year.png" width="380"><br>
<sub>Fig. 3 — summer vs. other months, by year</sub>
</td>
</tr>
</table>

**Reading Figures 2–3 together:** both gaps are recent (open 2018–19, close in 2020–21 — plausibly COVID — then widen every year from 2022 on) the two ways of asking "which rises faster" give different answers, so state both rather than picking one:
- **By 2025's absolute gap:** Summer vs. Other (≈1.4pp: 3.05% vs. 1.64%) is *wider* than Indoor vs. Outdoor (1.14pp).
- **By each side's own year-on-year trend coefficient:** Outdoor's own slope is **~2.9x** Indoor's (vs. Summer's slope being **~2.4x** Other's) — so indoor/outdoor is the *relatively* faster-accelerating split, even though summer/other has the bigger gap at the end.

### Content dimensions over time (racialised subset, N=7,768)

| Dim. | Label | Indoor % | Outdoor % | Summer % | Other % | Trend 2021–25 (log-odds/yr) |
|---|---|---|---|---|---|---|
| Theme | Crowd | 45.6* | 47.0* | 48.9** | 45.3** | −0.066** (declining) |
| Theme | Etiquette | 33.6*** | 28.6*** | 28.5*** | 32.4*** | −0.039 (ns) |
| Theme | Staff | 26.6 | 25.3 | 24.5* | 27.0* | +0.023 (ns) |
| Theme | Belonging | 19.8 | 19.3 | 19.4 | 19.6 | +0.049 (ns) |
| Theme | **Clothing** | 12.7*** | 18.2*** | 19.4*** | 12.9*** | **+0.212*** (steepest rise of any label)** |
| Context | Generalised | 83.5 | 83.6 | 84.0 | 83.1 | −0.097*** |
| Context | Incident | 40.8 | 41.5 | 41.4 | 41.2 | +0.111*** (rising) |
| Connotation | Negative | 88.1 | 88.5 | 88.8 | 87.9 | +0.216*** (rising) |
| Connotation | Neutral | 9.5 | 8.3 | 8.0 | 9.5* | −0.177*** |
| Connotation | Positive | 3.8 | 4.7 | 4.7 | 4.0 | −0.129* |
| Approach | Explicit | 80.2** | 76.9** | 76.8** | 79.4** | −0.070** |
| Approach | Implicit | 13.4*** | 17.3*** | 17.4*** | 14.4*** | +0.071* |
| Approach | Coded | 9.6 | 9.7 | 9.9 | 9.4 | +0.103** |

`*p<0.05, **p<0.01, ***p<0.001` (unmarked = not significant). % columns = share of racialised reviews carrying that label within each season group. Caution: all three `approach` labels are now significant on trend, but `approach` has near-chance inter-annotator agreement — read that row with care.

### All WHEN questions, answered

- **Is racialised discourse increasing over time (2016–2025)?**
  *How:* logistic regression of racialised (0/1) on year, fit on the full window and again restricted to 2021–2025.
  *Found:* yes, and accelerating — 2016–2025 slope = 0.0824 log-odds/yr (p=6.8e-65); the 2021–2025 slope is **2.7x steeper** (0.2216, p=5.4e-92). In plain terms: 0.77% (2021) → 2.30% (2025), ~3x (p=1.2e-73). **Significant, and a genuine acceleration.**

- **Does the rate vary by season (indoor/outdoor, summer/other)?**
  *How:* chi-square test, pooled across all years.
  *Found:* yes, significant but small — outdoor 1.80% vs. indoor 1.19% (p=3.3e-70, V=0.025); summer 1.92% vs. other 1.25% (p=5.5e-80, V=0.027). **Significant; effect size is modest — large N means even a small gap clears significance easily.**

- **Is that seasonal gap itself trending, or has it always been there?** *(Figures 2–3 above)*
  *How:* trend fit separately per season/period + year×season interaction term, then repeated per year.
  *Found:* the gap is recent, not longstanding — absent 2016–17, opens 2018–19, closes again 2020–21 (plausibly COVID), widens every year from 2022 on. Interaction term itself significant (p=1.5e-11). **Significant — a real, distinct effect, not two coincidentally different slopes.**

- **Do the four content dimensions shift over time?** *(table above)*
  *How:* same year-level model applied within the racialised subset, tested per individual label.
  *Found:* `Crowd` declining (~60%→40% of racialised reviews, 2018–2025), incident-framing rising (~27%→49%), negative connotation rising (85%→92-94%). **`Clothing` is the single standout label** — steepest post-2021 rise of any label and the largest season-gap effect size. **Significant for most labels**; treat `approach` results cautiously (weak inter-annotator agreement).

- **Is the post-2021 rise nationwide, or driven by a few constituencies?**
  *How:* classified every constituency by whether it has racialised reviews before/after 2021.
  *Found:* nationwide — 288/299 (96.3%) constituencies have racialised reviews both before and from 2021 onward. **Not an artifact of a few new places appearing** — rules out a fake national trend built from a handful of constituencies.

---

## WHERE — how does the rate vary geographically?

**Figure 4. Every constituency, sorted by racialised rate, with Wilson 95% CI** (red = below the reliability bar of ≥10 racialised reviews)
`descriptive_analysis_figures/racialised_rate_per_constituency_wilson_ci.png`

![Constituency rate lineup](descriptive_analysis_figures/racialised_rate_per_constituency_wilson_ci.png)

<table>
<tr>
<td width="50%">

**Figure 5. Choropleth — racialised rate by constituency** (black hatching = below the reliability bar, shown rather than removed so "noisy estimate" isn't confused with "no data")
`descriptive_analysis_figures/rr_choropleth_labeled.png`

![Racialised rate choropleth](descriptive_analysis_figures/rr_choropleth_labeled.png)

</td>
<td width="50%">

**Figure 6. LISA cluster map — where is the spatial clustering statistically real?**
`descriptive_analysis_figures/lisa_clusters.png`

![LISA clusters](descriptive_analysis_figures/lisa_clusters.png)

</td>
</tr>
</table>

**On the range (checked against Figure 4):** the **national average is 1.54% [95% CI 1.51%, 1.58%]**. At **state** level the range is ~4x: Mecklenburg-Vorpommern (0.81%) up to Saarland/Berlin/Bremen (~3.0–3.1%) — those top three are small states/city-states, so read as "elevated on a smaller base," not a flat ranking. At the finer **constituency** level (Figure 4) the range is actually wider than the state range — down to ~0.25% in the most reliable rural constituencies, up to individual outliers like Bochum I (5.7%, on a solid 526-review base, not sampling noise).

**Reliability bar:** the binding constraint turned out to be the *racialised-review count* (numerator), not pool/review coverage (every constituency has ≥4 pools, ≥678 text reviews). Raising the bar from ≥5 to ≥10 racialised reviews moved exclusions from 11/299 to **52/299 (17.4%) excluded, 247/299 (82.6%) reliable** — the hatched constituencies in Figure 5.

### WHERE: correlation & significance tests

| Test | ρ | p | n | Significant? |
|---|---|---|---|---|
| Rate vs. % of a constituency's pools carrying ≥1 racialised review | 0.65 | ≪0.001 | 299 | *** |
| Rate vs. share of a constituency's racialised reviews from its single top pool | 0.09 | n.s. | 299 | no |
| Dominant pool significantly elevated vs. rest of its own constituency | — | — | 247 | **99/247 (40%)** |
| Racialised rate vs. distance to nearest land border, all constituencies | −0.055 | 0.341 | 299 | no |
| Racialised rate vs. distance to nearest land border, reliable_n only | −0.000 | 0.996 | 247 | no |

### All WHERE questions, answered

- **Is the geographic clustering in the choropleth real, or just a suggestive color scale?** *(Figures 5–6)*
  *How:* Queen contiguity spatial weights from actual constituency polygons; Global Moran's I tested via 999 permutations.
  *Found:* **Global Moran's I = 0.28** (expected ≈0 under randomness, z=7.91, **p=0.001**) — moderate, highly significant positive spatial autocorrelation. LISA finds two coherent High-High hotspot clusters: a **Ruhr-area NRW cluster** and a **Berlin cluster**, plus a smaller Baden-Württemberg pair. Low-Low coldspots are broad and diffuse (most of Schleswig-Holstein/M-V, a Saxony-Anhalt/Thuringia block, most of Bavaria south of Nuremberg). **Significant — the clustering visible on the map is real.**

- **Is a high constituency rate really an area pattern, or one bad venue skewing it?**
  *How:* correlate the rate against (a) % of pools affected, (b) how much of the rate comes from one top pool.
  *Found:* area pattern, not one venue — strong correlation with % of pools affected (ρ=0.65, p≪0.001), no correlation with top-pool concentration (ρ=0.09, ns). Gelsenkirchen (90% of pools affected), Essen II (83%) are broadly affected areas, not one outlier venue. **Significant for the area-pattern test; the single-venue test is null (as expected — it's evidence *against* the one-bad-venue explanation).**

- **Is any single "dominant" pool itself a genuine outlier?**
  *How:* two-proportion z-test, each constituency's top pool vs. the rest of its own constituency.
  *Found:* **99/247 (40%) of dominant pools are statistically significantly elevated** vs. the rest of their own constituency. Of those, 54/247 also carry >50% of local review volume ("big" — mostly explained by size); but **90/247 (36%) carry only a minority of local volume and are still significantly elevated** — a real, venue-specific effect in a substantial minority of cases (e.g. Lübeck/Mansfeld at 23.8% vs. ~0.7% constituency baseline).

- **Does distance to a neighbouring country's land border predict the rate?**
  *How:* Spearman correlation, land-neighbour distance only (coastline excluded).
  *Found:* **null** — no significant relationship at any grain (all 299: ρ=−0.055, p=0.341; reliable_n only: ρ=−0.000, p=0.996). Different border segments pull opposite ways (France/Switzerland runs high, Denmark runs average-to-low) and cancel out. **Not significant — borders don't matter, at least not in aggregate.**

**Overall WHERE finding:** a real, formally-confirmed spatial cluster (Ruhr-area NRW + Berlin) that's broad-based across pools within a constituency — not one bad venue, not a border-adjacency artifact — while individual dominant pools are still genuine outliers in a real minority (40%) of constituencies.

---

## WHO — who writes racialised reviews, and does it differ by identity?

### Reviewer-ID structure — are racialised reviews driven by repeat posters?

| Grain | Unique reviewer IDs | Singleton (1 review) | Repeat (2+) |
|---|---|---|---|
| Entire corpus | 923,168 | 78.8% | 21.2% |
| `social == yes` | 424,420 | 86.9% | 13.1% |
| `racialised == yes` | 7,627 | **98.4%** | 1.6% |

**Finding:** racialised reviews are overwhelmingly one-off. The singleton share climbs steadily and monotonically as you narrow to the subset that matters — repeat-reviewer confounding is not a real concern here.

### Name→country ethnicity classifier — coverage, and does it fail differently on racialised content?

| Grain | Unique names | Singleton-name coverage | Repeating-name coverage |
|---|---|---|---|
| Entire corpus | 758,558 | 95.2% | 95.3% |
| `social == yes` | 356,220 | 95.0% | 94.1% |
| `racialised == yes` | 7,389 | 91.1% | 68.4%† |

† Only 269 repeating names at the racialised grain (vs. 194,909 corpus-wide); most are coincidental collisions between different reviewers sharing a generic name (bare initials, or the literal placeholder *"Vorname Nachname"*), not genuine repeat posting.

**One-liner finding:** the classifier fails on the *same kind* of name everywhere — bare initials, generic placeholders — not specifically on racialised content, and the unclassified ("Unknown") group looks the same whether you're in the total corpus, the social subset, or the racialised subset. **No significant difference — this is a classifier blind spot, not a signal.**

### Figure 7. Ethnic composition: racialised=yes vs. racialised=no (within social reviews)
`descriptive_analysis_figures/ethnicity_composition_racialised_yes_vs_no.png`

![Ethnicity composition racialised yes vs no](descriptive_analysis_figures/ethnicity_composition_racialised_yes_vs_no.png)

*(Note: this compares racialised=yes vs. racialised=no — the two groups that actually differ, per the notebook. The corpus-wide/social-wide breakdown is nearly identical to each other, so racialised=yes is where the real shift shows up.)*

**Table — the same composition, as numbers** (social=yes, n=504,072; overall association χ²=1573.1, p≪0.001, Cramér's V=0.056):

| Category | Social (%) | Racialised (%) | Fold change |
|---|---|---|---|
| Germany | 39.62 | 27.63 | 0.70x |
| Arab | 1.92 | 5.95 | 3.1x |
| Türkiye | 1.81 | 4.78 | 2.6x |
| Other (real, non-top-10) | 23.27 | 25.54 | 1.1x |
| Unknown (no prediction) | 7.74 | 11.07 | 1.4x |

### Figure 8. Within-group racialised rate — is a review from a given ethnicity group itself more likely to be racialised?
`descriptive_analysis_figures/within_group_racialised_rate.png`

![Within-group racialised rate](descriptive_analysis_figures/within_group_racialised_rate.png)

### Figure 9. German vs. non-German — how does racialisation content differ by name-origin, across all four dimensions?
`descriptive_analysis_figures/german_vs_nongerman_gap_by_dimension.png`

![German vs non-German gap by dimension](descriptive_analysis_figures/german_vs_nongerman_gap_by_dimension.png)

*(Shown before the Arab/Türkiye/Europe breakdown below — this is the broader German-vs-non-German cut; the breakdown then shows it's specifically Arab+Türkiye driving the non-German side, not a uniform effect.)*

### All WHO questions, answered

- **How unique are reviewers — does racialised discourse come from repeat posters?** *(table above)*
  *Found:* 98.4% of racialised reviewer IDs have exactly one review. **Not a repeat-poster phenomenon.**

- **How complete is the name classifier, and does coverage differ for racialised content?** *(table above)*
  *Found:* ~90–95% coverage everywhere; the racialised-grain gap (91.1% singleton vs. 68.4% repeating) is a tiny-sample artifact of generic-name collisions, not a racialised-specific failure. **No significant difference.**

- **Does reviewer ethnicity composition differ between racialised and non-racialised reviews?** *(Figure 7)*
  *How:* chi-square test of `ethnicity_final` × racialised, within social=yes.
  *Found:* significant, small effect (χ²=1573.1, p≪.001, Cramér's V=0.056). `Germany` share drops 39.8%→27.6%; **`Arab` (1.9%→5.9%, ~3.1x)** and **`Türkiye` (1.8%→4.8%, ~2.6x)** both roughly triple. **Significant but modest — real shift, not a large one.**

- **Is a review from a given ethnicity group itself more likely to be racialised (holding group size fixed)?** *(Figure 8)*
  *How:* within-group racialised rate — each group's own reviews as the denominator.
  *Found:* yes — **`Arab` 4.78% [4.38–5.23%]**, **`Türkiye` 4.06% [3.68–4.49%]**, both ~3–4x the 1.541% baseline; **`Germany` lowest at 1.08% [1.03–1.12%]**. Two independent ways of asking the question land on the same story. **Significant, real in-group effect.**

- **Does review language show the same pattern as ethnicity?**
  *How:* same three-way + chi-square design on `org_review_lang`.
  *Found:* significant but smaller effect (χ²=645.4, p=3.3e-132, V=0.036) — German drops 95.19%→90.69%, French/Arabic/Czech/Polish rise 1.6–3.4x. **Significant, smaller effect than ethnicity.**

- **Does ethnicity signal actually correspond to the language a review is written in?**
  *How:* chi-square + Cramér's V, ethnicity × language, judged against each language's own baseline rate.
  *Found:* real identity signal, moderate effect (V=0.240) — concentrated where a "home language" exists (France→French ~16x baseline, Poland→Polish ~33x, Arab→Arabic ~12x, Türkiye→Turkish ~13x). **Significant — the name-origin signal is a real identity signal, not noise.**

- **Does content (theme/context/connotation/approach) differ between German- and non-German-origin reviewers, and is that driven by specific groups?** *(Figure 9, and the breakdown that follows it)*
  *How:* column-normalised cross-tabs, then the same test on the full ethnicity scheme and the consolidated 4-group scheme (Germany/Arab/Türkiye/Europe/Other).
  *Found:* concentrated in `theme` (V=0.180) and `context` (V=0.085); `connotation` nearly flat (ethnicity tracks *what* is said, not *how negative*). It's specifically **`Arab` + `Türkiye` vs. `Germany`**, not a uniform non-German effect: both skew far less `Crowd`/`Etiquette`, far more `Staff`/`Access`, and much more `incident`-framed than the grand mean — roughly 2–3x further from it than Germany's own (smaller, opposite-direction) deviation. `Europe`/`Other` land in between, closer to Germany's profile. **Significant, and specific to two groups, not "non-German" in general.**

- **Does East/West/Berlin geography modulate any of the ethnicity-content patterns — is this a WHERE effect hiding inside WHO?**
  *How:* chi-square test per ethnicity group × dimension (20 tests: 5 groups × 4 dimensions), rather than one pooled test, since the hypothesis was specifically about regional variation *within* a group.
  *Found:* **no — ruled out.** Only 3/20 tests reach p<0.05, all with small effect sizes, and **none involve `Germany`, `Arab`, or `Türkiye`** — the three groups the hypothesis was actually about (all p≥0.155 there). A simpler, ethnicity-blind version (does region alone shift theme/connotation) finds the same: connotation doesn't shift with region at all (p=0.152); theme shifts modestly (V=0.056) but not enough to call region a driver. **Not significant for the groups that matter — this was checked and is a real null result, not an oversight.**

**Overall WHO finding:** a real, modest, and *specific* ethnicity/language signal — concentrated in Arab/Türkiye vs. Germany, not a uniform "non-German" effect — showing up in both who writes (composition, in-group rate) and what they write about (theme, context), with how negative it is (connotation) untouched throughout. Geography was checked twice, from two angles, and ruled out both times.

---

## What's next (explicitly out of scope for this document)

Everything above is descriptive only — no socioeconomic or political data anywhere in this document. The next layer (deprivation, political vote share, and whether the WHERE/WHO patterns above survive controlling for those) is a separate piece of work.
