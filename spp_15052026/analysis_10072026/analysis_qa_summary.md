# Racialised Discourse in German Swimming Pool Reviews — Data, Results & Story

One file. Format throughout: **what we asked** → **how we answered it** → **what we found**.

---

# PART A — DATA

## A.1 Data collection

**What:** how we got from "German public swimming pools" to a final corpus of 1,265,806 text reviews across 5,529 verified POIs.
**How:** two independent candidate-pool lists — an administrative registry (`baederleben`, 5,230→5,204 deduped pools) and Google Maps POIs collected by scraping across Germany (11,848 candidates, filtered to swimming-pool place types) — reconciled by spatial matching (BallTree) then fuzzy address matching, with remaining unmatched pools resolved by manual Google Maps search. "Verified" = confirmed swimming-pool place type + located inside Germany (checked against a Germany shapefile). Reviews were scraped via a custom three-stage ETL pipeline (places → reviews → reviewer profiles, multi-threaded, paginated, stored in Postgres).
**Found:** final verified-POI set = **5,529 places** (`places_filtered.parquet`); ~2.79M reviews collected total, filtered to reviews at verified pools (2,410,565), then to text-only reviews → **1,265,806** (`master_corpus_final.parquet`). Every review row is asserted to join to exactly one verified POI (100%).
**Caveat to know:** the exact scrape mechanism (which Google endpoint) isn't documented anywhere — only proxy/threaded-request config exists. Not a blocker, just not citable in the methods section as a named API.

## A.2 Annotation & codebook

**What:** how reviews were labelled `social` → `racialised` → four content dimensions.
**How:** a hierarchical codebook (racialisation = attribution to a group defined by race/ethnicity/nationality/culture/religion, any valence, explicit/implicit/coded) applied via a **single LLM, zero-shot** (Claude Sonnet 4.6, via OpenRouter), one classifier script per stage, structured-JSON output, no fine-tuning. A human-coded validation sample exists (two annotators) compared against model output via precision/recall — but see A.3, this isn't finished for every dimension yet.
**Found — the exact taxonomy:**
- `social` (binary): yes = references other people/staff/atmosphere; no = physical environment/facilities/pricing only.
- `racialised` (binary, social=yes only): yes = attributes anything to an ethnic/national/cultural/religious group; no = classist/gendered/generic-conduct complaints with no ethnic anchor.
- Four dimensions (racialised=yes only): **theme** (multi-label, ≤3: Staff, Crowd, Belonging, Language, Clothing, Hygiene, Etiquette, Security, Safety, Access, Other), **context** (Incident, Generalised), **approach** (Explicit, Coded, Implicit), **connotation** (Positive, Negative, Neutral).
**Classification results:** 504,072 / 1,265,806 (39.8%) social=yes; 7,768 / 504,072 (1.54%) racialised=yes; 100% dimension coverage on all four dimensions for the racialised=yes subset.

## A.3 Inter-annotator agreement & LLM validation — corrected 2026-08-24, was stale

**This section previously said "not reported yet." That was wrong as of this correction** — `inter_annotator_agreement.ipynb` and `data_annotation_validation/iaa_results/iaa_summary.csv` contain completed two-annotator (Ethel, Keziah) hand-coding against the model for every dimension except `theme`. Found during a folder-wide consolidation audit; see
`analysis_10072026/supervisor_run_2026-08-16/ANALYSIS_LOG.md` item 1.7 for the full citation trail.

**`social` gate:** recall (n=100) ethel-vs-model raw agreement 0.99, κ=0.980 [0.94,1.00]; keziah-vs-model 0.86, κ=0.709. Precision (n=50, model-yes only): ethel 100.0% [92.9,100%]; keziah 88.0% [76.2,94.4%]. Human-human (ethel vs keziah): 0.87 raw, κ=0.729 on recall; κ=0.0 on precision (degenerate — class-skew kappa paradox, not genuine disagreement, see the notebook).

**`racialised` gate** (the label the whole study rests on): precision (n=250) ethel 98.0% [95.4,99.1%], keziah 99.6% [97.8,99.9%]. Recall (n=100, ~all model-`no`): estimated false-negative rate 0/97 for both annotators (Wilson CI 0.0–3.8%) — no evidence of missed racialised reviews in this sample, though the CI's upper bound is still consistent with corpus-level misses if real.

**Sub-dimensions:** `approach`, `context`, `connotation` all have completed hand-coded validation (exact-set-agreement / mean-Jaccard, since these are multi-label) — see `iaa_summary.csv`. **`theme` alone remains unvalidated** — genuinely incomplete, not enough human-coded comparison data yet. That is the one part of the original "not reported yet" framing that's still accurate.

**Bottom line, corrected:** the two pipeline gates (`social`, `racialised`) have strong, completed hand-coded validation, not just rerun/back-translation stability (§A.4). Three of four content dimensions do too. Only `theme` is still open. Flag `theme` as "in progress" if asked; the rest is real, reportable validation, not partial numbers to hedge on.

## A.4 Robustness checks — what we're reporting instead

Four checks, all in `robustness_check.ipynb`, all using the same design: **rerun the identical production classifier** on a held-out stratified sample and measure exact-label-match stability, plus a **back-translation (EN→DE→EN) sensitivity check** to see if paraphrase-level noise (not the classifier itself) explains any instability. Paired McNemar-style test distinguishes "translation-specific" instability from ordinary rerun noise.

- **`social` gate** (n=300, stratified by canonical label; backtrans n=151): relabel stability **0.910**, backtranslation stability **0.927**, no significant translation effect (p=0.727). **Holds up well.**
- **`racialised` gate** (stage_2_racial_reclassed only — the classifier of record; the earlier, superseded `stage_2_racial` pass is deliberately excluded) (n=300, 50/50 case-control since base rate is ~1.5%; backtrans n=150): relabel stability **0.940**, backtranslation stability **0.940**, no significant translation effect (p=0.607). **The single most stable check.**
- **Sub-dimensions, same-classifier rerun** (n=387, stratified by approach×connotation×context combo): exact-match stability — approach 0.912, context 0.910, connotation 0.894, **theme 0.801 (weakest, consistent with it being the least-validated dimension)**.
- **Sub-dimensions, back-translation** (n=155): stability drops further under paraphrase — approach 0.845, context 0.884, connotation 0.826, theme 0.716. Translation effect is **statistically significant for connotation** (p=0.013) and marginal for theme (p=0.044); not significant for approach/context.

**Bottom line to report:** the two pipeline *gates* (social, racialised) are highly stable (0.91–0.94) and insensitive to translation-level noise. The four *content dimensions* are less stable, `theme` weakest of all four — consistent with §A.3's caution and with WHEN's own IAA caveat on `approach`. This is a legitimate, complete robustness story even without full IAA.

---

# PART B — RESULTS: PREVALENCE (WHEN / WHERE / WHO)

## B.0 The timestamp detour (needed before WHEN)

**What happened:** building the WHEN trend required `review_timestamp`, which wasn't fully populated — a real data-quality detour before any temporal claim could be trusted. This was a **two-stage fix**, not one:

- **Stage 1 — the `gmaps_poi` DB refresh.** Originally only **~69.5%** of the entire corpus had a parsed absolute timestamp at all, and that ~69.5% coverage was **roughly uniform across the corpus** (not concentrated in social/racialised vs. not — no structured gap to worry about at this stage). A refresh of the `gmaps_poi` database (unrelated to anything we computed — an upstream data-source fix) lifted this to **99.5%** (1,260,044 / 1,265,806) corpus-wide before we did anything else.
- **Stage 2 — our own recovery step.** From that 99.5% starting point, we recovered the rest ourselves: missing timestamps parsed from the relative-time text (`review_time`, e.g. "2 years ago") anchored to `place_retrieved_time` (the scrape date), rather than dropping incomplete rows.

**How we handled stage 2, exact numbers** (starting point = post-refresh, i.e. the 0.5% still missing at that point):

| Grain | n | Missing before recovery | Missing after recovery |
|---|---|---|---|
| Entire corpus | 1,265,806 | 5,762 | **40** |
| `social == 'yes'` | 504,072 | 2,910 (0.58%) | 22 |
| `racialised == 'yes'` | 7,768 | 138 (1.78%) | 1 |

(Both columns are post-DB-refresh — recovery is the last-mile step covered here; the 69.5%→99.5% DB-refresh jump above is a separate, earlier event with no per-grain breakdown available.)

**Cause of the unrecoverable 40:** all 40 have no `review_time` text at all (nothing to parse, not a parsing failure). Of the 40: 22 social=yes/18 social=no; of the social=yes ones, 21 racialised=no/1 racialised=yes. Negligible at every grain (≤0.02% of any subset) — a completeness statement, not a caveat that weakens the trend.

**Report both stages, not just the second one** — "69.5% → 99.5% → 100%" is the honest full story; presenting only the last mile (99.5%→100%) understates how incomplete the timestamp data originally was.

## B.1 WHEN — temporal trend

- **Q: Is racialised discourse increasing over time (2016–2025)?**
  **How:** logistic regression of `racialised` (0/1) on year (centered, continuous) — the standard trend-test approach for a rate observed at several time points; fit once on the full 2016–2025 window, once restricted to 2021–2025.
  **Found:** yes, highly significant — 2016–2025 slope = 0.0824 log-odds/year (p=6.8e-65); 2021–2025 slope is **2.7x steeper** (0.2216, p=5.4e-92), confirming a genuine acceleration, not just "the same trend zoomed in." Plainest version: 0.77% (2021) → 2.30% (2025), ~3x (two-proportion z-test, p=1.2e-73).
- **Q: Does the rate vary by season (indoor/outdoor, summer/other)?**
  **How:** chi-square test of independence (racialised × season), pooled across all years; Cramér's V for effect size.
  **Found:** yes, significant but small — outdoor 1.80% vs. indoor 1.19% (χ²=313.8, p=3.3e-70, V=0.025); summer 1.92% vs. other 1.25% (χ²=358.7, p=5.5e-80, V=0.027). Large N makes even a modest gap easily significant — the p-value confirms it's real, V says the size is modest.
- **Q: Is that seasonal gap itself trending, or stable over time?**
  **How:** trend model fit separately within each season/period, plus a combined model with a year×season interaction term (the formal test for "do the two trends actually differ") — then repeated per-year to see exactly when the gap became real.
  **Found:** the gap is recent, not longstanding. Absent 2016–17 (p>0.24), opens 2018–19, **closes again in 2020–21** (plausibly COVID — review-volume patterns shifted then too), then **widens every year from 2022 on** (indoor/outdoor gap grows 0.61pp→1.14pp 2022→2025; by 2025, summer is ~2x other months). Interaction term is itself significant (coef=0.0681, p=1.5e-11) — outdoor/summer's faster rise is a real, distinct effect, not two coincidentally-different positive slopes.
- **Q: Do the four content dimensions (theme/context/connotation/approach) shift over time?**
  **How:** same year-level lens applied within the racialised=yes subset only (N=7,768); formal significance tests repeated per individual label, ranked by effect size.
  **Found:** yes — `Crowd` theme declining (~60%→40% of racialised reviews, 2018→2025), incident-framing rising (~27%→49%), negative connotation rising (85%→92-94%). **`Clothing` is the single standout label**: steepest post-2021 rise of any label (+0.212, p=6.5e-12) *and* the largest season-gap effect size (V=0.088). One caution: all three `approach` labels are now significant on trend where none were before — read cautiously given `approach`'s near-chance inter-annotator agreement (§A.3/A.4).
- **Q: Is the 2021+ rise a nationwide pattern or concentrated in a few constituencies?**
  **How:** classified every constituency by whether its racialised reviews fall before/after 2021.
  **Found:** nationwide — 288/299 (96.3%) constituencies have racialised reviews both before and from 2021 onward; only 1 constituency is pre-2021-only (a single review), 10 are 2021+-only (all thin, scattered across 7 states, no shared pattern). Rules out "a few newly-appearing places are faking a national trend."

**Overall WHEN finding:** a real, accelerating (not just rising — accelerating), recently-reshaped temporal trend, confirmed formally rather than eyeballed off a line chart — not a data artifact.

## B.2 WHERE — geographic distribution

- **Q: Is pool/review coverage adequate to trust constituency-level rates at all?**
  **How:** built a full constituency-level aggregate table (`const_agg`) via a **left join** from the complete list of 299 constituencies (not just ones appearing in the corpus), so zero-coverage areas would show up as zero rather than as missing rows.
  **Found:** every one of 299 constituencies has ≥4 pools, ≥678 text reviews — no true zero-coverage constituency anywhere. But the right reliability question turned out to be different: `racialised_reviews` count (the numerator), not pool/review coverage (which was never binding — min `social_reviews`=307). Revised the reliability bar from `≥5` to `≥10` racialised reviews → **247/299 (82.6%) reliable**, 52 excluded (was 11 at the old bar).
- **Q: What is the prevalence, with proper uncertainty, by state and constituency?**
  **How:** Wilson score interval (not the normal approximation — better-behaved for a small proportion near zero, which this is) on racialised/social reviews per unit.
  **Found:** **1.54% overall [95% CI 1.51%, 1.58%]** — a tight interval, so the headline number is solid. State range ~4x: Mecklenburg-Vorpommern (0.81%) up to Saarland/Berlin/Bremen (~3.0–3.1%) — the top three are small states/city-states with wider CIs, read as "elevated on a smaller base," not a flat ranking.
- **Q: Is the geographic clustering visible on the choropleth actually statistically real, or just a suggestive color scale?**
  **How:** Queen contiguity spatial weights (constituencies sharing a border are neighbours) built from the actual Wahlkreis polygons, row-standardized; **Global Moran's I** tested via permutation (999 relabellings, not a normal-theory p-value, since the rate is skewed).
  **Found:** real — **Global Moran's I = 0.28** (expected ≈0 under randomness, z=7.91, **p=0.001** over 999 permutations) — moderate, highly significant positive spatial autocorrelation. **LISA** identifies two coherent High-High hotspot clusters: a **Ruhr-area NRW cluster** (Gelsenkirchen, Bochum I, Essen II/III, Duisburg I, etc.) and a **Berlin cluster** (Neukölln, Mitte, Tempelhof-Schöneberg, etc.) — plus a smaller Baden-Württemberg pair extending the Offenburg/Lörrach signal. Low-Low coldspots are broad/diffuse (most of Schleswig-Holstein/Mecklenburg-Vorpommern, a Saxony-Anhalt/Thuringia block, most of Bavaria south of Nuremberg).
- **Q: Is a high constituency rate really an area pattern, or secretly one bad venue?**
  **How:** Spearman correlation of `racialised_rate` against (a) the % of a constituency's social-active pools that carry ≥1 racialised review, and (b) the share of racialised reviews contributed by its single most-implicated pool.
  **Found:** area pattern, not one venue — rate correlates strongly with **% of pools affected** (ρ=0.65, p≪0.001) but not with **top-pool concentration** (ρ=0.09, not significant). High-rate constituencies (Gelsenkirchen 90% of pools affected, Essen II 83%) are broadly affected, not one outlier venue carrying an otherwise-quiet area.
- **Q: Is any single "dominant" pool itself a genuine outlier, though — big by size, or big by rate?**
  **How:** two-proportion z-test comparing each constituency's single most-implicated pool against the rest of its own constituency's pools combined (not against pools nationally); cross-checked against non-overlapping Wilson CIs.
  **Found:** **99/247 (40%) of dominant pools are statistically significantly elevated** vs. the rest of their own constituency. Of those, 54/247 (22%) also carry >50% of local review volume ("big" — mostly explained by size alone, only 9 of those 54 are also significantly elevated); but **90/247 (36%) carry only a minority of local volume and are still significantly elevated** — a real, venue-specific effect in a substantial minority of cases (e.g. `Lübeck`/`Mansfeld` at 23.8% vs. ~0.7% constituency baseline, p≈2e-18).
- **Q: Does distance to a neighbouring country's land border predict the rate?**
  **How:** Spearman correlation of `racialised_rate` against `distance_to_neighbor_country_km` (land-neighbour polygons only, coastline explicitly excluded so a coastal-but-inland seat isn't miscounted as a border seat).
  **Found:** null — no significant relationship at any grain (all 299: ρ=-0.055, p=0.341; reliable_n only: ρ=-0.000, p=0.996). Different border segments pull opposite ways (France/Switzerland stretch runs high, Denmark stretch runs average-to-low) and cancel out in the aggregate — not evidence of a general "border effect."

**Overall WHERE finding:** a real, formally-confirmed spatial cluster (Ruhr-area NRW + Berlin, Moran's I=0.28, p=0.001) that is broad-based across pools within a constituency — not one bad venue, and not a border-adjacency artifact — though individual dominant pools are themselves genuine outliers in a real minority (40%) of constituencies.

## B.3 WHO — reviewer identity, ethnicity, language, content

- **Q: How unique are reviewers, and does racialised discourse come from repeat posters?**
  **How:** compared singleton- vs. repeat-`reviewer_id` shares across three grains (entire corpus / social=yes / racialised=yes).
  **Found:** racialised reviews are overwhelmingly one-off — **98.4% of racialised reviewer_ids have exactly one review** (vs. 86.9% at social grain, 78.8% at corpus grain) — repeat-reviewer structure shrinks sharply and monotonically toward the subset that matters, so repeat-reviewer confounding isn't a real concern here.
- **Q: How complete is the name→country ethnicity classifier, and does that coverage differ for racialised reviews?**
  **How:** compared classified/unclassified shares for singleton vs. repeating names, across the three grains.
  **Found:** ~90–95% coverage at corpus/social grain; racialised grain is lower overall (90.2%) with a wide singleton (91.1%) vs. repeating-name (68.4%) gap — but repeating names are a tiny slice there (269 of 7,389), and mostly generic-name collisions between different reviewers (`M`, `Vorname Nachname`), not genuine repeat posting. The classifier fails on the same kind of name (bare initials) everywhere, not specifically on racialised content.
- **Q: Does reviewer ethnicity (name-origin) composition differ between racialised and non-racialised reviews?**
  **How:** chi-square test of independence (12-category `ethnicity_final` × racialised), within social=yes, plus a three-way comparison (entire corpus / social / racialised) to confirm any shift isn't just an artifact of the social=yes filter.
  **Found:** significant, small effect (χ²=1573.1, p≪.001, **Cramér's V=0.056**) — real but modest, as expected at this N. `Germany` share drops 39.8%→27.6%; **`Arab` (1.9%→5.9%, ~3.1x)** and **`Türkiye` (1.8%→4.8%, ~2.6x)** both roughly triple. The three-way check confirms the shift is concentrated entirely between social and racialised (not an artifact of filtering to social=yes — entire-corpus and social-grain shares are nearly identical).
- **Q: Is a review from a given ethnicity group itself more likely to be racialised (holding group size fixed)?**
  **How:** within-group racialised rate — each `ethnicity_final` group's own reviews as the denominator, not its share of an already-rare event.
  **Found:** yes, a real in-group effect matching the composition finding — **`Arab` 4.78% [95% CI 4.38–5.23%]** and **`Türkiye` 4.06% [3.68–4.49%]**, both ~3–4x the 1.541% baseline; **`Germany` lowest at 1.08% [1.03–1.12%]**, clearly below baseline. Two independent ways of asking the question (composition share vs. own-group base rate) land on the same story.
- **Q: Does review language show the same pattern as ethnicity?**
  **How:** same three-way (entire corpus/social/racialised) + chi-square design, applied to `org_review_lang` (after a data-quality fix: 65,908 null-language rows corpus-wide were mechanically mislabeled English, not missing — recovered 17 racialised rows that `pd.crosstab` had been silently dropping).
  **Found:** significant but smaller effect (χ²=645.4, dof=10, p=3.3e-132, **V=0.036**) — German drops 95.19%→90.69%, French/Arabic/Czech/Polish rise 1.6–3.4x. Unlike ethnicity, language does **not** stay stable between entire-corpus and social grain (German 88.6%→95.2%) — a mechanical artifact of many short, generic non-social reviews being non-German.
- **Q: Does ethnicity signal actually correspond to the language a review is written in?**
  **How:** chi-square + Cramér's V on ethnicity (13-cat) × language (15-cat), judged against each language's own corpus-wide baseline rate (not raw row-dominance, which is uninformative since German dominates every row by default).
  **Found:** real identity signal, moderate effect (χ²=5349.8, dof=168, p≈0, **V=0.240**) — concentrated in categories with a plausible "home language": `France` 49.7% French (vs. 3.1% baseline, ~16x), `Poland` 28.8% Polish (~33x), `Russian Federation` 25.5% Russian (~22x), `Arab` 8.4% Arabic (~12x), `Türkiye` 3.5% Turkish (~13x). Categories without an obvious home language (Brazil, Indonesia, Philippines, US, Other, Unknown) show no elevation — a construct-availability limit, not necessarily an absence of signal.
- **Q: Does content (theme/context/connotation/approach) differ between German- and non-German-origin racialised reviews?**
  **How:** column-normalised cross-tabs (German vs. non-German name-origin) per dimension; Cramér's V to compare effect size across dimensions on the same scale.
  **Found:** yes, concentrated in `theme` (V=0.180) and `context` (V=0.085); `connotation` nearly flat (86.8% vs. 87.0% negative — ethnicity signal tracks *what*, not *how negative*). German-authored reviews skew `Crowd` (+8.9pp)/`Etiquette` (+6.6pp), framed `generalised`; non-German skew `Staff` (-7.9pp)/`Access` (-4.0pp), framed as a specific `incident`.
- **Q: Is that really a uniform "non-German" effect, or driven by specific groups?**
  **How:** repeated the same per-dimension test on the full 12-category ethnicity scheme, then consolidated into 4 theoretically-motivated groups (`Germany`/`Arab`/`Türkiye`/`Europe`/`Other`).
  **Found:** specifically **`Arab` + `Türkiye` vs. `Germany`**, not uniform — both show far less `Crowd`/`Etiquette` and far more `Staff`/`Access`/`incident`-framing than the grand mean, 2–3x further from it than `Germany`'s own (smaller) deviation in the opposite direction. `Poland`/`Russian Federation` actually lean mildly *toward* the Germany pattern; `US`/`France`/`Italy`/`Indonesia`/`Philippines`/`Brazil`/`Other` sit near the grand mean with no strong signature.
- **Q: Does East/West/Berlin geography modulate any of the ethnicity-content patterns?**
  **How:** chi-square test per ethnicity group × dimension (20 tests total: 5 groups × 4 dimensions), rather than one pooled test, since the hypothesis was specifically about regional variation *within* a group.
  **Found:** no — ruled out. Only 3/20 tests reach p<0.05, all with small Cramér's V (≤0.122), and **none involve `Germany`, `Arab`, or `Türkiye`** (all p≥0.155) — the three groups the hypothesis was actually about. Region isn't worth carrying into the constituency-level work.

**Overall WHO finding:** a real, modest, and specific ethnicity/language signal — concentrated in `Arab`/`Türkiye` vs. `Germany`, not a uniform non-German effect — showing up in both composition and content (what's said, how concretely), with connotation (how negative) untouched throughout. Geography checked twice and ruled out.

## B.4 Bringing WHO + WHERE together — is this ethnicity, or just density?

This section exists because WHO and WHERE can't be read independently — a constituency's ethnicity mix and its density/volume are themselves correlated (immigrant populations concentrate in cities), so any "ethnicity explains the rate" claim has to survive controlling for density before it means anything.

- **Q: Is the constituency-level rate just a volume artifact (bigger/denser constituencies happen to show a different rate for unrelated reasons)?**
  **How:** Spearman correlation of `racialised_rate` against `social_reviews` and `pool_count`, restricted to the 247 reliable_n constituencies (already excludes thin-sample noise).
  **Found:** real and not small — negatively correlated with both `social_reviews` (ρ=-0.337, p=5.5e-08) and, more strongly, `pool_count` (ρ=-0.556, p=1.8e-21). Combined with B.2's pool-concentration finding (a high rate isn't one bad venue), this is a genuine constituency-scale pattern, not sampling noise.
- **Q: Does a constituency's ethnicity composition (`ethnicity_group4` share) predict its rate, beyond volume?**
  **How:** Spearman correlation of each group's share of local racialised reviews against `racialised_rate`, across all 247 reliable_n constituencies (not just hand-picked high/low examples).
  **Found:** yes, at first pass — **`Arab` share ρ=+0.211 (p=0.001)**, **`Türkiye` share ρ=+0.180 (p=0.004)**, **`Germany` share ρ=-0.161 (p=0.011)**; `Europe` trends the same direction as Arab/Türkiye but doesn't clear significance (ρ=+0.105, p=0.100); `Other` shows no relationship (ρ=-0.036, p=0.573). Specificity matters — it isn't "any non-German presence," it's specifically Arab/Türkiye presence.
- **Q: Why do low-rate constituencies skew *more* `Staff`-themed despite having *less* Arab/Türkiye presence (who individually skew Staff-heavy)?**
  **How:** a standard decomposition — hold each tier's actual ethnicity mix fixed and ask what its `Staff`-share would be if it kept the *other* tier's within-group `Staff` rates (if the gap mostly disappears under the swap, it's compositional; if it survives, it's a real place-level effect).
  **Found:** a real within-group effect, not compositional — every single ethnicity group independently writes about `Staff` roughly 2x more in the low-rate tier (e.g. `Arab` 22.0%→44.4%, `Germany` 5.2%→14.4%). The decomposition confirms it: giving the low tier the high tier's ethnicity mix but keeping its own within-group rates still predicts 19.5%, almost exactly the observed 18.5% — the ethnicity-mix shift explains essentially none of the gap. Survives controlling for `pool_count` too (partial ρ=-0.155, p=0.015). Mechanism remains genuinely open (flagged for qualitative follow-up), but it is **not** an artifact of which ethnicities are present.
- **Q: Is the WHERE pattern actually a density/urban-rural story rather than an ethnicity story?**
  **How:** Spearman correlation of `racialised_rate` and each `ethnicity_group4` share against `population_density_per_km2`; rank-residualized partial correlations to see what survives controlling for it.
  **Found:** decisively yes — **`population_density_per_km2` correlates with the rate at ρ=0.597 (p=3.1e-25), the single strongest relationship found anywhere in Part 1**, and explains much of why `pool_count` behaves the way it does (density vs. pool_count ρ=-0.794 — Germany's equal-population constituency boundaries force sparse rural areas to be geographically huge, accumulating more small pools). Once density is controlled: **`Germany` share's link to rate disappears completely** (ρ=-0.014, p=0.828, vs. raw -0.161); **`Türkiye` share weakens to non-significance** (ρ=+0.082, p=0.198, vs. raw +0.180); **only `Arab` share survives**, weakened but real (ρ=+0.132, p=0.039, vs. raw +0.211). Density and pool_count each survive controlling for the other (partial ρ=0.307 and -0.169 respectively) — neither fully subsumes the other.

**Overall B.4 finding:** the WHERE geographic pattern is mostly a density/urban-rural story, not an ethnicity story — two of three ethnicity leads from WHO evaporate or weaken to nothing once density is accounted for. **`Arab` share is the one lead that survives**, and is exactly what motivates bringing in socioeconomic data next: density alone doesn't fully explain it, so the natural next question is deprivation.

---

# PART C — LAYERING IN SOCIOECONOMIC & POLITICAL DATA

## C.1 Data quality first

**What we checked:** is the socioeconomic/political file corrupt, and does every column carry real constituency-level information?
**How:** ran a full corruption check on the raw file (decimal parsing, scaling, range sanity — all passed cleanly), then a separate audit script scanning every substantive column in the cleaned file (66 columns) for values that are actually just **repeated across all of a state's constituencies** (a state-wide figure stamped onto every constituency, not independently measured per constituency).
**Found:** **43 of 66 columns** are flagged as state-repeated somewhere (mostly Berlin/Hamburg/Bremen — small city-states where "constituency-level" and "city-level" collapse together). Of those, **11 columns feed directly into this analysis's actual predictor set** (the six age-share columns, disposable income, GDP per capita, SGB-II rate, employment rate, unemployment rate) — flagged everywhere they appear and re-tested with/without those constituencies throughout, not silently dropped. *(43/66 is the current, correct figure — not 140.)*

## C.2 Ignoring that caveat and going ahead — what do the predictors show?

- **Which predictors correlate with racialised rate?** 17/18 significant after FDR correction. Strongest: `sgb_ii_recipients_pct` (ρ=0.47), `foreign_population_pct` (ρ=0.46), `unemployment_total` (ρ=0.43). Political: left/SPD/Greens positive; CDU/CSU and **AfD negative** (ρ=-0.20) — counter to naive expectation.
- **Does that hold up once WHERE's density confound is controlled for?** Only partly — controlling for `pool_count`, just 6/18 stay significant (deprivation cluster + turnout + two vote-share variables). **The AfD association collapses to ~0 (ρ=+0.03, ns)** — confirmed further: also killed by controlling for `foreign_population_pct` alone. **This was never an independent AfD effect** — a clean, reportable negative result.
- **Is the deprivation-cluster finding robust?** Yes — survives excluding venue-concentration outliers (shift ≤0.02) and, mostly, excluding the Berlin/Hamburg/Bremen city-states (one weaker predictor, disposable income, loses significance without them — exactly the column already flagged in C.1).
- **Does the `Arab`-share lead from WHO get resolved?** Yes — it tracks the deprivation/labor-market cluster specifically (not general foreign-population presence), and — unlike `Germany`/`Türkiye` — survives the density control, just attenuated by about half.
- **Does it matter which racialised rate we use — `/social` or `/total_text_reviews`?** No. Both correlation (§2.4) and the joint NB2 model (§3.3b) give near-identical results under either denominator — same significant predictors, same signs, magnitudes within 0.03 rho / comparable IRRs. The `/social` definition (Part 1's original choice) isn't costing anything; no result in this section depends on which one is used.

**Overall socioeconomic finding:** the geographic pattern in racialised discourse is explained by **deprivation, not political lean** — with the AfD hypothesis explicitly tested and rejected, and the one surviving ethnicity lead (`Arab` share) now explained by the same deprivation story.

---

# PART D — PREDICTIVE MODELLING (optional for this pass)

A single joint model (negative binomial, `foreign_population_pct` + 5 other predictors + `pool_count`) confirms `foreign_population_pct` survives jointly (IRR=1.034, p<0.0001) and improves out-of-sample prediction ~12% over density alone — **confirmatory**, not a new finding: it re-tests what C.2 already established one-at-a-time. Residual spatial clustering (Moran's I=0.118, about half the raw rate's 0.28) is the one honest gap left, flagged for a future spatial-lag/state-random-effect model.

## Is this enough without a full predictive section?

**Yes.** Parts A–C already give: a real temporal trend, a real and geographically-explained spatial pattern, a real content-differentiated WHO signal, and a real socioeconomic explanation for the geography that survives every robustness check thrown at it (density confound, outlier venues, city-state contamination, multiple-comparison correction) — with the most obvious reviewer objection (AfD) pre-empted and answered, not left as an awkward loose end.

What Part D adds beyond that: one joint-model confirmation, one quotable effect-size number for the abstract ("each 1pp more foreign-population share → 3.4% higher rate"), and an out-of-sample check. Nice-to-have, not load-bearing. **If time is short: write up A–C as the results section, mention Part D as a one-paragraph robustness/future-work note, and spend the saved time on the discussion.**

---

# The whole story, one paragraph

We collected and verified 5,529 German swimming pools and 1,265,806 text reviews, classified them through a single-LLM pipeline (social → racialised → four content dimensions) whose two decision gates are highly stable under rerun and back-translation (0.91–0.94) even though full human-validation isn't finished for every dimension yet. Within that corpus, racialised discourse is rare (1.54%) but real: rising steadily since 2021, spatially clustered in the Ruhr area and Berlin (confirmed via Moran's I/LISA), and differentiated by reviewer name-origin in both who writes it and what they write about. Bringing in socioeconomic and political data — after confirming it isn't corrupt and flagging the handful of columns that don't vary independently at the city-state level — shows that geographic pattern is explained by **deprivation** (benefit dependency, unemployment, foreign-population share), not political lean: the AfD hypothesis a reviewer would expect to see confirmed is instead explicitly tested and ruled out. That's a complete, defensible results section on its own; a predictive model would only confirm it jointly and attach one clean effect-size number, not add a new finding.

---

# Open threads / limitations (honest, not buried)

- Why low-rate (rural) constituencies skew `Staff`-themed — real (WHO/WHERE), mechanism still unexplained by density or ethnicity composition.
- `sgb_ii_recipients_pct` vs. `unemployment_total` — too collinear (VIF ~26) to separate individual effects in the predictive model.
- Residual spatial clustering (Moran's I=0.118) in the predictive model — needs a spatial-lag or state-level random effect.
- `theme`/`approach` IAA and LLM validation — not yet complete; `approach` has near-chance human IAA (κ≈0–0.09) and `theme` is the weakest dimension in the robustness rerun (0.80) — read findings using either with extra caution.
