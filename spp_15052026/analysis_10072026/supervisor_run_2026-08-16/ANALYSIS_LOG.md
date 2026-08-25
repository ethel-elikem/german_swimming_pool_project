# Analysis log — racialised discourse in German swimming-pool reviews

Read this first. Reorganised 2026-08-24 around the project's three research questions
(previously organised by analysis thread and scoped only to the NB-modelling thread —
see the note at the end of this section). Now covers **both** threads:

- The WHEN/WHERE/WHO/socioeconomic thread: `../analysis_qa_summary.md` (canonical summary),
  `../full_analysis.ipynb`, `../descriptive_analysis.ipynb`/`.md`, `../icwsm_results.ipynb`,
  `../02_socioeconomic_political_relations.ipynb`, `../redo_socio_political_relations.ipynb`,
  `../inter_annotator_agreement.ipynb`, and `/home/ethel/Desktop/swimming_pool_project/spp_15052026/robustness_check/`.
- The five-predictor NB-modelling thread: `notebooks/01_data_and_measures.ipynb` through
  `04_robustness_and_open_issues.ipynb`, with the three superseded source notebooks in
  `archive/` (see `archive/README.md`).

**RQ1** — How prevalent is racialised discourse, what form does it take, is it changing?
**RQ2** — Who produces it, and how are they expressing or reporting it?
**RQ3** — Where does it concentrate, and what kind of places produce it?

**September paper scope: RQ1 + RQ3.** RQ2 is documented in full below as completed work,
but every RQ2 row is marked **NOT IN SCOPE** — carried to a second paper. Numbering
throughout this file is fixed and does not get renumbered as items resolve.

*(Filename note: this consolidation was requested against a file named
`analysis_my_24082026.txt`, which doesn't exist anywhere in the repo. This document — the
one the request describes — is `ANALYSIS_LOG.md` in this directory; treating that as the
intended target rather than creating a second file. Flag if that's wrong.)*

---

## Part I — Chronological narrative

**Phase 0: corpus and codebook (shared).** 5,529 verified German swimming pools
reconciled from an administrative registry (`baederleben`, 5,230→5,204 deduped) and a
Google Maps scrape (11,848 candidates), by spatial (BallTree) then fuzzy-address
matching. 1,265,806 text reviews, classified through a single zero-shot LLM pipeline
(Claude Sonnet 4.6 via OpenRouter): `social` → `racialised` → four content dimensions
(theme/context/approach/connotation), hierarchical by construction.

**Phase A: WHEN/WHERE/WHO (the other thread's Part 1).** Built the constituency spatial
join, then worked through prevalence-over-time (a two-stage timestamp-recovery detour
first — 69.5%→99.5%→100% coverage), geographic distribution (Moran's I / LISA), and
reviewer-identity/ethnicity composition, ending in a synthesis asking whether the
WHERE pattern is really an ethnicity story or a density story (density won, mostly —
only the Arab-share lead survived controlling for it).

**Phase B: socioeconomic layering (the other thread's Part 2).** Brought in
census/political data after a corruption/state-repetition audit (43/66 columns flagged).
Found a deprivation cluster of predictors correlating with the rate, and explicitly
tested and rejected an AfD-vote-share hypothesis that ran counter to naive expectation —
it collapsed to ~0 once density/foreign-population share were controlled for.

**Phase C: a first predictive model (the other thread's Part 3), scoped as confirmatory.**
A joint NB model confirmed `foreign_population_pct` survives jointly and improves
out-of-sample prediction, explicitly framed as re-testing what Phase B already
established one-at-a-time, not a new finding. Residual spatial clustering
(Moran's I=0.118) flagged and left open.

**Phase D: the supervisor's four questions (2026-08-16/19 run, the NB-modelling thread).**
A parallel, more rigorous NB specification of the same rate (OLS→WLS→NB ladder, corrected
predictor set) plus Task B — does reviewer group (German/non-German) moderate the
predictor-outcome relationship. Building Task B surfaced a volume anomaly (non-German had
*more* review volume than German, backwards for reviews of German pools), which led
directly to discovering the "United States" name-classifier bug.

**Phase E: this consolidation, round 1 (2026-08-24, morning).** Reorganised the
NB-modelling thread alone into `01`–`04` + `ANALYSIS_LOG.md`, and closed five gaps:
corrected heteroskedasticity tests, verified (not re-derived) influence diagnostics,
three US-category Task B refits, a 500-rep permutation test on the resulting p-movement,
and reproducible seeded name spot-checks that found the same classifier bug, smaller, in
the France-predicted category.

**Phase F: this consolidation, round 2 (2026-08-24, this pass).** Absorbed the
WHEN/WHERE/WHO/socioeconomic thread, reorganised everything around RQ1/RQ2/RQ3 instead of
by thread, and audited the whole project against an explicit sub-question checklist
covering prevalence, form, change, who, how, where, place-type, and four cross-cutting
concerns (ethics, reproducibility, undocumented runs, a limitations register). That audit
is Part V below, and it is the part that matters most for what happens before September.

**Phase G: working the gap list (2026-08-24/25).** Surveyed the whole `spp_15052026/`
tree rather than trusting Phase F's file-by-file searches alone — surfaced
`data_annotation_validation/` (the directionality pilot and IAA results actually exist
and are committed, contra Phase F's assumption) and `data/coverage_report.parquet` (a
scraping-completeness audit nobody had cross-referenced against the final corpus).
Closed six of Part V's seven gaps: corrected `analysis_qa_summary.md` §A.3's stale IAA
claim; hand-checked a 60-review sample for two candidate uncaptured forms (1.10); reran
the WHEN trend on a fixed pool panel (1.13); fit and evaluated actual spatial-error/lag
models rather than just flagging the gap (3.13); froze the environment and wrote a
reproducibility note (X.2).

The other two gaps took real back-and-forth to get right, worth recording honestly
rather than only stating the final answer:

- **X.3** (orphaned `outputs/deck/` files) — first pass reproduced the NB-modelling
  cluster's numbers independently (confirming a joint-MLE alpha specification that fits
  better than this thread's moment-matched one, conclusions unchanged) but wrongly
  concluded the ethnicity/direction cluster's source code was gone for good. Asked to
  check scratch files and archives again rather than accept "genuinely lost," a wider
  whole-repo sweep for the cluster's most distinctive strings turned up
  `analysis_10072026/presentation_18082026.md` — a 9-item data-request brief (R1–R9)
  written for an earlier AI-assisted session, giving literal code and formulas for
  nearly everything in `outputs/{presentation,deck}/`. Not lost, just never saved into a
  notebook — the brief survived, the ephemeral session that ran it didn't. That closed
  X.3 fully and, as a side effect, surfaced a real formal COVID structural-break test
  (R6) that resolved 1.14 too, and fixed 2.3/2.11/2.12/3.9's citations, previously
  pointing at "orphaned" files.
- **1.5** (corpus representativeness) — first pass found a real scraping-completeness
  bias but reported it as "15.5% of the corpus missing," dividing an all-review-type
  shortfall (`data/coverage_report.parquet`'s numerator and denominator both count all
  review types — its generating script, `scripts/reviews_data_eda.py`, runs the check
  *before* the text-only filter) by the text-only corpus size, the wrong denominator.
  Corrected to 8.1% against the right population — then pushed further: that 8.1% was
  itself an *estimate* (all-type shortfall × an assumed text-rate), not a count, since
  the missing reviews were never scraped and their content was never seen. A second
  plausible assumption moved the number to 9.6%, proving it was assumption-dependent.
  Final version drops the estimate entirely and states only what's directly measured:
  the 52 flagged pools contribute 3.81% of the actual text corpus against a 0.94%
  pool-count share — no "missing" figure attached, because none is measurable here.
  Three drafts of one number in one afternoon (15.5% → 8.1% → 3.81%, only the last
  actually counted rather than inferred) — the lesson worth keeping for the rest of this
  file: state what's counted, not what's inferred through an assumption, when the two
  get labelled the same way in prose.

Only X.1 (the ethics statement) remains untouched.

---

## Part II — Hierarchical question tree

```
RQ1. How prevalent is racialised discourse, what form does it take, is it changing?
   Prevalence
     1.1 Corpus size and coverage?                    [PARTIAL - 5,529 pools/1.27M reviews, no coverage-%, 1.1]
     1.2 Social/racialised shares, hierarchy?          [SETTLED - 39.8%/1.54%, 1.2]
     1.3 Distribution across pools?                    [SETTLED - widespread, breadth not one venue, 1.3]
     1.4 Pool-level sparsity?                           [PARTIAL - handled at constituency level only, 1.4]
     1.5 Corpus representativeness?                     [PARTIAL - scraping under-covers large pools, 1.5]
   Form
     1.6 The codebook?                                  [SETTLED - hierarchical, 4 dims, 1.6]
     1.7 Classification accuracy (social/racialised)?   [SETTLED - kappa/precision/recall exist, 1.7]
     1.8 Dimension distribution?                        [SETTLED - 1.8]
     1.9 Dimension co-occurrence?                        [PARTIAL - 1/6 pairs tested, 1.9]
     1.10 Recurring uncaptured forms?                    [PARTIAL - both real, ~10% each, single-coder n=60, 1.10]
   Change
     1.11 Overall temporal trend?                        [SETTLED - accelerating rise, 1.11]
     1.12 Per-dimension change?                           [SETTLED - 1.12]
     1.13 Trend vs. compositional artifact?               [SETTLED - survives fixed-panel rerun, 1.13]
     1.14 Discontinuities (COVID)?                        [SETTLED - formal covid-term panel test found, 1.14]

RQ2. Who produces it, and how are they expressing or reporting it?  [NOT IN SCOPE for September - carried to paper 2]
   Who
     2.1 How was reviewer group inferred?                 [NOT IN SCOPE - 2.1]
     2.2 Distribution of predicted groups?                [NOT IN SCOPE - 2.2]
     2.3 Classifier accuracy / failure modes?              [NOT IN SCOPE - 2.3]
     2.4 Formal validation status?                          [NOT IN SCOPE - 0/599 coded, 2.4]
     2.5 Does group moderate the relationship?              [NOT IN SCOPE - p=0.133, 2.5]
     2.6 Is that null a classifier artifact?                [NOT IN SCOPE - 2.6]
     2.7 Is "who" answerable with this data?                [NOT IN SCOPE - 2.7]
   How (attribution vs. reporting)
     2.8 What distinguishes the two?                        [NOT IN SCOPE - 2.8]
     2.9 How often does each occur?                          [NOT IN SCOPE - 40/8/2 pilot, 2.9]
     2.10 Codeable reliably?                                  [NOT IN SCOPE - 2.10]
     2.11 Separable from the 4 existing dimensions?           [NOT IN SCOPE - 2.11]
     2.12 Correlates with predicted ethnicity?                [NOT IN SCOPE - 2.12]
     2.13 Enough volume to model at constituency level?       [NOT IN SCOPE - ~5.2/constituency, 2.13]
     2.14 What does the scheme not yet handle?                [NOT IN SCOPE - 2.14]

RQ3. Where does it concentrate, and what kind of places produce it?
   Where
     3.1 Unit of analysis, spatial join?                     [SETTLED - 299 constituencies, 3.1]
     3.2 Geographic distribution, descriptive?                [SETTLED - 3.2]
     3.3 Spatial clustering?                                   [SETTLED - Moran's I=0.281, p=0.001, 3.3]
     3.4 Local clusters?                                       [SETTLED - Ruhr/Berlin hotspots, 3.4]
   What kind of places
     3.5 Outcome measure, why?                                 [SETTLED - 3.5]
     3.6 Predictors, sources, construction?                    [SETTLED - 3.6]
     3.7 Why negative binomial?                                [SETTLED - 3.7]
     3.8 Which predictors, how strong?                         [SETTLED - 3.8]
     3.9 Is the model well specified?                          [SETTLED - 3.9]
     3.10 Robust to influential constituencies?                [PARTIAL - category-robust, not influence-specific, 3.10]
     3.11 Predictor-set selection stable?                       [SETTLED - unstable, documented, 3.11]
     3.12 rate_total agreement?                                 [SETTLED - 3.12]
     3.13 Residual spatial dependence resolved?                 [SETTLED - spatial correction attempted, insufficient, accepted as limitation, 3.13]

Cross-cutting
   X.1 Ethics / broader-impact statement?                       [NOT RUN - no location, X.1]
   X.2 Reproducibility documentation?                            [SETTLED - requirements.txt + README written, X.2]
   X.3 Undocumented prior runs, swept?                            [SETTLED - traced to presentation_18082026.md's brief, X.3]
   X.4 One consolidated limitations register?                     [PARTIAL - 2 separate registers, X.4]
```

---

## Part III — Status table

| id | sub-question | section | notebook | what was run | answer | status | what would change it |
|---|---|---|---|---|---|---|---|
| 1.1 | Corpus size and coverage | §A.1; date-range cell | `analysis_qa_summary.md`; `descriptive_analysis.md` | Reconciliation provenance (BallTree + fuzzy match); timestamp range read off the corpus | 5,529 pools, 1,265,806 reviews; span 2009–2026, working window 2016–2025; no stated %-of-Germany coverage | PARTIAL | an external total-pool-count baseline to turn this into a coverage % |
| 1.2 | Social/racialised shares, hierarchy | §A.2 | `analysis_qa_summary.md` | Classification pipeline counts | 504,072/1,265,806 (39.8%) social; 7,768/504,072 (1.54%) racialised; racialised implies social by construction | SETTLED | — |
| 1.3 | Distribution across pools | Choropleths section; §B.2 | `descriptive_analysis.md`; `analysis_qa_summary.md` | Pool-level racialised counts; per-constituency breadth (%-pools-affected) and dominant-pool tests | 1,802/5,528 pools (32.6%) have ≥1 racialised review; rate tracks breadth (ρ=0.65) not top-pool share (ρ=0.09) — widespread, though 90/247 dominant pools are independently significant outliers | SETTLED | — |
| 1.4 | Pool-level sparsity | `full_analysis.ipynb` ~cell 99; `descriptive_analysis.md` ~L4241 | `full_analysis.ipynb`; `descriptive_analysis.md` | A pool-level reliability floor was proposed, tested, and explicitly rejected | Floor rejected as arbitrary (would've excluded 62% of constituencies); sparsity handled via constituency-level `reliable_n` instead | PARTIAL | a full per-pool total-review histogram — never plotted |
| 1.5 | Corpus representativeness | `data/coverage_report.parquet`, generated by `scripts/reviews_data_eda.py::check_coverage`; `analysis_10072026/gmaps_data/master_corpus_final.parquet` | `data/` (scraping-completeness audit, other thread); `analysis_10072026/gmaps_data` | Cross-referenced the scraping-completeness report against the final corpus; Spearman(pool size, coverage %); **corrected 2026-08-25 to use only directly-measured text-review counts, no all-type-to-text conversion factor** | The original question (matched vs. unmatched pool *selection* bias) is still unanswered. A real bias was found, stated now in text-review terms throughout, nothing estimated: coverage % (Google's all-review count vs. what got scraped) falls significantly as pool size rises (ρ=−0.272, p=4e-121) — larger, busier pools are systematically under-scraped by Google's own count. Flagging the 52/5,528 final-corpus pools (0.94% of pools) that fall below that completeness threshold and checking the actual text corpus directly: those 52 pools contribute **48,269 of the corpus's 1,265,806 text reviews — 3.81%, a 4.1× over-representation relative to their 0.94% pool-count share.** That's the whole of what's directly knowable: incompletely-scraped-by-Google's-count pools are still disproportionately large *text* contributors, because they're large venues. Whether a complete scrape of them would add still more text reviews, and in what proportion, is not something this data can answer — there's no independent measure of "how many text reviews this pool should have," so no version of that number should be reported as known (earlier drafts of this row did exactly that, first at 15.5%, then "corrected" to 8.1%/9.6% — both were estimates dressed as findings; retracted) | PARTIAL | the selection-bias question (matched vs. unmatched) still needs the original candidate lists — see Part V |
| 1.6 | The codebook | §A.2 | `analysis_qa_summary.md` | Codebook definition | Hierarchical: social → racialised → 4 dimensions; single zero-shot LLM | SETTLED | — |
| 1.7 | Classification accuracy (social/racialised) | full notebook | `inter_annotator_agreement.ipynb` | Two-annotator hand-coding vs. model output, precision/recall/Cohen's κ | social: κ=0.980/0.709, precision 100%/88%; racialised: precision 98.0%/99.6%, recall false-negative rate 0/97 (CI 0–3.8%); `theme` not yet validated | SETTLED | a larger recall sample — 3.8% CI upper bound still permits corpus-level misses. `analysis_qa_summary.md` §A.3 calling this "not reported yet" is stale (see X.3/X.4) |
| 1.8 | Dimension distribution | §A.2, §B.2 | `analysis_qa_summary.md` | Dimension counts on racialised=yes | 100% coverage on 4 dimensions; theme most granular (11 categories, Crowd largest) | SETTLED | — |
| 1.9 | Dimension co-occurrence | "Content baseline" cell | `descriptive_analysis.md` | χ² + Cramér's V, theme×connotation only | theme×connotation: χ²=413.9, df=20, p=2.6e-75, V=0.122; other 5 pairs untested for pure co-occurrence; theme confirmed multi-label (1.77 avg, 61.6% >1 label) | PARTIAL | testing the remaining 5 dimension pairs |
| 1.10 | Recurring uncaptured forms | — | this consolidation, single-coder pass | Read 60 randomly-sampled racialised reviews (seed=99) and flagged each for the two candidate forms | Both are real and recurring: **"presence as complaint"** (e.g. "Excellent but not suitable for Arab families"; "Always too crowded. Lots of foreign people") in ~10% (6/60); **"decline"** narratives (e.g. "what's become of..."; "what Germany will look like in 10-20 years"; "since a certain group... became more prevalent") in ~8–10% (5–6/60). Both cut across existing themes (mostly Crowd/Belonging/Etiquette) rather than being a distinct theme category — that's likely why the taxonomy didn't surface them as their own label | PARTIAL | a real, codebook-governed sample (not a single uncalibrated coder's 60-review pass) — this is a first-pass estimate, not a validated prevalence figure |
| 1.11 | Overall temporal trend | §B.1 | `analysis_qa_summary.md` | Logistic regression on centered year | 2016–2025 slope 0.0824 log-odds/yr (p=6.8e-65); 2021–2025 2.7× steeper (p=5.4e-92) | SETTLED | — |
| 1.12 | Per-dimension change | §B.1 | `analysis_qa_summary.md` | Year-level tests per label | Crowd declining (60%→40%), incident-framing rising (27%→49%), negative connotation rising (85%→92–94%), Clothing steepest riser | SETTLED | — |
| 1.13 | Trend vs. compositional artifact | — | this consolidation | Rebuilt the year-trend logistic regression restricted to the fixed panel of pools with ≥1 social review in both 2016–2018 and 2023–2025 (3,856/5,288 pools, 72.9%; 95.1% of the window's reviews) | Trend survives essentially unchanged on the fixed panel: slope 0.0837 (p=1.3e-65) vs. the full-corpus 0.0824 (p=6.8e-65) — a 1.6% difference. The rise is not explained by new pools or a growing reviewer population entering Google Maps over the window | SETTLED | a result outside this ~1.6% band, or a similar null on a stricter panel definition |
| 1.14 | Discontinuities (COVID) | "Pre-2021 vs. post-2021" section; `outputs/presentation/R6_panel_covid_models.csv` | `descriptive_analysis.md`; `analysis_10072026/presentation_18082026.md` R6 (recovered spec, see X.3) | Quantified before/after volume + composition comparison; **a real formal test, found this pass:** constituency-year NB panel refit three ways — plain trend, trend+covid-dummy (2020–21), trend+post2021-dummy | 2020/2021 volume collapse (33,191/38,093 vs. 46,451–80,124); composition flat. **R6:** plain trend rate ratio 1.077 [1.059,1.094]; with an explicit COVID term, rate ratio drops to 1.066 [1.050,1.083] and fits better (AIC 10364 vs. 10425) — part of the raw trend was pandemic-inflated, but a real, highly significant rise (p=1.2e-15) survives once COVID is accounted for | SETTLED | — |
| 2.1 | How reviewer group was inferred | cell 5; §4.2.1 | `archive/supervisor_run_2026-08-16.ipynb`; `04` | Name→country tool + Arab-League consolidation + group5/group4 mapping | Germany/Arab/Türkiye/Unknown kept distinct; everything else folds to Other (group5) or Europe/Other (group4) | NOT IN SCOPE | carried to paper 2 |
| 2.2 | Distribution of predicted groups | §4.3 | `04` | `value_counts` on `ethnicity_country_name` | Germany 500,951; United States 170,807; Russia 28,567; Poland 25,981; Italy 22,195; Türkiye 21,671 | NOT IN SCOPE | carried to paper 2 |
| 2.3 | Classifier accuracy / failure modes | §4.3; `outputs/deck/F15/F16`, traced to `presentation_18082026.md` R1 (see X.3); researcher's notes (uncommitted) | `04`; `analysis_10072026/outputs/deck` + its recovered spec; uncommitted | Seeded name spot-checks (this thread); R1's degraded-name-flag method (exact regex rules for digits/single-token/repeats/no-letters/etc., given in the brief) applied and reflected in F15's "clean-names-only" cut; researcher's own manual review (other two failure modes) | Three failure modes: (a) population-prior misdirection (US/France), file-backed; (b) script/geography conflation (Ukrainian labelled Russian); (c) confident labels on junk/pseudonym handles — (b)/(c) researcher-reported only. Two independent supporting checks, now fully traced: **F16** — per-constituency non-German-predicted share correlates strongly with independently-measured census `foreign_population_pct` (ρ=0.802, p=1.5e-68, n=299), a real aggregate convergent-validity signal despite (a)'s individual-level noise; **F15** — Arab:Germany and Türkiye:Germany own-group rate ratios are essentially unchanged restricting to unambiguous ("clean") names only (R1's degraded-name flags removed), i.e. the group-level rate story isn't an artifact of the ambiguous names (c) flags | NOT IN SCOPE | carried to paper 2; (b)/(c) need the same seeded-sample treatment (a) got |
| 2.4 | Formal validation status | full notebook | `archive/ethnicity_validation_scoring.ipynb` | Sampling + scoring pipeline built | 0/599 hand-coded; no confidence score anywhere in the pipeline (verified directly). F16's aggregate convergent validity (above) is real support at the *aggregate* level but doesn't substitute for individual-level precision/recall | NOT IN SCOPE | carried to paper 2 — but see Part V, worth starting early given hand-coding lead time |
| 2.5 | Does group moderate the relationship | §4.2.3 | `04` | Stacked interaction NB model, cluster SE | Joint Wald χ²=8.45, df=5, p=0.133 — no significant difference detected | NOT IN SCOPE | carried to paper 2 |
| 2.6 | Is that null a classifier artifact | §4.2.5/4.2.6 | `04` | 3 US-category refits + 500-rep permutation test | p moves to 0.077–0.087 under 2/3 fixes, but 38% of random same-sized reassignments do the same; the US group's own low racialised rate vs. random draws is the more informative signal | NOT IN SCOPE | carried to paper 2 |
| 2.7 | Is "who" answerable with this data | §B.3/B.4 (other thread); §4 (this thread) | `analysis_qa_summary.md`; `04` | Synthesis across both threads | Composition: answerable, real/modest/specific (Arab survives density control, Türkiye doesn't). Moderation: not detectable at current power, and that null is entangled with uncorrected classifier noise | NOT IN SCOPE | carried to paper 2; firms up once 2.4 closes |
| 2.8 | Attribution vs. reporting, definition | — | researcher's notes (uncommitted) | Conceptual definition | Attribution ascribes a category to others; reporting alleges unfavourable treatment linked to one; separated by whether a harm claim attaches | NOT IN SCOPE | carried to paper 2 |
| 2.9 | How often each occurs | `data_annotation_validation/directionality/directionality_hand_code_50.csv` | `data_annotation_validation/directionality/` (already committed — corrected during this pass) | 50-review hand-coded pilot, via `scripts/generate_directionality_pilot.py` | 40 attribution / 8 reporting / 2 both (verified directly against the file) — reporting present in ~20% (Wilson CI ~11–33%) | NOT IN SCOPE | carried to paper 2 |
| 2.10 | Codeable reliably | `data_annotation_validation/directionality/directionality_pilot_100_key.csv` | `data_annotation_validation/directionality/` | A 100-review candidate sheet exists, pre-attached with theme/context/connotation/approach and `predicted_group5` — ready for a second-coder pass, but not yet coded | Infrastructure built, coding not done | NOT IN SCOPE | run the second-coder/LLM pass on the existing 100-review sheet — the sample itself doesn't need building |
| 2.11 | Separable from the 4 existing dimensions | `outputs/deck/T11_theme_trajectory_by_direction.csv`, exact formula traced to `presentation_18082026.md` R9 (see X.3) | `analysis_10072026/presentation_18082026.md` R9; `outputs/deck` | A theme-by-direction trajectory analysis exists at corpus scale, on a fully transparent **rule-based proxy** for direction (R9's exact definitions: `proxy_attributed = context=='generalised' & theme in [Crowd,Etiquette]`; `proxy_reported = context=='incident' & theme in [Staff,Access,Belonging,Language]`) — not the real 50-review hand-coded labels, and by construction partly circular with theme/context (the proxy is *built from* theme+context, so "direction correlates with theme" is partly true by definition here) | "Reported"-proxy themes (Staff/Access/Belonging/Language) rise 2016→2025 (Access ×3.1, Belonging ×1.5, Staff ×1.4) while "attributed"-proxy themes (Crowd/Etiquette) fall or stay flat (×0.7, ×0.9) — informative as a rough stand-in, but the real question (is direction separable from theme/context, or redundant with them) can't be answered by a proxy defined *from* theme/context | PARTIAL | correlating the *real* 50-review (or a larger) hand-coded direction label against theme directly — the only way to escape the proxy's circularity |
| 2.12 | Correlates with predicted ethnicity | `outputs/deck/T12_proxy_directionality_ordered.csv`, same R9 proxy (see X.3) | `analysis_10072026/presentation_18082026.md` R9; `outputs/deck` | Same rule-based proxy, cross-tabulated against `ethnicity_group4` | Germany: 17.3% proxy-"reported" / 69.0% proxy-"attributed"; Unknown/Other/Europe closer to balanced (~24–28% / ~52–55%) — the proxy suggests Germans skew toward attributing, other groups more mixed, consistent with 2.7's composition story | PARTIAL | the retrospective test still needs the *real* hand-coded direction label (2.9's pilot, or a scaled-up version) cross-tabbed against predicted ethnicity — the proxy is suggestive, not the real test |
| 2.13 | Enough volume for constituency-level modelling | — | computed here, from the 2.9 pilot rate | 7,768 × 0.20 ÷ 299 | ≈5.2 reporting-labeled racialised reviews per constituency on average — below Task B's own median-5 feasibility bar before even accounting for skew | NOT IN SCOPE | carried to paper 2; needs direction coded at full-corpus scale, not a 50-review pilot |
| 2.14 | Unhandled cases | — | researcher's notes (uncommitted) | None formalised | Attribution aimed at Germans ("typisch deutsch"); self-reference with no ascription/harm; reviews contesting racism occurred; regional/East-West coding; "target" is 2 variables that split under reporting | NOT IN SCOPE | carried to paper 2 |
| 3.1 | Unit of analysis, spatial join | §1.10; cell 4 | `01`; `archive/supervisor_run_2026-08-16.ipynb` | Constituency-level join, nearest-constituency assignment inside the Germany boundary | n=299 constituencies (Wahlkreise, EPSG:25832) | SETTLED | — |
| 3.2 | Geographic distribution, descriptive | §3.4; §B.2 | `03`; `analysis_qa_summary.md` | Choropleths, East/West split | State range ~4× (Mecklenburg-Vorpommern 0.81% to Saarland/Berlin/Bremen ~3.0–3.1%); `is_east`, 48/299 | SETTLED | — |
| 3.3 | Spatial clustering | §3.4; §B.2 | `03`; `analysis_qa_summary.md` | Global Moran's I, Queen contiguity, 999 permutations | I=0.281, p=0.001 — cross-confirmed independently in both threads | SETTLED | — |
| 3.4 | Local clusters | §3.4; §B.2 | `03`; `analysis_qa_summary.md` | LISA (Moran_Local) | Ruhr-NRW + Berlin High-High hotspots, plus a smaller Baden-Württemberg pair; coldspots broad/diffuse | SETTLED | — |
| 3.5 | Outcome measure, why | §1.4/1.7/1.9 | `01` | NB refits of social/share/total | `rate_social` primary; additivity holds to 0.024; denominator effect ~1.6% at mean | SETTLED | — |
| 3.6 | Predictors, sources, construction | §1.10 | `01` | — | 5 final predictors + sources; `is_east`/`in_split_city` construction; border distance EPSG:25832 | SETTLED | — |
| 3.7 | Why negative binomial | §2.1–2.3 | `02` | OLS→WLS→NB ladder, corrected heteroskedasticity, dispersion ladder | 94.2% OLS→WLS shift; no heteroskedasticity (corrected tests); Poisson dispersion 9.60 → NB 0.826 | SETTLED | — |
| 3.8 | Which predictors, how strong | §3.1/3.2 | `03` | Final NB model; predicted rates at p25/p75 | `foreign_population_pct_c`, `log_population_density_c`, `sgb_ii_recipients_per_1k_c` significant; `is_east` marginal (p=0.070); `in_split_city` negative; 1.23%→1.65% across foreign-pop IQR | SETTLED | — |
| 3.9 | Is the model well specified | §3.3 | `03` | Dispersion, VIF, Cook's distance; independently verified against an alternative alpha-estimator — requested verbatim in `presentation_18082026.md` R4.4 (see X.3), executed as `outputs/deck/T6-T8`/`outputs/presentation/REFIT_*` | Dispersion 0.826; max VIF 4.61; top-5 influence identified (Offenburg, Lörrach, Duisburg II, Dahme-Spreewald, Chemnitz). **Caveat found this pass:** `03`'s dispersion figure uses a moment-matched (plug-in) α=0.375. R4.4 asks for the version that estimates α jointly by MLE instead — that gives α=0.1854513, fits notably better (AIC 2196.6 vs. 2241.7), but reads as *overdispersed* (1.502) instead of underdispersed (0.826) — reproduced independently and confirmed exact (`statsmodels.NegativeBinomial`, joint MLE). Coefficients, signs, and every significance call are essentially unchanged between the two (max rate-ratio difference 0.003) — the substantive results in `03` are robust to this choice, but the "well-specified, dispersion 0.826" framing is itself estimator-dependent and should say so | SETTLED | a joint-MLE refit whose coefficients diverge substantively from the reported ones — checked, they don't |
| 3.10 | Robust to influential constituencies | `B10`; §3.3 | `archive/supervisor_run_2026-08-16.ipynb`; `03` | 5-subset robustness (no-split-city, ≥10 racialised, West-only, no-NRW) | Coefficients stable across all 5 subsets | PARTIAL | a refit excluding specifically the top-5-by-Cook's-distance constituencies — the subsets tested are category-based, not influence-based |
| 3.11 | Predictor-set selection stability | §2.6 | `02` | Stepwise-prune vs. full-5 comparison | Pruned AIC worse (2300.8 vs. 2287.8); coefficient inflated (0.0546 vs. 0.0356) | SETTLED | — |
| 3.12 | `rate_total` agreement | §4.1 | `04` | Full-5-predictor `rate_total` refit | Same sign, overlapping CI on all 5 predictors | SETTLED | — |
| 3.13 | Residual spatial dependence resolved | §4.6; this consolidation | `04` | Consolidated Moran's I table (§4.6); this pass additionally fit ML spatial-error and spatial-lag models (`spreg.ML_Error`/`ML_Lag`, Queen weights) on `log(rate_social)` as a direct test of whether spatial correction actually helps | Range 0.082–0.281 across models, significant everywhere. The spatial-error fit found real spatial structure (λ=0.18) but its residual Moran's I (0.084, p=0.012) is barely different from plain OLS on the same scale (0.081, p=0.012) — **spatial-error/lag correction, at this specification, does not meaningfully reduce the residual dependence.** **Decision:** accept the residual spatial dependence as a stated limitation rather than pursue a full spatially-structured count model (e.g. a spatial random-effects NB) — the simple fix doesn't work, and the complex one is out of scope for this timeline | SETTLED | evidence that a spatially-structured NB (not the OLS-scale approximation tested here) behaves differently — genuinely untested |
| X.1 | Ethics / broader-impact / data statement | — | — | Repo-wide search | None found anywhere | NOT RUN | an ICWSM-format broader-impact/ethics statement — reviewer-identifier handling, anonymisation, scraping ToS position |
| X.2 | Reproducibility documentation | `../../README.md`; `../../requirements.txt` | repo root | Froze the active `.venv` (`pip freeze`); wrote a project-level reproducibility note (environment, seed convention, what regenerates from where, pointer to X.3's open items) | `requirements.txt` — 197 packages, Python 3.12.3. `README.md` — no single project-wide seed policy exists (each analysis sets its own; documented rather than imposed retroactively), plus the outputs→source map X.2 asked for | SETTLED | — |
| X.3 | Undocumented prior runs, swept | `outputs/B10_cooks_distance_top10.csv`; `analysis_10072026/presentation_18082026.md` (recovered spec for the whole `outputs/{presentation,deck}` cluster) | `supervisor_run_2026-08-16`; `analysis_10072026/` | Sampling sweep found the orphaned files; a second, wider sweep (prompted by "check your scratch files and archives") found `presentation_18082026.md` — a 9-item data-request brief (R1–R9), written for an earlier AI-assisted session, that specifies exactly what every one of these files computes | **Not lost — traced.** `presentation_18082026.md` gives literal code/formulas for what became `outputs/presentation/*` and (renamed into presentation-ready F#/T# labels) `outputs/deck/*`: R4.4 → the joint-MLE-alpha model (T6/T7/T8/`REFIT_*`, reproduced exactly here, see 3.9); R1's ACUA degraded-name flags → F15's clean-names-only check; R9's exact proxy-direction formula (`proxy_attributed = context=='generalised' & theme in [Crowd,Etiquette]`; `proxy_reported = context=='incident' & theme in [Staff,Access,Belonging,Language]`) → T11/T12/F14, now fully transparent, see 2.11/2.12; **R6 → a real, previously-missed formal COVID structural-break test, see 1.14.** The likely mechanism: an earlier AI-assisted session executed this brief in an ephemeral scratch environment and saved only the output files, never a notebook — the same failure mode this session's own scratch scripts are exposed to (see the `README.md` reproducibility note, updated with this as an explicit process lesson) | SETTLED | — |
| X.4 | One consolidated limitations register | "Open threads/limitations" (4 bullets); §6 (4 bullets) | `analysis_qa_summary.md`; `robustness_check/ROBUSTNESS_CHECK_REPORT.md` | — | Two separate 4-bullet registers exist, each scoped to its own thread; never merged | PARTIAL | one merged, project-wide limitations list |

---

## Part IV — Paper mapping

**Data section:** RQ1's prevalence/form material (`01_data_and_measures.ipynb` +
`analysis_qa_summary.md` §A) and RQ3's data/measures material (`01`). Nothing blocks
writing this. Do include the scraping-completeness finding from 1.5 as a stated
data-quality caveat: large pools are scraped less completely by Google's own review
count, and the 52 most-affected final-corpus pools still account for 3.81% of all text
reviews against a 0.94% pool-count share (4.1× over-representation) — real, directly
measured, and better disclosed than discovered by a reviewer. Do **not** state a
"% of the corpus missing" figure — that number was estimated twice in this log's history
(15.5%, then 8.1%/9.6%) and retracted both times; it isn't measurable from this data.

**Methods section:** RQ1's classification methodology (1.6/1.7 — genuinely strong,
kappa/precision/recall all exist, and `analysis_qa_summary.md` §A.3 is now corrected to
say so) plus RQ3's modelling methodology (`02`). 3.11's selection-instability finding and
3.9's alpha-estimator sensitivity (moment-matched vs. joint-MLE — same conclusions,
different dispersion reading) both belong here as stated limitations, not buried.

**Results section:** RQ1's prevalence/change results (`analysis_qa_summary.md` §B.1,
now including 1.13's fixed-panel confirmation that the trend isn't a corpus-growth
artifact) plus RQ3's spatial and model results (`03`, and the WHERE section of the other
thread). Clean and complete as written on both sides.

**Limitations/discussion:** `04_robustness_and_open_issues.ipynb` plus this file's Part V.
**One hard blocker for September: X.1** — no ethics/broader-impact statement exists, and
ICWSM's submission checklist requires one. Everything else that was open going into this
pass (1.13, 3.13, X.2 fully; 1.5, 1.10, X.3 partially) has now been worked — see Part V's
note on what residual threads remain inside the PARTIAL rows.

**RQ2 — explicitly out of scope for this paper.** Fully documented above (2.1–2.14) for
continuity into a second paper. The single highest-value action for that future work,
independent of this paper's timeline, is starting 2.4's hand-coding early — kappa
computation and the downstream re-tests (2.6, 2.7's firm verdict) all sit behind it, and
hand-coding has the longest lead time of anything in this file.

---

## Part V — Gaps

Every Part III row with status OPEN or NOT RUN (PARTIAL rows are write-up-ready with a
caveat, not gaps; NOT IN SCOPE rows are RQ2, carried to paper 2 on its own timeline).

**Updated 2026-08-25, third pass.** Six of the original seven rows got worked and closed
this pass — 1.13, 3.13, X.2 fully SETTLED; 1.5 and 1.10 PARTIAL with a real remaining
thread (see the note below the table); X.3 initially marked PARTIAL, then fully SETTLED
on a follow-up pass that traced the "orphaned" files to `presentation_18082026.md` (see
Part I, Phase H), which also resolved 1.14 as a side effect. Only **X.1 remains a true,
untouched gap.**

| id | gap | what closing it requires | who can do it | blocking September? | estimated effort |
|---|---|---|---|---|---|
| X.1 | No ethics/broader-impact statement | Write an ICWSM-format statement: reviewer-identifier handling, anonymisation, position on Google's scraping ToS | Researcher — requires an institutional/ethical judgment call, not just code | **YES — ICWSM's submission checklist requires one** | 0.5–1 day |

**Residual threads inside rows now marked PARTIAL** (not full Part-V gaps under this
file's schema, since PARTIAL rows are write-up-ready — but real open work remains):

- **1.5** — the completeness bias (large pools under-scraped) is now settled and
  quantified; the *original* question (do pools that failed baederleben/Google-Maps
  reconciliation differ systematically from those that succeeded) is still fully
  unanswered and would need the original unreconciled candidate lists, if they still
  exist.
- **1.10** — real forms confirmed at ~10% each on a 60-review single-coder pass; a
  properly-coded, codebook-governed sample (ideally by the researcher, or a second
  coder for agreement) would turn this into a citable number rather than a first-pass
  estimate.
- **X.3 is now fully closed**, not just partially — see Part I Phase H. The whole
  `outputs/{presentation,deck}` cluster traces to `presentation_18082026.md`'s R1–R9
  brief; the NB-model pieces (`T6`–`T8`, `REFIT_*`) were independently reproduced to full
  precision, and the ethnicity/direction pieces (`F14`–`F16`, `T11`–`T14`) now have an
  exact, transparent formula (2.11/2.12) rather than being unexplained numbers.
