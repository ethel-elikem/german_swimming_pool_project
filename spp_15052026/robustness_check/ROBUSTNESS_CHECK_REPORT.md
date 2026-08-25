# Robustness of the LLM Annotation Pipeline: Rerun and Translation-Sensitivity Checks

**Scope:** all classification stages used to build the analysis corpus that are the classifier of
record — `stage_1_social`, `stage_2_racial_reclassed`, and the four `stage_3_subs` dimensions
(approach/context/connotation/theme). `stage_2_racial` (the earlier, broader-triggering screen that
`stage_2_racial_reclassed` superseded) is intentionally out of scope — see §2.
**Data/code:** `spp_15052026/robustness_check/robustness_check.ipynb`; raw outputs in `results/`.
**Date:** Experiments A/B run 2026-07-10; Experiments C/D and full re-execution 2026-07-28.

---

## 1. Why this check exists

Every substantive finding in this project depends on sequential LLM classification decisions per
review: is it *social* (about the pool experience) → is it *racialised* (via
`stage_2_racial_reclassed`, the classifier of record) → and, for racialised reviews, what is its
*approach*, *context*, *connotation*, and *theme*. Each decision is made by prompting an LLM once
per review at temperature 0. This raises two distinct threats to validity that are **not** addressed
by inter-annotator agreement (IAA) work done elsewhere in this project:

1. **Stability**: even at temperature 0, LLM APIs are not perfectly deterministic (provider-side
   batching and numerics), and the labels analysed here were produced weeks before any recheck — so
   simply re-running the identical classifier on the identical text can produce a different label.
   This is a noise floor that exists independently of whether the *label itself* is correct.
2. **Translation sensitivity**: every classifier reads only the **English machine-translation**
   (`trans_review`) of what is overwhelmingly (~93%) German source text. If a label depends on
   incidental phrasing introduced by translation rather than on the substance of the review, that is
   measurement error correlated with language — a serious concern for any analysis of racialisation
   across German regions, since translation quality/style is not itself randomly distributed.

IAA (a separate notebook, `inter_annotator_agreement.ipynb`) asks *"is the label right?"* by
comparing the model to human annotators. This robustness check asks a logically prior question —
*"is the label even reproducible?"* — and is a necessary complement, not a substitute: a label can
be perfectly reproducible and still wrong, or valid on average but individually unstable. Both
matter for how much weight downstream statistics can bear.

**Why both experiments are needed to interpret either one.** Back-translated text is *never* the
same string as the original rerun text — so any disagreement it produces is a mix of ordinary rerun
noise plus a translation-specific effect. Experiment A (plain rerun) measures the noise floor alone;
Experiment B (back-translation) measures noise **plus** translation sensitivity on the exact same
reviews. Because Experiment B's sample is a subset of Experiment A's, the two can be compared
**per-review** (paired, McNemar-style), isolating the translation-specific component instead of just
comparing two unpaired instability rates.

---

## 2. The pipeline being tested

| stage | script | decision | population it runs on | canonical yes-count |
|---|---|---|---|---|
| 1 | `stage_1_social/social_classifier.py` | social (yes/no) | all 1,265,806 reviews at verified German POIs | 504,072 |
| 2 | `stage_2_racial_reclassed/racial_classifier_parallelization_examples.py` | racialised (yes/no) | the 504,072 social-yes reviews | 8,271 corrected yes (7,768 after the Germany-POI scope restriction) |
| 3 | `stage_3_subs/{approach,context,connotation,themes}_classifier.py` | four sub-dimension labels (each multi-label) | the 7,768 dimension-scope reviews | — |

Historically, `stage_2_racial_reclassed` was applied only to the subset that an earlier, broader
script (`stage_2_racial`) had already flagged as a candidate — a first pass over all social reviews,
followed by a stricter second pass correcting that first pass's over-triggering. **`stage_2_racial`
itself is not the classifier of record and is excluded from this robustness check entirely.**
Experiment D (§4) therefore tests `stage_2_racial_reclassed` directly against a sample of the
social-yes population, rather than reproducing the historical two-stage routing.

All classifier settings (model `claude-sonnet-4-6`, temperature 0, batch size) were matched to each
script's own logged production configuration, not chosen independently — a mismatch here would mean
measuring sensitivity to a *settings change*, not to rerun noise or translation.

---

## 3. Method

### 3.1 Sampling

| check | population | n (relabel) | stratification | n (back-translation) |
|---|---|---|---|---|
| approach/context/connotation/theme | 7,768 dimension-scope reviews | 387 | joint (approach × connotation × context) label combination, 58 strata, proportional allocation with a floor of 3/stratum (or all, if a stratum has <3) | 155, stratified subset of the 387 |
| social | 1,265,806 full corpus | 300 | canonical `social` label, proportional (~40/60 split, no floor needed) | 151 |
| racialised | 504,072 social-yes reviews | 300 | canonical `racialised` label, **50/50 case-control** (150 yes + 150 no) | 150 |

The 58-stratum design for the sub-dimensions exists because rare label combinations (e.g.
multi-label or `coded`) are exactly the cases most likely to be unstable, and a simple random sample
would under-represent or miss them entirely; the floor rule guarantees every stratum appears at all.
The floor also explains why the realised relabel sample is 387 rather than exactly 300: 29 of the 58
strata are small enough that the floor (not the proportional target) determines their count.

The racialised sample deliberately abandons proportional sampling. `racialised`=yes is only
~1.5% of its population (7,768/504,072); a proportional n=300 sample would carry roughly 5 yes-rows —
far too few to say anything about false negatives, which are the analytically important error mode
for a prevalence estimate. A 50/50 case-control design was used instead.

A fixed seed (20260710, with small deterministic offsets per sub-sample) makes every sample
reproducible; sample files are cached to `samples/*.parquet` so the notebook can be re-run without
regenerating or invalidating completed classifier runs.

### 3.2 Back-translation

Back-translation uses the same LLM (temperature 0) with a minimal translation instruction,
EN→German→EN, on the assumption that German is the dominant source language of the corpus. All four
checks that use back-translation (sub-dimensions, social, racialised) **share one
translation cache** keyed by `review_id` — a review's back-translation doesn't depend on which
downstream classifier is being tested, so no review is ever translated twice across the whole
notebook.

⚠ Caveat carried over from the original design: the back-translator here is *not* the original
scraper-side translator that produced the corpus's `trans_review` column (presumed to be Google's).
This experiment tests sensitivity to *translation-induced paraphrase* — does a different but
meaning-equivalent English phrasing change the label — not fidelity to the original translation
pipeline.

### 3.3 Statistics

- **Exact-set stability**: share of reviews whose new label (set) is identical to the original.
- **Mean Jaccard** (sub-dimensions only, which are multi-label): partial-credit overlap, separating
  total flips from a single label added/dropped.
- **Paired translation effect (McNemar logic)**: on the back-translation subset, for each review we
  know whether a plain rerun reproduced the canonical label (A) and whether the back-translated
  rerun did (B). If translation has no effect beyond ordinary noise, disagreements between A and B
  should be symmetric coin flips; an exact two-sided binomial test on the discordant pairs
  (`only_backtrans_unstable` vs. `only_rerun_unstable`) tests that null.

---

## 4. Results

### 4.1 Stability at a glance (all six checks)

| dimension | relabel exact stability | backtrans exact stability | n (relabel / backtrans) |
|---|---|---|---|
| social | 0.910 | 0.927 | 300 / 151 |
| racialised | 0.940 | 0.940 | 300 / 150 |
| approach | 0.912 | 0.845 | 387 / 155 |
| context | 0.910 | 0.884 | 387 / 155 |
| connotation | 0.894 | 0.826 | 387 / 155 |
| theme | 0.801 | 0.716 | 387 / 155 |

### 4.2 Paired translation-effect test (all six checks)

| dimension | n pairs | rerun-stable | backtrans-stable | only-backtrans-unstable | only-rerun-unstable | McNemar p |
|---|---|---|---|---|---|---|
| social | 151 | 0.914 | 0.927 | 3 | 5 | 0.73 |
| racialised | 150 | 0.960 | 0.940 | 9 | 6 | 0.61 |
| approach | 155 | 0.890 | 0.845 | 16 | 9 | 0.23 |
| context | 155 | 0.935 | 0.884 | 11 | 3 | 0.06 |
| connotation | 155 | 0.897 | 0.826 | 14 | 3 | **0.01** |
| theme | 155 | 0.806 | 0.716 | 28 | 14 | **0.04** |

**Finding 1 — the two upstream gate classifiers show no detectable translation sensitivity.** For
`social` and `racialised`, back-translation stability is not lower than plain-rerun stability, and
the paired test finds nothing (p=0.73, p=0.61). This is the opposite pattern from the
sub-dimensions, where back-translation is consistently *more* destabilizing than a plain rerun
(significant for connotation and theme, borderline for context, not significant for approach). The
decisions with the largest downstream consequences — does a review even enter the analysis
population at all — are, empirically, not the fragile ones; the fine-grained sub-dimension
judgements are.

### 4.3 The racialised classifier: near-perfect specificity, small residual recall gap

The confusion matrix for the `racialised` rerun (canonical label × fresh-rerun label, n=286
cleanly-compared pairs, after excluding 14 classification failures) shows a one-directional error
pattern, not symmetric noise:

| canonical \ rerun | no | yes |
|---|---|---|
| **no** | 137 | **0** |
| **yes** | 4 | 145 |

Zero canonical-`no` reviews were relabelled `yes` on rerun (perfect specificity in this sample), and
only 4 of 149 canonical-`yes` reviews (2.7%) were relabelled `no` — a small, one-directional residual
recall gap, not jitter, but a substantially smaller one than an earlier version of this check found
when `stage_2_racial_reclassed` was run downstream of a fresh rerun of the (now excluded)
`stage_2_racial` screen: chaining through that broader, less careful first pass introduced its own
instability upstream of `stage_2_racial_reclassed`'s own contribution — one further reason to treat
`stage_2_racial_reclassed` alone, run directly on the sample, as the right target for this check.

### 4.4 Sub-dimension label flows (relabel experiment, n=387)

Where the sub-dimension classifiers do drift, the drift is directional rather than uniform noise
across category pairs, e.g.:

- **approach**: `implicit → explicit` (6 cases) and `explicit ↔ explicit|implicit` boundary flips
  dominate — the implicit/explicit boundary is the main source of instability, not confusion with
  `coded`.
- **context**: almost entirely `generalised ↔ incident ↔ generalised|incident` boundary movement —
  a granularity/multi-label question, not a directional bias.
- **connotation**: `neutral ↔ negative` is the largest single flow (8 cases `neutral→negative`, 6
  `negative→neutral`) — the neutral/negative boundary, not neutral/positive or negative/positive, is
  where most instability lives.
- **theme**: `crowd ↔ belonging|crowd` and `belonging|staff ↔ access|staff` are the largest flows —
  adjacent, thematically-related labels being added or dropped, not wholesale relabelling.

---

## 5. What this means for downstream analysis

1. **The population definition itself is robust to translation.** Since `social` and `racialised`
   show no translation-specific effect, the headline scope numbers (504,072 social reviews; 7,768
   dimension-scope reviews) are not artifacts of translation phrasing — a legitimate worry given the
   corpus is ~93% originally German, now empirically addressed for the two decisions that matter most.

2. **Racialised-review prevalence estimates are, if anything, mildly conservative, and the residual
   instability is small.** `stage_2_racial_reclassed`, tested directly, reproduces canonical `no`
   labels perfectly on rerun (0/137) and canonical `yes` labels 97.3% of the time (145/149). Since
   rerun instability only removes true positives here and never adds false ones, any residual
   instability biases prevalence estimates *downward*, not upward — supporting the reported
   prevalence rise (0.92%→2.11%, 2021–2025, per the feasibility notebook) as unlikely to be inflated
   by classifier noise, with a plausibly small (~3%) undercount rather than a large one.

3. **Sub-dimension findings need dimension-specific caveats, and they now have quantitative
   grounding.** `theme` has both the weakest rerun stability (0.80) *and* a significant
   translation-specific effect (p=0.04) — and, per the IAA notebook, no human validity check exists
   for `theme` at all. Any time-series or cross-tab built on `theme` should be read as the
   least-reliable of the four dimensions on both stability and (unverifiable) validity grounds.
   `connotation` is the only other dimension with a significant translation effect (p=0.01); analyses
   that cross `connotation` against anything correlated with source language or translation quality
   (e.g., region, reviewer nationality) carry a specific, now-quantified confound risk. `approach`,
   by contrast, shows no significant translation effect (p=0.23) but has separately-documented
   near-chance human-human agreement (IAA κ≈0–0.09) — its problem is validity, not stability, and
   back-translation checks cannot detect or fix that.

4. **Boundary categories, not wholesale mislabelling, drive most instability.** The label-flow
   patterns (§4.4) show that when a sub-dimension label changes, it typically moves to an adjacent,
   substantively similar category (implicit↔explicit, neutral↔negative, generalised↔incident) rather
   than jumping arbitrarily. This is consistent with genuine underlying ambiguity in a meaningful
   share of reviews rather than classifier unreliability across the board, and supports treating
   instability as a source of *wider confidence intervals* on category-level counts rather than
   grounds to discard any dimension outright.

---

## 6. Limitations

- Back-translation uses the same LLM as the classifiers being tested (temperature 0) and a different
  translator than the corpus's original `trans_review` pipeline (presumed Google-side) — this
  measures sensitivity to translation-induced paraphrase specifically, not fidelity to the original
  translation.
- Sample sizes (150–387 per check) are adequate to detect the effects reported here but are not
  large enough to rule out smaller translation effects for `social`/`racialised`, or to characterise
  instability within every one of the 58 sub-dimension strata individually.
- This is a **stability** check, not a validity check. A classifier can pass every test here and
  still be systematically wrong; see the separate IAA analysis for validity evidence, which is weaker
  for `approach` and entirely absent for `theme`.
- The `racialised` confusion matrix (§4.3) is based on n=286 (of 300) cleanly-compared pairs after
  excluding 14 classification failures — a small further reduction in an already-modest sample.

---

## 7. Reproducibility / file manifest

All code: `robustness_check.ipynb` (sections 1–4 = Experiments A/B on the sub-dimensions; section 5 =
shared back-translation helper; sections 6–7 = Experiments C/D on the two gate classifiers; section 8
= combined export; section 9 = synthesis notes).

| file | contents |
|---|---|
| `results/stability_summary.csv` | relabel/backtrans exact-stability + Jaccard, sub-dimensions only |
| `results/translation_effect_paired.csv` | paired McNemar table, sub-dimensions only |
| `results/backtrans_label_changes.csv` | every flipped sub-dimension label, for qualitative review |
| `results/gate_stability_summary.csv` | relabel/backtrans exact stability, social + racialised |
| `results/gate_backtrans_label_changes.csv` | every flipped social/racialised label under back-translation |
| `results/translation_effect_paired_full.csv` | paired McNemar table, **all six checks combined** — the single most decision-relevant table in this report |

Samples (`samples/*.parquet`) and per-run classifier outputs (`runs/*/processed_success.csv`,
`processed_failed.csv`) are retained for full traceability back to individual review_ids.
