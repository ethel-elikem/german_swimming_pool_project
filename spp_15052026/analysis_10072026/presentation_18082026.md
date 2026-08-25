# Claude Code — Data Request for Supervisor Presentation

Nine requests. **R1 is essential** — it determines whether RQ2 is presentable. R2–R4 and R9 are presentation material. R5–R8 fill gaps.

Same rules: numbers and tables, no interpretation. Save to `outputs/presentation/` with the request ID in the filename.

---

# R1 — Ethnicity classification: noise sensitivity ★ ESSENTIAL

**Purpose.** Manual inspection suggests roughly 1 in 6 Arab, 1 in 11 Turkish, and 1 in 27 German predictions rest on degraded name strings. This tests whether the group rate differential survives when those are excluded.

## R1.0 — Check ACUA's output first

Before anything else, print all columns ACUA returns:

```python
print([c for c in df.columns if 'ethnic' in c.lower() or 'acua' in c.lower()])
df[[c for c in df.columns if 'ethnic' in c.lower()]].head(20)
```

**If ACUA returns a confidence score or class probabilities, report the column name and its distribution per group.** Confidence-based filtering is preferable to the string heuristics below, and if it exists, run R1.3 as well.

## R1.1 — Flag degraded name strings

```python
name = df.reviewer_name.fillna('')

flags = pd.DataFrame({
    'has_digit':      name.str.contains(r'\d', na=False),
    'single_token':   name.str.strip().str.split().str.len() <= 1,
    'very_short':     name.str.replace(r'\W', '', regex=True).str.len() < 3,
    'char_repeat':    name.str.contains(r'(.)\1{2,}', na=False, regex=True),
    'no_letters':     ~name.str.contains(r'[A-Za-zÀ-ÿ\u0600-\u06FF\u0400-\u04FF]{2,}',
                                         na=False, regex=True),
    'all_caps':       name.str.isupper() & (name.str.len() > 3),
    'has_emoji':      name.str.contains(r'[\U0001F300-\U0001FAFF]', na=False, regex=True),
})
flags['any_flag'] = flags.any(axis=1)
```

**Output A — flag prevalence by group:**

| group | n reviewers | has_digit | single_token | very_short | char_repeat | no_letters | all_caps | has_emoji | **any_flag** | **% flagged** |
|---|---|---|---|---|---|---|---|---|---|---|

Compute at **reviewer** level, not review level — one reviewer writing 30 reviews shouldn't count 30 times. Also report at review level separately, since that's what the rate calculation uses.

**Output B — duplicate names:**
```
n distinct reviewer names
n names appearing >1 time, by group
top 20 most frequent names per group, with counts
```

## R1.2 — Group rates three ways

Recompute B6 on three samples:

| group | n_social | rate | CI_lo | CI_hi |
|---|---|---|---|---|
| **Full sample** (all reviews) | | | | |
| **Clean only** (`~any_flag`) | | | | |
| **Degraded only** (`any_flag`) | | | | |

For each of the five `ethnicity_group5` values.

**Then output the key comparison:**
```
Arab : Germany rate ratio, full sample:      ___
Arab : Germany rate ratio, clean names only: ___
Türkiye : Germany, full:                     ___
Türkiye : Germany, clean only:                ___
```

**Also output** the rate among degraded-name reviewers pooled across all groups, and compare to the overall 1.54%. If degraded names racialise at roughly the base rate, contamination dilutes the differential rather than creating it.

## R1.3 — Confidence-based sensitivity (only if ACUA emits confidence)

Group rates by confidence decile or band. Then group rates restricted to the top 50% and top 25% by confidence.

**Output:** the Arab:Germany ratio at each confidence threshold. If it strengthens as confidence rises, that is strong evidence the differential is real.

## R1.4 — Recompute the proxy directionality result on clean names

The proxy result (Germany 68.8% attributed / 17.2% reported; Arab 17.1% / 50.9%; V = 0.27–0.39) rests on the same classifications.

Rerun the full proxy table and Cramér's V on clean names only. **Output:** the six-group table and both V values, full sample vs clean only, side by side.

## R1.5 — Gold-standard sample for manual coding

Export a stratified sample for manual validation with Keziah.

```
200 reviewers total, stratified:
  50 Germany, 50 Arab, 50 Türkiye, 25 Other, 25 Unknown
sampled within stratum: 70% clean names, 30% flagged
```

Columns: `reviewer_id`, `reviewer_name`, `n_reviews`, `languages_used`, `acua_prediction`, `acua_confidence` (if available), plus **empty columns** `human_label_1`, `human_label_2`, `notes`.

**Shuffle the row order and do not sort by prediction** — the coders must not be able to infer the model's guess from position.

## R1.6 — Partial external validation

Language is ~90% German so it cannot validate directly, but enrichment can:

```
Among reviewers predicted Arab:    % who ever wrote a review in Arabic
Among reviewers predicted Türkiye: % who ever wrote in Turkish
Among reviewers predicted Germany: % who ever wrote in Arabic, and in Turkish
```

**Output:** a 2×2-style enrichment table. Even small absolute percentages are informative if the ratio between groups is large.

---

# R2 — Predictor audit table

**Purpose.** The deck walks through predictor selection. This is the table behind it.

One row per candidate variable considered — all ~80 from the Strukturdaten plus the political variables.

| variable | conceptual group | transformation | status | reason | evidence |
|---|---|---|---|---|---|

- **status:** `retained` / `dropped-redundant` / `dropped-construction` / `dropped-no-signal` / `dropped-not-relevant` / `control`
- **reason:** short phrase, e.g. "ρ = 0.96 with sgb_ii", "sums to 100 with other age bands", "L-shaped, driven by 4 constituencies"
- **evidence:** the number, where one exists

Also output a **collapsed summary**: count of variables per conceptual group, and how many were retained from each.

---

# R3 — Split-city demonstration

**Purpose.** One concrete example explains the duplication problem better than a description.

## R3.1 — Duisburg side by side

```
constituency | social_reviews | racialised | racialised_rate |
foreign_population_pct | unemployment_total_pct | sgb_ii_recipients_per_1k |
school_grads_higher_education_entrance_pct | age_60_plus_pct |
disposable_income_eur_per_inhabitant | population_density_per_km2 |
second_vote_afd_pct | turnout_pct
```

For `Duisburg I` and `Duisburg II`. Mark which columns are identical.

## R3.2 — The same for München (4 constituencies) and Berlin

Berlin specifically: show that education *does* vary across the 12 Berlin constituencies while income and benefits do not. That's the "duplication is variable-specific" point.

## R3.3 — Full cluster list

The 12 clusters, 33 constituencies: cluster name, n constituencies, which variables are identical, and the min/max racialised rate within the cluster.

**Output the key contrast:** for each cluster, the ratio of highest to lowest racialised rate, alongside the fact that the structural predictors are constant. Duisburg's 7.6% / 3.0% = 2.5× is the headline.

## R3.4 — What the political variables do by contrast

Same table for the Berlin 12 and München 4 showing `second_vote_afd_pct` and `turnout_pct` **do** vary. This is why the political variables are unaffected by the problem.

---

# R4 — Full coefficient tables, all model families, both scales

**Purpose.** The deck explains intercepts, coefficients, and what they mean. That needs both the log scale and the multiplicative scale visible together.

## R4.1 — One table per family

For each of **Poisson**, **quasi-Poisson**, **negative binomial** — same formula, same offset, clustered SEs by state:

| term | β | SE(β) | z | p | β CI_lo | β CI_hi | **exp(β)** | **exp CI_lo** | **exp CI_hi** |
|---|---|---|---|---|---|---|---|---|---|

Include the intercept. Add a note row giving `exp(intercept)` as a rate in percent.

## R4.2 — Model-level statistics for each family

```
n observations
df residual
log-likelihood                (n/a for quasi-Poisson)
AIC                           (n/a for quasi-Poisson)
BIC                           (n/a for quasi-Poisson)
deviance
Pearson chi2
dispersion (Pearson chi2 / df)
alpha (NB) or phi (quasi-Poisson)
pseudo-R² (Cragg-Uhler)
```

## R4.3 — Side-by-side comparison table

| | Poisson | quasi-Poisson | negative binomial |
|---|---|---|---|
| foreign pop: exp(β) | | | |
| foreign pop: SE(β) | | | |
| density: exp(β) | | | |
| density: SE(β) | | | |
| benefits: exp(β) | | | |
| benefits: SE(β) | | | |
| is_east: exp(β) | | | |
| in_split_city: exp(β) | | | |
| dispersion | | | |
| AIC | | n/a | |
| n significant at p<0.05 | | | |

**The point to make visible:** quasi-Poisson coefficients are identical to Poisson with wider SEs; negative binomial re-estimates the coefficients. Confirm this in the output.

## R4.4 — Also fit the fully-estimated negative binomial

Currently `alpha` is estimated from a Poisson fit, then supplied as fixed to the GLM. Fit the version that estimates `β` and `alpha` jointly (`statsmodels.discrete.discrete_model.NegativeBinomial`) and report the coefficient table plus the estimated alpha.

**Output:** the two alphas and the two coefficient sets side by side. If they agree, one sentence in methods closes the question.

## R4.5 — Sequential deviance tables

For Poisson and negative binomial, in this fixed order:

```
~ 1
+ foreign_population_pct_c
+ log_population_density_c
+ sgb_ii_recipients_per_1k_c
+ is_east
+ in_split_city
```

| step | added term | deviance | Δdeviance | df | p | dispersion | AIC |
|---|---|---|---|---|---|---|---|

Also run in **reverse order** and report both — the order-dependence is worth knowing given foreign population and density correlate at 0.78.

---

# R5 — Venue concentration (B8, missing)

Never produced output. Needed.

## R5.1 — Per constituency
```
n pools with any review
n pools with ≥1 social review
n pools with ≥1 racialised review
pct_pools_with_racialised
top_pool_share  (largest single pool's share of the constituency's racialised reviews)
top2_pool_share
```

## R5.2 — Distributions
Histograms of `pct_pools_with_racialised` and `top_pool_share`. Five-number summary of each.

## R5.3 — Dominant pool vs. rest
For each constituency: the dominant pool's own racialised rate vs. the pooled rate of all its other pools. Two-proportion test, p-value, and a significance flag.

```
n constituencies tested
n where dominant pool significantly higher (p<0.05)
n where dominant pool is above the rest but not significantly
n where dominant pool is below the rest
```

## R5.4 — Pool-level concentration overall
```
Gini coefficient of racialised reviews across all pools with ≥1 social review
share of all racialised reviews accounted for by the top 1% / 5% / 10% of pools
```

---

# R6 — Panel trend model with a covid term

Y2 gave 1.077/year [1.059, 1.095]. A single constant growth rate across a period containing a pandemic shutdown is a strong assumption.

Refit the constituency-year panel three ways, all with `offset = log(social)`, clustered SEs by constituency:

- **M-a** `rac ~ t`
- **M-b** `rac ~ t + covid` where `covid = year in [2020, 2021]`
- **M-c** `rac ~ t + post2021` where `post2021 = year >= 2022`

**Output:** rate ratio on `t` with CI for each, plus AIC, plus n rows and n constituencies. Report whether the slope steepens once 2020–21 are absorbed.

---

# R7 — Variance and structure summary

One table gathering everything about what the model explains. Currently scattered.

| quantity | before predictors | after predictors | explained |
|---|---|---|---|
| dispersion (constituency) | 14.278 | 9.602 | 35.2% |
| Moran's I | 0.281 | 0.105 | |
| pseudo-R² | — | | |
| residual deviance / df | | | |

Plus, for the caveat about comparing across groupings:
```
total social reviews
mean per year / mean per constituency / ratio
dispersion (year, intercept only)
```

---

# R8 — Pipeline funnel

Exact N at each stage, for one slide:

```
total reviews scraped
reviews with text
pools
constituencies
→ Stage 1: social = yes / no
→ Stage 2: racialised = yes / no
→ Stage 3: dimensionally annotated
reviews after 2026 cutoff exclusion
```

Plus annotation validation figures: inter-annotator agreement per dimension, LLM-human match rate per dimension, sample sizes used.

---

# R9 — Example reviews for the fifth dimension ★

**Purpose.** The deck needs the Directionality dimension defined with concrete examples. Definitions alone won't persuade.

Using the proxy categories, export candidate examples:

```
proxy_attributed = (context == 'generalised') & theme.isin(['Crowd','Etiquette'])
proxy_reported   = (context == 'incident')    & theme.isin(['Staff','Access','Belonging','Language'])
proxy_both       = (context contains both generalised AND incident)
```

For each category export **10 candidates**, sampled to span:
- reviewer group (Germany / Arab / Türkiye)
- year (pre-2021 and post-2021)
- theme

Columns: `review_id`, `original_text`, `english_translation`, `theme`, `context`, `connotation`, `acua_prediction`, `review_language`, `year`, `pool_type`.

**Prioritise short reviews** — under 60 words — so they fit on a slide.

Also export 5 candidates that are **hard to classify** under the proposed scheme. Those are more valuable for the meeting than clean cases, because they show where the scheme will need a decision rule.

---

# Priority

1. **R1** — gates RQ2
2. **R9** — needed for the deck's closing argument
3. **R5** — likely to be asked
4. **R3**, **R4** — presentation aids
5. **R2**, **R6**, **R7**, **R8**