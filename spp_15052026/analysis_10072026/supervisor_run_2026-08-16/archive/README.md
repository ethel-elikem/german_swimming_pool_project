# Archive — superseded by `../notebooks/`

Consolidated 2026-08-24. Nothing below is deleted; the four active notebooks in
`../notebooks/` reorganise this material around the shape of the paper's methods/results
sections and close five specific gaps (§2.2 heteroskedasticity, §3.3 influence — already
existed here, verified not re-derived, §4.2.5 US-category specs, §4.2.6 permutation test,
§4.3 seeded name samples). See `../ANALYSIS_LOG.md` for the full history and a
question-by-question status table.

## `supervisor_run_2026-08-16.ipynb` (last run 2026-08-19/24)

The main modelling notebook: outcome definition (§0), prevalence/form/change (§1),
WHO (§2), WHERE + model selection (§3), the final NB model (`B9`), Task B
(German/non-German split, `BSPLIT1`–`BSPLIT5`), and diagnostics (`D3`–`D6`, `B10`).

Absorbed into:
- Setup, outcome definition → `01_data_and_measures.ipynb` §1.1–1.4, 1.10
- `R1` (OLS/WLS/binomial baseline) → `02_modelling_strategy.ipynb` §2.1 (binomial rows kept
  for the OLS→WLS comparison only, not carried forward as candidate models)
- `D3`/`D4` (NB spec, deviance) → `02_modelling_strategy.ipynb` §2.3
- `B9` (final model) → `03_results.ipynb` §3.1, §3.2
- `B10`/`D6` (VIF, influence, residuals) → `03_results.ipynb` §3.3; `04_robustness_and_open_issues.ipynb`
  §4.6 (residual Moran's I)
- `B7` (spatial) → `03_results.ipynb` §3.4; `04_robustness_and_open_issues.ipynb` §4.6
- Task B (`BSPLIT1`–`BSPLIT5`) → `04_robustness_and_open_issues.ipynb` §4.2.1–4.2.4, 4.2.7
  (§4.2.5/4.2.6 are new refits run directly in the consolidated notebook, not relocated
  from here)

## `rate_total_analysis.ipynb` (last run 2026-08-24)

Task C: the `rate_total` sensitivity analysis, including the `share_social` bridge model
(originally fit as binomial GLM — **removed from the active path**, see below) and its
NB-with-offset refit (`C7b`/`C7c`).

Absorbed into:
- `C1` (distribution, scatter) → `01_data_and_measures.ipynb` §1.5, 1.6
- `C7b`/`C7c` (NB refit, additivity check) → `01_data_and_measures.ipynb` §1.7
- `C3a`–`C3d` (selection, LOWESS, instability) → `02_modelling_strategy.ipynb` §2.4–2.6
- `C5`/`C8` (diagnostics, comparison table) → `04_robustness_and_open_issues.ipynb` §4.1, §4.6

**Removed from the active path, preserved here only:** the original binomial GLM fit of
`share_social` (`C7`) — superseded by `C7b`'s NB-with-offset refit, which is the version
carried into `01_data_and_measures.ipynb` §1.7. Any narrative referring to logit links,
odds scales, or `log(k/(s-k))` for this quantity is likewise not carried forward; the
binomial fit is still readable in this file for reference.

## `ethnicity_validation_scoring.ipynb` (last run 2026-08-19)

Task D: the name→ethnicity classifier validation scoring pipeline (confusion matrix,
precision/recall/F1, error-by-feature, threshold check, self-check).

Absorbed into:
- Validation status, blocked-on-coding statement → `04_robustness_and_open_issues.ipynb` §4.4
- Everything gated on hand-coding (0/599 coded) remains **not runnable** until the
  researcher codes the sample — this notebook is the pipeline that will produce the
  confusion matrix once that happens; nothing here was recomputed for the consolidation.
