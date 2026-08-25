# Socio-Economic Data Processing

Last updated: 2026-07-13

This folder contains scripts that clean the federal constituency baseline data,
derive analysis columns, audit repeated state-level values, and merge selected
Bundestag 2025 political variables.

## Main inputs

- `federal_baseline_eng.csv`: translated federal baseline CSV. Original numeric
  values use German decimal commas.
- `../../../SOCIAL_INTERACTIONS_2025/voters_share - second_vote_constituency.csv`:
  constituency-level Bundestag 2025 second-vote results.

## Generated files

- `federal_baseline_eng_clean.csv`: full cleaned federal file with snake_case
  headers, decimal dots, scaled population columns, and derived columns.
- `federal_baseline_eng_selected.csv`: smaller constituency-level file with
  selected socio-economic columns and no state or Germany total rows.
- `federal_baseline_eng_selected_political.csv`: selected socio-economic file
  merged with selected Bundestag 2025 second-vote columns.
- `state_repeated_columns_report.csv`: audit report showing which columns repeat
  within constituencies of the same state.

## Scripts

Run from this folder:

```bash
python3 clean_federal_baseline_eng.py
python3 merge_political_data.py
python3 audit_state_repeated_columns.py
```

`clean_federal_baseline_eng.py` creates both the full cleaned file and the
selected socio-economic file. `merge_political_data.py` expects the selected file
to exist and creates the socio-economic plus political file.

## Core transformations

- Decimal commas are converted to decimal dots.
- Columns marked as `(in 1,000s)`, `(in 1000s)`, or `(in 100s)` are multiplied
  by `1000`.
- `as of 31.12.2023` is removed from column names.
- Final output column names use snake_case.
- Rows where `constituency_name` is `Land insgesamt` or `Insgesamt` are excluded
  from the selected file.

## Derived columns

The selected socio-economic file includes derived count or total columns:

- `foreign_population = total_population - german_population` (not derived from
  `foreign_population_pct`, which is only published to 1 decimal place and
  would not sum back to `total_population`; subtracting the two directly
  reported actual counts guarantees `german_population + foreign_population
  == total_population` exactly — verified for all 299 constituencies
  2026-07-13)
- `age_*_population = total_population * age_*_pct / 100`
- `employees_total = total_population * employees_per_1k / 1000`
- `sgb_ii_recipients_total = total_population * sgb_ii_recipients_per_1k / 1000`
- `disposable_income_total_eur = total_population * disposable_income_eur_per_inhabitant`
- `gdp_total_eur = total_population * gdp_eur_per_inhabitant`
- `employees_* = employees_total * employees_*_pct / 100`
- `sgb_ii_unable_to_work = sgb_ii_recipients_total * sgb_ii_unable_to_work_pct / 100`
- `sgb_ii_foreign_nationals = sgb_ii_recipients_total * sgb_ii_foreign_nationals_pct / 100`

The derived values are rounded to whole numbers.

## Political merge

The political merge uses second-vote results, not first-vote results. Second
vote is the better variable for party preference/support. The merged file keeps:

- `direct_mandate_party`
- `eligible_voters`
- `voters`
- `turnout_pct`
- `valid_votes`
- `second_vote_spd_pct`
- `second_vote_cdu_csu_pct`
- `second_vote_greens_pct`
- `second_vote_afd_pct`
- `second_vote_left_pct`
- `second_vote_bsw_pct`

CDU and CSU are combined into `second_vote_cdu_csu_pct`.

## Calibration notes

Some federal indicators are repeated within city-states, especially Berlin and
Hamburg. For those cases, derived counts still vary by constituency if
`total_population` varies, but the variation comes from allocating a shared
state-level rate across constituencies. Treat these as estimates rather than
directly observed constituency-level measurements.

The audit on 2026-07-13 found no column repeated within every state. Repetition
was concentrated in Berlin, Hamburg, and for a few columns Bremen.

Unemployment rates are kept as rates. They are not converted to counts because
the proper denominator is the labor force, not total population.
