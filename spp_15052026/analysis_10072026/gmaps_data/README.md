# Analysis dataset — built 2026-07-10, consolidated 2026-07-13

Canonical, verified dataset for the swimming-pool review analysis.
**Use the files in this folder for analysis; treat everything in `../data/` as
upstream sources.** Rebuild any time with:

```
python build_analysis_dataset.py
```

One script now covers the whole chain — base classification layers, place
metadata, parsed Google-review timestamps, and name-to-country ethnicity
prediction — so there is a single place documenting every transformation
applied to the corpus. It reads only authoritative sources and **asserts
every link in the chain**: if an upstream file, the `gmaps_poi` Postgres DB,
or the prediction zip changes, the build either produces a correct dataset
or fails loudly. No intermediate parquet files are written; every stage runs
in memory and only the outputs below are written to disk.

## Files

| file | rows | what |
|---|---|---|
| `master_corpus_final.parquet` | 1,265,806 | canonical analysis corpus: classification layers, place metadata, DB timestamps, name-to-country ethnicity prediction |
| `racialised_reviews_analysis.parquet` | 7,768 | the racialised (dimension-scope) subset, same columns as the master corpus |
| `reviewer_ethnicity_lookup.parquet` | 923,168 | reviewer-id lookup table used for the ethnicity merge |
| `build_report.json` | — | counts + verification results of the latest build, one section per stage |
| `Prediction Result.zip` | — | name-level ethnicity prediction input (external tool output) |
| `reviewer_names_dedup.csv` | — | name list submitted to the external ethnicity-prediction tool (kept for provenance; not read by the build script) |

## Provenance chain (all verified in the build)

```
places_filtered.parquet                    5,529 verified POIs in Germany
                                           (join key: unique_identifier — NOT place_id,
                                            which is non-unique: 4,918 distinct)
  └─ social_classified_text_reviews.parquet   1,265,806 reviews (100% within verified POIs)
       social: yes 504,072 | no 761,733 | 1 unclassified
       └─ racial model classification (stage_2_racial/processed_success.csv)
            run on ALL 541,248 social-yes reviews of the pre-places-filter era
            (541,248 = 504,072 in verified POIs + 37,176 outside)
            model: yes 11,471 | no 529,777   ← the 11,471 IS the reclass pool
            └─ stage-2 racial RE-classification of the 11,471
                 corrected: yes 8,271 | no 3,200
                 └─ racialised ∩ verified-Germany = 7,768  ← dimension scope
                      (the other 503 corrected-yes are from non-verified POIs)
                      approach / theme / connotation / context
                      each covers 7,768/7,768 (verified)
                      └─ place metadata merge (places_filtered.parquet, key: unique_identifier)
                      └─ gmaps_poi Postgres timestamp merge (local DB, key: review_id)
                      └─ name-to-country ethnicity prediction merge (key: reviewer_id)
                         + drop of low-use metadata columns
```

## Columns added to the corpus metadata

- `social` — yes/no from the social classifier.
- `racialised` — final label: stage-2 **corrected** value where the review was in
  the reclassification pool, otherwise the model value. `NaN` for social-no rows.
  yes = 7,768 | no = 496,304 | NaN = 761,734. Every social-yes review has a label
  (asserted in the build).
- `approach`, `theme`, `connotation`, `context` — lists of labels; only set where
  the review is in dimension scope. Sources: `approach` from the June-10 reclass run
  (`stage_3_subs/approach/reclass_approach/`); the other three from the May runs
  filtered to the 7,768 (their extra ~3,703 out-of-scope rows are dropped here).
- `place_retrieved_time`, `place_name`, `place_rating`, `place_total_reviews`,
  `primary_place_type`, `place_subtype`, `latitude`, `longitude`, `opening_hours`
  — place-level metadata from `places_filtered.parquet` (join key: `unique_identifier`,
  not `place_id`, which has blanks and duplicates there). On 2026-07-13 the merge
  matched all 1,265,806 review rows; one place row had no reviews in the corpus.
- `review_timestamp`, `review_edited_timestamp`, `reply_time_text` — from the
  local Postgres `gmaps_poi` DB, table `reviews` (join key: `review_id`).
  `review_timestamp` is the parsed absolute timestamp derived in the DB from the
  relative Google Maps review time text. The original corpus columns `review_time`
  and `retrieved_time` are preserved unchanged. On 2026-07-13, all rows matched
  the DB, but `review_timestamp` was non-null for only 879,504 rows; 386,302
  matched DB rows still lack a parsed review timestamp. Not merged: the DB's
  `retrieved_time`/`review_time_text` looked like duplicates of the corpus's
  own `retrieved_time`/`review_time` at first, but only matched ~95-97% of
  rows (the rest differ by hours/days, not just a timezone offset) — a
  later/different `gmaps_poi` retrieval snapshot, not a restatement of the
  same value, so they were dropped rather than kept as a second, drifting
  copy. `reply_timestamp`/`reply_edited_timestamp` and
  `gmaps_review_created_at`/`updated_at` are also not merged — not used in
  analysis.
- `ethnicity_country_code`, `ethnicity_country_name` — name-to-country
  prediction. The prediction file (`Prediction Result.zip`) is name-level,
  not reviewer-id-level, so the build selects the most frequent observed
  `reviewer_name` per `reviewer_id` (via `reviewer_ethnicity_lookup.parquet`,
  which also keeps `reviewer_name_for_ethnicity_prediction`,
  `reviewer_name_prediction_id_count`, and `reviewer_id_name_count` for
  diagnosing shared-name collisions), merges the prediction onto that name,
  then merges just the two country columns back onto the corpus by
  `reviewer_id`. Age and gender predictions are intentionally excluded.
  Country abbreviations are expanded to full country names using ISO country
  metadata, with explicit handling for historical/special codes such as `AN`
  and `XK`. Calibration note: on 2026-07-13, 51,935 reviewer names were shared
  by more than one reviewer ID, affecting 313,200 review rows — interpret this
  as name-origin/ethnicity signal, not verified individual identity.

Columns dropped from the final corpus (present in upstream sources, not needed
for analysis): `needs_summary`, `racialised_source`, `in_dimension_scope`,
`contributor_link`, `profile_pic`, `uploaded_pics_links`, plus (dropped
2026-07-13) `reply_timestamp`, `reply_edited_timestamp`,
`gmaps_review_created_at`, `gmaps_review_updated_at`,
`reviewer_name_for_ethnicity_prediction`, `reviewer_name_prediction_id_count`,
`reviewer_id_name_count` (these three remain in `reviewer_ethnicity_lookup.parquet`).

## ⚠ Stale files in ../data/ — do not use for analysis

| file | problem |
|---|---|
| `social_reviews.parquet` (541,248) | pre-dates the June-10 places filter; contains 37,176 reviews from non-verified POIs |
| `racialised_classified_text_reviews.parquet` (504,071) | rebuilt June-10 for the places filter but labels are **pre-correction**: 3,024 of its `yes` rows were reclassified `no` |
| `racialised_reviews.parquet` (11,471) | reclassification *pool*, `racialised` column is an all-`yes` placeholder, not a result |
| `racialised_merged_data_01062026.parquet` / `_exploded_` | June-1 merges of the pre-reclass dimension outputs |

## Known edge cases (documented, resolved)

- Two review IDs (`Ci9D…T1hOdU9IYxAB`, `Ci9D…VjJkM1ozYxAB`) reliably trigger a
  token-repetition loop in the classifier model when a prompt asks it to echo the
  ID back. Fixed July 2026 in the racial classifier and all four dimension
  classifiers (`classify_single` no longer echoes IDs); both reviews are now
  successfully labelled (the second one is racialised=no, classified 2026-07-10).
- The 9 old `invalid_value` failures in context and 2 in themes are all
  out-of-scope rows (corrected racialised=no or non-Germany) — irrelevant.

## History

Originally built as four separate scripts (`build_analysis_dataset.py` for the
base corpus, then `update_master_with_place_data.py`,
`update_master_with_gmaps_poi_db.py`, and `merge_name_ethnicity_predictions.py`
each writing their own intermediate parquet). Consolidated into the single
`build_analysis_dataset.py` above on 2026-07-13 so the whole transformation
chain — and every change made to it — lives in one documented place. The
intermediate parquet files (`master_corpus.parquet`,
`master_corpus_with_place_data.parquet`, `master_corpus_with_gmaps_poi.parquet`)
and a stale, superseded `create_slim_master_corpus.py` output
(`master_corpus_slim.parquet`, `master_corpus_slim_ethnicity.parquet`) were
removed as part of that consolidation.
