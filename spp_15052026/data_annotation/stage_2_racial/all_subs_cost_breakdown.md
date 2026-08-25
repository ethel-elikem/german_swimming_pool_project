All-subs classification cost breakdown
Date: 2026-05-18

What I produced
- `spp_15052026/data_annotation/stage_2_racial/racialised_yes_reviews.csv` — merged `processed_succes.csv` with `data/social_reviews.parquet`, filtered where `racialised` == 'yes'.
- `spp_15052026/data_annotation/stage_2_racial/all_subs_cost_estimate.json` — raw numbers used for the estimate.

Selection & counts
- Rows in `processed_succes.csv`: 409,593
- Rows in `social_reviews.parquet`: 541,248
- Rows in `proccessed_failed.csv`: 127,795
- Remaining (unprocessed) = 541,248 - (409,593 + 127,795) = 3,860
- Total considered = already classified + remaining = 409,593 + 3,860 = 413,453
- Number to classify (2% of already + 2% of remaining) = ceil(0.02 * 413,453) = 8,270 reviews

Token estimates (method)
- I measured average review length across `social_reviews.parquet` (non-empty `trans_review`) = 309.2 characters → estimate tokens ≈ 309.2 / 4 = 77.3 tokens/review (approx).
- `all_subs_classifier` uses `SYSTEM_PROMPT_COMBINED` (found in `spp_15052026/data_annotation/stage_3_subs/all_subs/all_subs_classifier.py`), length = 9,998 characters → estimate tokens ≈ 9,998 / 4 = 2,499.5 tokens (this is the system prompt that is sent once per API call and therefore fully counted when `batch_size=1`).
- Small user-message overhead (wrapping instructions, JSON wrapper) assumed = 10 tokens.
- Therefore per-review input tokens (batch_size=1) ≈ 2,499.5 + 77.3 + 10 = 2,586.8 tokens.
- Completion tokens: assumed 80 tokens/review (output JSON with multiple fields). You can change this assumption if you expect longer/shorter outputs.

Prices
- Input: $3.00 per 1,000,000 tokens
- Output: $15.00 per 1,000,000 tokens

Per-review cost (batch_size=1)
- Input cost = 2,586.8 / 1e6 * $3 = $0.0077604
- Output cost = 80 / 1e6 * $15 = $0.0012
- Total per review ≈ $0.0089604

Total cost for 8,270 reviews
- Total = 8,270 * $0.0089604 = $74.14 (rounded)

If you change assumptions
- System prompt tokenisation: shorter system prompt (or using grouped batched calls) drastically reduces cost. Example: if you batch B reviews, the system prompt token share per review ≈ sys_tokens / B.
- If completion tokens differ, adjust `per_review_output` and recompute: cost delta per 1 token = $15 / 1e6 = 0.000015 USD.

Files created
- [racialised_yes_reviews.csv](spp_15052026/data_annotation/stage_2_racial/racialised_yes_reviews.csv#L1)
- [all_subs_cost_estimate.json](spp_15052026/data_annotation/stage_2_racial/all_subs_cost_estimate.json#L1)

Next steps (recommended)
1. If you want more precise costs, run a small pilot (e.g., 50 with `batch_size=1`) using `all_subs_classifier` to measure actual `usage` tokens; I can run that if you want.  
2. Alternatively, reduce `batch_size` amortisation impact by batching >1 (but you requested `batch_size=1`).