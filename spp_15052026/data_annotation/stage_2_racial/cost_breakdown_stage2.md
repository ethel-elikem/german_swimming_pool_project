Cost breakdown — Stage 2 (racial) rerun
Date: 2026-05-18

Summary
- Remaining reviews to classify: 131,655
- Pilot sample: 50 reviews (ran using the first key from stage_2 `.env`)

Measured pilot (50 reviews)
- Total input tokens: 9,517
- Total output tokens: 2,548
- Per-review (average): input = 190.34 tokens, output = 50.96 tokens

Unit prices used
- Input (prompt) price: $3.00 per 1,000,000 tokens
- Output (completion) price: $15.00 per 1,000,000 tokens

Derived per-review costs (from pilot averages)
- Input cost / review = 190.34 / 1,000,000 * $3 = $0.00057102
- Output cost / review = 50.96 / 1,000,000 * $15 = $0.00076440
- Total per-review = $0.00133542 (≈ $0.00134)

Total cost — pass‑1 only (no retries)
- Total input tokens = 131,655 * 190.34 = 25,059,212.7 → 25,059,213 tokens
- Total output tokens = 131,655 * 50.96 = 6,709,138.8 → 6,709,139 tokens
- Input cost = 25,059,212.7 / 1e6 * $3 = $75.1776381 → $75.18
- Output cost = 6,709,138.8 / 1e6 * $15 = $100.637082 → $100.64
- Total cost (pass‑1 only) = $75.1776381 + $100.637082 = $175.8147201 → $175.81

Historical retry overhead (empirical multiplier)
- Historical runs produced a retry multiplier ≈ 1.1178 (i.e., full-run cost ≈ 11.78% higher after retries)
- Total cost including historical retries = $175.81 * 1.1178 ≈ $196.52

Convenience figures
- Cost / 1,000 reviews (pass‑1 only) = 1,000 * $0.00133542 = $1.3354 → $1.34
- Cost / 1,000 reviews (with retries) ≈ $1.49
- Pilot (50 reviews) measured cost = $0.0668 (matches CostTracker output)

Alternative: grouped-output target (example)
- If we can reduce completion tokens to ~13 tokens/review (grouped output) while input tokens remain the same:
  - Output tokens total = 13 * 131,655 = 1,711,515 → output cost = 1.711515 / 1e6 * $15 = $25.67
  - Input cost (unchanged) = $75.18
  - Total = $100.85 (pass‑1 only)
  - With historical retry multiplier (×1.1178) ≈ $112.68

Assumptions & notes
- Pilot averages include all prompt tokens (system + user messages + per-review text) and completion tokens; they therefore implicitly capture batch/system overhead averaged across the batch size used (batch_size=5 for the pilot). If you change batch size, small differences in per‑review tokens may occur (system prompt is amortised across the batch).
- The historical retry multiplier is empirical from prior runs; its value can change depending on HTTP errors and key exhaustion behavior.
- Rounding: token totals shown above include fractional results from averages and are rounded to integers where appropriate for token counts; monetary figures rounded to 2 decimals for reporting.

How to recompute locally (Python one-liner)
```python
# Replace remaining_n, in_tok_per_review, out_tok_per_review, in_price_m, out_price_m
remaining_n = 131655
in_tok = 190.34
out_tok = 50.96
in_cost = in_tok * remaining_n / 1e6 * 3
out_cost = out_tok * remaining_n / 1e6 * 15
total = in_cost + out_cost
print(total)
```

Where I placed this file
- [spp_15052026/data_annotation/stage_2_racial/cost_breakdown_stage2.md](spp_15052026/data_annotation/stage_2_racial/cost_breakdown_stage2.md#L1)
