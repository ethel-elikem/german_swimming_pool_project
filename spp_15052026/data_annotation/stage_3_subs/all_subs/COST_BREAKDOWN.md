Cost breakdown — Stage 3 `all_subs` classification (batch_size=1)

Summary
- Target: classify 2% of social reviews (2% of 541,248)
- Reviews to process: 10,825

Per-review token estimates
- System prompt length: 10,101 chars → ≈ 2,526 tokens
- Average review length: 453 chars → ≈ 114 tokens
- Message/wrapper overhead: ≈ 10 tokens
- Total input tokens / review: 2,650 tokens
- Estimated output tokens / review (JSON): 120 tokens

Pricing (used for estimate)
- Input (prompt/messages): $3.00 per 1,000,000 tokens
- Output (completion): $15.00 per 1,000,000 tokens

Per-review cost (batch_size=1)
- Input cost: 2,650 * $3/1e6 = $0.00795
- Output cost: 120 * $15/1e6 = $0.00180
- Total per review: $0.00975

Totals for 10,825 reviews
- Total input tokens: 2,650 * 10,825 = 28,686,250 tokens
- Total output tokens: 120 * 10,825 = 1,299,000 tokens
- Base cost (no retries): 10,825 * $0.00975 = $105.54
- Historical retry multiplier used: ×1.1178
- Estimated cost with retries: $105.54 × 1.1178 ≈ $117.98

Assumptions and notes
- Token conversion: 1 token ≈ 4 chars (used for chars→tokens)
- System prompt measured from `SYSTEM_PROMPT_COMBINED` (10,101 chars)
- Average review length measured from racialised subset (453 chars)
- Output token size (120) is an estimate for the JSON schema (`theme`, `connotation`, `approach`, `context`) and up to 3 themes; adjust if you expect larger outputs
- Batch_size=1 was the setting used for this estimate (the full system prompt is sent per call, major cost driver)

Pilot & artifacts
- Successful 5-review pilot saved: spp_15052026/data_annotation/stage_3_subs/all_subs/pilot_runs/all_subs_5_success.json
- Earlier 50-review run showed truncation/finish_reason="length" when batch_size=1 and large prompt; consider changing strategy

Recommendations to reduce cost
- Increase batch_size (e.g., 5–20) to amortize system prompt tokens across multiple reviews per call
- Shorten / externalize the system prompt where possible (reduce per-call input tokens)
- Run a 50-review pilot (batch_size=1 or small batches) to measure real token counts after the tightened prompt; then re-run estimates using measured tokens

Next actions (suggested)
- If you want, I can run a 50-review pilot with the current tightened prompt (choose batch_size=1 or batch_size=5)
- Or I can recompute estimates for a chosen batch_size (5, 10, 20) to show savings

Document created by automation on 2026-05-18

Measured 50-review pilot (batch_size=1)
-------------------------------------
- Pilot run: 50 reviews from `racialised_yes_reviews.csv` using the API key in `.env`
- Measured totals:
	- Input tokens: 158,057
	- Output tokens: 4,972
	- Total cost recorded: $0.5488
	- Cost per review: $0.010976

Extrapolation using measured per-review cost (batch_size=1)
- Per-review measured cost: $0.010976
- Scaling to 2% run (10,825 reviews):
	- Base extrapolated cost (no retries): 10,825 × $0.010976 ≈ $118.82
	- With historical retry multiplier ×1.1178: ≈ $132.81

Comparison to earlier estimate
- Earlier base estimate (chars→tokens conversion): $105.54
- Earlier retry-adjusted estimate: $117.98
- Measured-based extrapolation is slightly higher because the actual per-call input tokens were larger than the original conservative estimate (measured input tokens per review ≈ 3,161 vs estimated 2,650).

Implications
- The measured pilot confirms our cost model directionally. The main driver of variance is the per-call input token count (system prompt + review text). Use the measured per-review tokens when computing final budget requests.

Next steps
- If you want exact accounting, I can update `SYSTEM_PROMPT_COMBINED` to shorten or further compact wording and re-run a 50-review pilot to measure savings, or run a 50-review pilot with `batch_size=5` to measure amortised prompt savings.

