# Cost Analysis Report — Stage 1 Social Classification

Date: 2026-05-13

TL;DR: Your 13-token output assumption was too low. Even strict JSON with review_id needs ~18 output tokens/review; measured output was ~49.57. That gap, plus higher input tokens, drove the Stage 1 overrun.

## Executive summary

- Input reviews: 1,366,980
- Canonical successful classifications: 1,366,979
- Your budget estimate: $620.00
- Actual main-run cost: $1,394.79
- Total observed across all logs: $1,418.30
- Overrun factor: 2.28× (actual main-run ÷ estimated)

"Overrun" means actual spending exceeded your budgeted estimate; here you spent ~2.28× more than planned.

---

## Your initial estimates

- Tokens pricing used: input $3 / 1M tokens, output $15 / 1M tokens
- Assumed tokenization: 1 token ≈ 4 chars
- Assumed prompt/system tokens per call: 175 tokens
- Assumed average review length: 50 tokens (≈ 200 chars)
- Assumed batch size: 5 reviews per API call
- Assumed model output per review: 13 tokens

Per-batch estimate (input): 175 + 5×50 = 425 tokens
Per-batch estimate (output): 5×13 = 65 tokens

Estimated total cost (your calculation): ≈ $615.14 → rounded to $620 budget

---

## What actually happened (measured)

### Review length (input text)
- Average characters per review: 191 chars
- Median characters per review: 89 chars
- Average tokens per review (chars/4): 47 tokens
- Median tokens per review: 22 tokens
- 95th percentile characters: 705 chars

Verdict: your review-length assumption (50 tokens) was accurate (actual mean 47). Review length did not cause the cost overrun.

### Token metrics (from main classification log: classification_20260511_173002.log)
- Total reviews processed in main run: 1,274,820
- Total input tokens (main run): 149,029,827
- Total output tokens (main run): 63,180,451

Calculated per-call and per-review:
- Average input tokens per API call (batch): 584.51 tokens
- Average output tokens per API call (batch): 247.80 tokens
- Average input tokens per review: 116.906
- Median input tokens per review: 113.0
- Average output tokens per review: 49.567
- Median output tokens per review: 49.2

(These medians were computed from per-batch in/out token records in the log and normalized by batch size per batch type.)

### API calls and costs
- Total API calls in main run: 254,964 (pass-1) + 117 (retry-2) + 20 (retry-3) = 255,101 calls
- Main-run cost breakdown:
  - Input cost: 149,029,827 tokens × $3 / 1,000,000 = $447.09
  - Output cost: 63,180,451 tokens × $15 / 1,000,000 = $947.71
  - Total main-run cost: $1,394.79

- Retry cost (main run): $0.79 (gretry2 + gretry3)
- Total across all logs (including reruns): $1,418.30 (extra ≈ $23.51 for later reruns)


## Root cause analysis — why the budget was exceeded

1. OUTPUT TOKENS (PRIMARY DRIVER):
   - You assumed ~13 output tokens per review; actual was ~49.6 tokens per review (3.8× higher).
   - Output tokens accounted for ~68% of main-run cost ($947.71 of $1,394.79).
   - Reason: model responses contained classification + confidence/explanation text rather than a short label only.

2. INPUT TOKENS (SECONDARY CONTRIBUTOR):
   - Per-batch input tokens were 584.5 vs your estimate 425 (≈1.38× higher).
   - This is due to a larger-than-expected system/prompt size (the fixed prompt contributed ~70 tokens per review on average).

3. REVIEW LENGTH: Not a driver — your review-length estimate was essentially correct.

4. RETRIES: Negligible cost.
   - Retry cost (main run) = $0.79 (0.06% of main-run cost).
   - Reruns/extra logs contributed ~$23.51 to the grand total, but these are separate resumed/rerun jobs, not retries within the main pass.

## Additional Quantification of Underestimates

Using the canonical total of **1,366,979** classified reviews, we can quantify the underestimates that caused the overrun:

- **Output tokens underestimation**
  - Assumed output per review: **13 tokens**
  - Actual average output per review: **49.56 tokens**
  - Extra output tokens per review: **36.56 tokens**
  - Extra output tokens total: 36.56 × 1,366,979 ≈ **49,976,752 tokens**
  - Extra output cost: 49,976,752 × $15/1,000,000 ≈ **$749.65**

- **Input tokens underestimation**
  - Assumed input per review (prompt share + review): 175 prompt/5 + 50 = **85 tokens/review**
  - Actual average input per review: **116.91 tokens**
  - Extra input tokens per review: **31.91 tokens**
  - Extra input tokens total: 31.91 × 1,366,979 ≈ **43,614,831 tokens**
  - Extra input cost: 43,614,831 × $3/1,000,000 ≈ **$130.84**

- **Combined extra cost (approx):** $749.65 + $130.84 ≈ **$880.49**

Note: this combined estimate is slightly higher than the observed main-run overrun ($774.79) because the logged main run used resumption filtering (not all reviews were reprocessed in pass-1) and because small reruns/recovery steps affect the final logged billing. The token-based estimate explains the bulk of the discrepancy and shows the primary driver was the much larger-than-expected model output.

## Output Token Breakdown Per Review (vs your 13-token estimate)

Using the strict output schema in the classifier (`{"review_id":"...","social":"yes|no"}`) and the observed average review_id length:

- Average review_id length: **38.54 chars** -> **9.64 tokens** (chars/4)
- JSON structure only (braces, field names, quotes, punctuation): **31 chars** -> **7.75 tokens**
- Label value (`yes` or `no`): **2-3 chars** -> **0.50-0.75 tokens**

Minimum expected output per review for strict minified JSON is therefore:

- **17.89-18.14 tokens/review** (= 9.64 + 7.75 + 0.50..0.75)

Comparison to your estimate:

- Your estimate: **13 tokens/review**
- Strict JSON floor: **~18 tokens/review**
- Gap from structure + id alone: **~5 tokens/review above your estimate**

And the measured average output was **49.57 tokens/review**, which is much higher than even the strict-JSON floor. So the 13-token estimate was too low even before accounting for tokenizer effects and any additional formatting/content the model emitted.

## Direct answers to your specific questions

- "What do you mean by overrun?"
  - Overrun = actual spend − budgeted amount. Here main-run overrun = $1,394.79 − $620 = $774.79.

- "So my estimated output tokens were wrong?"
  - Yes. You estimated 13 tokens/review; actual average output was ~49.56 tokens/review (≈3.8× higher).

- "The output is {review_id, social: yes} or {review_id, social: yes}"
  - I believe you mean the labels are `{review_id, social: yes}` or `{review_id, social: no}`. The classification output stored in `processed_success.csv` contains the label, but the model responses also included explanation text. That extra text produced the large output token totals.

- "How much was spent on retries alone?"
  - Retries during the main run (gretry2 + gretry3) cost **$0.79** total.
  - Additional rerun jobs across other logs cost ≈ **$23.51** in aggregate; these are reruns/resumptions, not low-level retries.

- "You still did not give me the average/median input and output tokens from the logs"
  - Average input tokens per review: **116.906**
  - Median input tokens per review: **113.0**
  - Average output tokens per review: **49.5668**
  - Median output tokens per review: **49.2**

## Recommendations

1. If you want to reduce cost, constrain model output to a concise, structured format (e.g., strictly JSON with a short label field), and validate via a small pilot.
2. Measure prompt token length (system + instructions) precisely before large runs; trimming prompts can reduce input cost significantly.
3. Run a 1k-review pilot to measure real input/output token distributions and compute a realistic budget.

---

Report generated by analysis tools; the file is saved at `/home/ethel/Desktop/swimming_pool_project/spp_15052026/data_annotation/stage_1_social/cost_analysis_report.md`.
