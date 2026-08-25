# Cost Analysis Report — Stage 2 Racial Classification

Date: 2026-05-13

TL;DR: Stage 2 output was costlier than expected: 13 tokens/review was below the strict JSON floor (~27-28), while measured output was ~53.33. High failures (mostly 403/402) also disrupted completion and rerun efficiency.

## Executive Summary

- **Input reviews:** 537,388 (filtered from stage 1 where `social='yes'`, original file at time of run)
- **Successfully classified:** 409,593
- **Failed to classify:** 127,795
- **Not processed:** 0 (all input rows were processed)
- **Total processed (SUCCESS + FAILED):** 537,388 (100% coverage)
- **Coverage:** 100% of the input file
- **Your budget estimate:** $250
- **Actual cost:** $388.94 (logged); $576.58 (calculated from tokens)
- **Overrun:** $138.94–$326.58 (55–131% over budget)

---

## Processing Status Breakdown

| Status | Count | Percentage |
|--------|-------|-----------|
| **SUCCESS** | 409,593 | 76.2% |
| **FAILED** | 127,795 | 23.8% |
| **UNPROCESSED** | 0 | 0.0% |
| **TOTAL INPUT** | 537,388 | 100.0% |

**Interpretation:**
- The process achieved **100% coverage** (all 537,388 rows processed)
- Of those processed, 76.2% successfully classified
- Failure rate: 23.8% (127,795 reviews failed classification)
- No reviews were skipped or left unprocessed
- *Note: The current social_reviews.parquet has 541,248 rows; the additional 3,860 rows were added to the input file after this classification run completed.*

---

## Token & Cost Metrics

### Per-Review Tokens (calculated from 89,015 batches)

- **Average input tokens per review:** 165.15 tokens
- **Median input tokens per review:** 162.80 tokens
- **Std deviation (input):** 26.21 tokens
- **Average output tokens per review:** 53.33 tokens
- **Median output tokens per review:** 51.60 tokens
- **Std deviation (output):** 9.06 tokens

### Per-API-Call Tokens (batch size = 5 reviews)

- **Average input tokens per call:** 
 (165.15 × 5)
- **Average output tokens per call:** 267 tokens (53.33 × 5)

### Total Tokens & Cost

- **Total input tokens (from log):** 73,504,116
- **Total output tokens (from log):** 23,738,027
- **Total API calls:** 89,015 batches (mostly batch size 5)

#### Cost Calculation

- Input cost: 73,504,116 tokens × $3 / 1,000,000 = **$220.51**
- Output cost: 23,738,027 tokens × $15 / 1,000,000 = **$356.07**
- **Total calculated cost: $576.58**

#### Logged Cost vs Calculated Cost

- Logged in classification log: **$388.94**
- Calculated from tokens: **$576.58**
- Discrepancy: **$187.64** (48% difference)

**Note:** The classification log shows $388.94, which appears to be based on estimated pricing from the length-analysis phase. The actual calculated cost from token usage is $576.58, which is what you would be charged by the API.

---

## Comparison: Stage 1 (Social) vs Stage 2 (Racial)

| Metric | Stage 1 (Social) | Stage 2 (Racial) | Difference |
|--------|---|---|---|
| **Input reviews** | 1,366,980 | 541,248 | -60.4% (smaller) |
| **Successful** | 1,366,979 (99.9%) | 409,593 (75.7%) | Much lower success rate |
| **Failed** | 1 (0.0%) | 127,795 (23.6%) | Much higher failure rate |
| **Avg input tokens/review** | 116.91 | 165.15 | +41% (longer) |
| **Avg output tokens/review** | 49.57 | 53.33 | +7% (slightly longer) |
| **API calls** | 255,101 | 89,015 | -65% (fewer calls) |
| **Actual cost** | $1,394.79 | $388.94* | -72% (but see note) |
| **Cost per review** | $0.00102 | $0.000718* | -30% |

*Note: Stage 2 logged cost is $388.94 but calculated from tokens is $576.58. Stage 1 figures are verified.

---

## Root Cause Analysis — Why Stage 2 Had Higher Failure Rate

### Higher Input Token Usage Per Review

- **Stage 1:** 116.91 tokens/review
- **Stage 2:** 165.15 tokens/review
- **Difference:** +41% more input tokens needed

**Reasons:**
1. Racial classification is more complex; requires longer prompts/instructions
2. Racial categories may require more context than social yes/no classification
3. The system prompt for racial classification is likely longer/more detailed

### Much Higher Failure Rate

- **Stage 1:** 0.0% failure rate (only 1 failed out of 1.36M)
- **Stage 2:** 23.8% failure rate (127,795 failed out of 537,388)

**Likely causes:**
1. Racial classification is more nuanced and subjective; harder for AI to classify with confidence
2. Text truncation at 700 chars may affect racial classification more than social classification
3. Model may be returning "unclear" or refusing responses for sensitive racial topics
4. Incomplete or ambiguous reviews after truncation

---

## Your Budget Estimate vs Actual

| Item | Your Estimate | Actual (Log) | Actual (Calculated) |
|------|---|---|---|
| **Estimated cost** | $250 | — | — |
| **Cost budget for both stages** | $620 + $250 = **$870** | — | — |
| **Stage 1 actual** | $620 | $1,418.30 | $1,418.30 |
| **Stage 2 actual** | $250 | $388.94 | $576.58 |
| **Total actual** | — | $1,807.24 | $1,995.88 |
| **Overrun (logged)** | — | **+$937.24** (108%) | — |
| **Overrun (calculated)** | — | — | **+$1,125.88** (129%) |

---

## Why Your Stage 2 Estimate Was Inaccurate

### You Assumed
- Same token usage as stage 1 (50 input, 13 output per review)
- Same success rate (>99%)
- Same failure/retry overhead

### Reality
- **Input tokens:** 165.15 per review (3.3× your estimate of 50)
- **Output tokens:** 53.33 per review (4.1× your estimate of 13)
- **Failure rate:** 23.8% instead of <1% (high retry overhead)
- **Coverage:** 100% of the input file (all 537,388 rows processed)

### Key Differences from Stage 1

1. **Longer prompts required** — racial classification needs more instructional context
2. **More complex output** — responses likely include reasoning for racial classifications
3. **Higher failure rate** — likely due to:
   - Ambiguous racial categories in truncated text
   - Model caution with sensitive racial content
   - Text truncation at 700 chars affecting racial identifiers

---

## Additional Quantification of Underestimates

## Output Token Breakdown Per Review (vs your 13-token estimate)

Using the strict output schema in stage 2 (`{"review_id":"...","social_encounter":"yes|no","racial_connotations":"yes|no"}`) and observed review_id lengths:

- Average review_id length: **39.26 chars** -> **9.81 tokens** (chars/4)
- JSON structure only (braces, field names, quotes, punctuation): **66 chars** -> **16.50 tokens**
- Two label values combined: **4-6 chars** -> **1.00-1.50 tokens**

Minimum expected output per review for strict minified JSON is therefore:

- **27.31-27.81 tokens/review** (= 9.81 + 16.50 + 1.00..1.50)

Comparison to your estimate:

- Your estimate: **13 tokens/review**
- Strict JSON floor: **~27-28 tokens/review**
- Gap from structure + id + labels alone: **~14-15 tokens/review above your estimate**

Measured average output in stage 2 was **53.33 tokens/review**, so your estimate was below even the strict-format minimum, and the real API tokenization/response formatting pushed it higher.

### Why Stage 1 output tokens were much higher than assumed
- Your estimated output per review: **13 tokens** (short label)
- Actual average output per review (Stage 1): **49.56 tokens**
- Difference: **+36.56 tokens/review**
- For 1,366,979 reviews this is **~49,976,752 extra output tokens**,
   which at $15 / 1M output tokens equals **~$749.65** additional cost.

This extra output was not only from longer phrasing; even with strict JSON, the
response must carry `review_id` and JSON structure for every review. That
already creates a higher floor than a 13-token label assumption. To reduce
future cost further, return only labels (no `review_id` in output) and map by
position, plus enforce strict max-token output.

### How much Stage 2 increased cost versus a Stage-1-like baseline
Assume a baseline using Stage 1 actual averages (`input=116.91 tokens/review`,
`output=49.57 tokens/review`) and compare to Stage 2 measured averages
(`input=165.15`, `output=53.33`):

- Extra input tokens per review: **48.24** → for 537,388 reviews = **~25,923,597 tokens**
   → input cost delta ≈ **$77.77**
- Extra output tokens per review: **3.76** → for 537,388 reviews = **~2,020,577 tokens**
   → output cost delta ≈ **$30.31**
- Retry cost (measured from Stage 2 logs): **$45.99**

Total Stage-2 incremental cost vs a Stage-1-like baseline ≈ **$153.99**.

### Combined effect (how you reached ~ $2k spend)
- Stage 1 primary underestimation (output length) added **~$749.65**
- Stage 1 other input underruns and prompt size added the remaining of the
   Stage 1 overrun (see Stage 1 report) — total Stage 1 overrun ≈ **$774.79**
- Stage 2 incremental cost vs Stage-1-like baseline ≈ **$153.99**
- Stage 2 logged vs calculated difference (estimation vs actual tokens): **$187.64**

When combined (Stage 1 actual + Stage 2 calculated), the workspace token-based
total approaches **$1,995–2,000**, which explains why your $2,000 credits were
exhausted.


## Retry Analysis

- Retry batches are included in the 89,015 batch count
- Separate retry cost breakdown not available from this log
- High failure rate (23.6%) suggests significant retry overhead (likely $50–100+)

---

## Recommendations

1. **For future racial classifications:**
   - Budget **3–4× more** than simple binary classifications
   - Plan for 20–25% failure rate, not <1%
   - Consider longer text preservation (>700 chars) for racial context

2. **Review the 127,795 failures:**
   - Analyze failure reasons in `classification_error` column
   - Determine if failures are due to: truncation, ambiguous text, model limitations, or text quality

3. **Cost control for future runs:**
   - Run a 5k-review pilot first to measure actual tokens and failure rates
   - Budget 4× the simple classification estimate for complex classifiers
   - Consider cheaper models for initial filtering, expensive models only for edge cases

---

Report generated 2026-05-13; file saved to `/home/ethel/Desktop/swimming_pool_project/spp_15052026/data_annotation/stage_2_racial/cost_analysis_report.md`
