# Cost Breakdown for 13,000 Reviews Analysis

## Themes Classifier
| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |
| :--- | :--- | :--- | :--- |
| Prompt input | 516 | 6,708,000 | 20.12 |
| Message input | 0 | 0 | 0.00 |
| Review input | 66 | 858,000 | 2.57 |
| **Total Input** | **582** | **7,566,000** | **22.70** |
| Output | 25 | 325,000 | 4.88 |
| **Base Total** | - | - | **27.57** |
| Retries (10%) | - | - | 2.76 |

### Retry Sensitivity
| Retry Rate | Addl. Cost ($) | Total Cost ($) |
| :--- | :--- | :--- |
| 5% | 1.38 | 28.95 |
| 10% | 2.76 | 30.33 |
| 20% | 5.51 | 33.09 |

## Connotation Classifier
| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |
| :--- | :--- | :--- | :--- |
| Prompt input | 516 | 6,708,000 | 20.12 |
| Message input | 0 | 0 | 0.00 |
| Review input | 66 | 858,000 | 2.57 |
| **Total Input** | **582** | **7,566,000** | **22.70** |
| Output | 25 | 325,000 | 4.88 |
| **Base Total** | - | - | **27.57** |
| Retries (10%) | - | - | 2.76 |

### Retry Sensitivity
| Retry Rate | Addl. Cost ($) | Total Cost ($) |
| :--- | :--- | :--- |
| 5% | 1.38 | 28.95 |
| 10% | 2.76 | 30.33 |
| 20% | 5.51 | 33.09 |

## Approach Classifier
| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |
| :--- | :--- | :--- | :--- |
| Prompt input | 410 | 5,330,000 | 15.99 |
| Message input | 106 | 1,378,000 | 4.13 |
| Review input | 66 | 858,000 | 2.57 |
| **Total Input** | **582** | **7,566,000** | **22.70** |
| Output | 25 | 325,000 | 4.88 |
| **Base Total** | - | - | **27.57** |
| Retries (10%) | - | - | 2.76 |

### Retry Sensitivity
| Retry Rate | Addl. Cost ($) | Total Cost ($) |
| :--- | :--- | :--- |
| 5% | 1.38 | 28.95 |
| 10% | 2.76 | 30.33 |
| 20% | 5.51 | 33.09 |

## Context Classifier
| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |
| :--- | :--- | :--- | :--- |
| Prompt input | 432 | 5,616,000 | 16.85 |
| Message input | 84 | 1,092,000 | 3.28 |
| Review input | 66 | 858,000 | 2.57 |
| **Total Input** | **582** | **7,566,000** | **22.70** |
| Output | 25 | 325,000 | 4.88 |
| **Base Total** | - | - | **27.57** |
| Retries (10%) | - | - | 2.76 |

### Retry Sensitivity
| Retry Rate | Addl. Cost ($) | Total Cost ($) |
| :--- | :--- | :--- |
| 5% | 1.38 | 28.95 |
| 10% | 2.76 | 30.33 |
| 20% | 5.51 | 33.09 |

## Grand Totals
| Metric | Cost ($) |
| :--- | :--- |
| Total Base Cost | **110.29** |
| Total with 10% Retries | **121.32** |

## Assumptions
- Character to token conversion factor: 0.25.
- Input pricing: $3.0 per 1M tokens.
- Output pricing: $15.0 per 1M tokens.
- Retries are modeled as additional full calls (input + output).
- Batch size = 1.
