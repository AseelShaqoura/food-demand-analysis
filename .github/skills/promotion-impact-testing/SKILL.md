---
name: promotion-impact-testing
description: 'Use when testing or explaining the association between email promotion or homepage featuring and meal orders in this project, or when calculating the discount-demand relationship. Use the specified matched center-meal comparison, Mann–Whitney U test, and Spearman correlation. Do not use for unrelated analysis.'
---

# Promotion and Discount Analysis

## When to use
- The question is whether observed orders differ with email promotion or homepage featuring, or how discount rate relates to orders.
- Do not invoke for unrelated programming tasks or analyses that do not involve these measures.

## Procedure
1. Use training observations only and retain the original rows.
2. For each binary promotion separately, include only `(center_id, meal_id)` strata that contain both promoted and unpromoted observations.
3. Report both sample sizes, group medians, median percentage difference, two-sided Mann–Whitney U statistic/p-value, and the number of eligible strata.
4. Calculate `discount_rate = (base_price - checkout_price) / base_price`; report Spearman rho, p-value, and sample size.
5. Explicitly note that promotions co-occur with other conditions, observations are not randomized, and neither a small p-value nor correlation establishes causation.
6. Treat `checkout_price > base_price` as an observed price relationship, not an automatic data error; do not silently discard those rows.

## Verification
- Check that only strata with both states contribute to each promotion comparison.
- Verify sample sizes and statistics by rerunning the analysis notebook.
- Include the test method and the non-causal caveat in every client-facing summary.
