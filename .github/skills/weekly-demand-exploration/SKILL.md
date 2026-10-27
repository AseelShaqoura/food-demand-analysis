---
name: weekly-demand-exploration
description: 'Use when analyzing weekly food demand, joining the supplied meal and fulfilment-center data, checking data quality, calculating demand KPIs, ranking categories or meals, comparing centers, or preparing this project''s demand charts and findings. Do not use for unrelated coding or general questions.'
---

# Weekly Demand Exploration

## When to use
- The task concerns weekly order patterns, category or meal concentration, center performance, demand variability, or source-data quality in this project.
- Do not invoke for unrelated topics such as writing UI code or general Python questions.

## Procedure
1. Read the local CSVs without changing or copying them.
2. Validate identifier uniqueness and join coverage. Use many-to-one merge validation and confirm row count and total orders are unchanged.
3. Report missing values, duplicates, price comparisons, order outliers, and each meal's observed week coverage. Treat `week` as an index, not a date.
4. Compute weekly demand, category/meal totals and cumulative share, center and center-type summaries, revenue proxy, discount rate, promotion rates, and meal-level demand variability.
5. Label every chart with units and an outcome-focused title. Keep all valid observations unless a documented, justified rule requires otherwise.
6. Ground each conclusion in a printed table or calculated statistic; separate fact from interpretation and state the limitations.

## Verification
- Re-run the notebook from its first cell.
- Assert join row counts and total orders match the original training data.
- Confirm generated chart files are saved under `reports/figures/`.
