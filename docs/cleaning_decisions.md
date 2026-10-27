# Data preparation and cleaning decisions

## Inputs and handling

The supplied source files are preserved as-is in `data/`. The notebook reads them directly; it does not overwrite or relocate them. Since these are local raw data files, the project `.gitignore` excludes the entire `data/` directory.

## Verified checks

| Check | Observed result | Decision |
|---|---:|---|
| Training rows and `num_orders` total | 456,548 rows; 119,557,485 orders | Keep as the pre-join baseline. |
| Missing values in training, center, and meal inputs | 0 in every column | No imputation needed. |
| Exact duplicate training rows | 0 | No row removal needed. |
| Repeated `(center_id, meal_id, week)` keys | 0 | No key-based deduplication needed. |
| Duplicate center and meal dimension keys | 0 | Use validated many-to-one joins. |
| Training keys absent from center/meal dimensions | 0 | All training rows have dimension matches. |
| `checkout_price > base_price` | 116,101 rows | Retain. A checkout price above the reference/base price may be a valid price state, not evidence of corruption. Compute signed discount as specified. |
| High-order observations above the global Tukey upper fence | 32,937 rows (7.21%) | Flag for interpretation; retain because the project asks for demand analysis and these may be real peak demand. |
| Weeks observed per meal | 87 to 145 of 145 | Retain. Report coverage; absence of a meal-week row is not proof of zero demand or a stockout. |
| Weeks in training | 1–145 | Use as an ordered week index only; there are no real dates. |
| Test file | 32,573 rows, weeks 146–155, no `num_orders` target | Do not use it to calculate historical outcomes or validate a supervised forecast. |

## Transformations

The notebook derives `discount_rate = (base_price - checkout_price) / base_price`, `revenue = checkout_price * num_orders`, and weekly meal order totals. All dimension joins are validated as many-to-one; assertions confirm that training row count and total orders are unchanged after both joins.

No observations are dropped or winsorized. Missing week-meal combinations are not filled with zero because an absent record could reflect availability/listing behavior, not necessarily zero customer demand. Statistical tests use only center-meal strata that contain both states for the promotion being tested.
