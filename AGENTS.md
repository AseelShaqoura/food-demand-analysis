# Project Guidelines

## Project and data
- This project analyzes weekly meal demand from the supplied, anonymized CSV files in `data/`.
- Treat all supplied data as read-only. Never edit, relocate, duplicate, or publish the original files.
- The source contains no real calendar dates; interpret `week` only as an ordered index.

## Analysis and reporting
- Validate join keys and row/order totals before and after each merge.
- Keep valid high-demand observations; flag outliers and explain decisions rather than silently deleting rows.
- Every reported number must be reproducible from `notebooks/analysis.ipynb` and its source data.
- Distinguish observed association from causation. State sample sizes, test, and limitations alongside statistical claims.
- Keep reports client-friendly, use labeled charts, and include a limitations section.

## Running
- From the project root, run `python notebooks/analysis.py` to regenerate the figures and client PDF.
- The notebook is generated from the same `# %%` source and should run from first cell to last.
