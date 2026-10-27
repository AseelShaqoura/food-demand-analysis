# Food Demand Analysis

## Business question

An anonymized meal-delivery business needs a better evidence base for weekly purchasing: demand varies across meals and fulfilment centers, while promotion and pricing choices are not yet well understood. This project describes the patterns in the supplied training data and offers measurable, non-causal recommendations.

## Verified headline findings

- The training data contains **119,557,485 orders** across **145 weeks** (456,548 rows). Average observed demand is about **824,534 orders per week**.
- **Beverages, Rice Bowl, Sandwich, Salad, and Pizza together account for more than 80% of orders**; the top three categories contribute about **66.1%**.
- In center-meal combinations where both states were observed, promoted rows had higher medians: **126.8% higher for email promotion** and **136.5% for homepage featuring**. These are associations, not causal lift estimates.

All metrics are computed by `notebooks/analysis.ipynb`; the notebook also checks source quality, validates joins, tests promotion differences and discount association, and generates the charts and client PDF.

## Project contents

- `data/` — supplied source files, kept unchanged and excluded from Git.
- `notebooks/analysis.ipynb` — runnable analysis notebook.
- `notebooks/analysis.py` — the notebook's `# %%` source; run it to regenerate deliverables.
- `reports/figures/` — generated result charts.
- `reports/client_report.pdf` — designed Arabic client report with a short presentation-style summary.
- `docs/cleaning_decisions.md` — verified quality checks and documented treatment decisions.
- `.github/skills/` — two task-specific Copilot skills.
- `AGENTS.md` — project-wide analysis and data-handling rules.

## Run locally

Run from the project root with Python 3.10+ and the following packages:

```powershell
python -m pip install pandas numpy "matplotlib>=3.9" scipy reportlab arabic-reshaper python-bidi
python notebooks\analysis.py
```

The command reads only the local CSVs and writes PNG charts and the PDF beneath `reports/`. To run interactively, open `notebooks/analysis.ipynb` and execute all cells from top to bottom with the project root as the working directory.

## Methods and limits

The training table is joined to the two dimension tables on their unique IDs with validated many-to-one joins; row count and order total are asserted unchanged. Promotion comparisons use a two-sided Mann–Whitney U test within eligible center-meal strata, and the signed discount rate is compared with orders using Spearman correlation. Orders are observational and promotion assignment is not randomized, so these results do not prove causal effects. Weeks have no calendar dates, cities are coded, missing meal-week rows are not assumed to be zero demand, and revenue is only the supplied checkout price multiplied by orders (not profit).

## Use of AI

Copilot assisted with drafting analysis and project documentation. The available workspace skill catalog did not include a ready-made skill specific to this demand-analysis task, so the project includes two focused skills for weekly exploration and promotion testing. The results in the report are computed from the local data by the notebook and were checked by running the analysis; generated interpretations are not treated as evidence without a corresponding calculation.

## Custom skill trigger checks

- **Weekly demand exploration:** “حلّل الطلب الأسبوعي، وحصة الفئات، وأداء المراكز من ملفات المشروع.” is in scope. “أنشئ صفحة HTML لتسجيل الدخول.” is out of scope.
- **Promotion impact testing:** “اختبر فرق الطلب مع البريد والظهور واستخدم Mann–Whitney وSpearman.” is in scope. “أصلح تنسيق CSS لصفحة الإعدادات.” is out of scope.

These are scope/description checks; automatic activation could not be exercised from this batch environment.
