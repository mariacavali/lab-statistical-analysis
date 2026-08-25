# In God we trust, everyone else bring the data

This submission analyzes the public, simulated Kaggle [Campaign Effectiveness dataset](https://www.kaggle.com/datasets/sanak2000/campaign-effectiveness). It compares ten marketing channels using campaign-level CPA, aggregate conversion rates, multiple-testing corrections, bootstrap confidence intervals, and a reproducible power simulation. The business interpretation and constrained $500K allocation are in `executive_memo.md`.

## Run the analysis

Python 3.9+ is required. The source workbook is included, so no Kaggle credentials are needed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
pytest
```

`run_analysis.py` regenerates the cleaned CSV, tables, memo, and figures from the raw workbook. Simulation and bootstrap seeds are fixed for reproducibility.

## File map

- `data/raw/campaign_effectiveness.xlsx` — unchanged Kaggle source workbook.
- `dataset_documentation.md` — source choice, schema, quality, and cleaning decisions.
- `data_exploration.py` — validates and cleans data, calculates metrics, and creates exploration charts.
- `statistical_analysis.py` — runs pairwise tests, corrections, confidence intervals, power analysis, and budget allocation.
- `run_analysis.py` — executes the complete workflow.
- `marketing_data.csv` — prepared campaign-level dataset.
- `tables/` — machine-readable summaries and all statistical results.
- `*.png` — required performance, distribution, test, correction, and power visualizations.
- `executive_memo.md` — findings, $500K recommendation, caveats, and next steps.
- `reflection.md` — short learning reflection.
- `tests/` — automated checks for calculations and generated deliverables.
- `instructions.md` and `rubric.md` — original assignment and grading criteria.

The source has no date field. Results demonstrate the required method but should not be interpreted as observed company performance or causal evidence.
