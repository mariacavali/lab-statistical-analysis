import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_exploration import group_performance, safe_divide
from statistical_analysis import (
    MONTHLY_BUDGET,
    apply_corrections,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    constrained_budget_allocation,
    effect_interpretation,
)

ROOT = Path(__file__).parents[1]


def test_safe_divide_handles_zero_and_infinity():
    result = safe_divide(pd.Series([4, 3]), pd.Series([2, 0]))
    assert result.iloc[0] == 2
    assert np.isnan(result.iloc[1])


def test_effect_size_and_interpretation():
    assert np.isfinite(cohens_d([1, 2, 3], [2, 3, 4]))
    assert effect_interpretation(0.1) == "negligible"
    assert effect_interpretation(0.3) == "small"
    assert effect_interpretation(0.6) == "medium"
    assert effect_interpretation(0.9) == "large"


def test_bh_adjustment_is_monotone_in_pvalue_order():
    p_values = np.array([0.04, 0.001, 0.02, 0.8])
    adjusted = benjamini_hochberg(p_values)
    ordered = adjusted[np.argsort(p_values)]
    assert np.all(np.diff(ordered) >= -1e-12)
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_corrections_add_required_columns():
    frame = pd.DataFrame({"p_value": [0.001, 0.04, 0.5]})
    corrected = apply_corrections(frame)
    assert {"significant_bonferroni", "p_value_fdr", "significant_fdr"}.issubset(corrected.columns)


def test_bootstrap_ci_is_reproducible_and_contains_mean():
    values = [1, 2, 3, 4, 5]
    first = bootstrap_ci(values, n_bootstrap=500, seed=7)
    second = bootstrap_ci(values, n_bootstrap=500, seed=7)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_budget_allocation_totals_exactly_500k():
    summary = pd.DataFrame(
        {"channel": list("ABCDE"), "cpa": [5, 6, 7, 8, 9], "roas": [2, 1.8, 1.6, 1.4, 1.2]}
    )
    allocation = constrained_budget_allocation(summary)
    assert allocation["allocation_dollars"].sum() == MONTHLY_BUDGET
    assert np.isclose(allocation["allocation_pct"].sum(), 1)


def test_generated_outputs_are_complete_and_consistent():
    required_files = [
        "marketing_data.csv",
        "group_metrics_overview.png",
        "group_distributions.png",
        "metric_comparison_heatmap.png",
        "rate_comparison.png",
        "correction_comparison.png",
        "power_analysis_cpa.png",
        "executive_memo.md",
    ]
    for relative in required_files:
        path = ROOT / relative
        assert path.exists() and path.stat().st_size > 0
    allocation = pd.read_csv(ROOT / "tables" / "budget_allocation.csv")
    assert int(allocation["allocation_dollars"].sum()) == MONTHLY_BUDGET
    summary = json.loads((ROOT / "tables" / "analysis_summary.json").read_text())
    assert summary["budget_total"] == MONTHLY_BUDGET
    assert summary["total_comparisons"] > 0
