"""Run pairwise tests, corrections, power analysis, CIs, and recommendations."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data_exploration import CLEAN_DATA, ROOT, TABLES, group_performance, run_exploration

ALPHA = 0.05
RANDOM_SEED = 20260825
MONTHLY_BUDGET = 500_000
PRIMARY_METRIC = "cpa"


def cohens_d(group_a: Iterable[float], group_b: Iterable[float]) -> float:
    """Return Cohen's d using the lab's average-variance pooled SD formula."""
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(b) - np.mean(a)) / pooled_sd) if pooled_sd > 0 else np.nan


def effect_interpretation(effect: float) -> str:
    magnitude = abs(effect)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def pairwise_t_tests(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    rows = []
    groups = sorted(df["channel"].unique())
    for group_a, group_b in itertools.combinations(groups, 2):
        a = df.loc[df["channel"].eq(group_a), metric].replace([np.inf, -np.inf], np.nan).dropna()
        b = df.loc[df["channel"].eq(group_b), metric].replace([np.inf, -np.inf], np.nan).dropna()
        test = stats.ttest_ind(a, b, equal_var=False)
        mean_a, mean_b = float(a.mean()), float(b.mean())
        difference = mean_b - mean_a
        effect = cohens_d(a, b)
        rows.append(
            {
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference_b_minus_a": difference,
                "percentage_difference": difference / mean_a * 100 if mean_a else np.nan,
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohens_d": effect,
                "effect_size": effect_interpretation(effect),
                "significant_uncorrected": bool(test.pvalue < ALPHA),
            }
        )
    return pd.DataFrame(rows)


def pairwise_fisher_tests(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("channel").agg(conversions=("conversions", "sum"), clicks=("clicks", "sum"))
    rows = []
    for group_a, group_b in itertools.combinations(sorted(totals.index), 2):
        conversions_a = int(totals.loc[group_a, "conversions"])
        conversions_b = int(totals.loc[group_b, "conversions"])
        clicks_a = int(totals.loc[group_a, "clicks"])
        clicks_b = int(totals.loc[group_b, "clicks"])
        non_conversions_a = clicks_a - conversions_a
        non_conversions_b = clicks_b - conversions_b
        odds_ratio, p_value = stats.fisher_exact(
            [[conversions_a, non_conversions_a], [conversions_b, non_conversions_b]],
            alternative="two-sided",
        )
        rate_a, rate_b = conversions_a / clicks_a, conversions_b / clicks_b
        difference = rate_b - rate_a
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "conversions_a": conversions_a,
                "non_conversions_a": non_conversions_a,
                "attempts_a": clicks_a,
                "conversions_b": conversions_b,
                "non_conversions_b": non_conversions_b,
                "attempts_b": clicks_b,
                "rate_a": rate_a,
                "rate_b": rate_b,
                "rate_difference_b_minus_a": difference,
                "percentage_difference": difference / rate_a * 100 if rate_a else np.nan,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
                "significant_uncorrected": bool(p_value < ALPHA),
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if hasattr(stats, "false_discovery_control"):
        return np.asarray(stats.false_discovery_control(values, method="bh"))
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def apply_corrections(results: pd.DataFrame) -> pd.DataFrame:
    corrected = results.copy()
    corrected["bonferroni_alpha"] = ALPHA / len(corrected)
    corrected["significant_bonferroni"] = corrected["p_value"] < corrected["bonferroni_alpha"]
    corrected["p_value_fdr"] = benjamini_hochberg(corrected["p_value"])
    corrected["significant_fdr"] = corrected["p_value_fdr"] < ALPHA
    return corrected


def bootstrap_ci(data: Iterable[float], n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = RANDOM_SEED) -> tuple[float, float]:
    values = np.asarray(list(data), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(n_bootstrap, len(values)), replace=True), axis=1)
    alpha = 1 - ci_level
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def empirical_power_cpa(true_diff_pct: float, base_cpa: float, n_days: int, n_sim: int = 1000, alpha: float = 0.05) -> float:
    """Estimate two-sample CPA-test power with 15% SD and a 50%-of-base floor."""
    seed = RANDOM_SEED + int(round(true_diff_pct * 10000)) + n_days * 101 + n_sim
    rng = np.random.default_rng(seed)
    standard_deviation = base_cpa * 0.15
    mean_b = base_cpa * (1 + true_diff_pct)
    detections = 0
    for _ in range(n_sim):
        group_a = np.clip(rng.normal(base_cpa, standard_deviation, n_days), base_cpa * 0.5, None)
        group_b = np.clip(rng.normal(mean_b, standard_deviation, n_days), base_cpa * 0.5, None)
        if stats.ttest_ind(group_a, group_b, equal_var=False).pvalue < alpha:
            detections += 1
    return detections / n_sim


def run_power_analysis(base_cpa: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = [0.05, 0.10, 0.15, 0.20]
    sample_sizes = [30, 60, 90, 120, 180]
    rows = [
        {"effect_size_pct": effect * 100, "n_days": n, "power": empirical_power_cpa(effect, base_cpa, n)}
        for effect in effects
        for n in sample_sizes
    ]
    results = pd.DataFrame(rows)
    requirements = []
    for effect in effects:
        subset = results.loc[results["effect_size_pct"].eq(effect * 100)]
        qualifying = subset.loc[subset["power"] >= 0.8, "n_days"]
        minimum = int(qualifying.min()) if not qualifying.empty else np.nan
        power_90 = float(subset.loc[subset["n_days"].eq(90), "power"].iloc[0])
        requirements.append(
            {
                "effect_size_pct": effect * 100,
                "minimum_tested_days_for_80pct_power": minimum,
                "power_at_90_days": power_90,
                "is_90_days_sufficient": power_90 >= 0.8,
            }
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for effect, subset in results.groupby("effect_size_pct"):
        ax.plot(subset["n_days"], subset["power"], marker="o", label=f"{effect:.0f}% difference")
    ax.axhline(0.8, color="black", linestyle="--", label="80% target")
    ax.set(ylim=(0, 1.02), xlabel="Days per group", ylabel="Empirical power", title="CPA power simulation (1,000 simulations per point)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "power_analysis_cpa.png", dpi=180)
    plt.close(fig)
    return results, pd.DataFrame(requirements)


def plot_pvalue_heatmap(results: pd.DataFrame, groups: list[str]) -> None:
    matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)
    for row in results.itertuples():
        matrix.loc[row.group_a, row.group_b] = row.p_value
        matrix.loc[row.group_b, row.group_a] = row.p_value
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="RdYlGn", vmin=0, vmax=0.1, annot=True, fmt=".3f", ax=ax, cbar_kws={"label": "p-value"})
    ax.set_title("Pairwise Welch t-test p-values for campaign-level CPA")
    fig.tight_layout()
    fig.savefig(ROOT / "metric_comparison_heatmap.png", dpi=180)
    plt.close(fig)


def plot_conversion_rates(summary: pd.DataFrame) -> None:
    plot_data = summary.sort_values("conversion_rate")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=plot_data, y="channel", x="conversion_rate", color="#8b5cf6", ax=ax)
    for patch, value in zip(ax.patches, plot_data["conversion_rate"]):
        ax.text(value, patch.get_y() + patch.get_height() / 2, f" {value:.2%}", va="center")
    ax.set(xlabel="Aggregate conversion rate", ylabel="", title="Conversion rate by channel")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(ROOT / "rate_comparison.png", dpi=180)
    plt.close(fig)


def correction_summary(t_tests: pd.DataFrame, fisher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in [("CPA t-tests", t_tests), ("Conversion Fisher tests", fisher)]:
        rows.extend(
            [
                {"test_family": family, "method": "Uncorrected", "significant_results": int(frame["significant_uncorrected"].sum())},
                {"test_family": family, "method": "Bonferroni", "significant_results": int(frame["significant_bonferroni"].sum())},
                {"test_family": family, "method": "FDR (BH)", "significant_results": int(frame["significant_fdr"].sum())},
            ]
        )
    return pd.DataFrame(rows)


def plot_correction_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=summary, x="test_family", y="significant_results", hue="method", ax=ax)
    ax.set(xlabel="", ylabel="Significant pairwise results", title="Effect of multiple-comparison correction")
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(ROOT / "correction_comparison.png", dpi=180)
    plt.close(fig)


def confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (channel, group) in enumerate(df.groupby("channel")):
        values = group["cpa"].dropna()
        lower, upper = bootstrap_ci(values, seed=RANDOM_SEED + index)
        rows.append({"channel": channel, "n": len(values), "mean_cpa": values.mean(), "ci_95_lower": lower, "ci_95_upper": upper})
    return pd.DataFrame(rows).sort_values("mean_cpa")


def constrained_budget_allocation(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary[["channel", "cpa", "roas"]].copy()
    n_groups = len(ranked)
    ranked["cpa_rank"] = ranked["cpa"].rank(method="average", ascending=True)
    ranked["roas_rank"] = ranked["roas"].rank(method="average", ascending=False)
    ranked["cpa_score"] = n_groups + 1 - ranked["cpa_rank"]
    ranked["roas_score"] = n_groups + 1 - ranked["roas_rank"]
    ranked["composite_score"] = 0.5 * ranked["cpa_score"] + 0.5 * ranked["roas_score"]

    minimum, maximum = 0.05, 0.15
    weights = ranked["composite_score"].to_numpy(dtype=float)
    weights /= weights.sum()
    for _ in range(100):
        clipped = np.clip(weights, minimum, maximum)
        fixed = (clipped == minimum) | (clipped == maximum)
        remainder = 1 - clipped[fixed].sum()
        if (~fixed).sum() == 0:
            weights = clipped
            break
        free_raw = weights[~fixed]
        clipped[~fixed] = remainder * free_raw / free_raw.sum()
        if np.allclose(clipped, weights, atol=1e-12):
            weights = clipped
            break
        weights = clipped
    weights /= weights.sum()

    raw_dollars = weights * MONTHLY_BUDGET
    allocations = np.floor(raw_dollars / 100) * 100
    remaining_units = int(round((MONTHLY_BUDGET - allocations.sum()) / 100))
    order = np.argsort(-(raw_dollars - allocations))
    allocations[order[:remaining_units]] += 100

    ranked["allocation_dollars"] = allocations.astype(int)
    ranked["allocation_pct"] = ranked["allocation_dollars"] / MONTHLY_BUDGET
    ranked["rationale"] = ranked.apply(
        lambda row: f"Heuristic 50/50 CPA–ROAS rank; CPA rank {row.cpa_rank:.0f}, ROAS rank {row.roas_rank:.0f}; constrained to 5%–15%.",
        axis=1,
    )
    if int(ranked["allocation_dollars"].sum()) != MONTHLY_BUDGET:
        raise AssertionError("Budget allocation does not total exactly $500,000")
    return ranked.sort_values(["composite_score", "channel"], ascending=[False, True]).reset_index(drop=True)


def observed_power(t_tests: pd.DataFrame, base_cpa: float) -> pd.DataFrame:
    rows = []
    for row in t_tests.loc[t_tests["significant_fdr"]].itertuples():
        baseline = min(row.mean_a, row.mean_b)
        effect = abs(row.difference_b_minus_a) / baseline if baseline else np.nan
        rows.append(
            {
                "group_a": row.group_a,
                "group_b": row.group_b,
                "observed_difference_pct": effect * 100,
                "power_at_90_days": empirical_power_cpa(effect, base_cpa, 90) if np.isfinite(effect) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["group_a", "group_b", "observed_difference_pct", "power_at_90_days"])


def write_executive_memo(
    summary: pd.DataFrame,
    t_tests: pd.DataFrame,
    fisher: pd.DataFrame,
    corrections: pd.DataFrame,
    power_requirements: pd.DataFrame,
    ci: pd.DataFrame,
    allocation: pd.DataFrame,
    quality: dict,
) -> None:
    best_cpa = summary.loc[summary["cpa"].idxmin()]
    best_roas = summary.loc[summary["roas"].idxmax()]
    best_conversion = summary.loc[summary["conversion_rate"].idxmax()]
    top_ci = ci.loc[ci["channel"].eq(best_cpa["channel"])].iloc[0]
    fdr_t = t_tests.loc[t_tests["significant_fdr"]]
    fdr_fisher = fisher.loc[fisher["significant_fdr"]]
    total_comparisons = len(t_tests) + len(fisher)
    budget_rows = "\n".join(
        f"| {row.channel} | ${row.allocation_dollars:,.0f} | {row.allocation_pct:.1%} | {row.rationale} |"
        for row in allocation.itertuples()
    )
    power_rows = "\n".join(
        f"| {row.effect_size_pct:.0f}% | {row.power_at_90_days:.1%} | {'Yes' if row.is_90_days_sufficient else 'No'} | "
        f"{'>' + str(180) if pd.isna(row.minimum_tested_days_for_80pct_power) else int(row.minimum_tested_days_for_80pct_power)} |"
        for row in power_requirements.itertuples()
    )
    correction_rows = "\n".join(
        f"| {row.test_family} | {row.method} | {row.significant_results} |" for row in corrections.itertuples()
    )
    largest_rate_gap = fisher.loc[fisher["significant_fdr"], "rate_difference_b_minus_a"].abs().max()
    memo = f"""# Marketing Channel Statistical Analysis

**Date:** 25 August 2026

**Analyst:** Marija Kavaliauskaite

**Dataset:** Campaign Effectiveness (Kaggle; simulated public campaign data)

**Period analyzed:** No date field; 5,500 independent campaign executions

## Executive Summary

The 10-channel dataset shows no CPA pair surviving Benjamini–Hochberg FDR correction, while {len(fdr_fisher)} of {len(fisher)} aggregate conversion-rate pairs do survive. **{best_cpa['channel']}** has the lowest aggregate CPA (${best_cpa['cpa']:,.2f}), **{best_roas['channel']}** has the highest ROAS ({best_roas['roas']:.3f}), and **{best_conversion['channel']}** has the highest conversion rate ({best_conversion['conversion_rate']:.2%}). Even the largest FDR-significant conversion-rate gap is only {largest_rate_gap:.2%}, so practical importance is modest. Because the source is simulated and does not establish causal channel effects, the proposed $500K budget is a constrained planning heuristic, not a forecast.

## Key Findings

- Lowest aggregate CPA: **{best_cpa['channel']}**, ${best_cpa['cpa']:,.2f}; its campaign-level mean CPA is ${top_ci['mean_cpa']:,.2f} with a bootstrap 95% CI of ${top_ci['ci_95_lower']:,.2f}–${top_ci['ci_95_upper']:,.2f}.
- Highest aggregate ROAS: **{best_roas['channel']}**, {best_roas['roas']:.3f}; aggregate profit is ${best_roas['profit']:,.0f}.
- Highest aggregate conversion rate: **{best_conversion['channel']}**, {best_conversion['conversion_rate']:.2%}.
- Pairwise CPA results: {int(t_tests['significant_uncorrected'].sum())} of {len(t_tests)} uncorrected, {int(t_tests['significant_bonferroni'].sum())} Bonferroni-significant, and {len(fdr_t)} FDR-significant.
- Fisher conversion-rate results: {int(fisher['significant_uncorrected'].sum())} of {len(fisher)} uncorrected, {int(fisher['significant_bonferroni'].sum())} Bonferroni-significant, and {len(fdr_fisher)} FDR-significant.
- Across {total_comparisons} tests, alpha 0.05 implies {total_comparisons * ALPHA:.1f} expected false positives under the complete null. Conversion-rate differences remain statistically detectable after correction, but their small absolute size and the simulated, non-randomized source do not support a causal winner claim.

| Test family | Method | Significant results |
| --- | --- | ---: |
{correction_rows}

## Data Adequacy / Power Analysis

The required simulation assumes campaign-level CPA has a standard deviation equal to 15% of baseline, which is narrower than the actual heterogeneous campaign-level distribution; its power estimates are planning illustrations, not a fitted model of this dataset. At 90 observations per group, the simulation gives:

| True CPA difference | Power at 90 days | 90 days sufficient? | Minimum tested days for ~80% power |
| ---: | ---: | :---: | ---: |
{power_rows}

The current per-channel samples ({min(quality['sample_sizes'].values())}–{max(quality['sample_sizes'].values())} campaigns) exceed 180 observations, but rows are campaign executions rather than dated daily observations. No FDR-significant CPA pair exists for an observed-effect power follow-up.

## $500K Monthly Budget Recommendation

| Channel | Allocation | Allocation % | Rationale |
| --- | ---: | ---: | --- |
{budget_rows}
| **Total** | **${allocation['allocation_dollars'].sum():,.0f}** | **{allocation['allocation_pct'].sum():.1%}** | Constrained heuristic total |

The allocation combines descriptive CPA and ROAS ranks at 50% each, enforces a 5% floor and 15% ceiling, and rounds to $100 while totaling exactly $500,000. Larger allocations indicate comparatively favorable historical point estimates, not proven incremental returns. The ceiling limits overcommitment where corrected evidence is weak.

## Statistical Caveats

- Kaggle describes this as a **simulated** public dataset; it is not audited company performance and has no date field, geography, targeting assignment, or channel-selection process.
- Campaigns are observational groupings, not randomized channel assignments. Correlation does not establish that shifting spend will cause the same outcome.
- Welch t-tests compare campaign-level CPA means; CPA is right-skewed and influenced by campaign scale. Fisher tests aggregate clicks/conversions and assume independent attempts, which cannot be verified from the source.
- Bonferroni controls family-wise error and is conservative; FDR controls the expected false-discovery proportion and is the primary screening rule here. Neither correction fixes confounding or measurement quality.
- Confidence intervals quantify sampling uncertainty under the available campaigns, not data-generation or model uncertainty. Statistical significance and practical value are different questions.
- The power simulation uses a stylized normal CPA model and fixed 15% variability. Historical results cannot validate a future $500K budget or predict saturation.

## Next Steps

1. Run randomized, geo- or audience-matched incrementality tests before materially reallocating live spend.
2. Collect daily campaign data with audience, placement, attribution window, and margin-adjusted revenue for at least the sample durations indicated above.
3. Monitor CPA, ROAS, conversion quality, and marginal return weekly; hold the 5%–15% guardrails until replicated evidence supports changes.
4. Re-run the same corrected analysis on observed company data and pre-register the primary metric and decision thresholds.
"""
    (ROOT / "executive_memo.md").write_text(memo, encoding="utf-8")


def run_statistical_analysis() -> dict:
    TABLES.mkdir(exist_ok=True)
    if not CLEAN_DATA.exists():
        df, summary, quality = run_exploration()
    else:
        df = pd.read_csv(CLEAN_DATA)
        summary = group_performance(df)
        quality = json.loads((TABLES / "data_quality_summary.json").read_text(encoding="utf-8"))

    t_tests = apply_corrections(pairwise_t_tests(df))
    fisher = apply_corrections(pairwise_fisher_tests(df))
    corrections = correction_summary(t_tests, fisher)
    ci = confidence_intervals(df)
    allocation = constrained_budget_allocation(summary)
    base_cpa = float(df["cpa"].median())
    power, power_requirements = run_power_analysis(base_cpa)
    observed = observed_power(t_tests, base_cpa)

    t_tests.to_csv(TABLES / "cpa_pairwise_tests.csv", index=False)
    fisher.to_csv(TABLES / "fisher_conversion_tests.csv", index=False)
    corrections.to_csv(TABLES / "correction_summary.csv", index=False)
    power.to_csv(TABLES / "power_analysis.csv", index=False)
    power_requirements.to_csv(TABLES / "power_sample_requirements.csv", index=False)
    observed.to_csv(TABLES / "observed_effect_power.csv", index=False)
    ci.to_csv(TABLES / "cpa_confidence_intervals.csv", index=False)
    allocation.to_csv(TABLES / "budget_allocation.csv", index=False)

    plot_pvalue_heatmap(t_tests, sorted(df["channel"].unique()))
    plot_conversion_rates(summary)
    plot_correction_summary(corrections)
    write_executive_memo(summary, t_tests, fisher, corrections, power_requirements, ci, allocation, quality)

    result = {
        "rows": len(df),
        "channels": df["channel"].nunique(),
        "primary_metric": PRIMARY_METRIC,
        "total_comparisons": len(t_tests) + len(fisher),
        "expected_false_positives_at_0_05": (len(t_tests) + len(fisher)) * ALPHA,
        "cpa_uncorrected_significant": int(t_tests["significant_uncorrected"].sum()),
        "cpa_bonferroni_significant": int(t_tests["significant_bonferroni"].sum()),
        "cpa_fdr_significant": int(t_tests["significant_fdr"].sum()),
        "fisher_uncorrected_significant": int(fisher["significant_uncorrected"].sum()),
        "fisher_bonferroni_significant": int(fisher["significant_bonferroni"].sum()),
        "fisher_fdr_significant": int(fisher["significant_fdr"].sum()),
        "budget_total": int(allocation["allocation_dollars"].sum()),
        "best_aggregate_cpa_channel": str(summary.loc[summary["cpa"].idxmin(), "channel"]),
        "best_aggregate_roas_channel": str(summary.loc[summary["roas"].idxmax(), "channel"]),
        "best_aggregate_conversion_channel": str(summary.loc[summary["conversion_rate"].idxmax(), "channel"]),
        "fdr_cpa_pairs": t_tests.loc[t_tests["significant_fdr"], ["group_a", "group_b"]].to_dict("records"),
        "fdr_conversion_pairs": fisher.loc[fisher["significant_fdr"], ["group_a", "group_b"]].to_dict("records"),
    }
    (TABLES / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_statistical_analysis()
