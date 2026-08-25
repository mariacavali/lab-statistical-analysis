"""Clean the Kaggle Campaign Effectiveness data and create exploration outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
RAW_DATA = ROOT / "data" / "raw" / "campaign_effectiveness.xlsx"
CLEAN_DATA = ROOT / "marketing_data.csv"
TABLES = ROOT / "tables"

DATASET_NAME = "Campaign Effectiveness"
DATASET_URL = "https://www.kaggle.com/datasets/sanak2000/campaign-effectiveness"

COLUMN_MAP = {
    "Campaign_ID": "campaign_id",
    "Campaign_Manager": "campaign_manager",
    "Marketing_Channel_Type": "channel",
    "Creative_Format": "creative_format",
    "Impressions": "impressions",
    "Clicks": "clicks",
    "Conversions": "conversions",
    "Campaign_Cost": "cost",
    "Revenue": "revenue",
    "Retention_Rate": "retention_rate",
    "Avg_Order_Value": "average_order_value",
    "Purchase_Frequency_per_Year": "purchase_frequency_per_year",
    "Weather_Condition_Launch_Day": "launch_weather",
    "Customer_Lifespan_Years": "customer_lifespan_years",
    "Campaign_Color_Theme": "campaign_color_theme",
    "Discount_Rate": "discount_rate",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while returning NaN for zero denominators and non-finite results."""
    result = numerator.astype(float).div(denominator.astype(float).replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def load_and_clean(raw_path: Path = RAW_DATA) -> tuple[pd.DataFrame, dict]:
    """Load, validate, standardize, and enrich the source workbook."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing source workbook: {raw_path}")
    raw = pd.read_excel(raw_path, sheet_name="Sheet1")
    missing_columns = set(COLUMN_MAP).difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Source is missing columns: {sorted(missing_columns)}")

    df = raw.rename(columns=COLUMN_MAP).copy()
    numeric_columns = [
        "impressions",
        "clicks",
        "conversions",
        "cost",
        "revenue",
        "retention_rate",
        "average_order_value",
        "purchase_frequency_per_year",
        "customer_lifespan_years",
        "discount_rate",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rows_before = len(df)
    duplicate_rows = int(df.duplicated(subset="campaign_id").sum())
    df = df.drop_duplicates(subset="campaign_id")
    required = ["campaign_id", "channel", "impressions", "clicks", "conversions", "cost", "revenue"]
    missing_required_rows = int(df[required].isna().any(axis=1).sum())
    df = df.dropna(subset=required)

    valid = (
        (df["impressions"] > 0)
        & (df["clicks"] >= 0)
        & (df["clicks"] <= df["impressions"])
        & (df["conversions"] >= 0)
        & (df["conversions"] <= df["clicks"])
        & (df["cost"] > 0)
        & (df["revenue"] >= 0)
    )
    invalid_metric_rows = int((~valid).sum())
    df = df.loc[valid].copy()

    df["ctr"] = safe_divide(df["clicks"], df["impressions"])
    df["conversion_rate"] = safe_divide(df["conversions"], df["clicks"])
    df["cpa"] = safe_divide(df["cost"], df["conversions"])
    df["roas"] = safe_divide(df["revenue"], df["cost"])
    df["profit"] = df["revenue"] - df["cost"]
    df["profit_margin"] = safe_divide(df["profit"], df["revenue"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("campaign_id").reset_index(drop=True)

    quality = {
        "dataset_name": DATASET_NAME,
        "source": "Kaggle",
        "url": DATASET_URL,
        "source_is_simulated": True,
        "raw_shape": list(raw.shape),
        "clean_shape": list(df.shape),
        "raw_columns": raw.columns.tolist(),
        "clean_columns": df.columns.tolist(),
        "raw_dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "raw_missing_values": {column: int(value) for column, value in raw.isna().sum().items()},
        "duplicate_campaign_ids_removed": duplicate_rows,
        "rows_missing_required_values_removed": missing_required_rows,
        "invalid_funnel_or_financial_rows_removed": invalid_metric_rows,
        "rows_removed_total": rows_before - len(df),
        "groups": sorted(df["channel"].unique().tolist()),
        "sample_sizes": {str(k): int(v) for k, v in df["channel"].value_counts().sort_index().items()},
        "date_range": None,
        "date_note": "The source has no date field; each row is an independent campaign execution.",
    }
    return df, quality


def group_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate funnel and financial performance by marketing channel."""
    grouped = (
        df.groupby("channel", as_index=False)
        .agg(
            campaigns=("campaign_id", "count"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            cost=("cost", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    grouped["ctr"] = safe_divide(grouped["clicks"], grouped["impressions"])
    grouped["conversion_rate"] = safe_divide(grouped["conversions"], grouped["clicks"])
    grouped["cpa"] = safe_divide(grouped["cost"], grouped["conversions"])
    grouped["roas"] = safe_divide(grouped["revenue"], grouped["cost"])
    grouped["profit"] = grouped["revenue"] - grouped["cost"]
    grouped["profit_margin"] = safe_divide(grouped["profit"], grouped["revenue"])
    return grouped.sort_values("channel").reset_index(drop=True)


def create_overview_plot(summary: pd.DataFrame) -> None:
    metrics = [
        ("cpa", "CPA ($)", False),
        ("roas", "ROAS", True),
        ("conversion_rate", "Conversion rate", True),
        ("conversions", "Conversions", True),
        ("cost", "Cost ($)", False),
        ("profit", "Profit ($)", True),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for ax, (metric, label, ascending) in zip(axes.flat, metrics):
        plot_data = summary.sort_values(metric, ascending=ascending)
        sns.barplot(data=plot_data, y="channel", x=metric, ax=ax, color="#3b82f6")
        ax.set_title(f"{label} by channel")
        ax.set_ylabel("")
        ax.set_xlabel(label)
        if metric in {"conversion_rate"}:
            ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
        if metric == "profit":
            ax.axvline(0, color="black", linewidth=0.8)
    fig.suptitle("Campaign channel performance overview", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(ROOT / "group_metrics_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_distribution_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    sns.boxplot(data=df, x="channel", y="cpa", ax=axes[0], color="#93c5fd", showfliers=False)
    axes[0].set_title("Campaign-level CPA distribution by channel (outlier markers hidden)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    sns.boxplot(data=df, x="channel", y="roas", ax=axes[1], color="#86efac", showfliers=False)
    axes[1].set_title("Campaign-level ROAS distribution by channel (outlier markers hidden)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(ROOT / "group_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_exploration() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    TABLES.mkdir(exist_ok=True)
    df, quality = load_and_clean()
    summary = group_performance(df)
    df.to_csv(CLEAN_DATA, index=False)
    summary.to_csv(TABLES / "group_performance_summary.csv", index=False)
    (TABLES / "data_quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    create_overview_plot(summary)
    create_distribution_plot(df)
    print(f"Loaded {quality['raw_shape'][0]:,} rows and retained {len(df):,} across {df['channel'].nunique()} channels.")
    print("Sample sizes by channel:")
    print(df["channel"].value_counts().sort_index().to_string())
    print("\nGroup performance summary:")
    print(summary.round(4).to_string(index=False))
    return df, summary, quality


if __name__ == "__main__":
    run_exploration()
