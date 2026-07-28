from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_DIRS = [
    PROJECT_ROOT / ".python_packages_local",
    PROJECT_ROOT / ".python_packages",
]
for package_dir in LOCAL_PACKAGE_DIRS:
    if package_dir.exists() and str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bhar_outlier_policy import select_outlier_candidates
from src.analysis.time_varying_analysis import (
    attach_universe_snapshot,
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
    load_stock_universe_snapshots,
)
from analysis._analysis_shared import AnalysisOutputManager
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR


DATA_DIR = PROJECT_DATA_DIR
OUTPUTS = AnalysisOutputManager(__file__)
DEFAULT_COLUMN = "BHAR_2_20"
DEFAULT_ABSOLUTE_THRESHOLD_PCT = 50.0
DEFAULT_COLUMNS_TO_SHOW = [
    "Formation_Year",
    "Instrument",
    "Name",
    "Ann_Date",
    "Report_Frequency",
    "Forecast_Analyst_Count",
    "SUE",
    "SUE_Group",
    "BHAR_0_1",
    "BHAR_2_20",
    "BHAR_2_30",
    "BHAR_2_60",
    "BHAR_2_90",
    "BHAR_0_60",
    "Price_Lag_5",
    "Price",
    "Market_Cap_Current",
    "BM_French",
    "BM",
    "Benchmark_Portfolio",
    "Announcement_Trading_Date",
    "Window_End_Trading_Date",
    "Event_ID",
    "Selection_Method",
    "Selection_Value",
    "Selection_Tail",
    "Selection_Threshold",
    "Replacement_Value",
    "Selected_Rank",
    "Stock_Return_Inspection_Window",
    "Benchmark_Return_Inspection_Window",
    "Abnormal_Return_Inspection_Window",
    "Return_Driver_Label",
]


def parse_bhar_window(column_name: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"BHAR_(-?\d+)_(-?\d+)", str(column_name).strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _compound_percent_return(values: pd.Series) -> float | pd.NA:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return pd.NA
    return float(((1.0 + (numeric_values / 100.0)).prod() - 1.0) * 100.0)


def build_inspection_window_return_frame(
    abnormal_returns: pd.DataFrame,
    inspection_column: str,
) -> pd.DataFrame:
    window = parse_bhar_window(inspection_column)
    if window is None:
        return pd.DataFrame(columns=["Event_ID"])

    start_day, end_day = window
    required_columns = {"Event_ID", "Relative_Day", "Security_Return", "Benchmark_Return"}
    missing_columns = required_columns.difference(abnormal_returns.columns)
    if missing_columns:
        raise KeyError(
            "Cannot build inspection-window return columns because abnormal returns are "
            f"missing: {sorted(missing_columns)}."
        )

    relative_days = pd.to_numeric(abnormal_returns["Relative_Day"], errors="coerce")
    window_rows = abnormal_returns.loc[
        relative_days.ge(start_day) & relative_days.le(end_day)
    ].copy()
    if window_rows.empty:
        return pd.DataFrame(columns=["Event_ID"])

    inspection_returns = (
        window_rows.groupby("Event_ID", as_index=False)
        .agg(
            Stock_Return_Inspection_Window=(
                "Security_Return",
                _compound_percent_return,
            ),
            Benchmark_Return_Inspection_Window=(
                "Benchmark_Return",
                _compound_percent_return,
            ),
        )
    )
    return inspection_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate BHAR outliers using the raw event-level sample. "
            "By default, use an absolute-return threshold aligned with the K-style "
            "return diagnostics. Optional top-N or percentile selectors remain available."
        )
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"BHAR column to inspect (default: {DEFAULT_COLUMN}).",
    )
    parser.add_argument(
        "--absolute-threshold-pct",
        type=float,
        default=DEFAULT_ABSOLUTE_THRESHOLD_PCT,
        help=(
            "Keep only observations with absolute outlier-column value at or above this "
            f"percentage threshold (default: {DEFAULT_ABSOLUTE_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional rank-based selector after thresholding.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Select events beyond this percentile, expressed as a decimal (for example 0.99).",
    )
    parser.add_argument(
        "--tail",
        choices=("upper", "lower"),
        default="upper",
        help="Which tail of the distribution to inspect.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom stem for the saved CSV file.",
    )
    return parser.parse_args()


def build_output_stem(
    *,
    column: str,
    absolute_threshold_pct: float,
    top_n: int | None,
    percentile: float | None,
    tail: str,
    output_name: str | None,
) -> str:
    if output_name:
        return output_name
    threshold_stem = f"abs_{str(float(absolute_threshold_pct)).replace('.', 'p')}pct"
    if top_n is not None:
        selector = f"top_{top_n}"
    elif percentile is not None:
        selector = f"pct_{int(round(float(percentile) * 10000))}"
    else:
        selector = "threshold_only"
    return f"{column}_{tail}_{threshold_stem}_{selector}"


def select_threshold_candidates(
    event_level: pd.DataFrame,
    *,
    column: str,
    absolute_threshold_pct: float,
    tail: str,
    top_n: int | None,
    percentile: float | None,
) -> pd.DataFrame:
    working = event_level.copy()
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.loc[working[column].notna()].copy()
    if working.empty:
        return working

    threshold_value = float(abs(absolute_threshold_pct))
    working = working.loc[working[column].abs() >= threshold_value].copy()
    if working.empty:
        return working

    working["Selection_Method"] = "absolute_threshold"
    working["Selection_Value"] = threshold_value
    working["Selection_Threshold"] = threshold_value
    working["Replacement_Value"] = pd.NA
    working["Selection_Tail"] = tail

    if percentile is not None or top_n is not None:
        selected = select_outlier_candidates(
            working,
            column=column,
            top_n=top_n,
            percentile=percentile,
            tail=tail,
        )
        selected["Base_Absolute_Threshold_Pct"] = threshold_value
        return selected

    ascending = tail == "lower"
    selected = working.sort_values([column, "Event_ID"], ascending=[ascending, True]).reset_index(drop=True)
    selected["Selected_Rank"] = range(1, len(selected) + 1)
    return selected


def _format_plain_number(value: float, _position: int) -> str:
    if pd.isna(value):
        return ""
    if value == 0:
        return "0"
    magnitude = abs(float(value))
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_year_tick(value: float, _position: int) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value)))}"


def build_flagged_summary(candidates: pd.DataFrame, *, inspection_column: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "Inspection_Column",
                "Flagged_Event_Count",
                "Formation_Years_Covered",
                "Unique_Instruments",
                "Max_Selection_Value",
                "Median_Selection_Value",
                "Min_Selection_Value",
            ]
        )

    selection_values = pd.to_numeric(candidates[inspection_column], errors="coerce")
    return pd.DataFrame(
        [
            {
                "Inspection_Column": inspection_column,
                "Flagged_Event_Count": int(len(candidates)),
                "Formation_Years_Covered": int(candidates["Formation_Year"].nunique())
                if "Formation_Year" in candidates.columns
                else pd.NA,
                "Unique_Instruments": int(candidates["Instrument"].nunique())
                if "Instrument" in candidates.columns
                else pd.NA,
                "Max_Selection_Value": float(selection_values.max()) if selection_values.notna().any() else pd.NA,
                "Median_Selection_Value": float(selection_values.median()) if selection_values.notna().any() else pd.NA,
                "Min_Selection_Value": float(selection_values.min()) if selection_values.notna().any() else pd.NA,
            }
        ]
    )


def build_driver_decomposition(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "Formation_Year",
                "Instrument",
                "Name",
                "Ann_Date",
                "Event_ID",
                "Stock_Return_Inspection_Window",
                "Benchmark_Return_Inspection_Window",
                "Abnormal_Return_Inspection_Window",
                "Absolute_Abnormal_Return_Inspection_Window",
                "Return_Driver_Label",
                "Selection_Method",
                "Selection_Tail",
                "Selected_Rank",
            ]
        )

    working = candidates.copy()
    working["Stock_Return_Inspection_Window"] = pd.to_numeric(
        working.get("Stock_Return_Inspection_Window"), errors="coerce"
    )
    working["Benchmark_Return_Inspection_Window"] = pd.to_numeric(
        working.get("Benchmark_Return_Inspection_Window"), errors="coerce"
    )
    working["Abnormal_Return_Inspection_Window"] = (
        working["Stock_Return_Inspection_Window"] - working["Benchmark_Return_Inspection_Window"]
    )
    working["Absolute_Abnormal_Return_Inspection_Window"] = (
        working["Abnormal_Return_Inspection_Window"].abs()
    )

    stock_abs = working["Stock_Return_Inspection_Window"].abs()
    bench_abs = working["Benchmark_Return_Inspection_Window"].abs()
    working["Return_Driver_Label"] = "mixed_or_unclear"
    working.loc[
        stock_abs.notna() & bench_abs.notna() & (stock_abs >= bench_abs * 1.5),
        "Return_Driver_Label",
    ] = "stock_return_dominates"
    working.loc[
        stock_abs.notna() & bench_abs.notna() & (bench_abs >= stock_abs * 1.5),
        "Return_Driver_Label",
    ] = "benchmark_return_dominates"
    working.loc[
        stock_abs.notna() & bench_abs.notna() & (stock_abs < bench_abs * 1.5) & (bench_abs < stock_abs * 1.5),
        "Return_Driver_Label",
    ] = "both_move_materially"

    preferred_columns = [
        "Formation_Year",
        "Instrument",
        "Name",
        "Ann_Date",
        "Event_ID",
        "Stock_Return_Inspection_Window",
        "Benchmark_Return_Inspection_Window",
        "Abnormal_Return_Inspection_Window",
        "Absolute_Abnormal_Return_Inspection_Window",
        "Return_Driver_Label",
        "Selection_Method",
        "Selection_Tail",
        "Selected_Rank",
    ]
    available_columns = [column for column in preferred_columns if column in working.columns]
    return working[available_columns].sort_values(
        ["Absolute_Abnormal_Return_Inspection_Window", "Selected_Rank"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def plot_bhar_vs_stock_returns_by_portfolio_year(
    candidates: pd.DataFrame,
    *,
    inspection_column: str,
    output_path: Path,
) -> Path | None:
    required_columns = {
        inspection_column,
        "Stock_Return_Inspection_Window",
        "Formation_Year",
        "Benchmark_Portfolio",
    }
    if candidates.empty or not required_columns.issubset(candidates.columns):
        return None

    plot_frame = candidates.copy()
    plot_frame[inspection_column] = pd.to_numeric(plot_frame[inspection_column], errors="coerce")
    plot_frame["Stock_Return_Inspection_Window"] = pd.to_numeric(
        plot_frame["Stock_Return_Inspection_Window"], errors="coerce"
    )
    plot_frame["Formation_Year"] = pd.to_numeric(plot_frame["Formation_Year"], errors="coerce")
    plot_frame["Benchmark_Portfolio"] = (
        plot_frame["Benchmark_Portfolio"].astype("string").str.strip()
    )
    plot_frame = plot_frame.loc[
        plot_frame[inspection_column].notna()
        & plot_frame[inspection_column].gt(0)
        & plot_frame["Stock_Return_Inspection_Window"].notna()
        & plot_frame["Formation_Year"].notna()
        & plot_frame["Benchmark_Portfolio"].notna()
        & plot_frame["Benchmark_Portfolio"].ne("")
    ].copy()
    if plot_frame.empty:
        return None

    plot_frame["Portfolio_Year_Label"] = (
        plot_frame["Formation_Year"].astype(int).astype(str)
        + " | "
        + plot_frame["Benchmark_Portfolio"].astype(str)
    )
    labels = sorted(plot_frame["Portfolio_Year_Label"].unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    color_map = {label: cmap(index) for index, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label in labels:
        subset = plot_frame.loc[plot_frame["Portfolio_Year_Label"] == label]
        ax.scatter(
            subset["Stock_Return_Inspection_Window"],
            subset[inspection_column],
            s=48,
            alpha=0.8,
            color=color_map[label],
            edgecolors="none",
            label=label,
        )

    combined = pd.concat(
        [
            plot_frame["Stock_Return_Inspection_Window"],
            plot_frame[inspection_column],
        ],
        ignore_index=True,
    ).dropna()
    if not combined.empty:
        max_abs = float(combined.abs().max())
        if max_abs > 0:
            diagonal = pd.Series([-max_abs, max_abs], dtype="float64")
            ax.plot(
                diagonal,
                diagonal,
                linestyle="--",
                linewidth=1.2,
                color="black",
                alpha=0.8,
                label="BHAR = stock return (benchmark return = 0)",
            )

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["Stock_Return_Inspection_Window"].dropna().astype(float)
        y_values = plot_frame[inspection_column].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Stock return over inspection window (%)")
    ax.set_ylabel(f"{inspection_column} (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    if not combined.empty:
        x_values = plot_frame["Stock_Return_Inspection_Window"].dropna().astype(float)
        y_values = plot_frame[inspection_column].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_outlier_returns_vs_sue_group(
    candidates: pd.DataFrame,
    *,
    inspection_column: str,
    output_path: Path,
) -> Path | None:
    if candidates.empty or inspection_column not in candidates.columns or "SUE_Group" not in candidates.columns:
        return None

    plot_frame = candidates.copy()
    plot_frame[inspection_column] = pd.to_numeric(plot_frame[inspection_column], errors="coerce")
    plot_frame["SUE_Group"] = pd.to_numeric(plot_frame["SUE_Group"], errors="coerce")
    plot_frame = plot_frame.loc[
        plot_frame[inspection_column].notna()
        & plot_frame[inspection_column].gt(0)
        & plot_frame["SUE_Group"].notna()
    ].copy()
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(
        plot_frame["SUE_Group"],
        plot_frame[inspection_column],
        s=48,
        alpha=0.8,
        color="#1f77b4",
        edgecolors="none",
    )
    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["SUE_Group"].dropna().astype(float)
        y_values = plot_frame[inspection_column].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("SUE quintile")
    ax.set_ylabel(f"{inspection_column} (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_outlier_returns_vs_year(
    candidates: pd.DataFrame,
    *,
    inspection_column: str,
    output_path: Path,
) -> Path | None:
    required_columns = {inspection_column, "Formation_Year", "SUE_Group"}
    if candidates.empty or not required_columns.issubset(candidates.columns):
        return None

    plot_frame = candidates.copy()
    plot_frame[inspection_column] = pd.to_numeric(plot_frame[inspection_column], errors="coerce")
    plot_frame["Formation_Year"] = pd.to_numeric(plot_frame["Formation_Year"], errors="coerce")
    plot_frame["SUE_Group"] = pd.to_numeric(plot_frame["SUE_Group"], errors="coerce")
    plot_frame = plot_frame.loc[
        plot_frame[inspection_column].notna()
        & plot_frame[inspection_column].gt(0)
        & plot_frame["Formation_Year"].notna()
        & plot_frame["SUE_Group"].notna()
    ].copy()
    if plot_frame.empty:
        return None

    sue_groups = sorted(plot_frame["SUE_Group"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("viridis", max(len(sue_groups), 1))
    color_map = {group: cmap(index) for index, group in enumerate(sue_groups)}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for sue_group in sue_groups:
        subset = plot_frame.loc[plot_frame["SUE_Group"].astype(int) == sue_group]
        ax.scatter(
            subset["Formation_Year"],
            subset[inspection_column],
            s=48,
            alpha=0.8,
            color=color_map[sue_group],
            edgecolors="none",
            label=f"Q{sue_group}",
        )

    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["Formation_Year"].dropna().astype(float)
        y_values = plot_frame[inspection_column].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_year_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Formation year")
    ax.set_ylabel(f"{inspection_column} (%)")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(title="SUE quintile", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    abnormal_returns = load_abnormal_returns_with_groups(
        DATA_DIR,
        apply_saved_outlier_policy=False,
    )
    event_level = collapse_to_event_level(abnormal_returns)
    stock_universe = load_stock_universe_snapshots(DATA_DIR)
    event_level = attach_universe_snapshot(event_level, stock_universe)
    inspection_window_returns = build_inspection_window_return_frame(
        abnormal_returns,
        args.column,
    )
    if not inspection_window_returns.empty:
        event_level = event_level.merge(
            inspection_window_returns,
            on="Event_ID",
            how="left",
            validate="1:1",
        )

    if "Name" in stock_universe.columns:
        stock_names = (
            stock_universe[["Formation_Year", "Instrument", "Name"]]
            .dropna(subset=["Formation_Year", "Instrument"])
            .drop_duplicates(["Formation_Year", "Instrument"])
            .copy()
        )
        event_level = event_level.merge(
            stock_names,
            on=["Formation_Year", "Instrument"],
            how="left",
            validate="m:1",
        )

    candidates = select_threshold_candidates(
        event_level,
        column=args.column,
        absolute_threshold_pct=args.absolute_threshold_pct,
        top_n=args.top_n,
        percentile=args.percentile,
        tail=args.tail,
    )

    driver_decomposition = build_driver_decomposition(candidates)
    if not driver_decomposition.empty:
        candidates = candidates.merge(
            driver_decomposition[
                [
                    column
                    for column in [
                        "Event_ID",
                        "Abnormal_Return_Inspection_Window",
                        "Return_Driver_Label",
                    ]
                    if column in driver_decomposition.columns
                ]
            ],
            on="Event_ID",
            how="left",
            validate="1:1",
        )
    flagged_summary = build_flagged_summary(candidates, inspection_column=args.column)

    available_columns = [column for column in DEFAULT_COLUMNS_TO_SHOW if column in candidates.columns]
    preview = candidates[available_columns].copy()
    if "Ann_Date" in preview.columns:
        preview["Ann_Date"] = pd.to_datetime(preview["Ann_Date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    output_stem = build_output_stem(
        column=args.column,
        absolute_threshold_pct=args.absolute_threshold_pct,
        top_n=args.top_n,
        percentile=args.percentile,
        tail=args.tail,
        output_name=args.output_name,
    )
    output_dir = OUTPUTS.get_output_dir()
    output_path = output_dir / f"{output_stem}.csv"
    summary_path = output_dir / f"{output_stem}_summary.csv"
    driver_path = output_dir / f"{output_stem}_drivers.csv"
    sue_group_plot_path = output_dir / f"{output_stem}_vs_sue_quintile.png"
    year_plot_path = output_dir / f"{output_stem}_vs_year_by_sue_quintile.png"
    stock_bhar_plot_path = output_dir / f"{output_stem}_bhar_vs_stock_returns_by_portfolio_year.png"

    flagged_summary.to_csv(summary_path, index=False)
    preview.to_csv(output_path, index=False)
    driver_decomposition.to_csv(driver_path, index=False)
    saved_sue_group_plot = plot_outlier_returns_vs_sue_group(
        candidates,
        inspection_column=args.column,
        output_path=sue_group_plot_path,
    )
    saved_year_plot = plot_outlier_returns_vs_year(
        candidates,
        inspection_column=args.column,
        output_path=year_plot_path,
    )
    saved_stock_bhar_plot = plot_bhar_vs_stock_returns_by_portfolio_year(
        candidates,
        inspection_column=args.column,
        output_path=stock_bhar_plot_path,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"Saved flagged summary to: {summary_path}")
    print(f"Saved candidate outlier inspection table to: {output_path}")
    print(f"Saved driver decomposition table to: {driver_path}")
    if saved_sue_group_plot is not None:
        print(f"Saved SUE quintile scatter plot to: {saved_sue_group_plot}")
    if saved_year_plot is not None:
        print(f"Saved year scatter plot to: {saved_year_plot}")
    if saved_stock_bhar_plot is not None:
        print(f"Saved stock-vs-BHAR scatter plot to: {saved_stock_bhar_plot}")
if __name__ == "__main__":
    main()


