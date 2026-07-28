from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from src.analysis.time_varying_analysis import FORMATION_YEAR_COLUMN, load_stock_universe_snapshots
from _analysis_shared import AnalysisOutputManager
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.core.year_context import build_year_context
from src.pead.french_benchmarks import STANDARD_PORTFOLIO_LABELS


DATA_DIR = PROJECT_DATA_DIR
OUTPUTS = AnalysisOutputManager(__file__)

PORTFOLIO_ORDER = list(STANDARD_PORTFOLIO_LABELS)
SIZE_LABELS = {"S": "Small", "B": "Big"}
BM_LABELS = {"G": "Growth", "N": "Neutral", "V": "Value"}
SIZE_Q_TO_GROUP = {1: "S", 2: "B"}
BM_Q_TO_GROUP = {1: "G", 2: "N", 3: "V"}
PORTFOLIO_DISPLAY_LABELS = {
    "SG": "Small - Low BM",
    "SN": "Small - Medium BM",
    "SV": "Small - High BM",
    "BG": "Big - Low BM",
    "BN": "Big - Medium BM",
    "BV": "Big - High BM",
}

ASSIGNMENT_COLUMNS = ["Benchmark_Size_Q", "Benchmark_BM_Q", "Benchmark_Portfolio"]
CHARACTERISTIC_COLUMNS = ["Market_Cap_Current", "BM_French"]

def format_benchmark_portfolio_axis(ax):
    positions = range(len(PORTFOLIO_ORDER))

    # First x-axis level: BM bucket within each size group
    bm_labels = [BM_LABELS[portfolio[1]] for portfolio in PORTFOLIO_ORDER]
    ax.set_xticks(positions)
    ax.set_xticklabels(bm_labels, rotation=45, ha="right")

    # Vertical separator between Small and Big portfolios
    ax.axvline(2.5, color="black", linewidth=0.8, alpha=0.35)

    # Second x-axis level: size group labels
    for center, group_code in ((1, "S"), (4, "B")):
        ax.text(
            center,
            -0.18,
            SIZE_LABELS[group_code],
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
        )

    # Overall x-axis title below the second-level labels
    ax.set_xlabel("Book-to-market group within size group")
    ax.xaxis.set_label_coords(0.5, -0.27)

stock_universe = load_stock_universe_snapshots(DATA_DIR)

for column in [
    FORMATION_YEAR_COLUMN,
    "Benchmark_Size_Q",
    "Benchmark_BM_Q",
    *CHARACTERISTIC_COLUMNS,
]:
    if column in stock_universe.columns:
        stock_universe[column] = pd.to_numeric(stock_universe[column], errors="coerce")

missing_required_columns = [
    column
    for column in [FORMATION_YEAR_COLUMN, *ASSIGNMENT_COLUMNS, *CHARACTERISTIC_COLUMNS]
    if column not in stock_universe.columns
]

if missing_required_columns:
    raise KeyError(
        "stock_universe.csv snapshots are missing required French benchmark assignment columns: "
        f"{sorted(missing_required_columns)}. Run scripts/01_build_french_benchmarks.py "
        "and scripts/02_build_universe_and_market_data.py first."
    )

stock_universe["Benchmark_Portfolio"] = stock_universe["Benchmark_Portfolio"].astype("string")
reconstructed_portfolios = (
    stock_universe["Benchmark_Size_Q"].map(SIZE_Q_TO_GROUP).astype("string")
    + stock_universe["Benchmark_BM_Q"].map(BM_Q_TO_GROUP).astype("string")
)
needs_reconstruction = (
    stock_universe["Benchmark_Portfolio"].isna()
    & stock_universe["Benchmark_Size_Q"].notna()
    & stock_universe["Benchmark_BM_Q"].notna()
)
stock_universe.loc[needs_reconstruction, "Benchmark_Portfolio"] = reconstructed_portfolios.loc[
    needs_reconstruction
]
stock_universe["Benchmark_Portfolio"] = pd.Categorical(
    stock_universe["Benchmark_Portfolio"],
    categories=PORTFOLIO_ORDER,
    ordered=True,
)


missing_assignment_audit = (
    stock_universe.assign(
        Missing_Benchmark_Size_Q=stock_universe["Benchmark_Size_Q"].isna(),
        Missing_Benchmark_BM_Q=stock_universe["Benchmark_BM_Q"].isna(),
        Missing_Benchmark_Portfolio=stock_universe["Benchmark_Portfolio"].isna(),
        Missing_Any_Assignment=stock_universe[ASSIGNMENT_COLUMNS].isna().any(axis=1),
    )
    .groupby(FORMATION_YEAR_COLUMN, observed=True)
    .agg(
        Universe_Firm_Count=("Instrument", "nunique"),
        Missing_Benchmark_Size_Q=("Missing_Benchmark_Size_Q", "sum"),
        Missing_Benchmark_BM_Q=("Missing_Benchmark_BM_Q", "sum"),
        Missing_Benchmark_Portfolio=("Missing_Benchmark_Portfolio", "sum"),
        Missing_Any_Assignment=("Missing_Any_Assignment", "sum"),
    )
    .reset_index()
    .rename(columns={FORMATION_YEAR_COLUMN: "Formation_Year"})
)

display(missing_assignment_audit)
OUTPUTS.save_table(missing_assignment_audit, "missing_assignment_audit")


yearly_portfolio_counts = (
    stock_universe.dropna(subset=["Benchmark_Portfolio"])
    .groupby([FORMATION_YEAR_COLUMN, "Benchmark_Portfolio"], observed=True)["Instrument"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=PORTFOLIO_ORDER, fill_value=0)
    .sort_index()
)

# -------------------------------------------------------------------
# Plot 1: grouped yearly counts by French benchmark portfolio
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 3.5))

x = pd.Series(range(len(PORTFOLIO_ORDER)), index=PORTFOLIO_ORDER)
formation_years = yearly_portfolio_counts.index.tolist()
if not formation_years:
    raise ValueError(
        "No benchmark portfolio assignments were available for plotting. "
        "Benchmark_Size_Q and Benchmark_BM_Q are present, but Benchmark_Portfolio "
        "labels did not resolve to the current 2x3 French benchmark portfolios "
        f"{PORTFOLIO_ORDER}."
    )

bar_width = 0.8 / len(formation_years)
offsets = [
    (i - (len(formation_years) - 1) / 2) * bar_width
    for i in range(len(formation_years))
]

for offset, formation_year in zip(offsets, formation_years):
    values = yearly_portfolio_counts.loc[formation_year].reindex(PORTFOLIO_ORDER, fill_value=0)

    ax.bar(
        x.values + offset,
        values.values,
        width=bar_width,
        label=str(int(formation_year)),
        alpha=0.85,
    )

ax.set_ylabel("Assigned firms")
format_benchmark_portfolio_axis(ax)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(title="Formation year", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=8)

plt.subplots_adjust(bottom=0.22)
OUTPUTS.save_figure(fig, "yearly_counts_by_french_benchmark_portfolio")
plt.show()

annual_portfolio_stats = (
    stock_universe.dropna(subset=["Benchmark_Portfolio"])
    .groupby([FORMATION_YEAR_COLUMN, "Benchmark_Portfolio"], observed=True)["Instrument"]
    .nunique()
    .groupby("Benchmark_Portfolio", observed=True)
    .agg(["mean", "min", "max"])
    .reindex(PORTFOLIO_ORDER)
    .fillna(0.0)
)

# -------------------------------------------------------------------
# Plot 2: average annual counts by French benchmark portfolio
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 3.5))

ax.bar(
    x,
    annual_portfolio_stats["mean"].values,
    color=plt.get_cmap("plasma")(0.45),
    width=0.8,
)

ax.errorbar(
    x.values,
    annual_portfolio_stats["mean"].values,
    yerr=np.vstack(
        [
            annual_portfolio_stats["mean"].values - annual_portfolio_stats["min"].values,
            annual_portfolio_stats["max"].values - annual_portfolio_stats["mean"].values,
        ]
    ),
    fmt="none",
    ecolor="black",
    elinewidth=1.1,
    capsize=3,
    alpha=0.85,
)

ax.set_ylabel("Average Assigned firms per Year")
format_benchmark_portfolio_axis(ax)
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
OUTPUTS.save_figure(fig, "average_annual_french_benchmark_portfolio_counts")
plt.show()


def build_portfolio_color_map() -> dict[str, tuple[float, float, float, float]]:
    # Matches the benchmark-portfolio constituent-count figure in 01b.
    return {
        "SG": mcolors.to_rgba("#007C91"),
        "SN": mcolors.to_rgba("#2A9D8F"),
        "SV": mcolors.to_rgba("#8FC9C3"),
        "BG": mcolors.to_rgba("#6F5C1E"),
        "BN": mcolors.to_rgba("#C28E00"),
        "BV": mcolors.to_rgba("#E9D8A6"),
    }


benchmark_scatter_rows = []
for formation_year in sorted(stock_universe[FORMATION_YEAR_COLUMN].dropna().astype(int).unique()):
    year_context = build_year_context(int(formation_year), DATA_DIR)

    if not year_context.benchmark_returns_path.exists():
        continue
    if not year_context.benchmark_constituents_path.exists():
        continue

    benchmark_returns = pd.read_csv(
        year_context.benchmark_returns_path,
        parse_dates=["Date"],
    )
    benchmark_constituents = pd.read_csv(year_context.benchmark_constituents_path)

    for portfolio in PORTFOLIO_ORDER:
        if portfolio not in benchmark_returns.columns:
            continue

        daily_returns = pd.to_numeric(benchmark_returns[portfolio], errors="coerce").dropna()
        cumulative_return = (
            ((1.0 + daily_returns / 100.0).prod() - 1.0) * 100.0
            if not daily_returns.empty
            else np.nan
        )
        firm_count = int(
            benchmark_constituents.loc[
                benchmark_constituents["Benchmark_Portfolio"] == portfolio,
                "Instrument",
            ].nunique()
        )
        size_bucket = 1 if portfolio.startswith("S") else 2
        bm_bucket = {"G": 1, "N": 2, "V": 3}[portfolio[1]]

        benchmark_scatter_rows.append(
            {
                "Formation_Year": int(formation_year),
                "Benchmark_Portfolio": portfolio,
                "Size_Q": size_bucket,
                "BM_Q": bm_bucket,
                "Firm_Count": firm_count,
                "Cumulative_Return": cumulative_return,
            }
        )

benchmark_scatter_df = pd.DataFrame(benchmark_scatter_rows)
display(benchmark_scatter_df.head())
OUTPUTS.save_table(benchmark_scatter_df, "benchmark_cumulative_return_scatter_inputs")

portfolio_results_table = (
    benchmark_scatter_df.groupby("Benchmark_Portfolio", observed=True)
    .agg(
        Minimum_Firm_Count=("Firm_Count", "min"),
        Mean_Firm_Count=("Firm_Count", "mean"),
        Median_Firm_Count=("Firm_Count", "median"),
        Maximum_Firm_Count=("Firm_Count", "max"),
        Mean_Cumulative_Return=("Cumulative_Return", "mean"),
        Median_Cumulative_Return=("Cumulative_Return", "median"),
        Cumulative_Return_SD=("Cumulative_Return", "std"),
        Minimum_Cumulative_Return=("Cumulative_Return", "min"),
        Maximum_Cumulative_Return=("Cumulative_Return", "max"),
    )
    .reindex(PORTFOLIO_ORDER)
    .reset_index()
)

# -------------------------------------------------------------------
# Plot 3: annual median B/M and market capitalization by French benchmark portfolio
# -------------------------------------------------------------------

portfolio_characteristics = (
    stock_universe.dropna(subset=["Benchmark_Portfolio", *CHARACTERISTIC_COLUMNS])
    .loc[lambda df: df["Market_Cap_Current"] > 0]
    .groupby([FORMATION_YEAR_COLUMN, "Benchmark_Portfolio"], observed=True)
    .agg(
        Median_BM=("BM_French", "median"),
        Median_Market_Cap=("Market_Cap_Current", "median"),
    )
    .reindex(
        pd.MultiIndex.from_product(
            [formation_years, PORTFOLIO_ORDER],
            names=[FORMATION_YEAR_COLUMN, "Benchmark_Portfolio"],
        )
    )
    .reset_index()
    .sort_values([FORMATION_YEAR_COLUMN, "Benchmark_Portfolio"])
    .reset_index(drop=True)
)

if portfolio_characteristics[["Median_BM", "Median_Market_Cap"]].isna().any().any():
    missing_portfolios = portfolio_characteristics.loc[
        portfolio_characteristics[["Median_BM", "Median_Market_Cap"]].isna().any(axis=1),
        "Benchmark_Portfolio",
    ].tolist()
    raise ValueError(
        "Annual median B/M and market-cap characteristics could not be calculated for "
        f"the following benchmark portfolios: {missing_portfolios}."
    )

OUTPUTS.save_table(portfolio_characteristics, "benchmark_portfolio_median_characteristics")

portfolio_median_of_yearly_medians = (
    portfolio_characteristics.groupby("Benchmark_Portfolio", observed=True)
    .agg(
        Median_Annual_Median_BM=("Median_BM", "median"),
        Median_Annual_Median_Market_Cap=("Median_Market_Cap", "median"),
    )
    .reindex(PORTFOLIO_ORDER)
    .reset_index()
)
OUTPUTS.save_table(
    portfolio_median_of_yearly_medians,
    "benchmark_portfolio_median_of_annual_medians",
)
display(portfolio_results_table)
OUTPUTS.save_table(portfolio_results_table, "benchmark_portfolio_results_summary")

portfolio_colors = build_portfolio_color_map()

fig, ax = plt.subplots(figsize=(7, 4.2))

for portfolio in PORTFOLIO_ORDER:
    portfolio_rows = portfolio_characteristics.loc[
        portfolio_characteristics["Benchmark_Portfolio"] == portfolio
    ]
    ax.scatter(
        portfolio_rows["Median_BM"],
        portfolio_rows["Median_Market_Cap"],
        s=42,
        color=portfolio_colors[portfolio],
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
        label=PORTFOLIO_DISPLAY_LABELS[portfolio],
    )
    portfolio_median = portfolio_median_of_yearly_medians.loc[
        portfolio_median_of_yearly_medians["Benchmark_Portfolio"] == portfolio
    ].iloc[0]
    ax.scatter(
        portfolio_median["Median_Annual_Median_BM"],
        portfolio_median["Median_Annual_Median_Market_Cap"],
        s=105,
        marker="D",
        color=portfolio_colors[portfolio],
        edgecolor="#1b1b1b",
        linewidth=0.9,
        zorder=3,
    )

ax.set_xlabel("Median book-to-market ratio")
ax.set_ylabel("Median market cap ($ millions)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.grid(alpha=0.25)
portfolio_legend = ax.legend(
    title="Benchmark portfolio",
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
    fontsize=8,
)
ax.add_artist(portfolio_legend)

plt.tight_layout()
OUTPUTS.save_figure(fig, "median_market_cap_vs_bm_by_french_benchmark_portfolio")
plt.close(fig)

point_sizes = 20 + benchmark_scatter_df["Firm_Count"].fillna(0) * 3.5
point_colors = benchmark_scatter_df["Benchmark_Portfolio"].map(portfolio_colors)

fig, ax = plt.subplots(figsize=(7, 4.2))

ax.scatter(
    benchmark_scatter_df["Formation_Year"],
    benchmark_scatter_df["Cumulative_Return"],
    s=point_sizes,
    c=point_colors,
    alpha=0.82,
    edgecolor="black",
    linewidth=0.4,
)

ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Formation year")
ax.set_ylabel("Cumulative benchmark return over benchmark period")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.set_xticks(sorted(benchmark_scatter_df["Formation_Year"].dropna().unique()))
ax.grid(alpha=0.25)

legend_handles = []
for portfolio in PORTFOLIO_ORDER:
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=portfolio_colors[portfolio],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
            label=portfolio,
        )
    )

size_legend_counts = [10, 50, 100, 150]
size_handles = [
    ax.scatter(
        [],
        [],
        s=20 + count * 3.5,
        color="#808080",
        alpha=0.5,
        edgecolor="black",
        linewidth=0.4,
        label=f"{count} firms",
    )
    for count in size_legend_counts
]

portfolio_legend = ax.legend(
    handles=legend_handles,
    title="Portfolio color",
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
    fontsize=8,
)
ax.add_artist(portfolio_legend)

ax.legend(
    handles=size_handles,
    title="Point size",
    bbox_to_anchor=(1.02, 0.38),
    loc="upper left",
    fontsize=8,
)

plt.tight_layout()
OUTPUTS.save_figure(fig, "cumulative_benchmark_returns_scatter_by_year_and_portfolio")
plt.close(fig)


benchmark_path_rows = []
for formation_year in sorted(stock_universe[FORMATION_YEAR_COLUMN].dropna().astype(int).unique()):
    year_context = build_year_context(int(formation_year), DATA_DIR)

    if not year_context.benchmark_returns_path.exists():
        continue

    benchmark_returns = pd.read_csv(
        year_context.benchmark_returns_path,
        parse_dates=["Date"],
    ).sort_values("Date").reset_index(drop=True)

    for portfolio in PORTFOLIO_ORDER:
        if portfolio not in benchmark_returns.columns:
            continue

        daily_returns = pd.to_numeric(benchmark_returns[portfolio], errors="coerce")
        valid_returns = daily_returns.dropna()
        if valid_returns.empty:
            continue

        valid_index = valid_returns.index
        cumulative_path = ((1.0 + valid_returns / 100.0).cumprod() - 1.0) * 100.0

        path_df = pd.DataFrame(
            {
                "Formation_Year": int(formation_year),
                "Benchmark_Portfolio": portfolio,
                "Relative_Period": np.arange(len(cumulative_path)),
                "Date": benchmark_returns.loc[valid_index, "Date"].to_numpy(),
                "Cumulative_Return": cumulative_path.to_numpy(),
            }
        )
        benchmark_path_rows.append(path_df)

if not benchmark_path_rows:
    raise ValueError(
        "No benchmark return paths were available. Check the yearly benchmark return files "
        "and confirm they contain the expected 2x3 portfolio columns."
    )

benchmark_paths_df = pd.concat(benchmark_path_rows, ignore_index=True)
OUTPUTS.save_table(benchmark_paths_df, "benchmark_cumulative_return_paths")

fig, ax = plt.subplots(figsize=(7, 3.5))

for portfolio in PORTFOLIO_ORDER:
    portfolio_df = benchmark_paths_df.loc[
        benchmark_paths_df["Benchmark_Portfolio"] == portfolio
    ].copy()
    if portfolio_df.empty:
        continue

    for formation_year, year_df in portfolio_df.groupby("Formation_Year", sort=True):
        ax.plot(
            year_df["Relative_Period"],
            year_df["Cumulative_Return"],
            color=portfolio_colors[portfolio],
            alpha=0.35,
            linewidth=1.0,
        )

ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Relative trading periods since benchmark start")
ax.set_ylabel("Cumulative benchmark return")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.grid(alpha=0.25)

portfolio_legend = ax.legend(
    handles=legend_handles,
    title="Portfolio color",
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
    fontsize=8,
)
ax.add_artist(portfolio_legend)

plt.tight_layout()
OUTPUTS.save_figure(fig, "cumulative_benchmark_return_progression_by_relative_time")
plt.close(fig)
