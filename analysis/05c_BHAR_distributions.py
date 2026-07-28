"""Plot full-sample BHAR distributions for the PEAD duration-test intervals."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from _analysis_shared import AnalysisOutputManager, DATA_DIR
from src.analysis.time_varying_analysis import (
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
)


plt.style.use("ggplot")

OUTPUTS = AnalysisOutputManager(__file__)
INTERVAL_BHAR_COLUMNS = ("BHAR_2_20", "BHAR_21_40", "BHAR_41_60")
CUMULATIVE_BHAR_COLUMNS = ("BHAR_2_20", "BHAR_2_40", "BHAR_2_60")
DESCRIPTIVE_BHAR_COLUMNS = tuple(
    dict.fromkeys(INTERVAL_BHAR_COLUMNS + CUMULATIVE_BHAR_COLUMNS)
)
BHAR_COLOR = "#9B5DE5"


def bhar_label(column: str) -> str:
    """Convert a BHAR column name to the notation used in figure labels."""
    _, start_day, end_day = column.split("_")
    return f"Buy-and-hold abnormal return (BHAR), days {start_day}--{end_day} (%)"


def bhar_distribution_statistics(values: pd.Series, *, window: str) -> dict[str, float | int | str]:
    """Return the complete set of statistics required by the BHAR thesis table."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        raise ValueError(f"No non-missing observations are available for {window}.")

    return {
        "Window": window,
        "Events": clean_values.size,
        "Mean": clean_values.mean(),
        "SD": clean_values.std(ddof=1),
        "Min": clean_values.min(),
        "Q1": clean_values.quantile(0.25),
        "Median": clean_values.median(),
        "Q3": clean_values.quantile(0.75),
        "Max": clean_values.max(),
    }


def format_distribution_table_rows(table: pd.DataFrame) -> str:
    """Format descriptive statistics as thesis-ready LaTeX table rows."""
    numeric_columns = ("Mean", "SD", "Min", "Q1", "Median", "Q3", "Max")
    rows = []
    for _, row in table.iterrows():
        formatted_values = " & ".join(
            f"{float(row[column]):.2f}" for column in numeric_columns
        )
        rows.append(f"${row['Window']}$ & {formatted_values} \\\\")
    return "\n".join(rows) + "\n"


def format_distribution_narrative(table: pd.DataFrame) -> str:
    """Format a numerical, thesis-ready interpretation of the interval distributions."""
    first, middle, last = (table.iloc[index] for index in range(3))
    return (
        f"Mean (median) \\ac{{BHAR}} is ${first['Mean']:.2f}$ (${first['Median']:.2f}$) "
        f"percentage points in $[{first['Window'].split('[')[1].rstrip(']')}]$, "
        f"${middle['Mean']:.2f}$ (${middle['Median']:.2f}$) percentage points in "
        f"$[{middle['Window'].split('[')[1].rstrip(']')}]$, and "
        f"${last['Mean']:.2f}$ (${last['Median']:.2f}$) percentage points in "
        f"$[{last['Window'].split('[')[1].rstrip(']')}]$. The corresponding standard "
        f"deviations are ${first['SD']:.2f}$, ${middle['SD']:.2f}$, and ${last['SD']:.2f}$ "
        "percentage points. Thus, the interval returns are dispersed around modest central values, "
        "with the difference between means and medians and the tail observations indicating "
        "non-normality. These descriptive statistics do not themselves test for \\ac{PEAD}, "
        "because they do not condition on the earnings-surprise rank; that relation is evaluated "
        "in the interval regressions in \\Cref{sec:pead-evidence-results}.\n"
    )


def plot_bhar_distribution(values: pd.Series, column: str) -> None:
    """Save a BHAR histogram over the 1st--99th percentile range."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        raise ValueError(f"No non-missing observations are available for {column}.")

    lower_bound, upper_bound = clean_values.quantile([0.01, 0.99])
    if lower_bound >= upper_bound:
        raise ValueError(
            f"Cannot set 1st--99th percentile limits for {column}: "
            "the two percentiles are identical."
        )

    bin_edges = np.linspace(float(lower_bound), float(upper_bound), 41)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(
        clean_values,
        bins=bin_edges,
        color=BHAR_COLOR,
        edgecolor="black",
        alpha=0.85,
    )
    x_label = (
        "Buy-and-hold abnormal return from day 2 to day 20"
        if column == "BHAR_2_20"
        else bhar_label(column)
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Earnings announcements")
    ax.set_xlim(float(lower_bound), float(upper_bound))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.95,
        "Horizontal axis truncated at\n1st and 99th percentiles",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    OUTPUTS.save_figure(fig, f"{column.lower()}_distribution_p1_p99")


def plot_bhar_2_20_qq(values: pd.Series) -> None:
    """Save a normal QQ plot for the full-sample BHAR(2,20) distribution."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean_values) < 2:
        raise ValueError("At least two BHAR(2,20) observations are required for a QQ plot.")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    stats.probplot(clean_values, dist="norm", plot=ax)
    qq_points, fitted_line = ax.get_lines()
    qq_points.set_color(BHAR_COLOR)
    qq_points.set_markerfacecolor(BHAR_COLOR)
    qq_points.set_markeredgecolor(BHAR_COLOR)
    qq_points.set_markersize(3)
    fitted_line.set_color("#FF3B30")
    fitted_line.set_linestyle("--")
    ax.set_title("")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("BHAR from day 2 to day 20")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(alpha=0.25)

    fig.tight_layout()
    OUTPUTS.save_figure(fig, "bhar_2_20_normal_qq_plot")


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(
    abnormal_returns,
    additional_bhar_columns=DESCRIPTIVE_BHAR_COLUMNS,
)

missing_columns = [
    column for column in DESCRIPTIVE_BHAR_COLUMNS if column not in event_level.columns
]
if missing_columns:
    raise KeyError(
        "The event-level data do not contain the required BHAR columns: "
        f"{missing_columns}."
    )

# Consecutive intervals used in the sequential duration-selection rule.
interval_bhar_distribution_table = pd.DataFrame(
    [
        bhar_distribution_statistics(
            event_level[column],
            window=f"BHAR[{column.split('_')[1]},{column.split('_')[2]}]",
        )
        for column in INTERVAL_BHAR_COLUMNS
    ]
)
OUTPUTS.save_table(interval_bhar_distribution_table, "bhar_distribution_statistics")
OUTPUTS.save_latex(
    "bhar_distribution_statistics_rows",
    format_distribution_table_rows(interval_bhar_distribution_table),
)
OUTPUTS.save_latex(
    "bhar_distribution_narrative",
    format_distribution_narrative(interval_bhar_distribution_table),
)

# Cumulative horizons provide the economic magnitude of PEAD through each
# candidate endpoint. They are descriptive and do not enter the sequential
# continuation rule.
cumulative_bhar_distribution_table = pd.DataFrame(
    [
        bhar_distribution_statistics(
            event_level[column],
            window=f"BHAR[{column.split('_')[1]},{column.split('_')[2]}]",
        )
        for column in CUMULATIVE_BHAR_COLUMNS
    ]
)
OUTPUTS.save_table(
    cumulative_bhar_distribution_table,
    "cumulative_bhar_distribution_statistics",
)
OUTPUTS.save_latex(
    "cumulative_bhar_distribution_statistics_rows",
    format_distribution_table_rows(cumulative_bhar_distribution_table),
)

plot_bhar_distribution(event_level["BHAR_2_20"], "BHAR_2_20")
# Not retained in thesis2/Figures.
# plot_bhar_distribution(event_level["BHAR_21_40"], "BHAR_21_40")
# plot_bhar_distribution(event_level["BHAR_41_60"], "BHAR_41_60")

plot_bhar_2_20_qq(event_level["BHAR_2_20"])
