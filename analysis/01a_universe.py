from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("ggplot")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
SAMPLE_SIZE_PATH = DATA_DIR / "sample_size_all_years.json"
from _analysis_shared import AnalysisOutputManager

OUTPUTS = AnalysisOutputManager(__file__)


def log_status(message: str) -> None:
    print(f"[01a_universe] {message}")


def load_sample_sizes():
    if not SAMPLE_SIZE_PATH.exists():
        log_status(
            "Skipping sample-size-dependent plots because "
            f"{SAMPLE_SIZE_PATH} is missing."
        )
        return pd.DataFrame()

    with SAMPLE_SIZE_PATH.open("r", encoding="utf-8") as file:
        sample_sizes = json.load(file)

    sample_size_df = pd.DataFrame(sample_sizes).T
    sample_size_df.index.name = "Formation year"
    return sample_size_df.apply(pd.to_numeric, errors="coerce").sort_index()


def load_stock_universe():
    universe_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/stock_universe.csv")
    )

    if not universe_files:
        log_status(
            "Skipping stock-universe-dependent plots because no "
            "data/yearly/<year>/stock_universe.csv files were found."
        )
        return pd.DataFrame()

    return pd.concat(
        (pd.read_csv(path) for path in universe_files),
        ignore_index=True,
    )


def load_earnings_events():
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_events.csv")
    )

    if not earnings_files:
        log_status(
            "Skipping earnings-event-dependent plots because no "
            "data/yearly/<year>/earnings_events.csv files were found."
        )
        return pd.DataFrame()

    earnings_events = pd.concat(
        (pd.read_csv(path) for path in earnings_files),
        ignore_index=True,
    )
    if "Ann_Date" in earnings_events.columns:
        earnings_events["Ann_Date"] = pd.to_datetime(earnings_events["Ann_Date"])

    return earnings_events


sample_size_df = load_sample_sizes()
stock_universe = load_stock_universe()
earnings_events = load_earnings_events()

SAMPLE_SIZE_STEP_ORDER = [
    "Raw Compustat sample",
    "Non-missing ISIN",
    "Successfully mapped to RIC",
    "Raw historical candidates",
    "Valid year-specific GBP candidates",
    "Ordinary/common shares",
    "Required accounting and market data available",
    "Positive market cap",
    "Market cap >= threshold",
    "Price >= threshold",
    "Positive book-to-market last fiscal year",
    "Rows before firm-level deduplication",
    "Unique firms (gvkey where available)",
    "Unique security identifiers",
    "All earnings announcements in event window",
    "Earnings events with valid actual EPS",
    "Earnings events with 90-day forecast median",
    "Earnings events with enough analyst forecasts",
    "Earnings events with complete stock return window",
]

# Set to a subset of step labels to restrict the plots; keep None to show all
# available sample-size steps.
SELECTED_SAMPLE_SIZE_STEPS = None


def filter_sample_size_steps(df, selected_steps):
    if selected_steps is None:
        return df

    available_steps = [step for step in selected_steps if step in df.columns]
    return df[available_steps]


def plot_sample_size_by_year(
    plot_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    legend_title: str = "Filtering step",
    color=None,
) -> None:
    fig_width = max(10, len(plot_df.index) * 0.9)
    ax = plot_df.plot(
        kind="bar",
        figsize=(7, 3.5),
        width=0.85,
        colormap="tab20" if color is None else None,
        color=color,
    )

    fig = ax.get_figure()

    ax.set_xlabel("Formation year")
    ax.set_ylabel("Sample size")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=(0.07, 0.08, 1, 1))
    OUTPUTS.save_figure(fig, output_name)
    plt.close(fig)


def plot_sample_size_line_by_year(
    plot_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    legend_title: str = "Filtering step",
    color=None,
    display_labels: dict[str, str] | None = None,
) -> None:
    fig_width = max(10, len(plot_df.index) * 0.35)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    x_values = pd.to_numeric(plot_df.index, errors="coerce")

    for i, column in enumerate(plot_df.columns):
        line_color = None if color is None else color[i]
        ax.plot(
            x_values,
            plot_df[column],
            linewidth=2,
            color=line_color,
            label=display_labels.get(column, column) if display_labels else column,
        )

    ax.set_xlabel("Formation year", labelpad=12)
    ax.set_ylabel("Sample size", labelpad=12)
    ax.set_xlim(x_values.min(), x_values.max())
    first_tick = int(np.ceil(x_values.min() / 5) * 5)
    last_tick = int(np.floor(x_values.max() / 5) * 5)
    ax.set_xticks(np.arange(first_tick, last_tick + 1, 5))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(axis="x", rotation=0)
    ax.margins(x=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=False,
    )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.47)
    OUTPUTS.save_figure(fig, output_name)
    plt.close(fig)

ordered_filters = [
    stage for stage in SAMPLE_SIZE_STEP_ORDER if stage in sample_size_df.columns
]
plot_df = filter_sample_size_steps(sample_size_df, ordered_filters)

if not plot_df.empty and len(plot_df.columns) > 0:
    # Not retained in thesis2/Figures.
    # plot_sample_size_by_year(
    #     plot_df,
    #     title="Sample size by Formation year and Filtering Step",
    #     output_name="sample_size_by_formation_year_and_filtering_step",
    # )

    selected_sample_size_steps = [
        "Raw Compustat sample",
        "Required accounting and market data available",
        "All earnings announcements in event window",
        "Earnings events with enough analyst forecasts",
    ]
    selected_plot_df = filter_sample_size_steps(sample_size_df, selected_sample_size_steps)

    if not selected_plot_df.empty and len(selected_plot_df.columns) > 0:
        # Not retained in thesis2/Figures.
        # plot_sample_size_by_year(
        #     selected_plot_df,
        #     title="Selected Sample size by Formation year and Filtering Step",
        #     output_name="selected_sample_size_by_formation_year_and_filtering_step",
        # )
        pass
    else:
        log_status(
            "Skipping selected sample size by formation year and filtering step "
            "because none of the requested step labels are available."
        )

    requested_sample_size_steps = [
        "Raw Compustat sample",
        "Ordinary shares",
        "Positive book-to-market last fiscal year",
        "Price >= threshold",
        "All earnings announcements in event window",
        "Earnings events with 90-day forecast median",
        "Earnings events with complete stock return window",
    ]
    requested_sample_size_display_labels = {
        "Raw Compustat sample": "Initial Compustat stock-year universe",
        "Ordinary shares": "Mapping to Datastream RIC & validation",
        "Positive book-to-market last fiscal year": "Required stock-level data available",
        "Price >= threshold": "Stock-year data cleaning filters",
        "All earnings announcements in event window": "All stock-year universe earnings announcements",
        "Earnings events with 90-day forecast median": "At least one valid analyst forecast available",
        "Earnings events with complete stock return window": "At least three valid analyst forecasts available",
    }
    requested_plot_df = filter_sample_size_steps(sample_size_df, requested_sample_size_steps)

    if not requested_plot_df.empty and len(requested_plot_df.columns) > 0:
        # Not retained in thesis2/Figures.
        # plot_sample_size_by_year(
        #     requested_plot_df,
        #     title="Requested Sample size by Formation year and Filtering Step",
        #     output_name="requested_sample_size_by_formation_year_and_filtering_step",
        # )

        plot_sample_size_line_by_year(
            requested_plot_df,
            title="Sample size by formation year and construction step",
            output_name="yearly_sample_size_plot",
            color=["#4c78a8", "#9ecae9", "#8e6bbd", "#2ca02c", "#f58518", "#eeca3b", "#e45756"],
            display_labels=requested_sample_size_display_labels,
        )
        log_status("Plotted requested sample size by formation year and filtering step as a line plot.")
    else:
        log_status(
            "Skipping requested sample size by formation year and filtering step "
            "because none of the requested step labels are available."
        )
else:
    if sample_size_df.empty:
        log_status(
            "Skipping sample size by formation year and filtering step because "
            "sample size data is unavailable."
        )
    else:
        available_columns = sample_size_df.columns.tolist()
        log_status(
            "Skipping sample size by formation year and filtering step because "
            "none of the current sample-size step labels are available. "
            f"Available columns: {available_columns}"
        )


aggregate_plot_df = filter_sample_size_steps(sample_size_df, ordered_filters)

if not aggregate_plot_df.empty and len(aggregate_plot_df.columns) > 0:
    aggregated_sample_size_plot_df = aggregate_plot_df.sum(axis=0).to_frame().T
    aggregated_sample_size_plot_df.index = ["All Formation years"]

    fig_width = max(12, len(aggregated_sample_size_plot_df.columns) * 0.6)
    ax = aggregated_sample_size_plot_df.plot(
        kind="bar",
        figsize=(7, 3.5),
        width=0.85,
        colormap="tab20",
    )

    fig = ax.get_figure()

    ax.set_ylabel("Sample size")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        title="Filtering step",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    # Not retained in thesis2/Figures.
    # OUTPUTS.save_figure(fig, "sample_size_by_filtering_step_all_years")
    plt.close(fig)
    log_status("Plotted sample size by filtering step for all years.")
else:
    if sample_size_df.empty:
        log_status(
            "Skipping aggregate sample size plot because sample "
            "size data is unavailable."
        )
    else:
        log_status(
            "Skipping aggregate sample size plot because no "
            "matching filtering-step columns are available."
        )
