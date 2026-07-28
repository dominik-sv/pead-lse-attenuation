from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
LOCAL_PACKAGE_DIRS = [
    PROJECT_ROOT / ".python_packages_local",
    PROJECT_ROOT / ".python_packages",
]
for package_dir in LOCAL_PACKAGE_DIRS:
    if package_dir.exists() and str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import PercentFormatter

from src.core.pipeline_config import COLOR_PALETTE
from src.core.pipeline_config import SUE_COMPUTATION_GROUP_COUNT, SUE_PLOT_GROUP_COUNT
from src.pead.sue_groups import SUE_GROUP_COLUMN, SUE_PLOT_GROUP_COLUMN, build_plot_group_labels
from src.analysis.time_varying_analysis import (
    FIRM_IDENTIFIER_COLUMN,
    FORMATION_YEAR_COLUMN,
    TIME_PERIOD_COLUMN,
    attach_universe_snapshot,
    assign_time_periods,
    build_period_axes,
    build_time_periods,
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
    load_stock_universe_snapshots,
    percentile_limits,
    prepare_bhar_path_events,
    summarize_plot_group_paths,
)
from _analysis_shared import AnalysisOutputManager


DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS = AnalysisOutputManager(__file__)
PERIOD_LENGTH_YEARS = 10
# Keep the descriptive time-period diagnostics aligned with the primary H2
# specification in analysis/11_regression_suite.py.  Automatic ten-year bins
# would otherwise produce 1990-1999, 2000-2009, 2010-2019, and 2020-2024.
EXPLICIT_PERIODS = [(1991, 2002), (2003, 2013), (2014, 2024)]


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
available_years = sorted(abnormal_returns[FORMATION_YEAR_COLUMN].dropna().astype(int).unique().tolist())
time_periods = build_time_periods(
    available_years,
    period_length=PERIOD_LENGTH_YEARS,
    explicit_periods=EXPLICIT_PERIODS,
)
period_labels = [str(period["label"]) for period in time_periods]
period_color = plt.get_cmap(COLOR_PALETTE)(0.65)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


event_level = collapse_to_event_level(abnormal_returns)
stock_universe = load_stock_universe_snapshots(DATA_DIR)
event_level = attach_universe_snapshot(event_level, stock_universe)
event_level = assign_time_periods(event_level, time_periods)
event_level = event_level.dropna(subset=[TIME_PERIOD_COLUMN]).copy()
event_level["Ann_Date"] = pd.to_datetime(event_level["Ann_Date"], errors="coerce")
event_level["Announcement_Month"] = event_level["Ann_Date"].dt.month.astype("Int64")
event_level["Forecast_Analyst_Count"] = pd.to_numeric(
    event_level["Forecast_Analyst_Count"], errors="coerce"
)
event_level["SUE"] = pd.to_numeric(event_level["SUE"], errors="coerce")
event_level["Market_Cap_Current"] = pd.to_numeric(
    event_level["Market_Cap_Current"], errors="coerce"
)
event_level["BM"] = pd.to_numeric(event_level["BM"], errors="coerce")
event_level["Log_Market_Cap_Current"] = np.where(
    event_level["Market_Cap_Current"] > 0,
    np.log10(event_level["Market_Cap_Current"]),
    np.nan,
)


period_summary = (
    event_level.groupby(TIME_PERIOD_COLUMN, observed=True)
    .agg(
        Formation_Years=(FORMATION_YEAR_COLUMN, lambda series: f"{int(series.min())}-{int(series.max())}"),
        Event_Count=("Event_ID", "nunique"),
        Firm_Count=(FIRM_IDENTIFIER_COLUMN, "nunique"),
        Median_Analyst_Count=("Forecast_Analyst_Count", "median"),
        Mean_Analyst_Count=("Forecast_Analyst_Count", "mean"),
        Median_SUE=("SUE", "median"),
        Mean_SUE=("SUE", "mean"),
        Median_Market_Cap_Current=("Market_Cap_Current", "median"),
        Median_BM=("BM", "median"),
    )
    .reset_index()
    .rename(columns={TIME_PERIOD_COLUMN: "Time_Period"})
)
display(period_summary)
OUTPUTS.save_table(period_summary, "period_summary")


scatter_frame = event_level.dropna(
    subset=[TIME_PERIOD_COLUMN, "Log_Market_Cap_Current", "BM"]
).copy()
x_limits = percentile_limits(scatter_frame["Log_Market_Cap_Current"], lower=0.01, upper=0.99)
y_limits = percentile_limits(scatter_frame["BM"], lower=0.01, upper=0.99)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.2, subplot_height=4.8)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = scatter_frame.loc[scatter_frame[TIME_PERIOD_COLUMN] == label].copy()
    full_count = len(subset)
    if full_count > 2000:
        subset = subset.sample(2000, random_state=0)

    axis.scatter(
        subset["Log_Market_Cap_Current"],
        subset["BM"],
        s=18,
        alpha=0.35,
        color=period_color,
        edgecolor="none",
    )
    axis.set_xlabel("log10(Market cap, USD mn)")
    axis.set_ylabel("Book-to-market")
    if np.isfinite(x_limits[0]) and np.isfinite(x_limits[1]):
        axis.set_xlim(x_limits)
    if np.isfinite(y_limits[0]) and np.isfinite(y_limits[1]):
        axis.set_ylim(y_limits)
    axis.grid(alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.93))
OUTPUTS.save_figure(fig, "formation_date_size_and_book_to_market_by_time_period")
plt.show()


fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.2, subplot_height=4.5)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = event_level.loc[
        event_level[TIME_PERIOD_COLUMN] == label,
        ["Event_ID", "Announcement_Month"],
    ].dropna()

    month_counts = (
        subset.groupby("Announcement_Month")["Event_ID"].nunique().reindex(range(1, 13), fill_value=0)
    )
    month_shares = month_counts / month_counts.sum() * 100 if month_counts.sum() else month_counts.astype(float)

    axis.bar(range(1, 13), month_shares, color=period_color, edgecolor="black", linewidth=0.6)
    axis.set_xticks(range(1, 13))
    axis.set_xticklabels(month_labels, rotation=45, ha="right")
    axis.set_ylabel("Share of events")
    axis.yaxis.set_major_formatter(PercentFormatter())
    axis.grid(axis="y", alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.93))
OUTPUTS.save_figure(fig, "announcement_month_distribution_by_time_period")
plt.show()


analyst_frame = event_level.dropna(
    subset=[TIME_PERIOD_COLUMN, "Forecast_Analyst_Count"]
).copy()
analyst_upper = int(np.ceil(analyst_frame["Forecast_Analyst_Count"].quantile(0.95)))
analyst_upper = max(analyst_upper, 1)
analyst_bins = np.arange(-0.5, analyst_upper + 1.5, 1.0)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.2, subplot_height=4.5)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = analyst_frame.loc[
        analyst_frame[TIME_PERIOD_COLUMN] == label, "Forecast_Analyst_Count"
    ].clip(upper=analyst_upper)

    axis.hist(
        subset,
        bins=analyst_bins,
        color=period_color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    axis.set_xlabel("Analysts following")
    axis.set_ylabel("Events")
    axis.grid(axis="y", alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.93))
OUTPUTS.save_figure(fig, "analyst_following_by_time_period")
plt.show()


sue_frame = event_level.dropna(subset=[TIME_PERIOD_COLUMN, "SUE"]).copy()
sue_limits = percentile_limits(sue_frame["SUE"], lower=0.01, upper=0.99)
sue_bins = np.linspace(sue_limits[0], sue_limits[1], 31)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.2, subplot_height=4.5)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = sue_frame.loc[sue_frame[TIME_PERIOD_COLUMN] == label, "SUE"]

    axis.hist(
        subset,
        bins=sue_bins,
        density=True,
        color=period_color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    axis.set_xlabel("SUE")
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.93))
OUTPUTS.save_figure(fig, "sue_distribution_by_time_period")
plt.show()


group_frame = event_level.dropna(subset=[TIME_PERIOD_COLUMN, SUE_GROUP_COLUMN]).copy()
group_frame[SUE_GROUP_COLUMN] = group_frame[SUE_GROUP_COLUMN].astype(int)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.2, subplot_height=4.5)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = group_frame.loc[group_frame[TIME_PERIOD_COLUMN] == label]
    group_counts = subset.groupby(SUE_GROUP_COLUMN)["Event_ID"].nunique().reindex(range(1, SUE_COMPUTATION_GROUP_COUNT + 1), fill_value=0)
    group_shares = group_counts / group_counts.sum() * 100 if group_counts.sum() else group_counts.astype(float)

    axis.bar(
        range(1, SUE_COMPUTATION_GROUP_COUNT + 1),
        group_shares,
        color=plt.get_cmap(COLOR_PALETTE)(np.linspace(0.08, 0.92, SUE_COMPUTATION_GROUP_COUNT)),
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xlabel("SUE group")
    axis.set_ylabel("Share of events")
    axis.set_xticks(range(1, SUE_COMPUTATION_GROUP_COUNT + 1))
    axis.yaxis.set_major_formatter(PercentFormatter())
    axis.grid(axis="y", alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.93))
OUTPUTS.save_figure(fig, "sue_group_distribution_by_time_period")
plt.show()


abnormal_returns = assign_time_periods(abnormal_returns, time_periods)
abnormal_returns = abnormal_returns.dropna(subset=[TIME_PERIOD_COLUMN]).copy()

post_path_events = prepare_bhar_path_events(abnormal_returns, day_start=0, day_end=60)
pre_path_events = prepare_bhar_path_events(abnormal_returns, day_start=-60, day_end=-1)

post_path_summary = summarize_plot_group_paths(post_path_events)
pre_path_summary = summarize_plot_group_paths(pre_path_events)

plot_group_colors = plt.get_cmap(COLOR_PALETTE)(
    np.linspace(0.08, 0.92, SUE_PLOT_GROUP_COUNT)
)
plot_group_labels = build_plot_group_labels(
    computation_group_count=SUE_COMPUTATION_GROUP_COUNT,
    plot_group_count=SUE_PLOT_GROUP_COUNT,
)


event_support = (
    post_path_events.groupby([TIME_PERIOD_COLUMN, SUE_PLOT_GROUP_COLUMN], observed=True)["Event_ID"]
    .nunique()
    .unstack(SUE_PLOT_GROUP_COLUMN)
    .reindex(index=period_labels, columns=range(1, SUE_PLOT_GROUP_COUNT + 1))
    .fillna(0)
    .astype(int)
)
event_support.columns = [
    f"Plot_Group_{plot_group}" for plot_group in event_support.columns
]
display(event_support)
OUTPUTS.save_table(event_support, "event_support")


post_plot_df = post_path_summary.copy()
post_baseline = pd.DataFrame(
    [
        {
            TIME_PERIOD_COLUMN: label,
            SUE_PLOT_GROUP_COLUMN: plot_group,
            "Relative_Day": -1,
            "Cumulative_BHAR": 0.0,
        }
        for label in period_labels
        for plot_group in range(1, SUE_PLOT_GROUP_COUNT + 1)
    ]
)
post_plot_df = pd.concat([post_baseline, post_plot_df], ignore_index=True)
post_limits = percentile_limits(post_plot_df["Cumulative_BHAR"], lower=0.01, upper=0.99)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.4, subplot_height=4.8)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = post_plot_df.loc[post_plot_df[TIME_PERIOD_COLUMN] == label]

    for color, plot_group in zip(
        plot_group_colors, range(1, SUE_PLOT_GROUP_COUNT + 1)
    ):
        line_df = subset.loc[
            subset[SUE_PLOT_GROUP_COLUMN] == plot_group
        ].sort_values("Relative_Day")
        axis.plot(
            line_df["Relative_Day"],
            line_df["Cumulative_BHAR"],
            color=color,
            linewidth=2,
            label=plot_group_labels[plot_group],
        )

    axis.axhline(0, color="black", linewidth=0.9, alpha=0.8)
    axis.axvline(0, color="black", linewidth=0.9, alpha=0.8)
    axis.set_xlabel("Relative trading day")
    axis.set_ylabel("Cumulative BHAR")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    if np.isfinite(post_limits[0]) and np.isfinite(post_limits[1]):
        axis.set_ylim(post_limits)
    axis.grid(alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
OUTPUTS.save_figure(fig, "pead_paths_by_sue_quintile_and_time_period")
plt.show()


pre_plot_df = pre_path_summary.copy()
pre_baseline = pd.DataFrame(
    [
        {
            TIME_PERIOD_COLUMN: label,
            SUE_PLOT_GROUP_COLUMN: plot_group,
            "Relative_Day": -61,
            "Cumulative_BHAR": 0.0,
        }
        for label in period_labels
        for plot_group in range(1, SUE_PLOT_GROUP_COUNT + 1)
    ]
)
pre_plot_df = pd.concat([pre_baseline, pre_plot_df], ignore_index=True)
pre_limits = percentile_limits(pre_plot_df["Cumulative_BHAR"], lower=0.01, upper=0.99)

fig, axes = build_period_axes(len(time_periods), ncols=2, subplot_width=7.4, subplot_height=4.8)
for axis, period in zip(axes, time_periods):
    label = str(period["label"])
    subset = pre_plot_df.loc[pre_plot_df[TIME_PERIOD_COLUMN] == label]

    for color, plot_group in zip(
        plot_group_colors, range(1, SUE_PLOT_GROUP_COUNT + 1)
    ):
        line_df = subset.loc[
            subset[SUE_PLOT_GROUP_COLUMN] == plot_group
        ].sort_values("Relative_Day")
        axis.plot(
            line_df["Relative_Day"],
            line_df["Cumulative_BHAR"],
            color=color,
            linewidth=2,
            label=plot_group_labels[plot_group],
        )

    axis.axhline(0, color="black", linewidth=0.9, alpha=0.8)
    axis.set_xlabel("Relative trading day")
    axis.set_ylabel("Cumulative BHAR")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    if np.isfinite(pre_limits[0]) and np.isfinite(pre_limits[1]):
        axis.set_ylim(pre_limits)
    axis.grid(alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
OUTPUTS.save_figure(fig, "pre_ead_paths_by_sue_quintile_and_time_period")
plt.close(fig)
