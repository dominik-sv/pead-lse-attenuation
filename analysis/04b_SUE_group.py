import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from _analysis_shared import AnalysisOutputManager
from src.core.pipeline_config import COLOR_PALETTE, SUE_COMPUTATION_GROUP_COUNT
from src.pead.sue_groups import (
    SUE_GROUP_COLUMN,
)
from src.analysis.time_varying_analysis import (
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
)

from _analysis_shared import DATA_DIR

OUTPUTS = AnalysisOutputManager(__file__)


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(abnormal_returns)

available_years = sorted(event_level["Formation_Year"].dropna().astype(int).unique().tolist())

plot_events = event_level.dropna(subset=[SUE_GROUP_COLUMN]).copy()
plot_events[SUE_GROUP_COLUMN] = plot_events[SUE_GROUP_COLUMN].astype(int)

group_counts = (
    plot_events.groupby(SUE_GROUP_COLUMN)
    .size()
    .rename("Event_Count")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7, 4.2))

group_bins = np.arange(0.5, SUE_COMPUTATION_GROUP_COUNT + 1.5, 1)
group_colors = plt.get_cmap(COLOR_PALETTE)(np.linspace(0.08, 0.92, SUE_COMPUTATION_GROUP_COUNT))

counts, bins, patches = ax.hist(
    plot_events[SUE_GROUP_COLUMN],
    bins=group_bins,  # type: ignore
    edgecolor="black",
    linewidth=0.8,
)

for patch, color in zip(patches, group_colors):  # type: ignore
    patch.set_facecolor(color)

ax.set_xlabel("SUE group")
ax.set_ylabel("Events")
ax.set_xticks(range(1, SUE_COMPUTATION_GROUP_COUNT + 1))
ax.grid(axis="y", alpha=0.25)

for count, left_edge in zip(counts, bins[:-1]):
    if count > 0:
        ax.text(
            left_edge + 0.5,
            count,  # type: ignore
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.tight_layout()
OUTPUTS.save_figure(fig, "frequency_of_sue_groups")
plt.close(fig)

distribution_events = event_level.dropna(subset=[SUE_GROUP_COLUMN, "SUE"]).copy()
distribution_events[SUE_GROUP_COLUMN] = distribution_events[SUE_GROUP_COLUMN].astype(int)

group_values = [
    distribution_events.loc[
        distribution_events[SUE_GROUP_COLUMN] == group,
        "SUE",
    ].to_numpy()
    for group in range(1, SUE_COMPUTATION_GROUP_COUNT + 1)
]

fig, ax = plt.subplots(figsize=(7, 4.2))

box = ax.boxplot(
    group_values,
    positions=range(1, SUE_COMPUTATION_GROUP_COUNT + 1),
    widths=0.65,
    patch_artist=True,
    medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    whiskerprops={"color": "#4d4d4d", "linewidth": 1.0},
    capprops={"color": "#4d4d4d", "linewidth": 1.0},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.2, "markeredgewidth": 0},
)

for patch, color in zip(box["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_xlabel("SUE group")
ax.set_ylabel("SUE")
ax.set_ylim(-0.15, 0.15)
ax.set_xticks(range(1, SUE_COMPUTATION_GROUP_COUNT + 1))
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
OUTPUTS.save_figure(fig, "sue_distribution_by_assigned_group")
plt.show()

distribution_events = event_level.dropna(subset=[SUE_GROUP_COLUMN, "SUE"]).copy()
distribution_events[SUE_GROUP_COLUMN] = distribution_events[SUE_GROUP_COLUMN].astype(int)

group_values = [
    distribution_events.loc[
        distribution_events[SUE_GROUP_COLUMN] == group,
        "SUE",
    ].to_numpy()
    for group in range(1, SUE_COMPUTATION_GROUP_COUNT + 1)
]

fig, ax = plt.subplots(figsize=(7, 4.2))

box = ax.boxplot(
    group_values,
    positions=range(1, SUE_COMPUTATION_GROUP_COUNT + 1),
    widths=0.65,
    patch_artist=True,
    medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    whiskerprops={"color": "#4d4d4d", "linewidth": 1.0},
    capprops={"color": "#4d4d4d", "linewidth": 1.0},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.2, "markeredgewidth": 0},
)

for patch, color in zip(box["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_xlabel("SUE group")
ax.set_ylabel("SUE")
ax.set_ylim(-0.02, 0.02)
ax.set_xticks(range(1, SUE_COMPUTATION_GROUP_COUNT + 1))
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
OUTPUTS.save_figure(fig, "sue_distribution_by_assigned_group_zoomed")
plt.show()

relationship_events = event_level.dropna(
    subset=[SUE_GROUP_COLUMN, "Forecast_Analyst_Count"]
).copy()
relationship_events[SUE_GROUP_COLUMN] = relationship_events[SUE_GROUP_COLUMN].astype(int)
relationship_events["Forecast_Analyst_Count"] = pd.to_numeric(
    relationship_events["Forecast_Analyst_Count"],
    errors="coerce",
)
relationship_events = relationship_events.dropna(subset=["Forecast_Analyst_Count"])
relationship_events["Forecast_Analyst_Count"] = relationship_events[
    "Forecast_Analyst_Count"
].astype(int)
sue_group_values = list(range(1, SUE_COMPUTATION_GROUP_COUNT + 1))
analyst_following_values = np.arange(
    relationship_events["Forecast_Analyst_Count"].min(),
    relationship_events["Forecast_Analyst_Count"].max() + 1,
)

cumulative_count_matrix = pd.DataFrame(index=analyst_following_values)

fig, ax = plt.subplots(figsize=(7, 4.2))

for group, color in zip(sue_group_values, group_colors):
    group_followings = relationship_events.loc[
        relationship_events[SUE_GROUP_COLUMN] == group,
        "Forecast_Analyst_Count",
    ].to_numpy()
    counts_by_following = (
        pd.Series(group_followings)
        .value_counts()
        .reindex(analyst_following_values, fill_value=0)
        .sort_index()
    )
    cumulative_counts = counts_by_following.cumsum()
    cumulative_count_matrix[group] = cumulative_counts.values

    ax.plot(
        analyst_following_values,
        cumulative_counts,
        color=color,
        linewidth=2,
        label=f"SUE {group}",
    )

ax.set_xlabel("Analysts following")
ax.set_ylabel("Cumulative events")
ax.grid(alpha=0.25)
ax.legend(ncol=2, frameon=False)

fig.tight_layout()
OUTPUTS.save_figure(fig, "cumulative_analyst_following_lines_by_sue_group")
plt.show()

relative_cumulative_count_matrix = cumulative_count_matrix.div(
    cumulative_count_matrix.iloc[-1],
    axis=1,
).fillna(0.0)

fig, ax = plt.subplots(figsize=(7, 4.2))

for group, color in zip(sue_group_values, group_colors):
    ax.plot(
        analyst_following_values,
        relative_cumulative_count_matrix[group],
        color=color,
        linewidth=2,
        label=f"SUE {group}",
    )

ax.set_xlabel("Analysts following")
ax.set_ylabel("Share of events within SUE group")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(ncol=2, frameon=False)

fig.tight_layout()
OUTPUTS.save_figure(fig, "relative_cumulative_analyst_following_lines_by_sue_group")
plt.close(fig)
