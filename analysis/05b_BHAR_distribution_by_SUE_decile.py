import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from _analysis_shared import AnalysisOutputManager, DATA_DIR
from src.analysis.time_varying_analysis import collapse_to_event_level, load_abnormal_returns_with_groups
from src.core.pipeline_config import COLOR_PALETTE, SUE_COMPUTATION_GROUP_COUNT
from src.pead.sue_groups import SUE_GROUP_COLUMN

OUTPUTS = AnalysisOutputManager(__file__)

BHAR_COLUMN = "BHAR_2_60"
GROUP_COUNT = SUE_COMPUTATION_GROUP_COUNT


def _group_label(group_count: int) -> str:
    if group_count == 10:
        return "decile"
    if group_count == 5:
        return "quintile"
    if group_count == 4:
        return "quartile"
    if group_count == 2:
        return "group"
    return "group"


GROUP_LABEL = _group_label(GROUP_COUNT)


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(abnormal_returns)

plot_events = event_level.dropna(subset=[SUE_GROUP_COLUMN, BHAR_COLUMN]).copy()
plot_events[SUE_GROUP_COLUMN] = pd.to_numeric(
    plot_events[SUE_GROUP_COLUMN],
    errors="coerce",
)
plot_events[BHAR_COLUMN] = pd.to_numeric(plot_events[BHAR_COLUMN], errors="coerce")
plot_events = plot_events.dropna(subset=[SUE_GROUP_COLUMN, BHAR_COLUMN]).copy()
plot_events[SUE_GROUP_COLUMN] = plot_events[SUE_GROUP_COLUMN].astype(int)

plot_events = plot_events.loc[
    plot_events[SUE_GROUP_COLUMN].between(1, GROUP_COUNT)
].copy()

group_values = [
    plot_events.loc[plot_events[SUE_GROUP_COLUMN] == group, BHAR_COLUMN].to_numpy()
    for group in range(1, GROUP_COUNT + 1)
]

non_empty_group_values = [values for values in group_values if len(values) > 0]
if not non_empty_group_values:
    raise ValueError("No BHAR observations were available after filtering for SUE groups.")

all_bhar_values = np.concatenate(non_empty_group_values)

x_min = float(np.nanmin(all_bhar_values))
x_max = float(np.nanmax(all_bhar_values))
x_padding = 0.05 * max(x_max - x_min, 1.0)
x_limits = (x_min - x_padding, x_max + x_padding)
hist_x_min = float(np.nanpercentile(all_bhar_values, 2.5))
hist_x_max = float(np.nanpercentile(all_bhar_values, 97.5))

bin_count = 40
hist_bin_edges = np.linspace(hist_x_min, hist_x_max, bin_count + 1)
hist_counts = [
    np.histogram(values, bins=hist_bin_edges)[0]
    for values in group_values
]
max_hist_count = max(
    int(counts.max()) for counts in hist_counts if len(counts) > 0
)

group_colors = plt.get_cmap(COLOR_PALETTE)(np.linspace(0.08, 0.92, GROUP_COUNT))

summary_table = (
    plot_events.groupby(SUE_GROUP_COLUMN)[BHAR_COLUMN]
    .agg(
        Event_Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max",
        Q25=lambda values: values.quantile(0.25),
        Q75=lambda values: values.quantile(0.75),
    )
    .reindex(range(1, GROUP_COUNT + 1))
)
OUTPUTS.save_table(summary_table, "bhar_distribution_summary_by_sue_decile")


histogram_columns = min(5, GROUP_COUNT)
histogram_rows = int(np.ceil(GROUP_COUNT / histogram_columns))
fig, axes = plt.subplots(
    histogram_rows,
    histogram_columns,
    figsize=(7, 3.5),
    sharex=True,
    sharey=True,
)
axes = np.atleast_1d(axes).flatten()
histogram_order = list(range(1, GROUP_COUNT + 1))

for group, ax in zip(histogram_order, axes):
    values = group_values[group - 1]
    color = group_colors[group - 1]
    ax.hist(
        values,
        bins=hist_bin_edges,
        color=color,
        edgecolor="white",
        alpha=0.9,
    )
    if len(values) > 0:
        ax.axvline(values.mean(), color="#1b1b1b", linestyle="--", linewidth=1.2)
    ax.set_xlim(hist_x_min, hist_x_max)
    ax.set_ylim(0, max_hist_count * 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.97,
        0.95,
        f"N = {len(values):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

for ax in axes[GROUP_COUNT:]:
    ax.set_visible(False)

for ax in axes[-histogram_columns:]:
    ax.set_xlabel("BHAR(2,60)")
for ax in axes[::histogram_columns]:
    ax.set_ylabel("Frequency")

fig.tight_layout()
OUTPUTS.save_figure(fig, "bhar_histograms_by_sue_decile")
plt.show()


fig, ax = plt.subplots(figsize=(7, 4.2))

box = ax.boxplot(
    group_values,
    vert=False,
    positions=range(1, GROUP_COUNT + 1),
    widths=0.65,
    patch_artist=True,
    whis=1.5,
    showfliers=True,
    medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    whiskerprops={"color": "#4d4d4d", "linewidth": 1.0},
    capprops={"color": "#4d4d4d", "linewidth": 1.0},
    flierprops={
        "marker": "o",
        "markersize": 3.5,
        "alpha": 0.45,
        "markeredgewidth": 0,
        "markerfacecolor": "#1b1b1b",
    },
)

for patch, color in zip(box["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_xlabel("BHAR(2,60)")
ax.set_ylabel(f"SUE {GROUP_LABEL}")
ax.set_xlim(*x_limits)
ax.set_yticks(range(1, GROUP_COUNT + 1))
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
OUTPUTS.save_figure(fig, "bhar_boxplots_by_sue_decile")
plt.show()


fig, ax = plt.subplots(figsize=(7, 4.2))

violin = ax.violinplot(
    group_values,
    positions=range(1, GROUP_COUNT + 1),
    vert=False,
    showmeans=False,
    showmedians=True,
    showextrema=True,
)

for body, color in zip(violin["bodies"], group_colors):
    body.set_facecolor(color)
    body.set_edgecolor("#4d4d4d")
    body.set_alpha(0.75)

for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
    if key in violin:
        violin[key].set_color("#1b1b1b")
        violin[key].set_linewidth(1.0)

for group, values, color in zip(range(1, GROUP_COUNT + 1), group_values, group_colors):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = values[(values < lower_fence) | (values > upper_fence)]
    if len(outliers) == 0:
        continue
    ax.scatter(
        outliers,
        np.full(len(outliers), group, dtype=float),
        s=12,
        alpha=0.45,
        color="#1b1b1b",
        edgecolors="none",
        zorder=3,
    )

ax.set_xlabel("BHAR(2,60)")
ax.set_ylabel(f"SUE {GROUP_LABEL}")
ax.set_xlim(*x_limits)
ax.set_yticks(range(1, GROUP_COUNT + 1))
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
OUTPUTS.save_figure(fig, "bhar_violin_plots_by_sue_decile")
plt.show()


year_plot_events = event_level.dropna(subset=["Formation_Year", BHAR_COLUMN]).copy()
year_plot_events["Formation_Year"] = pd.to_numeric(
    year_plot_events["Formation_Year"],
    errors="coerce",
)
year_plot_events[BHAR_COLUMN] = pd.to_numeric(
    year_plot_events[BHAR_COLUMN],
    errors="coerce",
)
year_plot_events = year_plot_events.dropna(subset=["Formation_Year", BHAR_COLUMN]).copy()
year_plot_events["Formation_Year"] = year_plot_events["Formation_Year"].astype(int)

available_years = sorted(year_plot_events["Formation_Year"].unique().tolist())
year_values = [
    year_plot_events.loc[year_plot_events["Formation_Year"] == year, BHAR_COLUMN].to_numpy()
    for year in available_years
]

year_colors = plt.get_cmap(COLOR_PALETTE)(np.linspace(0.08, 0.92, len(available_years)))

fig, ax = plt.subplots(figsize=(7, 3.5))

year_box = ax.boxplot(
    year_values,
    positions=range(1, len(available_years) + 1),
    widths=0.65,
    patch_artist=True,
    whis=1.5,
    showfliers=True,
    medianprops={"color": "#1b1b1b", "linewidth": 1.5},
    whiskerprops={"color": "#4d4d4d", "linewidth": 1.0},
    capprops={"color": "#4d4d4d", "linewidth": 1.0},
    flierprops={
        "marker": "o",
        "markersize": 3.5,
        "alpha": 0.45,
        "markeredgewidth": 0,
        "markerfacecolor": "#1b1b1b",
    },
)

for patch, color in zip(year_box["boxes"], year_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_xlabel("Formation year")
ax.set_ylabel("BHAR(2,60)")
ax.set_ylim(*x_limits)
ax.set_xticks(range(1, len(available_years) + 1))
ax.set_xticklabels(available_years, rotation=45)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
OUTPUTS.save_figure(fig, "bhar_boxplots_by_formation_year")
plt.close(fig)
