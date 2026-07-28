from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, LogLocator, MaxNLocator, MultipleLocator
from _analysis_shared import AnalysisOutputManager
import sys

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.time_varying_analysis import (
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
    load_stock_universe_snapshots,
)
from src.pead.sue_groups import SUE_GROUP_COLUMN

DATA_DIR = PROJECT_ROOT / "data"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
OUTPUTS = AnalysisOutputManager(__file__)


def load_earnings_events() -> pd.DataFrame:
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_events.csv")
    )

    if not earnings_files:
        raise FileNotFoundError(
            "No yearly earnings_events.csv files found under data/yearly/<year>/. "
            "Run scripts/04_build_earnings_and_sue.py first."
        )

    frames = []

    for path in earnings_files:
        frame = pd.read_csv(path)
        frame["Formation_Year"] = path.parent.name
        frames.append(frame)

    earnings_events = pd.concat(frames, ignore_index=True)

    for column in ("SUE", "Price_Lag_5", "Forecast_Analyst_Count"):
        earnings_events[column] = pd.to_numeric(earnings_events[column], errors="coerce")

    earnings_events = earnings_events.dropna(subset=["SUE"]).copy()

    return earnings_events


def distribution_statistics(values: pd.Series, *, variable: str) -> dict[str, float | int | str]:
    """Return the thesis-table distribution statistics for one numeric variable."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        raise ValueError(f"No non-missing observations are available for {variable}.")

    return {
        "Variable": variable,
        "Mean": clean_values.mean(),
        "SD": clean_values.std(ddof=1),
        "Min": clean_values.min(),
        "Q1": clean_values.quantile(0.25),
        "Median": clean_values.median(),
        "Q3": clean_values.quantile(0.75),
        "Max": clean_values.max(),
    }


earnings_events = load_earnings_events()

# Panel A of Table "SUE descriptive statistics and quintile classification".
# This deliberately uses every event with a computable raw SUE, including events
# that cannot subsequently be assigned to a prior-year quintile.
sue_distribution_statistics = pd.DataFrame(
    [distribution_statistics(earnings_events["SUE"], variable="Raw SUE")]
)
OUTPUTS.save_table(sue_distribution_statistics, "sue_raw_distribution_statistics")

# Annual statistics support the discussion of changes in the centre and
# dispersion of SUE through time in the supplementary appendix.
annual_sue_statistics = (
    earnings_events.assign(
        Formation_Year=pd.to_numeric(earnings_events["Formation_Year"], errors="coerce")
    )
    .dropna(subset=["Formation_Year"])
    .groupby("Formation_Year")["SUE"]
    .apply(
        lambda values: pd.Series(
            distribution_statistics(values, variable="Raw SUE")
        ).drop("Variable")
    )
    .unstack()
    .reset_index()
)
annual_sue_statistics["Formation_Year"] = annual_sue_statistics["Formation_Year"].astype(int)
OUTPUTS.save_table(annual_sue_statistics, "annual_sue_distribution_statistics")

# Panel B uses the final event-level analysis data, so it is aligned with the
# return construction and has one observation per earnings announcement.
abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(abnormal_returns)
required_event_columns = {"SUE", SUE_GROUP_COLUMN, "BHAR_0_1", "Formation_Year", "Instrument"}
missing_event_columns = required_event_columns.difference(event_level.columns)
if missing_event_columns:
    raise KeyError(
        "The event-level data are missing columns required for the SUE descriptive "
        f"outputs: {sorted(missing_event_columns)}."
    )

classified_events = event_level.dropna(subset=["SUE", SUE_GROUP_COLUMN]).copy()
classified_events[SUE_GROUP_COLUMN] = pd.to_numeric(
    classified_events[SUE_GROUP_COLUMN], errors="coerce"
).astype("Int64")
classified_events["BHAR_0_1"] = pd.to_numeric(
    classified_events["BHAR_0_1"], errors="coerce"
)
classified_events = classified_events.dropna(subset=[SUE_GROUP_COLUMN])

sue_quintile_statistics = (
    classified_events.groupby(SUE_GROUP_COLUMN, observed=True)
    .agg(
        Events=("SUE", "size"),
        Mean_SUE=("SUE", "mean"),
        Median_SUE=("SUE", "median"),
        Mean_BHAR_0_1=("BHAR_0_1", "mean"),
        Median_BHAR_0_1=("BHAR_0_1", "median"),
    )
    .reindex(range(1, 6))
    .rename_axis("SUE_Quintile")
    .reset_index()
)
sue_quintile_statistics["SUE_Quintile_Label"] = sue_quintile_statistics[
    "SUE_Quintile"
].map({1: "Q1 (lowest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (highest)"})
sue_total_statistics = pd.DataFrame(
    [
        {
            "SUE_Quintile": "Total",
            "SUE_Quintile_Label": "Total",
            "Events": classified_events["SUE"].size,
            "Mean_SUE": classified_events["SUE"].mean(),
            "Median_SUE": classified_events["SUE"].median(),
            "Mean_BHAR_0_1": classified_events["BHAR_0_1"].mean(),
            "Median_BHAR_0_1": classified_events["BHAR_0_1"].median(),
        }
    ]
)
sue_quintile_statistics = sue_quintile_statistics[
    [
        "SUE_Quintile",
        "SUE_Quintile_Label",
        "Events",
        "Mean_SUE",
        "Median_SUE",
        "Mean_BHAR_0_1",
        "Median_BHAR_0_1",
    ]
]
sue_quintile_statistics = pd.concat(
    [sue_quintile_statistics, sue_total_statistics], ignore_index=True
)
OUTPUTS.save_table(sue_quintile_statistics, "sue_quintile_classification_statistics")

# The supplementary firm-size diagnostic uses formation-date market
# capitalization and is deliberately kept separate from the SUE calculation.
stock_universe = load_stock_universe_snapshots(DATA_DIR)
firm_size_columns = ["Instrument", "Formation_Year", "Market_Cap_Current"]
missing_firm_size_columns = set(firm_size_columns).difference(stock_universe.columns)
if missing_firm_size_columns:
    raise KeyError(
        "The stock-universe snapshots are missing columns required for the SUE "
        f"firm-size diagnostic: {sorted(missing_firm_size_columns)}."
    )

firm_size_snapshot = stock_universe.loc[:, firm_size_columns].copy()
firm_size_snapshot["Formation_Year"] = pd.to_numeric(
    firm_size_snapshot["Formation_Year"], errors="coerce"
)
firm_size_snapshot["Market_Cap_Current"] = pd.to_numeric(
    firm_size_snapshot["Market_Cap_Current"], errors="coerce"
)
firm_size_snapshot = firm_size_snapshot.drop_duplicates(
    subset=["Instrument", "Formation_Year"], keep="first"
)
firm_size_by_sue = classified_events.merge(
    firm_size_snapshot,
    on=["Instrument", "Formation_Year"],
    how="left",
    validate="m:1",
).loc[lambda frame: frame["Market_Cap_Current"] > 0].copy()
firm_size_by_sue["Log_Market_Cap"] = np.log(firm_size_by_sue["Market_Cap_Current"])

firm_size_by_sue_statistics = (
    firm_size_by_sue.groupby(SUE_GROUP_COLUMN, observed=True)["Log_Market_Cap"]
    .agg(
        N="size",
        Mean="mean",
        SD="std",
        Q1=lambda values: values.quantile(0.25),
        Median="median",
        Q3=lambda values: values.quantile(0.75),
    )
    .reindex(range(1, 6))
    .rename_axis("SUE_Quintile")
    .reset_index()
)
OUTPUTS.save_table(firm_size_by_sue_statistics, "log_market_cap_by_sue_quintile_statistics")

firm_size_groups = [
    firm_size_by_sue.loc[firm_size_by_sue[SUE_GROUP_COLUMN] == quintile, "Market_Cap_Current"]
    .dropna()
    .to_numpy()
    for quintile in range(1, 6)
]
if any(values.size == 0 for values in firm_size_groups):
    raise ValueError("Each SUE quintile must contain positive formation-date market-cap observations.")

fig, ax = plt.subplots(figsize=(7, 4.2))
boxplot = ax.boxplot(
    firm_size_groups,
    positions=range(1, 6),
    widths=0.65,
    patch_artist=True,
    boxprops={"facecolor": "#00A651", "alpha": 0.8},
    medianprops={"color": "#006B3C", "linewidth": 1.5},
    whiskerprops={"color": "#006B3C"},
    capprops={"color": "#006B3C"},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.2, "markeredgewidth": 0},
)
ax.set_xlabel("SUE quintile")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
ax.set_ylabel(r"Market cap (\$ millions)")
ax.set_xticks(range(1, 6))
ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax.grid(axis="y", alpha=0.25)
ax.grid(axis="x", visible=False)
fig.tight_layout()
OUTPUTS.save_figure(fig, "log_market_cap_by_sue_quintile_boxplots")

sue_values = earnings_events["SUE"].dropna().to_numpy()
SUE_DISPLAY_LOWER, SUE_DISPLAY_UPPER = np.percentile(sue_values, [2, 98])
SUE_COLOR = "#FF7A00"
SUE_LIGHT_COLOR = "#FFB45C"
SUE_DARK_COLOR = "#B94700"
SUE_MID_COLOR = "#D95D00"
ANALYST_FOLLOWING_COLOR = "#0096C7"
ANALYST_FOLLOWING_DARK_COLOR = "#006C8F"
EARNINGS_ANNOUNCEMENT_COLOR = "#1A365D"

# Bin edges are defined within the displayed range, while the unmodified SUE
# values are passed to hist(). Observations outside the 2nd--98th percentile
# range are omitted from the figure rather than placed in edge bins.
histogram_bins = np.linspace(SUE_DISPLAY_LOWER, SUE_DISPLAY_UPPER, 41)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(
    sue_values,
    bins=histogram_bins,
    color=SUE_COLOR,
    edgecolor="black",
    alpha=0.85,
)
ax.set_xlim(SUE_DISPLAY_LOWER, SUE_DISPLAY_UPPER)
ax.set_xlabel("Standardized unexpected earnings (SUE)")
ax.set_ylabel("Earnings announcements")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.grid(axis="y", alpha=0.25)
ax.grid(axis="x", alpha=0.25)
ax.text(
    0.98,
    0.95,
    "Horizontal axis truncated at\n2nd and 98th percentiles",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
)

fig.tight_layout()
OUTPUTS.save_figure(fig, "distribution_of_sue")
# plt.show()

plot_data = earnings_events.dropna(subset=["SUE", "Price_Lag_5"])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(
    plot_data["Price_Lag_5"],
    plot_data["SUE"],
    alpha=0.35,
    s=20,
    color=SUE_COLOR,
    edgecolors="none",
)
ax.set_xscale("log")
ax.set_xlabel("Lagged price (Price_Lag_5)")

sue_2_5_percentile = np.percentile(plot_data["SUE"], 2.5)
sue_97_5_percentile = np.percentile(plot_data["SUE"], 97.5)
ax.set_ylim(bottom=sue_2_5_percentile, top=sue_97_5_percentile)
ax.set_ylabel("SUE")
fig.tight_layout()
# Not retained in thesis2/Figures.
# OUTPUTS.save_figure(fig, "sue_vs_lagged_price")
# plt.show()

variance_test_data = earnings_events.dropna(subset=["SUE", "Price_Lag_5"]).copy()
low_price_sue = variance_test_data.loc[variance_test_data["Price_Lag_5"] < 1, "SUE"].astype(float)
high_price_sue = variance_test_data.loc[variance_test_data["Price_Lag_5"] > 1, "SUE"].astype(float)

if len(low_price_sue) < 2 or len(high_price_sue) < 2:
    raise ValueError("Need at least two SUE observations in both price groups to compare variances.")

low_price_variance = low_price_sue.var(ddof=1)
high_price_variance = high_price_sue.var(ddof=1)
variance_ratio = low_price_variance / high_price_variance
f_df_num = len(low_price_sue) - 1
f_df_den = len(high_price_sue) - 1
one_sided_p_value = 1 - stats.f.cdf(variance_ratio, f_df_num, f_df_den)

variance_test_summary = pd.DataFrame(
    {
        "Group": ["Price < 1", "Price > 1"],
        "Observation_Count": [len(low_price_sue), len(high_price_sue)],
        "SUE_Variance": [low_price_variance, high_price_variance],
        "SUE_Std_Dev": [low_price_sue.std(ddof=1), high_price_sue.std(ddof=1)],
    }
)
variance_test_result = pd.DataFrame(
    {
        "Test": ["One-sided F-test of variances"],
        "Alternative": ["Var(SUE | Price < 1) > Var(SUE | Price > 1)"],
        "Variance_Ratio_F": [variance_ratio],
        "DF_Numerator": [f_df_num],
        "DF_Denominator": [f_df_den],
        "P_Value": [one_sided_p_value],
    }
)

OUTPUTS.save_table(variance_test_summary, "sue_variance_by_price_group")
OUTPUTS.save_table(variance_test_result, "sue_variance_f_test_price_lt1_vs_gt1")
print(variance_test_summary.to_string(index=False))
print()
print(variance_test_result.to_string(index=False))

plot_data = earnings_events.dropna(subset=["Formation_Year", "SUE"]).copy()
plot_data["Formation_Year"] = pd.to_numeric(plot_data["Formation_Year"], errors="coerce")
plot_data = plot_data.dropna(subset=["Formation_Year"])
plot_data["Formation_Year"] = plot_data["Formation_Year"].astype(int)
year_values = sorted(plot_data["Formation_Year"].unique().tolist())
year_groups = [
    plot_data.loc[plot_data["Formation_Year"] == year, "SUE"].values # type: ignore
    for year in year_values
]
def select_year_ticks(years: list[int]) -> list[int]:
    if len(years) <= 7:
        return years

    if len(years) <= 14:
        ticks = years[::2]
    else:
        ticks = [year for year in years if year % 5 == 0]

    for endpoint in (years[0], years[-1]):
        if endpoint not in ticks:
            ticks.append(endpoint)

    ticks = sorted(set(ticks))
    if len(ticks) <= 8:
        return ticks

    return [ticks[index] for index in np.linspace(0, len(ticks) - 1, 8, dtype=int)]


year_tick_values = select_year_ticks(year_values)

fig, ax_box = plt.subplots(figsize=(7, 4.2))
boxplot_artists = ax_box.boxplot(
    year_groups,
    positions=year_values,
    widths=0.6,
    patch_artist=True,
    boxprops={"facecolor": SUE_LIGHT_COLOR, "alpha": 0.8},
    medianprops={"color": SUE_DARK_COLOR, "linewidth": 1.5},
    whiskerprops={"color": SUE_MID_COLOR},
    capprops={"color": SUE_MID_COLOR},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.25, "markeredgewidth": 0},
)

whisker_values = np.concatenate(
    [whisker.get_ydata() for whisker in boxplot_artists["whiskers"]]
)
whisker_min = float(whisker_values.min())
whisker_max = float(whisker_values.max())
whisker_range = whisker_max - whisker_min
whisker_padding = max(whisker_range * 0.05, 1e-4)

ax_box.set_ylim(whisker_min - whisker_padding, whisker_max + whisker_padding)
ax_box.yaxis.set_major_locator(MultipleLocator(0.01))
ax_box.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax_box.set_ylabel("SUE")
ax_box.set_xlim(min(year_values) - 0.5, max(year_values) + 0.5)
ax_box.set_xticks(year_tick_values)
ax_box.set_xticklabels([str(year) for year in year_tick_values])
ax_box.set_xticks(year_values, minor=True)
ax_box.tick_params(axis="x", which="minor", length=3)
ax_box.set_xlabel("Formation year")
ax_box.grid(axis="y", alpha=0.25)
ax_box.grid(axis="x", which="major", alpha=0.25)
fig.tight_layout()
OUTPUTS.save_figure(fig, "sue_by_formation_year")
# plt.show()

plot_data = earnings_events.dropna(subset=["SUE", "Forecast_Analyst_Count"]).copy()
plot_data["Forecast_Analyst_Count"] = pd.to_numeric(
    plot_data["Forecast_Analyst_Count"],
    errors="coerce",
)
plot_data = plot_data.dropna(subset=["Forecast_Analyst_Count"])
plot_data["Forecast_Analyst_Count"] = plot_data["Forecast_Analyst_Count"].astype(int)
plot_data = plot_data.loc[plot_data["Forecast_Analyst_Count"] >= 3].copy()
plot_data["Analyst_Following_Group"] = plot_data["Forecast_Analyst_Count"].clip(upper=20)
analyst_group_values = sorted(plot_data["Analyst_Following_Group"].unique().tolist())
analyst_group_labels = [str(value) if value < 20 else "20+" for value in analyst_group_values]
analyst_group_sue = [
    plot_data.loc[plot_data["Analyst_Following_Group"] == value, "SUE"].values # type: ignore
    for value in analyst_group_values
]
fig, ax_scatter = plt.subplots(figsize=(7, 4.2))

# Boxplots (top)
ax_scatter.boxplot(
    analyst_group_sue,
    positions=analyst_group_values,
    widths=0.6,
    patch_artist=True,
    boxprops={"facecolor": SUE_LIGHT_COLOR, "alpha": 0.8},
    medianprops={"color": SUE_DARK_COLOR, "linewidth": 1.5},
    whiskerprops={"color": SUE_MID_COLOR},
    capprops={"color": SUE_MID_COLOR},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.25, "markeredgewidth": 0},
)
ax_scatter.set_ylabel("SUE")
ax_scatter.set_ylim(SUE_DISPLAY_LOWER, SUE_DISPLAY_UPPER)
ax_scatter.yaxis.set_major_locator(MaxNLocator(nbins=6))
analyst_group_label_map = dict(zip(analyst_group_values, analyst_group_labels))
analyst_tick_values = list(range(3, max(analyst_group_values) + 1))
ax_scatter.set_xlim(2.5, max(analyst_group_values) + 0.5)
ax_scatter.set_xticks(analyst_tick_values)
ax_scatter.set_xticklabels([
    analyst_group_label_map.get(value, str(value)) for value in analyst_tick_values
])
ax_scatter.set_xlabel("Analysts following")
ax_scatter.grid(axis="y", alpha=0.25)
ax_scatter.grid(axis="x", visible=False)
ax_scatter.text(
    0.98,
    0.95,
    "Vertical axis truncated at\n2nd and 98th percentiles",
    transform=ax_scatter.transAxes,
    ha="right",
    va="top",
    fontsize=9,
)

plt.tight_layout()
OUTPUTS.save_figure(fig, "sue_by_analyst_following")
plt.close(fig)
