from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use("ggplot")
from IPython.display import display
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
from _analysis_shared import AnalysisOutputManager

OUTPUTS = AnalysisOutputManager(__file__)


def load_stock_universe():
    universe_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/stock_universe.csv")
    )

    return pd.concat(
        (pd.read_csv(path) for path in universe_files),
        ignore_index=True,
    )


def load_earnings_events():
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_events.csv")
    )

    frames = []
    for path in earnings_files:
        frame = pd.read_csv(path)
        frame["Formation_Year"] = int(path.parent.name)
        frames.append(frame)

    earnings_events = pd.concat(frames, ignore_index=True)
    earnings_events["Ann_Date"] = pd.to_datetime(earnings_events["Ann_Date"])

    return earnings_events


def load_all_earnings_events_before_filtering():
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_events_full.csv")
    )

    earnings_events = pd.concat(
        (pd.read_csv(path) for path in earnings_files),
        ignore_index=True,
    )
    earnings_events["Ann_Date"] = pd.to_datetime(earnings_events["Ann_Date"])

    return earnings_events


stock_universe = load_stock_universe()
earnings_events = load_earnings_events()
all_earnings_events_before_filtering = load_all_earnings_events_before_filtering()

final_earnings_events = earnings_events.copy()


def comma_number_formatter(decimals=0):
    return FuncFormatter(lambda value, _: f"{value:,.{decimals}f}")


def positive_log_bins(series, n_bins=30):
    positive_values = pd.to_numeric(series, errors="coerce").dropna()
    positive_values = positive_values[positive_values > 0]
    if positive_values.empty:
        return n_bins

    lower = positive_values.min()
    upper = positive_values.max()
    if lower == upper:
        lower *= 0.9
        upper *= 1.1

    return np.geomspace(lower, upper, n_bins + 1)


def linear_bins(series, n_bins=30):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return n_bins

    lower = values.min()
    upper = values.max()
    if lower == upper:
        lower -= 0.5
        upper += 0.5

    return np.linspace(lower, upper, n_bins + 1)


MARKET_CAP_COLOR = "#00A651"
MARKET_CAP_DARK_COLOR = "#006B3C"
MARKET_CAP_LIGHT_COLOR = "#90EE90"
ANALYST_FOLLOWING_COLOR = "#0096C7"
PRICE_COLOR = "#FF3B30"
BOOK_TO_MARKET_COLOR = "#5D4037"
EARNINGS_ANNOUNCEMENT_COLOR = "#1A365D"


def describe_population(series, population_name):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "Population": population_name,
        "N": int(values.shape[0]),
        "Mean": values.mean(),
        "Std": values.std(),
        "Min": values.min(),
        "P1": values.quantile(0.01),
        "P5": values.quantile(0.05),
        "P25": values.quantile(0.25),
        "Median": values.median(),
        "P75": values.quantile(0.75),
        "P95": values.quantile(0.95),
        "P99": values.quantile(0.99),
        "Max": values.max(),
    }


def plot_event_characteristic_histogram(
    series,
    *,
    title,
    xlabel,
    output_name,
    bins,
    log_x=False,
    formatter=None,
    ylabel="Earnings announcements",
    color=BOOK_TO_MARKET_COLOR,
    x_min=None,
    note=None,
):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if log_x:
        values = values[values > 0]

    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=bins, edgecolor="black", color=color, alpha=0.85)

    if log_x:
        ax.set_xscale("log")
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)
    if x_min is not None:
        ax.set_xlim(left=x_min)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if note is not None:
        ax.text(
            0.98,
            0.95,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    OUTPUTS.save_figure(fig, output_name)
    # plt.close(fig)


final_announcement_characteristics = final_earnings_events.copy()

formation_characteristics = stock_universe[
    ["Instrument", "Formation_Year", "Market_Cap_Current", "BM", "Price"]
].copy()
formation_characteristics["Formation_Year"] = pd.to_numeric(
    formation_characteristics["Formation_Year"],
    errors="coerce",
)
formation_characteristics = formation_characteristics.drop_duplicates(
    subset=["Instrument", "Formation_Year"]
)

final_announcement_characteristics = final_announcement_characteristics.merge(
    formation_characteristics,
    on=["Instrument", "Formation_Year"],
    how="left",
)

formation_characteristics_stats = pd.DataFrame(
    [
        describe_population(
            final_announcement_characteristics.loc[
                final_announcement_characteristics["Market_Cap_Current"] > 0,
                "Market_Cap_Current",
            ],
            "Market cap",
        ),
        describe_population(
            final_announcement_characteristics["BM"],
            "Book-to-market ratio",
        ),
        describe_population(
            final_announcement_characteristics.loc[
                final_announcement_characteristics["Price"] > 0,
                "Price",
            ],
            "Price",
        ),
    ]
)

display(formation_characteristics_stats)
OUTPUTS.save_table(
    formation_characteristics_stats,
    "final_earnings_sample_formation_characteristics_stats",
)

plot_event_characteristic_histogram(
    final_announcement_characteristics["Market_Cap_Current"],
    title="Final Earnings Sample: Market cap",
    xlabel=r"Market cap (\$ millions)",
    output_name="final_earnings_sample_market_cap_at_formation_histogram",
    bins=positive_log_bins(final_announcement_characteristics["Market_Cap_Current"]),
    log_x=True,
    formatter=comma_number_formatter(decimals=0),
    ylabel="Firm-year observations",
    color=MARKET_CAP_COLOR,
)

book_to_market_upper_bound = final_announcement_characteristics["BM"].quantile(0.98)
book_to_market_histogram_values = final_announcement_characteristics.loc[
    final_announcement_characteristics["BM"] <= book_to_market_upper_bound,
    "BM",
]

plot_event_characteristic_histogram(
    book_to_market_histogram_values,
    title="Final Earnings Sample: Book-to-market ratio",
    xlabel="Book-to-market ratio",
    output_name="final_earnings_sample_book_to_market_at_formation_histogram",
    bins=linear_bins(book_to_market_histogram_values),
    formatter=comma_number_formatter(decimals=2),
    ylabel="Firm-year observations",
    color=BOOK_TO_MARKET_COLOR,
    x_min=0,
    note="Horizontal axis truncated at the 98th percentile",
)

plot_event_characteristic_histogram(
    final_announcement_characteristics["Price"],
    title="Final Earnings Sample: Price",
    xlabel=r"Price (\$)",
    output_name="final_earnings_sample_price_at_formation_histogram",
    bins=positive_log_bins(final_announcement_characteristics["Price"]),
    log_x=True,
    formatter=comma_number_formatter(decimals=2),
    ylabel="Firm-year observations",
    color=PRICE_COLOR,
)

analyst_following_before_filtering = pd.to_numeric(
    all_earnings_events_before_filtering["Forecast_Analyst_Count"],
    errors="coerce",
).dropna()
analyst_following_before_filtering = analyst_following_before_filtering[
    analyst_following_before_filtering >= 0
]
analyst_following_before_filtering_stats = pd.DataFrame(
    [
        describe_population(
            analyst_following_before_filtering,
            "Analyst following per earnings announcement before filtering",
        )
    ]
)

display(analyst_following_before_filtering_stats)
OUTPUTS.save_table(
    analyst_following_before_filtering_stats,
    "analyst_following_per_announcement_before_filtering_stats",
)

analyst_following_histogram_values = analyst_following_before_filtering.clip(upper=15)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(
    analyst_following_histogram_values,
    bins=np.arange(-0.5, 16.5, 1),
    edgecolor="black",
    color=ANALYST_FOLLOWING_COLOR,
    alpha=0.85,
)
analyst_following_ticks = np.arange(0, 16, 1)
analyst_following_tick_labels = [
    str(value) if value < 15 else "15+" for value in analyst_following_ticks
]
ax.set_xticks(analyst_following_ticks)
ax.set_xticklabels(analyst_following_tick_labels)
ax.set_xlim(-0.5, 15.5)
ax.set_xlabel("Analysts following")
ax.set_ylabel("Earnings announcements")
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
OUTPUTS.save_figure(
    fig,
    "analyst_following_per_announcement_before_filtering_histogram",
)
# plt.close(fig)

# Use the common pre-filter event file for every series.  This makes the
# populations nested solely by the analyst-count requirement; the saved
# min1/final event files also apply later forecast-median and price filters.
analyst_restriction_market_caps = all_earnings_events_before_filtering[
    ["Instrument", "Ann_Date", "Forecast_Analyst_Count"]
].copy()
analyst_restriction_market_caps["Formation_Year"] = (
    analyst_restriction_market_caps["Ann_Date"].dt.year
)
analyst_restriction_market_caps["Forecast_Analyst_Count"] = pd.to_numeric(
    analyst_restriction_market_caps["Forecast_Analyst_Count"], errors="coerce"
)
analyst_restriction_market_caps = analyst_restriction_market_caps.merge(
    formation_characteristics[["Instrument", "Formation_Year", "Market_Cap_Current"]],
    on=["Instrument", "Formation_Year"],
    how="left",
    validate="many_to_one",
)
analyst_restriction_market_caps["Market_Cap_Current"] = pd.to_numeric(
    analyst_restriction_market_caps["Market_Cap_Current"], errors="coerce"
)
analyst_restriction_market_caps = analyst_restriction_market_caps.loc[
    analyst_restriction_market_caps["Market_Cap_Current"] > 0
].copy()

market_cap_samples_by_analyst_restriction = [
    (
        "All events",
        analyst_restriction_market_caps["Market_Cap_Current"],
        MARKET_CAP_LIGHT_COLOR,
    ),
    (
        r"$\geq$1 analyst",
        analyst_restriction_market_caps.loc[
            analyst_restriction_market_caps["Forecast_Analyst_Count"] >= 1,
            "Market_Cap_Current",
        ],
        MARKET_CAP_COLOR,
    ),
    (
        r"$\geq$3 analysts",
        analyst_restriction_market_caps.loc[
            analyst_restriction_market_caps["Forecast_Analyst_Count"] >= 3,
            "Market_Cap_Current",
        ],
        MARKET_CAP_DARK_COLOR,
    ),
]

# The bin edges are equally spaced after the logarithmic transformation, and
# are shared by all three populations.
common_market_cap_bins = positive_log_bins(
    analyst_restriction_market_caps["Market_Cap_Current"], n_bins=30
)

fig, ax = plt.subplots(figsize=(7, 4.2))
for label, market_caps, color in market_cap_samples_by_analyst_restriction:
    ax.hist(
        market_caps,
        bins=common_market_cap_bins,
        edgecolor="black",
        color=color,
        alpha=0.85,
        label=label,
    )

ax.set_xscale("log")
ax.xaxis.set_major_formatter(comma_number_formatter(decimals=0))
ax.set_xlabel(r"Market cap (\$ millions)")
ax.set_ylabel("Firm-year observations")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="upper right", fontsize=9)

plt.tight_layout()
OUTPUTS.save_figure(
    fig,
    "market_cap_distribution_by_analyst_forecast_restriction",
)
# plt.close(fig)

analyst_following_by_year_df = all_earnings_events_before_filtering[
    ["Ann_Date", "Forecast_Analyst_Count"]
].copy()
analyst_following_by_year_df["Announcement_Year"] = (
    analyst_following_by_year_df["Ann_Date"].dt.year
)
analyst_following_by_year_df["Forecast_Analyst_Count"] = pd.to_numeric(
    analyst_following_by_year_df["Forecast_Analyst_Count"],
    errors="coerce",
)
analyst_following_by_year_df = analyst_following_by_year_df.dropna(
    subset=["Announcement_Year", "Forecast_Analyst_Count"]
)
analyst_following_by_year_df = analyst_following_by_year_df[
    analyst_following_by_year_df["Forecast_Analyst_Count"] > 0
]

analyst_following_yearly_summary = (
    analyst_following_by_year_df.groupby("Announcement_Year")["Forecast_Analyst_Count"]
    .agg(
        Mean="mean",
        P25=lambda values: values.quantile(0.25),
        Median="median",
        P75=lambda values: values.quantile(0.75),
    )
    .sort_index()
)

fig, ax = plt.subplots(figsize=(7, 4.2))
for column, linestyle in [
    ("Mean", "-"),
    ("P25", ":"),
    ("Median", "--"),
    ("P75", "-."),
]:
    ax.plot(
        analyst_following_yearly_summary.index.astype(str),
        analyst_following_yearly_summary[column],
        linewidth=2,
        color=ANALYST_FOLLOWING_COLOR,
        linestyle=linestyle,
        label=column,
    )

ax.set_xlabel("Announcement year")
ax.set_ylabel("Analysts following")
ax.tick_params(axis="x", rotation=90)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Statistic", frameon=False)

plt.tight_layout()
# Not retained in thesis2/Figures.
# OUTPUTS.save_figure(
#     fig,
#     "analyst_following_by_year_before_filtering",
# )
# plt.close(fig)

final_sample_firm_size_by_year = final_announcement_characteristics.dropna(
    subset=["Formation_Year", "Market_Cap_Current"]
).copy()
final_sample_firm_size_by_year = final_sample_firm_size_by_year[
    final_sample_firm_size_by_year["Market_Cap_Current"] > 0
]
firm_size_years = sorted(final_sample_firm_size_by_year["Formation_Year"].unique())
firm_size_values_by_year = [
    final_sample_firm_size_by_year.loc[
        final_sample_firm_size_by_year["Formation_Year"] == year,
        "Market_Cap_Current",
    ].values
    for year in firm_size_years
]

fig_width = max(12, len(firm_size_years) * 0.35)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.boxplot(
    firm_size_values_by_year,
    positions=range(len(firm_size_years)),
    widths=0.6,
    patch_artist=True,
    boxprops={"facecolor": MARKET_CAP_LIGHT_COLOR, "alpha": 0.8},
    medianprops={"color": MARKET_CAP_DARK_COLOR, "linewidth": 1.5},
    whiskerprops={"color": MARKET_CAP_DARK_COLOR},
    capprops={"color": MARKET_CAP_DARK_COLOR},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.25, "markeredgewidth": 0},
)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(comma_number_formatter(decimals=0))
ax.set_xticks(range(len(firm_size_years)))
ax.set_xticklabels([str(int(year)) for year in firm_size_years], rotation=90)
ax.set_xlabel("Formation year")
ax.set_ylabel(r"Market cap (\$ millions)")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
OUTPUTS.save_figure(
    fig,
    "final_earnings_sample_firm_size_by_year_boxplots",
)
# plt.close(fig)

n_unique_announcements = final_earnings_events.drop_duplicates(
    subset=["Instrument", "Ann_Date"]
).shape[0]

n_unique_firms = final_earnings_events["Instrument"].nunique()

eligible_firms = pd.Index(stock_universe["Instrument"].dropna().unique())

announcements_per_firm = (
    final_earnings_events
    .drop_duplicates(subset=["Instrument", "Ann_Date"])
    .groupby("Instrument")
    .size()
    .sort_values(ascending=False)
)

announcements_per_eligible_firm = (
    announcements_per_firm
    .reindex(eligible_firms, fill_value=0)
    .sort_values(ascending=False)
)

n_eligible_firms = len(eligible_firms)
n_zero_announcement_firms = int((announcements_per_eligible_firm == 0).sum())
n_single_event_firms = int((announcements_per_firm == 1).sum())
n_single_event_observations = n_single_event_firms
n_multiple_event_firms = int((announcements_per_firm >= 2).sum())
n_multiple_event_observations = int(
    announcements_per_firm.loc[announcements_per_firm >= 2].sum()
)

summary_stats = pd.DataFrame(
    {
        "Statistic": [
            "Unique earnings announcements",
            "Unique firms",
            "Firms with one earnings announcement",
            "Earnings announcements from single-event firms",
            "Firms with at least two earnings announcements",
            "Earnings announcements from firms with at least two events",
            "Eligible firms in saved stock universe",
            "Eligible firms with zero saved earnings announcements",
        ],
        "Value": [
            n_unique_announcements,
            n_unique_firms,
            n_single_event_firms,
            n_single_event_observations,
            n_multiple_event_firms,
            n_multiple_event_observations,
            n_eligible_firms,
            n_zero_announcement_firms,
        ],
    }
)

display(summary_stats)
OUTPUTS.save_table(summary_stats, "summary_stats")

fig, ax = plt.subplots(figsize=(6, 3.6))
announcements_per_reporting_eligible_firm = announcements_per_eligible_firm[
    announcements_per_eligible_firm > 0
]
announcements_per_reporting_eligible_firm_stats = pd.DataFrame(
    [
        describe_population(
            announcements_per_reporting_eligible_firm,
            "Earnings announcements per eligible firm with at least one announcement",
        )
    ]
)

display(announcements_per_reporting_eligible_firm_stats)
OUTPUTS.save_table(
    announcements_per_reporting_eligible_firm_stats,
    "earnings_announcements_per_reporting_eligible_firm_stats",
)

announcements_per_reporting_eligible_firm.plot(
    kind="hist",
    ax=ax,
    bins=range(1, announcements_per_reporting_eligible_firm.max() + 2) - np.array(0.5),
    edgecolor="black",
    color=EARNINGS_ANNOUNCEMENT_COLOR,
)

ax.set_xlabel("Earnings announcements per firm")
ax.set_ylabel("Firms")
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis="x", which="minor", length=3)
ax.set_xlim(
    announcements_per_reporting_eligible_firm.min() - 0.5,
    announcements_per_reporting_eligible_firm.max() + 0.5,
)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
OUTPUTS.save_figure(fig, "distribution_of_earnings_announcements_per_eligible_firm")
# plt.close(fig)

firm_level_plot_df = final_earnings_events[
    ["Instrument", "Ann_Date", "Forecast_Analyst_Count"]
].copy()
firm_level_plot_df["Formation_Year"] = firm_level_plot_df["Ann_Date"].dt.year
firm_level_plot_df["Forecast_Analyst_Count"] = pd.to_numeric(
    firm_level_plot_df["Forecast_Analyst_Count"],
    errors="coerce",
)

firm_size_df = stock_universe[
    ["Instrument", "Formation_Year", "Market_Cap_Current"]
].copy()
firm_size_df["Formation_Year"] = pd.to_numeric(
    firm_size_df["Formation_Year"],
    errors="coerce",
)

firm_level_plot_df = firm_level_plot_df.merge(
    firm_size_df,
    on=["Instrument", "Formation_Year"],
    how="left",
)
firm_level_plot_df = firm_level_plot_df.dropna(
    subset=["Market_Cap_Current", "Forecast_Analyst_Count"]
)
firm_level_plot_df = firm_level_plot_df[firm_level_plot_df["Market_Cap_Current"] > 0]

firm_level_plot_df = (
    firm_level_plot_df.groupby("Instrument", as_index=False)
    .agg(
        Market_Cap_Current=("Market_Cap_Current", "median"),
        Forecast_Analyst_Count=("Forecast_Analyst_Count", "mean"),
    )
)

spearman_firm_size_analyst_following = stats.spearmanr(
    firm_level_plot_df["Market_Cap_Current"],
    firm_level_plot_df["Forecast_Analyst_Count"],
)
spearman_firm_size_analyst_following_table = pd.DataFrame(
    [
        {
            "Statistic": "Spearman rank correlation: firm size and analyst following",
            "Spearman_Rho": spearman_firm_size_analyst_following.statistic,
            "P_Value": spearman_firm_size_analyst_following.pvalue,
            "N_Firms": int(len(firm_level_plot_df)),
        }
    ]
)
display(spearman_firm_size_analyst_following_table)
OUTPUTS.save_table(
    spearman_firm_size_analyst_following_table,
    "firm_size_and_analyst_following_spearman_correlation",
)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(
    firm_level_plot_df["Market_Cap_Current"],
    firm_level_plot_df["Forecast_Analyst_Count"],
    s=10,
    alpha=0.4,
    color=ANALYST_FOLLOWING_COLOR,
)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
ax.set_xlabel(r"Market cap (\$ millions)")
ax.set_ylabel("Average analyst following")
ax.grid(alpha=0.25)

plt.tight_layout()
# Not retained in thesis2/Figures.
# OUTPUTS.save_figure(fig, "firm_size_and_analyst_following")
# plt.close(fig)

boxplot_df = all_earnings_events_before_filtering[
    ["Instrument", "Ann_Date", "Forecast_Analyst_Count"]
].copy()
boxplot_df["Formation_Year"] = boxplot_df["Ann_Date"].dt.year
boxplot_df["Forecast_Analyst_Count"] = pd.to_numeric(
    boxplot_df["Forecast_Analyst_Count"],
    errors="coerce",
)
boxplot_df = boxplot_df.merge(
    firm_size_df,
    on=["Instrument", "Formation_Year"],
    how="left",
)
boxplot_df = boxplot_df.dropna(subset=["Market_Cap_Current", "Forecast_Analyst_Count"])
boxplot_df = boxplot_df[
    (boxplot_df["Market_Cap_Current"] > 0)
    & (boxplot_df["Forecast_Analyst_Count"] > 0)
]
boxplot_df["Forecast_Analyst_Count"] = (
    pd.to_numeric(boxplot_df["Forecast_Analyst_Count"], errors="coerce")
    .round()
    .astype("Int64")
)
boxplot_df = boxplot_df.dropna(subset=["Forecast_Analyst_Count"])
boxplot_df["Forecast_Analyst_Count"] = boxplot_df["Forecast_Analyst_Count"].astype(int)
boxplot_df["Analyst_Following_Group"] = boxplot_df["Forecast_Analyst_Count"].clip(
    upper=15
)
analyst_group_values = sorted(boxplot_df["Analyst_Following_Group"].unique().tolist())
analyst_group_labels = [
    str(value) if value < 15 else "15+" for value in analyst_group_values
]
analyst_group_market_caps = [
    boxplot_df.loc[
        boxplot_df["Analyst_Following_Group"] == value, "Market_Cap_Current"
    ].values
    for value in analyst_group_values
]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.boxplot(
    analyst_group_market_caps,
    positions=analyst_group_values,
    widths=0.6,
    patch_artist=True,
    boxprops={"facecolor": MARKET_CAP_LIGHT_COLOR, "alpha": 0.8},
    medianprops={"color": MARKET_CAP_DARK_COLOR, "linewidth": 1.5},
    whiskerprops={"color": MARKET_CAP_DARK_COLOR},
    capprops={"color": MARKET_CAP_DARK_COLOR},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.25, "markeredgewidth": 0},
)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
ax.set_xticks(analyst_group_values)
ax.set_xticklabels(analyst_group_labels)
ax.set_xlabel("Analysts following")
ax.set_ylabel(r"Market cap (\$ millions)")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
OUTPUTS.save_figure(fig, "firm_size_by_analyst_following_boxplots")
# plt.close(fig)

plot_df = stock_universe[
    ["Instrument", "Name", "Formation_Year", "Market_Cap_Current", "BM"]
].copy()
plot_df = plot_df.dropna(subset=["Market_Cap_Current", "BM"])
plot_df = plot_df[(plot_df["Market_Cap_Current"] > 0) & (plot_df["BM"] >= 0)]

bm_upper_bound = plot_df["BM"].quantile(0.99)
n_outliers = (plot_df["BM"] > bm_upper_bound).sum()
plot_df = plot_df[plot_df["BM"] <= bm_upper_bound]

fig, ax = plt.subplots(figsize=(7, 4.2))

for year, year_data in plot_df.groupby("Formation_Year"):
    ax.scatter(
        year_data["Market_Cap_Current"],
        year_data["BM"],
        s=5,
        alpha=0.35,
        color=BOOK_TO_MARKET_COLOR,
        label=str(year),
    )

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
ax.set_xlabel(r"Market cap (\$ millions)")
ax.set_ylabel("Book-to-market ratio")
ax.grid(alpha=0.25)
ax.text(
    0.01,
    0.99,
    f"Omitted BM outliers: {n_outliers}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)
ax.legend(
    title="Formation year",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

plt.tight_layout()
# Not retained in thesis2/Figures.
# OUTPUTS.save_figure(fig, "company_size_and_book_to_market_ratio")
# plt.close(fig)


# -------------------------------------------------------------------
# Thesis benchmark-portfolio constituent diagnostics
# -------------------------------------------------------------------
# The descriptive-results chapter uses the 1991--2024 period, which is also
# the period partitioned into early, middle, and late attenuation periods.
BENCHMARK_PORTFOLIO_ORDER = ["SG", "SN", "SV", "BG", "BN", "BV"]
BENCHMARK_PORTFOLIO_LABELS = {
    "SG": "Small\nLow BM",
    "SN": "Small\nMedium BM",
    "SV": "Small\nHigh BM",
    "BG": "Big\nLow BM",
    "BN": "Big\nMedium BM",
    "BV": "Big\nHigh BM",
}
BENCHMARK_PERIODS = (
    (1991, 2002, "Early (1991--2002)"),
    (2003, 2013, "Middle (2003--2013)"),
    (2014, 2024, "Late (2014--2024)"),
)
BENCHMARK_PORTFOLIO_COLORS = {
    "SG": "#007C91",
    "SN": "#2A9D8F",
    "SV": "#8FC9C3",
    "BG": "#6F5C1E",
    "BN": "#C28E00",
    "BV": "#E9D8A6",
}


def benchmark_time_period(formation_year: int) -> str:
    """Assign a formation year to the thesis attenuation-period label."""
    for start_year, end_year, label in BENCHMARK_PERIODS:
        if start_year <= formation_year <= end_year:
            return label
    raise ValueError(f"Formation year {formation_year} is outside the thesis period range.")


required_benchmark_columns = {"Formation_Year", "Instrument", "Benchmark_Portfolio"}
missing_benchmark_columns = required_benchmark_columns.difference(stock_universe.columns)
if missing_benchmark_columns:
    raise KeyError(
        "The stock-universe snapshots are missing columns required for benchmark "
        f"constituent diagnostics: {sorted(missing_benchmark_columns)}."
    )

benchmark_universe = stock_universe.loc[:, list(required_benchmark_columns)].copy()
benchmark_universe["Formation_Year"] = pd.to_numeric(
    benchmark_universe["Formation_Year"], errors="coerce"
)
benchmark_universe["Benchmark_Portfolio"] = benchmark_universe[
    "Benchmark_Portfolio"
].astype("string").str.strip()
benchmark_universe = benchmark_universe.loc[
    benchmark_universe["Formation_Year"].between(1991, 2024)
    & benchmark_universe["Benchmark_Portfolio"].isin(BENCHMARK_PORTFOLIO_ORDER)
].copy()
benchmark_universe["Formation_Year"] = benchmark_universe["Formation_Year"].astype(int)

if benchmark_universe.empty:
    raise ValueError("No 1991--2024 benchmark-portfolio assignments are available.")

benchmark_years = list(range(1991, 2025))
benchmark_portfolio_counts = (
    benchmark_universe.groupby(["Formation_Year", "Benchmark_Portfolio"], observed=True)[
        "Instrument"
    ]
    .nunique()
    .unstack(fill_value=0)
    .reindex(index=benchmark_years, columns=BENCHMARK_PORTFOLIO_ORDER, fill_value=0)
    .rename_axis(index="Formation_Year", columns="Benchmark_Portfolio")
    .reset_index()
    .melt(
        id_vars="Formation_Year",
        var_name="Benchmark_Portfolio",
        value_name="Constituent_Count",
    )
)
benchmark_portfolio_counts["Portfolio_Label"] = benchmark_portfolio_counts[
    "Benchmark_Portfolio"
].map(BENCHMARK_PORTFOLIO_LABELS)
benchmark_portfolio_counts["Time_Period"] = benchmark_portfolio_counts[
    "Formation_Year"
].map(benchmark_time_period)
benchmark_portfolio_counts = benchmark_portfolio_counts[
    [
        "Formation_Year",
        "Time_Period",
        "Benchmark_Portfolio",
        "Portfolio_Label",
        "Constituent_Count",
    ]
]
OUTPUTS.save_table(benchmark_portfolio_counts, "benchmark_portfolio_constituent_counts")

benchmark_portfolio_count_summary = (
    benchmark_portfolio_counts.groupby(["Benchmark_Portfolio", "Portfolio_Label"], observed=True)[
        "Constituent_Count"
    ]
    .agg(Mean_Annual_Count="mean", Minimum_Annual_Count="min", Maximum_Annual_Count="max")
    .reindex(BENCHMARK_PORTFOLIO_ORDER, level="Benchmark_Portfolio")
    .reset_index()
)
OUTPUTS.save_table(
    benchmark_portfolio_count_summary,
    "benchmark_portfolio_constituent_count_summary",
)

# The x-axis identifies the portfolio, so each portfolio's range, annual
# observations, and mean use that portfolio's dedicated colour.
portfolio_positions = {
    portfolio: position
    for position, portfolio in enumerate(BENCHMARK_PORTFOLIO_ORDER)
}

fig, ax = plt.subplots(figsize=(7, 4.2))
for portfolio in BENCHMARK_PORTFOLIO_ORDER:
    portfolio_data = benchmark_portfolio_counts.loc[
        benchmark_portfolio_counts["Benchmark_Portfolio"] == portfolio
    ]
    position = portfolio_positions[portfolio]
    minimum = portfolio_data["Constituent_Count"].min()
    maximum = portfolio_data["Constituent_Count"].max()
    mean = portfolio_data["Constituent_Count"].mean()
    portfolio_color = BENCHMARK_PORTFOLIO_COLORS[portfolio]

    ax.vlines(position, minimum, maximum, color=portfolio_color, linewidth=2.0, alpha=0.5, zorder=1)
    ax.scatter(
        np.full(portfolio_data.shape[0], position),
        portfolio_data["Constituent_Count"],
        s=28,
        color=portfolio_color,
        alpha=0.85,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        position,
        mean,
        s=72,
        color=portfolio_color,
        edgecolors="black",
        marker="D",
        zorder=3,
    )
    # Label each portfolio's mean annual constituent count. Big-portfolio
    # labels sit to the left of their mean marker and Small-portfolio labels
    # to the right, keeping the labels clear within each size group.
    is_big_portfolio = portfolio.startswith("B")
    ax.annotate(
        f"{mean:.1f}",
        xy=(position, mean),
        xytext=(-7 if is_big_portfolio else 7, 0),
        textcoords="offset points",
        ha="right" if is_big_portfolio else "left",
        va="center",
        fontsize=8,
        fontweight="semibold",
        color="black",
        zorder=4,
    )

ax.set_xlabel("Portfolio")
ax.set_ylabel("Constituent securities")
ax.set_xticks(range(len(BENCHMARK_PORTFOLIO_ORDER)))
ax.set_xticklabels(
    [BENCHMARK_PORTFOLIO_LABELS[portfolio] for portfolio in BENCHMARK_PORTFOLIO_ORDER]
)
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.grid(axis="y", which="major", alpha=0.35)
ax.grid(axis="y", which="minor", alpha=0.2)
ax.grid(axis="x", visible=False)
fig.tight_layout()
OUTPUTS.save_figure(fig, "benchmark_portfolio_constituent_dot_range_plot")
