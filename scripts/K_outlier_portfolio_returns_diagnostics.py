from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

for import_root in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from analysis._analysis_shared import AnalysisOutputManager
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR, resolve_yearly_data_dir
from src.core.year_context import build_year_context


DATA_DIR = PROJECT_DATA_DIR
YEARLY_DATA_DIR = resolve_yearly_data_dir(DATA_DIR)
OUTPUTS = AnalysisOutputManager(__file__)

BENCHMARK_DAY_THRESHOLD_PCT = 10.0
CONSTITUENT_DAY_THRESHOLD_PCT = 50.0
PORTFOLIO_ORDER = [
    f"S_Q{size_bucket}-BM_Q{bm_bucket}"
    for size_bucket in range(1, 6)
    for bm_bucket in range(1, 6)
]


def load_benchmark_return_frames() -> list[tuple[int, pd.DataFrame]]:
    benchmark_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/benchmark_portfolio_returns.csv")
    )
    if not benchmark_files:
        raise FileNotFoundError(
            "No yearly benchmark_portfolio_returns.csv files found under data/yearly/<year>/. "
            "Run the benchmark pipeline first."
        )

    frames: list[tuple[int, pd.DataFrame]] = []
    for path in benchmark_files:
        formation_year = int(path.parent.name)
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame["Formation_Year"] = formation_year
        frames.append((formation_year, frame))
    return frames


def build_flagged_daily_return_table(
    frames: list[tuple[int, pd.DataFrame]],
    threshold_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for formation_year, frame in frames:
        for portfolio in PORTFOLIO_ORDER:
            if portfolio not in frame.columns:
                continue

            daily_returns = pd.to_numeric(frame[portfolio], errors="coerce")
            flagged = frame.loc[
                daily_returns.abs() > threshold_pct,
                ["Date"],
            ].copy()
            if flagged.empty:
                continue

            flagged["Daily_Benchmark_Return"] = daily_returns.loc[flagged.index].astype(float)
            flagged["Absolute_Daily_Benchmark_Return"] = (
                flagged["Daily_Benchmark_Return"].abs()
            )
            flagged["Formation_Year"] = formation_year
            flagged["Benchmark_Portfolio"] = portfolio
            flagged["Benchmark_Label"] = f"{formation_year} | {portfolio}"
            rows.extend(flagged.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "Formation_Year",
                "Benchmark_Portfolio",
                "Benchmark_Label",
                "Date",
                "Daily_Benchmark_Return",
                "Absolute_Daily_Benchmark_Return",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["Absolute_Daily_Benchmark_Return", "Formation_Year", "Benchmark_Portfolio", "Date"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def load_year_cache(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = YEARLY_DATA_DIR / str(year) / "_cache"
    constituents_path = cache_dir / "benchmark_portfolio_constituents.csv"
    return_windows_path = cache_dir / "benchmark_return_windows.csv"
    if not constituents_path.exists():
        raise FileNotFoundError(f"Missing benchmark constituents file: {constituents_path}")
    if not return_windows_path.exists():
        raise FileNotFoundError(f"Missing benchmark return windows file: {return_windows_path}")

    constituents = pd.read_csv(constituents_path)
    returns = pd.read_csv(return_windows_path, parse_dates=["Date"])
    constituents["Benchmark_Weight"] = pd.to_numeric(
        constituents["Benchmark_Weight"], errors="coerce"
    )
    returns["TotalReturn"] = pd.to_numeric(returns["TotalReturn"], errors="coerce")
    return constituents, returns


def load_stock_universe_market_caps(year: int) -> pd.DataFrame:
    year_context = build_year_context(year, DATA_DIR)
    if not year_context.stock_universe_path.exists():
        return pd.DataFrame(columns=["Instrument", "Market_Cap_Current"])

    stock_universe = pd.read_csv(year_context.stock_universe_path)
    if "Instrument" not in stock_universe.columns:
        return pd.DataFrame(columns=["Instrument", "Market_Cap_Current"])

    market_cap_column = next(
        (
            column
            for column in [
                "Market_Cap_Current",
                "MarketCap_Current",
                "Market_Cap",
                "MarketCap",
            ]
            if column in stock_universe.columns
        ),
        None,
    )
    if market_cap_column is None:
        return pd.DataFrame(columns=["Instrument", "Market_Cap_Current"])

    market_caps = stock_universe.loc[:, ["Instrument", market_cap_column]].copy()
    market_caps["Instrument"] = market_caps["Instrument"].astype("string").str.strip()
    market_caps["Market_Cap_Current"] = pd.to_numeric(
        market_caps[market_cap_column], errors="coerce"
    )
    market_caps = market_caps.drop(columns=[market_cap_column], errors="ignore")
    market_caps = market_caps.dropna(subset=["Instrument"]).drop_duplicates(
        subset=["Instrument"], keep="first"
    )
    return market_caps.reset_index(drop=True)


def build_portfolio_summary(flagged_daily_returns: pd.DataFrame) -> pd.DataFrame:
    if flagged_daily_returns.empty:
        return pd.DataFrame(
            columns=[
                "Formation_Year",
                "Benchmark_Portfolio",
                "Benchmark_Label",
                "Extreme_Day_Count",
                "First_Extreme_Date",
                "Last_Extreme_Date",
                "Max_Abs_Daily_Return",
                "Max_Positive_Daily_Return",
                "Min_Daily_Return",
            ]
        )

    summary = (
        flagged_daily_returns.groupby(
            ["Formation_Year", "Benchmark_Portfolio", "Benchmark_Label"],
            as_index=False,
        )
        .agg(
            Extreme_Day_Count=("Date", "count"),
            First_Extreme_Date=("Date", "min"),
            Last_Extreme_Date=("Date", "max"),
            Max_Abs_Daily_Return=("Absolute_Daily_Benchmark_Return", "max"),
            Max_Positive_Daily_Return=("Daily_Benchmark_Return", "max"),
            Min_Daily_Return=("Daily_Benchmark_Return", "min"),
        )
        .sort_values(
            ["Max_Abs_Daily_Return", "Extreme_Day_Count", "Formation_Year", "Benchmark_Portfolio"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    return summary


def build_driver_decomposition(
    constituent_details: pd.DataFrame,
) -> pd.DataFrame:
    if constituent_details.empty:
        return pd.DataFrame(
            columns=[
                "Formation_Year",
                "Benchmark_Portfolio",
                "Benchmark_Label",
                "Date",
                "Daily_Benchmark_Return",
                "Extreme_Constituent_Count",
                "Top_Contributor_Instrument",
                "Top_Contributor_Name",
                "Top_Contributor_Return",
                "Top_Contributor_Weight",
                "Top_Contributor_Weighted_Return",
                "Sum_Weighted_Extreme_Returns",
            ]
        )

    working = constituent_details.copy()
    working["Weighted_TotalReturn"] = (
        pd.to_numeric(working["Benchmark_Weight"], errors="coerce")
        * pd.to_numeric(working["TotalReturn"], errors="coerce")
    )
    working = working.sort_values(
        [
            "Formation_Year",
            "Benchmark_Portfolio",
            "Date",
            "Absolute_TotalReturn",
        ],
        ascending=[True, True, True, False],
    )

    top_driver = (
        working.groupby(["Formation_Year", "Benchmark_Portfolio", "Date"], as_index=False)
        .first()
        .rename(
            columns={
                "Instrument": "Top_Contributor_Instrument",
                "Name": "Top_Contributor_Name",
                "TotalReturn": "Top_Contributor_Return",
                "Benchmark_Weight": "Top_Contributor_Weight",
                "Weighted_TotalReturn": "Top_Contributor_Weighted_Return",
            }
        )
    )

    aggregate = (
        working.groupby(["Formation_Year", "Benchmark_Portfolio", "Date"], as_index=False)
        .agg(
            Extreme_Constituent_Count=("Instrument", "count"),
            Sum_Weighted_Extreme_Returns=("Weighted_TotalReturn", "sum"),
        )
    )

    merged = aggregate.merge(
        top_driver[
            [
                "Formation_Year",
                "Benchmark_Portfolio",
                "Date",
                "Benchmark_Label",
                "Daily_Benchmark_Return",
                "Top_Contributor_Instrument",
                "Top_Contributor_Name",
                "Top_Contributor_Return",
                "Top_Contributor_Weight",
                "Top_Contributor_Weighted_Return",
            ]
        ],
        on=["Formation_Year", "Benchmark_Portfolio", "Date"],
        how="left",
        validate="1:1",
    )
    return merged.sort_values(
        ["Extreme_Constituent_Count", "Top_Contributor_Weighted_Return"],
        ascending=[False, False],
    ).reset_index(drop=True)


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


def build_constituent_extreme_tables(
    flagged_benchmark_days: pd.DataFrame,
    constituent_threshold_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if flagged_benchmark_days.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "Formation_Year",
                "Benchmark_Portfolio",
                "Benchmark_Label",
                "Date",
                "Daily_Benchmark_Return",
                "Constituent_Count_In_Portfolio",
                "Constituent_Returns_Observed",
                "Extreme_Constituent_Count",
                "Max_Abs_Constituent_Return",
                "Has_Over_Threshold_Constituent_Move",
            ]
        )
        empty_details = pd.DataFrame(
            columns=[
                "Formation_Year",
                "Benchmark_Portfolio",
                "Benchmark_Label",
                "Date",
                "Daily_Benchmark_Return",
                "Instrument",
                "Name",
                "Exchange_Code",
                "gvkey",
                "Market_Cap_Current",
                "Benchmark_Weight",
                "TotalReturn",
                "Absolute_TotalReturn",
            ]
        )
        return empty_summary, empty_details

    year_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for row in flagged_benchmark_days.itertuples(index=False):
        formation_year = int(row.Formation_Year)
        if formation_year not in year_cache:
            constituents, returns = load_year_cache(formation_year)
            stock_universe_market_caps = load_stock_universe_market_caps(formation_year)
            year_cache[formation_year] = (constituents, returns, stock_universe_market_caps)

        constituents, returns, stock_universe_market_caps = year_cache[formation_year]
        portfolio_constituents = constituents.loc[
            constituents["Benchmark_Portfolio"] == row.Benchmark_Portfolio
        ].copy()
        portfolio_constituents = portfolio_constituents.merge(
            stock_universe_market_caps,
            on="Instrument",
            how="left",
            suffixes=("", "_StockUniverse"),
        )
        if "Market_Cap_Current_StockUniverse" in portfolio_constituents.columns:
            portfolio_constituents["Market_Cap_Current"] = portfolio_constituents[
                "Market_Cap_Current_StockUniverse"
            ].combine_first(pd.to_numeric(portfolio_constituents.get("Market_Cap_Current"), errors="coerce"))
            portfolio_constituents = portfolio_constituents.drop(
                columns=["Market_Cap_Current_StockUniverse"], errors="ignore"
            )
        portfolio_returns = returns.loc[
            returns["Date"] == row.Date, ["Instrument", "TotalReturn"]
        ]

        merged = portfolio_constituents.merge(
            portfolio_returns,
            on="Instrument",
            how="left",
        )
        merged["Absolute_TotalReturn"] = merged["TotalReturn"].abs()

        extreme_constituents = merged.loc[
            merged["Absolute_TotalReturn"] >= constituent_threshold_pct
        ].copy()
        if not extreme_constituents.empty:
            extreme_constituents["Formation_Year"] = formation_year
            extreme_constituents["Benchmark_Portfolio"] = row.Benchmark_Portfolio
            extreme_constituents["Benchmark_Label"] = row.Benchmark_Label
            extreme_constituents["Date"] = row.Date
            extreme_constituents["Daily_Benchmark_Return"] = row.Daily_Benchmark_Return
            detail_rows.extend(
                extreme_constituents[
                    [
                        "Formation_Year",
                        "Benchmark_Portfolio",
                        "Benchmark_Label",
                        "Date",
                        "Daily_Benchmark_Return",
                        "Instrument",
                        "Name",
                        "Exchange_Code",
                        "gvkey",
                        "Market_Cap_Current",
                        "Benchmark_Weight",
                        "TotalReturn",
                        "Absolute_TotalReturn",
                    ]
                ].sort_values("Absolute_TotalReturn", ascending=False).to_dict(orient="records")
            )

        summary_rows.append(
            {
                "Formation_Year": formation_year,
                "Benchmark_Portfolio": row.Benchmark_Portfolio,
                "Benchmark_Label": row.Benchmark_Label,
                "Date": row.Date,
                "Daily_Benchmark_Return": row.Daily_Benchmark_Return,
                "Constituent_Count_In_Portfolio": int(len(portfolio_constituents)),
                "Constituent_Returns_Observed": int(merged["TotalReturn"].notna().sum()),
                "Extreme_Constituent_Count": int(len(extreme_constituents)),
                "Max_Abs_Constituent_Return": float(merged["Absolute_TotalReturn"].max())
                if merged["Absolute_TotalReturn"].notna().any()
                else None,
                "Has_Over_Threshold_Constituent_Move": bool(len(extreme_constituents) > 0),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "Has_Over_Threshold_Constituent_Move",
            "Extreme_Constituent_Count",
            "Max_Abs_Constituent_Return",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    details = pd.DataFrame(detail_rows)
    if not details.empty:
        details = details.sort_values(
            ["Absolute_TotalReturn", "Formation_Year", "Benchmark_Portfolio", "Date"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
    return summary, details


def plot_constituent_outlier_returns_vs_market_cap(
    constituent_details: pd.DataFrame,
) -> Path | None:
    if constituent_details.empty:
        return None

    plot_frame = constituent_details.copy()
    plot_frame["Market_Cap_Current"] = pd.to_numeric(
        plot_frame["Market_Cap_Current"], errors="coerce"
    )
    plot_frame["TotalReturn"] = pd.to_numeric(plot_frame["TotalReturn"], errors="coerce")
    plot_frame = plot_frame.loc[
        plot_frame["Market_Cap_Current"].notna()
        & plot_frame["Market_Cap_Current"].gt(0)
        & plot_frame["TotalReturn"].notna()
        & plot_frame["TotalReturn"].gt(0)
    ].copy()
    if plot_frame.empty:
        return None

    portfolios = sorted(plot_frame["Benchmark_Portfolio"].dropna().astype(str).unique().tolist())
    cmap = plt.cm.get_cmap("tab20", max(len(portfolios), 1))
    color_map = {portfolio: cmap(index) for index, portfolio in enumerate(portfolios)}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for portfolio in portfolios:
        subset = plot_frame.loc[plot_frame["Benchmark_Portfolio"].astype(str) == portfolio]
        ax.scatter(
            subset["Market_Cap_Current"],
            subset["TotalReturn"],
            s=42,
            alpha=0.8,
            color=color_map[portfolio],
            edgecolors="none",
            label=portfolio,
        )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["Market_Cap_Current"].dropna().astype(float)
        y_values = plot_frame["TotalReturn"].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Market cap (USD mn)")
    ax.set_ylabel("Constituent daily return (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(
        title="Benchmark portfolio",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    fig.tight_layout()

    output_path = OUTPUTS.get_output_dir() / "constituent_outlier_returns_vs_market_cap.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


benchmark_frames = load_benchmark_return_frames()
flagged_daily_returns = build_flagged_daily_return_table(
    benchmark_frames,
    threshold_pct=BENCHMARK_DAY_THRESHOLD_PCT,
)
portfolio_summary = build_portfolio_summary(flagged_daily_returns)
constituent_summary, constituent_details = build_constituent_extreme_tables(
    flagged_daily_returns,
    constituent_threshold_pct=CONSTITUENT_DAY_THRESHOLD_PCT,
)
driver_decomposition = build_driver_decomposition(constituent_details)
market_cap_plot_path = plot_constituent_outlier_returns_vs_market_cap(constituent_details)

portfolio_summary_path = OUTPUTS.save_table(
    portfolio_summary,
    (
        "benchmark_portfolios_with_daily_returns_exceeding_"
        f"{int(BENCHMARK_DAY_THRESHOLD_PCT)}pct"
    ),
)
flagged_daily_returns_path = OUTPUTS.save_table(
    flagged_daily_returns,
    (
        "flagged_daily_benchmark_returns_exceeding_"
        f"{int(BENCHMARK_DAY_THRESHOLD_PCT)}pct"
    ),
)
constituent_summary_path = OUTPUTS.save_table(
    constituent_summary,
    (
        "flagged_benchmark_days_with_constituent_"
        f"returns_exceeding_{int(CONSTITUENT_DAY_THRESHOLD_PCT)}pct"
    ),
)
constituent_details_path = OUTPUTS.save_table(
    constituent_details,
    (
        "constituents_exceeding_"
        f"{int(CONSTITUENT_DAY_THRESHOLD_PCT)}pct_on_flagged_benchmark_days"
    ),
)
driver_decomposition_path = OUTPUTS.save_table(
    driver_decomposition,
    (
        "benchmark_day_driver_decomposition_for_constituents_exceeding_"
        f"{int(CONSTITUENT_DAY_THRESHOLD_PCT)}pct"
    ),
)

if flagged_daily_returns.empty:
    summary_text = (
        f"No daily benchmark portfolio returns exceeded +/-{BENCHMARK_DAY_THRESHOLD_PCT:.0f}% "
        f"under {YEARLY_DATA_DIR}."
    )
else:
    worst = flagged_daily_returns.iloc[0]
    with_extreme_constituent = constituent_summary.loc[
        constituent_summary["Has_Over_Threshold_Constituent_Move"]
    ]
    summary_text = (
        f"Found {len(flagged_daily_returns)} flagged benchmark-day observations above "
        f"+/-{BENCHMARK_DAY_THRESHOLD_PCT:.0f}% across {portfolio_summary.shape[0]} benchmark-year portfolios.\n"
        f"Largest move: {worst['Benchmark_Label']} on {worst['Date'].date()} = "
        f"{worst['Daily_Benchmark_Return']:.2f}%.\n"
        f"{len(with_extreme_constituent)} flagged benchmark-day observations had at least one "
        f"constituent with |TotalReturn| >= {CONSTITUENT_DAY_THRESHOLD_PCT:.0f}%."
    )

OUTPUTS.save_text("summary", summary_text)
print(f"Saved portfolio summary to {portfolio_summary_path}")
print(f"Saved flagged benchmark-day details to {flagged_daily_returns_path}")
print(f"Saved constituent summary to {constituent_summary_path}")
print(f"Saved constituent details to {constituent_details_path}")
print(f"Saved driver decomposition to {driver_decomposition_path}")
if market_cap_plot_path is not None:
    print(f"Saved market-cap scatter plot to {market_cap_plot_path}")
