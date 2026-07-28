from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis._analysis_shared import AnalysisOutputManager
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR, resolve_yearly_data_dir
from src.core.year_context import build_year_context
from src.pead.market_data_fetch import read_price_window_cache


DATA_DIR = PROJECT_DATA_DIR
YEARLY_DATA_DIR = resolve_yearly_data_dir(DATA_DIR)
OUTPUTS = AnalysisOutputManager(__file__)

EXTREME_RETURN_THRESHOLD_PCT = 50.0
ACCURACY_TOLERANCE_PCT_POINTS = 20.0
SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT = 20.0
LONG_SPIKE_MAX_TRADING_DAY_GAP = 5
WINDOW_SEGMENTS: list[tuple[str, int, int]] = [
    ("0-1", 0, 1),
    ("2-20", 2, 20),
    ("21-30", 21, 30),
    ("31-60", 31, 60),
    ("61-90", 61, 90),
]
SEGMENT_COLOR_MAP = {
    "0-1": "#c44e52",
    "2-20": "#dd8452",
    "21-30": "#55a868",
    "31-60": "#4c72b0",
    "61-90": "#8172b3",
}


def _empty_error_flag_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Spike_Error_Candidate"] = False
    out["Spike_Error_Confirmed"] = False
    out["Spike_Error_Pair_Id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["Spike_Error_Role"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["Spike_Error_Trading_Day_Gap"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["Spike_Error_Adjusted_Constant_Price"] = np.nan
    out["Spike_Error_Adjusted_First_Return"] = np.nan
    out["Spike_Error_Adjusted_Second_Return"] = np.nan
    out["Long_Spike_Error_Candidate"] = False
    out["Long_Spike_Error_Confirmed"] = False
    out["Long_Spike_Error_Pair_Id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["Long_Spike_Error_Role"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["Long_Spike_Error_Trading_Day_Gap"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["Long_Spike_Error_Adjusted_Constant_Price"] = np.nan
    out["Long_Spike_Error_Adjusted_First_Return"] = np.nan
    out["Long_Spike_Error_Adjusted_Second_Return"] = np.nan
    return out


def build_price_return_table(price_windows: pd.DataFrame) -> pd.DataFrame:
    if price_windows.empty:
        return pd.DataFrame(
            columns=[
                "Instrument",
                "Trading_Date",
                "Instrument_First_Price_Date",
                "Previous_PriceClose",
                "Current_PriceClose",
                "Price_Return_From_Close",
            ]
        )

    work = price_windows.loc[:, ["Instrument", "Date", "PriceClose"]].copy()
    work["Instrument"] = work["Instrument"].astype("string").str.strip()
    work["Trading_Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["PriceClose"] = pd.to_numeric(work["PriceClose"], errors="coerce")
    work = work.drop(columns=["Date"])
    work = work.dropna(subset=["Instrument", "Trading_Date", "PriceClose"]).copy()
    work = work.sort_values(["Instrument", "Trading_Date"], kind="stable")
    work["Instrument_First_Price_Date"] = work.groupby("Instrument")["Trading_Date"].transform("min")
    work["Previous_PriceClose"] = work.groupby("Instrument")["PriceClose"].shift(1)
    work["Current_PriceClose"] = work["PriceClose"]
    work["Price_Return_From_Close"] = (
        work["Current_PriceClose"] / work["Previous_PriceClose"] - 1.0
    ) * 100.0
    return work.loc[
        :,
        [
            "Instrument",
            "Trading_Date",
            "Instrument_First_Price_Date",
            "Previous_PriceClose",
            "Current_PriceClose",
            "Price_Return_From_Close",
        ],
    ].reset_index(drop=True)


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


def classify_window_segment(relative_day: object) -> str | pd.NA:
    day = pd.to_numeric(relative_day, errors="coerce")
    if pd.isna(day):
        return pd.NA
    day_int = int(day)
    for label, start, end in WINDOW_SEGMENTS:
        if start <= day_int <= end:
            return label
    return pd.NA


def load_final_event_window_panel() -> pd.DataFrame:
    panel_frames: list[pd.DataFrame] = []
    year_dirs = sorted(YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]"))

    for year_dir in tqdm(year_dirs, desc="Loading kept BHAR event windows", unit="year"):
        year = int(year_dir.name)
        year_context = build_year_context(year, DATA_DIR)
        if (
            not year_context.earnings_abnormal_returns_path.exists()
            or not year_context.price_windows_path.exists()
        ):
            continue

        abnormal_returns = pd.read_csv(year_context.earnings_abnormal_returns_path)
        required_columns = {
            "Event_ID",
            "Instrument",
            "Trading_Date",
            "Relative_Day",
            "Security_Return",
            "Benchmark_Portfolio",
        }
        if not required_columns.issubset(abnormal_returns.columns):
            continue

        panel = abnormal_returns.loc[
            :,
            [
                "Event_ID",
                "Instrument",
                "Trading_Date",
                "Relative_Day",
                "Security_Return",
                "Benchmark_Portfolio",
            ],
        ].copy()
        panel["Instrument"] = panel["Instrument"].astype("string").str.strip()
        panel["Trading_Date"] = pd.to_datetime(panel["Trading_Date"], errors="coerce")
        panel["Relative_Day"] = pd.to_numeric(panel["Relative_Day"], errors="coerce")
        panel["TotalReturn"] = pd.to_numeric(panel["Security_Return"], errors="coerce")
        panel["Benchmark_Portfolio"] = panel["Benchmark_Portfolio"].astype("string").str.strip()
        panel = panel.drop(columns=["Security_Return"])
        panel = panel.loc[
            panel["Instrument"].notna()
            & panel["Trading_Date"].notna()
            & panel["Relative_Day"].notna()
            & panel["TotalReturn"].notna()
            & panel["Relative_Day"].between(0, 90, inclusive="both")
        ].copy()
        if panel.empty:
            continue

        price_windows = read_price_window_cache(year_context.price_windows_path)
        price_return_table = build_price_return_table(price_windows)
        stock_universe_market_caps = load_stock_universe_market_caps(year)

        panel = panel.merge(
            price_return_table,
            on=["Instrument", "Trading_Date"],
            how="left",
        )
        panel = panel.merge(
            stock_universe_market_caps,
            on="Instrument",
            how="left",
            suffixes=("", "_StockUniverse"),
        )
        if "Market_Cap_Current_StockUniverse" in panel.columns:
            existing_market_cap = (
                pd.to_numeric(panel["Market_Cap_Current"], errors="coerce")
                if "Market_Cap_Current" in panel.columns
                else pd.Series(np.nan, index=panel.index, dtype="float64")
            )
            panel["Market_Cap_Current"] = pd.to_numeric(
                panel["Market_Cap_Current_StockUniverse"], errors="coerce"
            ).combine_first(existing_market_cap)
            panel = panel.drop(columns=["Market_Cap_Current_StockUniverse"], errors="ignore")
        panel["Formation_Year"] = year
        panel["Absolute_Total_Return"] = panel["TotalReturn"].abs()
        panel["Return_Error"] = panel["Price_Return_From_Close"] - panel["TotalReturn"]
        panel["Absolute_Return_Error"] = panel["Return_Error"].abs()
        panel["Price_Return_Available"] = panel["Price_Return_From_Close"].notna()
        panel["Ignore_For_Validation"] = (
            panel["Current_PriceClose"].notna()
            & panel["Previous_PriceClose"].isna()
            & (panel["Trading_Date"] == panel["Instrument_First_Price_Date"])
        )
        panel["Validation_Status"] = np.where(
            ~panel["Price_Return_Available"],
            "missing_price_return",
            np.where(
                panel["Ignore_For_Validation"]
                | (panel["Absolute_Return_Error"] <= ACCURACY_TOLERANCE_PCT_POINTS),
                "accurate",
                "inaccurate",
            ),
        )
        panel["Window_Segment"] = panel["Relative_Day"].apply(classify_window_segment)
        panel_frames.append(panel)

    if not panel_frames:
        return pd.DataFrame()

    return pd.concat(panel_frames, ignore_index=True).sort_values(
        ["Formation_Year", "Event_ID", "Relative_Day", "Trading_Date"],
        kind="stable",
    ).reset_index(drop=True)


def select_outlier_stock_returns(event_window_panel: pd.DataFrame) -> pd.DataFrame:
    if event_window_panel.empty:
        return _empty_error_flag_columns(event_window_panel.copy())

    selected = event_window_panel.loc[
        event_window_panel["Absolute_Total_Return"] >= EXTREME_RETURN_THRESHOLD_PCT
    ].copy()
    if selected.empty:
        return _empty_error_flag_columns(selected)

    selected = _empty_error_flag_columns(selected)
    selected["Selection_Method"] = "absolute_threshold"
    selected["Selection_Value"] = EXTREME_RETURN_THRESHOLD_PCT
    selected["Selection_Threshold"] = EXTREME_RETURN_THRESHOLD_PCT
    selected["Selected_Rank"] = (
        selected["Absolute_Total_Return"].rank(method="first", ascending=False).astype(int)
    )
    return selected.sort_values(
        ["Absolute_Total_Return", "Formation_Year", "Event_ID", "Relative_Day", "Trading_Date"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def _find_constant_price_solution(
    previous_price: float,
    next_price: float,
) -> dict[str, float] | None:
    if (
        not np.isfinite(previous_price)
        or not np.isfinite(next_price)
        or previous_price <= 0
        or next_price <= 0
    ):
        return None

    lower_bound = max(
        previous_price * (1.0 - SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT / 100.0),
        next_price / (1.0 + SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT / 100.0),
    )
    upper_bound = min(
        previous_price * (1.0 + SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT / 100.0),
        next_price / (1.0 - SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT / 100.0),
    )
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound > upper_bound:
        return None

    adjusted_price = (lower_bound + upper_bound) / 2.0
    adjusted_first_return = (adjusted_price / previous_price - 1.0) * 100.0
    adjusted_second_return = (next_price / adjusted_price - 1.0) * 100.0
    if (
        abs(adjusted_first_return) >= SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT
        or abs(adjusted_second_return) >= SPIKE_ERROR_MAX_ABSOLUTE_RETURN_PCT
    ):
        return None

    return {
        "adjusted_price": adjusted_price,
        "adjusted_first_return": adjusted_first_return,
        "adjusted_second_return": adjusted_second_return,
    }


def detect_error_sequences(
    details: pd.DataFrame,
    event_window_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if details.empty or event_window_panel.empty:
        return details.copy(), pd.DataFrame(), pd.DataFrame()

    updated = details.copy()
    detail_keys = (
        updated.loc[:, ["Formation_Year", "Event_ID", "Instrument", "Trading_Date", "Relative_Day"]]
        .copy()
    )
    detail_keys["Formation_Year"] = pd.to_numeric(detail_keys["Formation_Year"], errors="coerce").astype("Int64")
    detail_keys["Instrument"] = detail_keys["Instrument"].astype("string").str.strip()
    detail_keys["Trading_Date"] = pd.to_datetime(detail_keys["Trading_Date"], errors="coerce")
    detail_keys["Relative_Day"] = pd.to_numeric(detail_keys["Relative_Day"], errors="coerce").astype("Int64")
    updated["_detail_key"] = list(
        zip(
            detail_keys["Formation_Year"],
            detail_keys["Event_ID"],
            detail_keys["Instrument"],
            detail_keys["Trading_Date"],
            detail_keys["Relative_Day"],
        )
    )
    key_to_index = pd.Series(updated.index.to_numpy(), index=updated["_detail_key"]).to_dict()
    detail_key_set = set(updated["_detail_key"].tolist())

    spike_rows: list[dict[str, object]] = []
    long_spike_rows: list[dict[str, object]] = []

    grouped_panel = (
        event_window_panel.assign(
            Instrument=event_window_panel["Instrument"].astype("string").str.strip(),
            Trading_Date=pd.to_datetime(event_window_panel["Trading_Date"], errors="coerce"),
            Relative_Day=pd.to_numeric(event_window_panel["Relative_Day"], errors="coerce"),
        )
        .sort_values(
            ["Formation_Year", "Event_ID", "Instrument", "Relative_Day", "Trading_Date"],
            kind="stable",
        )
        .groupby(["Formation_Year", "Event_ID", "Instrument"], sort=True)
    )

    for (year, event_id, instrument), group in grouped_panel:
        work = group.reset_index(drop=True)
        keys = [
            (
                int(year),
                event_id,
                str(instrument).strip(),
                pd.Timestamp(date),
                int(relative_day),
            )
            for date, relative_day in zip(
                work["Trading_Date"].tolist(),
                work["Relative_Day"].tolist(),
                strict=False,
            )
        ]
        extreme_flags = [key in detail_key_set for key in keys]

        for idx in range(len(work) - 1):
            if not extreme_flags[idx]:
                continue
            first = work.iloc[idx]
            first_key = keys[idx]
            first_index = key_to_index.get(first_key)
            if first_index is None:
                continue
            first_return = float(first["TotalReturn"])
            if not np.isfinite(first_return) or first_return == 0:
                continue

            for gap in range(1, min(LONG_SPIKE_MAX_TRADING_DAY_GAP, len(work) - idx - 1) + 1):
                if not extreme_flags[idx + gap]:
                    continue
                second = work.iloc[idx + gap]
                second_return = float(second["TotalReturn"])
                if not np.isfinite(second_return) or second_return == 0:
                    continue
                if np.sign(first_return) == np.sign(second_return):
                    continue

                second_key = keys[idx + gap]
                second_index = key_to_index.get(second_key)
                if second_index is None:
                    continue

                solution = _find_constant_price_solution(
                    previous_price=float(first["Previous_PriceClose"]),
                    next_price=float(second["Current_PriceClose"]),
                )
                pair_id = (
                    f"{int(year)}::{event_id}::{instrument}::"
                    f"{int(first['Relative_Day'])}::{int(second['Relative_Day'])}"
                )
                pair_indices = [first_index, second_index]

                row_base = {
                    "Formation_Year": int(year),
                    "Event_ID": event_id,
                    "Instrument": str(instrument).strip(),
                    "First_Outlier_Date": pd.Timestamp(first["Trading_Date"]),
                    "Second_Outlier_Date": pd.Timestamp(second["Trading_Date"]),
                    "First_Relative_Day": int(first["Relative_Day"]),
                    "Second_Relative_Day": int(second["Relative_Day"]),
                    "Pair_Id": pair_id,
                    "Trading_Day_Gap": gap,
                    "Calendar_Gap_Days": int(
                        (pd.Timestamp(second["Trading_Date"]) - pd.Timestamp(first["Trading_Date"])).days
                    ),
                    "First_TotalReturn": first_return,
                    "Second_TotalReturn": second_return,
                    "Previous_PriceClose": (
                        float(first["Previous_PriceClose"]) if pd.notna(first["Previous_PriceClose"]) else np.nan
                    ),
                    "Observed_First_Outlier_PriceClose": (
                        float(first["Current_PriceClose"]) if pd.notna(first["Current_PriceClose"]) else np.nan
                    ),
                    "Observed_Second_Outlier_PriceClose": (
                        float(second["Current_PriceClose"]) if pd.notna(second["Current_PriceClose"]) else np.nan
                    ),
                    "Adjusted_Constant_Price": float(solution["adjusted_price"]) if solution is not None else np.nan,
                    "Adjusted_First_Return": float(solution["adjusted_first_return"]) if solution is not None else np.nan,
                    "Adjusted_Second_Return": float(solution["adjusted_second_return"]) if solution is not None else np.nan,
                }

                if gap == 1:
                    updated.loc[pair_indices, "Spike_Error_Candidate"] = True
                    updated.loc[pair_indices, "Spike_Error_Pair_Id"] = pair_id
                    updated.loc[pair_indices, "Spike_Error_Trading_Day_Gap"] = gap
                    updated.at[first_index, "Spike_Error_Role"] = "first_outlier_return"
                    updated.at[second_index, "Spike_Error_Role"] = "second_outlier_return"
                    confirmed = solution is not None
                    if confirmed:
                        updated.loc[pair_indices, "Spike_Error_Confirmed"] = True
                        updated.loc[pair_indices, "Spike_Error_Adjusted_Constant_Price"] = float(solution["adjusted_price"])
                        updated.at[first_index, "Spike_Error_Adjusted_First_Return"] = float(solution["adjusted_first_return"])
                        updated.at[second_index, "Spike_Error_Adjusted_Second_Return"] = float(solution["adjusted_second_return"])
                    spike_rows.append({**row_base, "Spike_Error_Confirmed": confirmed})
                else:
                    updated.loc[pair_indices, "Long_Spike_Error_Candidate"] = True
                    updated.loc[pair_indices, "Long_Spike_Error_Pair_Id"] = pair_id
                    updated.loc[pair_indices, "Long_Spike_Error_Trading_Day_Gap"] = gap
                    updated.at[first_index, "Long_Spike_Error_Role"] = "first_outlier_return"
                    updated.at[second_index, "Long_Spike_Error_Role"] = "second_outlier_return"
                    confirmed = solution is not None
                    if confirmed:
                        updated.loc[pair_indices, "Long_Spike_Error_Confirmed"] = True
                        updated.loc[pair_indices, "Long_Spike_Error_Adjusted_Constant_Price"] = float(solution["adjusted_price"])
                        updated.at[first_index, "Long_Spike_Error_Adjusted_First_Return"] = float(solution["adjusted_first_return"])
                        updated.at[second_index, "Long_Spike_Error_Adjusted_Second_Return"] = float(solution["adjusted_second_return"])
                    long_spike_rows.append({**row_base, "Long_Spike_Error_Confirmed": confirmed})

    spike_report = pd.DataFrame(spike_rows)
    if not spike_report.empty:
        spike_report = spike_report.sort_values(
            ["Spike_Error_Confirmed", "Formation_Year", "Event_ID", "Instrument", "First_Relative_Day"],
            ascending=[False, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

    long_spike_report = pd.DataFrame(long_spike_rows)
    if not long_spike_report.empty:
        long_spike_report = long_spike_report.sort_values(
            ["Long_Spike_Error_Confirmed", "Trading_Day_Gap", "Formation_Year", "Event_ID", "Instrument", "First_Relative_Day"],
            ascending=[False, True, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

    return updated.drop(columns="_detail_key"), spike_report, long_spike_report


def build_flagged_summary(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(
            columns=[
                "Flagged_Event_Window_Returns",
                "Formation_Years_Covered",
                "Unique_Events",
                "Unique_Instruments",
                "Inaccurate_Count",
                "Accurate_Count",
                "Missing_Price_Return_Count",
                "Spike_Error_Candidate_Count",
                "Spike_Error_Confirmed_Count",
                "Long_Spike_Error_Candidate_Count",
                "Long_Spike_Error_Confirmed_Count",
            ]
        )

    return pd.DataFrame(
        [
            {
                "Flagged_Event_Window_Returns": int(len(details)),
                "Formation_Years_Covered": int(details["Formation_Year"].nunique()),
                "Unique_Events": int(details["Event_ID"].nunique()),
                "Unique_Instruments": int(details["Instrument"].nunique()),
                "Inaccurate_Count": int((details["Validation_Status"] == "inaccurate").sum()),
                "Accurate_Count": int((details["Validation_Status"] == "accurate").sum()),
                "Missing_Price_Return_Count": int(
                    (details["Validation_Status"] == "missing_price_return").sum()
                ),
                "Spike_Error_Candidate_Count": int(details["Spike_Error_Candidate"].sum()),
                "Spike_Error_Confirmed_Count": int(details["Spike_Error_Confirmed"].sum()),
                "Long_Spike_Error_Candidate_Count": int(details["Long_Spike_Error_Candidate"].sum()),
                "Long_Spike_Error_Confirmed_Count": int(details["Long_Spike_Error_Confirmed"].sum()),
            }
        ]
    )


def build_window_segment_summary(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(
            columns=[
                "Window_Segment",
                "Outlier_Return_Count",
                "Share_Of_Outlier_Returns",
                "Unique_Events",
                "Unique_Instruments",
                "Max_Absolute_Return",
                "Median_Absolute_Return",
            ]
        )

    summary = (
        details.groupby("Window_Segment", dropna=False, as_index=False)
        .agg(
            Outlier_Return_Count=("Event_ID", "size"),
            Unique_Events=("Event_ID", "nunique"),
            Unique_Instruments=("Instrument", "nunique"),
            Max_Absolute_Return=("Absolute_Total_Return", "max"),
            Median_Absolute_Return=("Absolute_Total_Return", "median"),
        )
        .sort_values(
            "Window_Segment",
            key=lambda series: series.map(
                {label: index for index, (label, _, _) in enumerate(WINDOW_SEGMENTS)}
            ),
        )
        .reset_index(drop=True)
    )
    summary["Share_Of_Outlier_Returns"] = (
        summary["Outlier_Return_Count"] / float(len(details)) * 100.0
    )
    return summary


def build_driver_decomposition(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(
            columns=[
                "Formation_Year",
                "Event_ID",
                "Instrument",
                "Trading_Date",
                "Relative_Day",
                "Window_Segment",
                "TotalReturn",
                "Price_Return_From_Close",
                "Return_Error",
                "Absolute_Return_Error",
                "Validation_Status",
                "Driver_Label",
                "Spike_Error_Candidate",
                "Spike_Error_Confirmed",
                "Long_Spike_Error_Candidate",
                "Long_Spike_Error_Confirmed",
            ]
        )

    out = details.loc[
        :,
        [
            "Formation_Year",
            "Event_ID",
            "Instrument",
            "Trading_Date",
            "Relative_Day",
            "Window_Segment",
            "TotalReturn",
            "Price_Return_From_Close",
            "Return_Error",
            "Absolute_Return_Error",
            "Validation_Status",
            "Spike_Error_Candidate",
            "Spike_Error_Confirmed",
            "Long_Spike_Error_Candidate",
            "Long_Spike_Error_Confirmed",
        ],
    ].copy()
    out["Driver_Label"] = "price_supported_or_missing"
    out.loc[out["Spike_Error_Candidate"], "Driver_Label"] = "adjacent_spike_candidate"
    out.loc[out["Long_Spike_Error_Candidate"], "Driver_Label"] = "long_spike_candidate"
    out.loc[out["Spike_Error_Confirmed"], "Driver_Label"] = "adjacent_spike_confirmed"
    out.loc[out["Long_Spike_Error_Confirmed"], "Driver_Label"] = "long_spike_confirmed"
    out.loc[out["Validation_Status"] == "inaccurate", "Driver_Label"] = "reported_return_mismatch"
    out.loc[out["Spike_Error_Candidate"], "Driver_Label"] = "adjacent_spike_candidate"
    out.loc[out["Long_Spike_Error_Candidate"], "Driver_Label"] = "long_spike_candidate"
    out.loc[out["Spike_Error_Confirmed"], "Driver_Label"] = "adjacent_spike_confirmed"
    out.loc[out["Long_Spike_Error_Confirmed"], "Driver_Label"] = "long_spike_confirmed"
    out.loc[
        out["Validation_Status"] == "missing_price_return",
        "Driver_Label",
    ] = "missing_price_support"
    return out.sort_values(
        ["Absolute_Return_Error", "Formation_Year", "Event_ID", "Relative_Day", "Trading_Date"],
        ascending=[False, True, True, True, True],
        na_position="last",
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


def plot_reported_vs_price_return(details: pd.DataFrame) -> Path | None:
    if details.empty:
        return None

    plot_frame = details.loc[
        details["Price_Return_From_Close"].notna()
        & details["TotalReturn"].notna()
        & details["TotalReturn"].gt(0)
        & details["Window_Segment"].notna()
    ].copy()
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, _, _ in WINDOW_SEGMENTS:
        subset = plot_frame.loc[plot_frame["Window_Segment"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["TotalReturn"],
            subset["Price_Return_From_Close"],
            s=38,
            alpha=0.75,
            color=SEGMENT_COLOR_MAP[label],
            edgecolors="none",
            label=label,
        )

    combined = pd.concat(
        [plot_frame["TotalReturn"], plot_frame["Price_Return_From_Close"]],
        ignore_index=True,
    ).dropna()
    if not combined.empty:
        max_abs = float(combined.abs().max())
        if max_abs > 0:
            diagonal = pd.Series([-max_abs, max_abs], dtype="float64")
            ax.plot(diagonal, diagonal, linestyle="--", linewidth=1.2, color="black", alpha=0.8)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["TotalReturn"].dropna().astype(float)
        y_values = plot_frame["Price_Return_From_Close"].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Reported event-window stock return (%)")
    ax.set_ylabel("Price-implied event-window stock return (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(title="Window segment", frameon=False)
    fig.tight_layout()
    return OUTPUTS.save_figure(fig, "reported_vs_price_implied_return_scatter", dpi=180)


def plot_outlier_returns_vs_year_with_market_cap_size(details: pd.DataFrame) -> Path | None:
    if details.empty:
        return None

    plot_frame = details.copy()
    market_cap_column = next(
        (
            column
            for column in [
                "Market_Cap_Current",
                "Market_Cap_Current_StockUniverse",
                "MarketCap_Current",
                "Market_Cap",
                "MarketCap",
            ]
            if column in plot_frame.columns
        ),
        None,
    )
    if market_cap_column is None:
        return None

    plot_frame["Formation_Year"] = pd.to_numeric(plot_frame["Formation_Year"], errors="coerce")
    plot_frame["TotalReturn"] = pd.to_numeric(plot_frame["TotalReturn"], errors="coerce")
    plot_frame["Market_Cap_Current"] = pd.to_numeric(
        plot_frame[market_cap_column], errors="coerce"
    )
    plot_frame = plot_frame.loc[
        plot_frame["Formation_Year"].notna()
        & plot_frame["TotalReturn"].notna()
        & plot_frame["TotalReturn"].gt(0)
        & plot_frame["Market_Cap_Current"].notna()
        & plot_frame["Market_Cap_Current"].gt(0)
        & plot_frame["Window_Segment"].notna()
    ].copy()
    if plot_frame.empty:
        return None

    log_market_cap = np.log10(plot_frame["Market_Cap_Current"])
    cap_min = float(log_market_cap.min())
    cap_range = float(log_market_cap.max() - cap_min)
    if cap_range > 0:
        normalized = (log_market_cap - cap_min) / cap_range
        plot_frame["Marker_Size"] = 40.0 + normalized * 180.0
    else:
        plot_frame["Marker_Size"] = 100.0

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, _, _ in WINDOW_SEGMENTS:
        subset = plot_frame.loc[plot_frame["Window_Segment"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["Formation_Year"],
            subset["TotalReturn"],
            s=subset["Marker_Size"],
            alpha=0.65,
            color=SEGMENT_COLOR_MAP[label],
            edgecolors="none",
            label=label,
        )
    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["Formation_Year"].dropna().astype(float)
        y_values = plot_frame["TotalReturn"].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Formation year")
    ax.set_ylabel("Outlier event-window stock return (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(title="Window segment", frameon=False)
    fig.tight_layout()
    return OUTPUTS.save_figure(fig, "outlier_returns_vs_year_market_cap_bubbles", dpi=180)


def plot_outlier_returns_vs_relative_day(details: pd.DataFrame) -> Path | None:
    if details.empty:
        return None

    plot_frame = details.loc[
        details["Relative_Day"].notna()
        & details["TotalReturn"].notna()
        & details["TotalReturn"].gt(0)
        & details["Window_Segment"].notna()
    ].copy()
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, _, _ in WINDOW_SEGMENTS:
        subset = plot_frame.loc[plot_frame["Window_Segment"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["Relative_Day"],
            subset["TotalReturn"],
            s=38,
            alpha=0.75,
            color=SEGMENT_COLOR_MAP[label],
            edgecolors="none",
            label=label,
        )

    for separator in [1, 20, 30, 60]:
        ax.axvline(separator, linestyle="--", linewidth=1.0, color="black", alpha=0.5)

    ax.set_yscale("symlog", linthresh=1.0)
    if not plot_frame.empty:
        x_values = plot_frame["Relative_Day"].dropna().astype(float)
        y_values = plot_frame["TotalReturn"].dropna().astype(float)
        if not x_values.empty:
            ax.set_xlim(float(x_values.min()), float(x_values.max()))
        if not y_values.empty:
            ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_plain_number))
    ax.set_xlabel("Relative day in kept BHAR event window")
    ax.set_ylabel("Outlier stock return (%)")
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.legend(title="Window segment", frameon=False)
    fig.tight_layout()
    return OUTPUTS.save_figure(fig, "outlier_returns_vs_relative_day", dpi=180)


event_window_panel = load_final_event_window_panel()
flagged_details = select_outlier_stock_returns(event_window_panel)
flagged_details, spike_pair_report, long_spike_pair_report = detect_error_sequences(
    flagged_details,
    event_window_panel,
)
flagged_summary = build_flagged_summary(flagged_details)
window_segment_summary = build_window_segment_summary(flagged_details)
driver_decomposition = build_driver_decomposition(flagged_details)
scatter_path = plot_reported_vs_price_return(flagged_details)
year_bubble_plot_path = plot_outlier_returns_vs_year_with_market_cap_size(flagged_details)
relative_day_plot_path = plot_outlier_returns_vs_relative_day(flagged_details)

summary_path = OUTPUTS.save_table(flagged_summary, "flagged_stock_return_summary")
segment_summary_path = OUTPUTS.save_table(window_segment_summary, "window_segment_summary")
details_path = OUTPUTS.save_table(flagged_details, "flagged_stock_return_details")
drivers_path = OUTPUTS.save_table(driver_decomposition, "flagged_stock_return_driver_decomposition")
spike_pairs_path = OUTPUTS.save_table(spike_pair_report, "spike_error_pair_report")
long_spike_pairs_path = OUTPUTS.save_table(long_spike_pair_report, "long_spike_error_pair_report")

if flagged_details.empty:
    summary_text = (
        f"No kept-BHAR event-window stock returns exceeded |return| >= "
        f"{EXTREME_RETURN_THRESHOLD_PCT:.0f}% under {YEARLY_DATA_DIR}."
    )
else:
    summary_text = (
        f"Found {len(flagged_details)} flagged kept-event return observations with "
        f"|TotalReturn| >= {EXTREME_RETURN_THRESHOLD_PCT:.0f}% across "
        f"{flagged_details['Event_ID'].nunique()} events and "
        f"{flagged_details['Instrument'].nunique()} instruments.\n"
        f"Inaccurate price-supported mismatches: "
        f"{int((flagged_details['Validation_Status'] == 'inaccurate').sum())}.\n"
        f"Confirmed adjacent spike rows: {int(flagged_details['Spike_Error_Confirmed'].sum())}; "
        f"confirmed long-spike rows: {int(flagged_details['Long_Spike_Error_Confirmed'].sum())}."
    )

print(f"Saved flagged summary to {summary_path}")
print(f"Saved window-segment summary to {segment_summary_path}")
print(f"Saved flagged details to {details_path}")
print(f"Saved driver decomposition to {drivers_path}")
print(f"Saved spike-error pair report to {spike_pairs_path}")
print(f"Saved long-spike-error pair report to {long_spike_pairs_path}")
if scatter_path is not None:
    print(f"Saved reported-vs-price return scatter plot to {scatter_path}")
if year_bubble_plot_path is not None:
    print(f"Saved year-vs-return bubble scatter plot to {year_bubble_plot_path}")
if relative_day_plot_path is not None:
    print(f"Saved relative-day scatter plot to {relative_day_plot_path}")
OUTPUTS.save_text("summary", summary_text)
