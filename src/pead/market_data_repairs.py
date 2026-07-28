from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.pipeline_config import (
    NEGATIVE_OUTLIER_RETURN_THRESHOLD_PCT,
    POSITIVE_OUTLIER_RETURN_THRESHOLD_PCT,
    PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS,
)
from .market_data_fetch import extract_price_history, extract_total_return_history

PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY = "Outlier return mismatches removed as missing"
TRAILING_ZERO_RETURN_REMOVAL_KEY = "Stock-days removed by padded trailing zero returns"
IDENTICAL_PRICE_REMOVAL_KEY = "Stock-days removed after 30 identical prices"
REVERSAL_OUTLIER_REMOVAL_KEY = "Stock-days removed by 100% then -50% reversal rule"
LOW_COVERAGE_DATE_REMOVAL_KEY = "Stock-days removed on low-coverage return dates"
NONPOSITIVE_PRICE_REMOVAL_KEY = "Stock-days removed with nonpositive prices"


def _is_outlier_return(value: float) -> bool:
    if not np.isfinite(value):
        return False
    return (
        value >= float(POSITIVE_OUTLIER_RETURN_THRESHOLD_PCT)
        or value <= float(NEGATIVE_OUTLIER_RETURN_THRESHOLD_PCT)
    )


def _price_return_from_levels(previous_price: float, current_price: float) -> float:
    return (current_price / previous_price - 1.0) * 100.0


def _build_long_histories(
    market_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_return_history = extract_total_return_history(market_data)
    price_history = extract_price_history(market_data)

    return_long = (
        total_return_history.stack()
        .rename("TotalReturn")
        .reset_index()
        .rename(columns={"level_1": "Instrument"})
    )
    return_long["Instrument"] = return_long["Instrument"].astype("string").str.strip()
    return_long["Date"] = pd.to_datetime(return_long["Date"], errors="coerce")
    return_long["TotalReturn"] = pd.to_numeric(return_long["TotalReturn"], errors="coerce")
    return_long = return_long.dropna(subset=["Instrument", "Date", "TotalReturn"]).copy()

    price_long = (
        price_history.stack()
        .rename("PriceClose")
        .reset_index()
        .rename(columns={"level_1": "Instrument"})
    )
    price_long["Instrument"] = price_long["Instrument"].astype("string").str.strip()
    price_long["Date"] = pd.to_datetime(price_long["Date"], errors="coerce")
    price_long["PriceClose"] = pd.to_numeric(price_long["PriceClose"], errors="coerce")
    price_long = price_long.dropna(subset=["Instrument", "Date", "PriceClose"]).copy()
    price_long = price_long.sort_values(["Instrument", "Date"], kind="stable")
    price_long["Instrument_First_Price_Date"] = price_long.groupby("Instrument")["Date"].transform("min")
    price_long["Previous_PriceClose"] = price_long.groupby("Instrument")["PriceClose"].shift(1)
    price_long["Current_PriceClose"] = price_long["PriceClose"]
    price_long["Price_Return_From_Close"] = (
        price_long["Current_PriceClose"] / price_long["Previous_PriceClose"] - 1.0
    ) * 100.0
    return return_long, price_long


def _write_series_updates(
    market_data: pd.DataFrame,
    *,
    instrument: str,
    dates: list[pd.Timestamp],
    total_returns: list[float],
    prices: list[float | None] | None = None,
) -> tuple[int, int]:
    return_column = ("TotalReturn", instrument)
    price_column = ("PriceClose", instrument)
    updated_return_days = 0
    updated_price_days = 0

    for position, date in enumerate(dates):
        matching_index = market_data.index == date
        if not matching_index.any():
            continue
        market_data.loc[matching_index, return_column] = float(total_returns[position])
        updated_return_days += int(matching_index.sum())
        if prices is not None and prices[position] is not None:
            market_data.loc[matching_index, price_column] = float(prices[position])
            updated_price_days += int(matching_index.sum())

    return updated_return_days, updated_price_days


def _remove_stock_days(
    market_data: pd.DataFrame,
    *,
    instrument: str,
    dates: list[pd.Timestamp],
) -> int:
    return_column = ("TotalReturn", instrument)
    price_column = ("PriceClose", instrument)
    updated_days = 0

    for date in dates:
        matching_index = market_data.index == date
        if not matching_index.any():
            continue
        had_any_data = (
            market_data.loc[matching_index, return_column].notna()
            | market_data.loc[matching_index, price_column].notna()
        )
        updated_days += int(had_any_data.sum())
        market_data.loc[matching_index, return_column] = np.nan
        market_data.loc[matching_index, price_column] = np.nan

    return updated_days


def _apply_stock_day_filters(
    market_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if market_data.empty:
        return market_data.copy(), {
            TRAILING_ZERO_RETURN_REMOVAL_KEY: 0,
            IDENTICAL_PRICE_REMOVAL_KEY: 0,
            REVERSAL_OUTLIER_REMOVAL_KEY: 0,
            LOW_COVERAGE_DATE_REMOVAL_KEY: 0,
            NONPOSITIVE_PRICE_REMOVAL_KEY: 0,
        }

    cleaned = market_data.copy()
    counts = {
        TRAILING_ZERO_RETURN_REMOVAL_KEY: 0,
        IDENTICAL_PRICE_REMOVAL_KEY: 0,
        REVERSAL_OUTLIER_REMOVAL_KEY: 0,
        LOW_COVERAGE_DATE_REMOVAL_KEY: 0,
        NONPOSITIVE_PRICE_REMOVAL_KEY: 0,
    }

    return_history = extract_total_return_history(cleaned)
    price_history = extract_price_history(cleaned)
    instruments = sorted(
        {
            str(instrument).strip()
            for instrument in set(return_history.columns.tolist()).union(price_history.columns.tolist())
            if pd.notna(instrument) and str(instrument).strip()
        }
    )

    for instrument in instruments:
        if instrument in return_history.columns:
            return_series = pd.to_numeric(return_history[instrument], errors="coerce")
            trailing_zero_dates: list[pd.Timestamp] = []
            zero_run = 0
            for date, value in zip(reversed(return_series.index.tolist()), reversed(return_series.tolist()), strict=False):
                if pd.isna(value):
                    break
                if float(value) == 0.0:
                    zero_run += 1
                    if zero_run >= 10:
                        trailing_zero_dates.append(pd.Timestamp(date))
                    continue
                break
            if trailing_zero_dates:
                counts[TRAILING_ZERO_RETURN_REMOVAL_KEY] += _remove_stock_days(
                    cleaned,
                    instrument=instrument,
                    dates=list(reversed(trailing_zero_dates)),
                )

        if instrument in price_history.columns:
            price_series = pd.to_numeric(price_history[instrument], errors="coerce")
            nonpositive_dates = [
                pd.Timestamp(date)
                for date, value in zip(price_series.index.tolist(), price_series.tolist(), strict=False)
                if pd.notna(value) and float(value) <= 0.0
            ]
            if nonpositive_dates:
                counts[NONPOSITIVE_PRICE_REMOVAL_KEY] += _remove_stock_days(
                    cleaned,
                    instrument=instrument,
                    dates=nonpositive_dates,
                )

    return_history = extract_total_return_history(cleaned)
    price_history = extract_price_history(cleaned)

    for instrument in instruments:
        if instrument in price_history.columns:
            price_series = pd.to_numeric(price_history[instrument], errors="coerce")
            identical_dates: list[pd.Timestamp] = []
            streak_value: float | None = None
            streak_count = 0
            for date, value in zip(price_series.index.tolist(), price_series.tolist(), strict=False):
                if pd.isna(value):
                    streak_value = None
                    streak_count = 0
                    continue
                current_value = float(value)
                if streak_value is None or not np.isclose(current_value, streak_value):
                    streak_value = current_value
                    streak_count = 1
                    continue
                streak_count += 1
                if streak_count > 30:
                    identical_dates.append(pd.Timestamp(date))
            if identical_dates:
                counts[IDENTICAL_PRICE_REMOVAL_KEY] += _remove_stock_days(
                    cleaned,
                    instrument=instrument,
                    dates=identical_dates,
                )

        if instrument in return_history.columns:
            return_series = pd.to_numeric(return_history[instrument], errors="coerce")
            reversal_dates: list[pd.Timestamp] = []
            dates = return_series.index.tolist()
            values = return_series.tolist()
            for idx in range(len(values) - 1):
                first = values[idx]
                second = values[idx + 1]
                if pd.isna(first) or pd.isna(second):
                    continue
                if float(first) > 100.0 and float(second) < -50.0:
                    reversal_dates.extend([pd.Timestamp(dates[idx]), pd.Timestamp(dates[idx + 1])])
            if reversal_dates:
                counts[REVERSAL_OUTLIER_REMOVAL_KEY] += _remove_stock_days(
                    cleaned,
                    instrument=instrument,
                    dates=list(dict.fromkeys(reversal_dates)),
                )

    return_history = extract_total_return_history(cleaned)
    if not return_history.empty:
        total_stocks = max(int(return_history.shape[1]), 1)
        valid_share_by_date = (return_history.notna() & return_history.ne(0)).sum(axis=1) / total_stocks
        low_coverage_dates = [
            pd.Timestamp(date)
            for date, share in valid_share_by_date.items()
            if float(share) < 0.005
        ]
        if low_coverage_dates:
            for instrument in [
                str(column).strip()
                for column in return_history.columns.tolist()
                if pd.notna(column) and str(column).strip()
            ]:
                counts[LOW_COVERAGE_DATE_REMOVAL_KEY] += _remove_stock_days(
                    cleaned,
                    instrument=instrument,
                    dates=low_coverage_dates,
                )

    return cleaned.sort_index(), counts


def apply_market_data_repairs(
    market_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if market_data.empty:
        return market_data.copy(), {
            TRAILING_ZERO_RETURN_REMOVAL_KEY: 0,
            IDENTICAL_PRICE_REMOVAL_KEY: 0,
            REVERSAL_OUTLIER_REMOVAL_KEY: 0,
            LOW_COVERAGE_DATE_REMOVAL_KEY: 0,
            NONPOSITIVE_PRICE_REMOVAL_KEY: 0,
            PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY: 0,
        }

    repaired_market_data, stock_day_filter_counts = _apply_stock_day_filters(market_data)
    return_long, price_long = _build_long_histories(repaired_market_data)
    if return_long.empty or price_long.empty:
        return repaired_market_data, {
            **stock_day_filter_counts,
            PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY: 0,
        }

    merged = return_long.merge(
        price_long.loc[
            :,
            [
                "Instrument",
                "Date",
                "Instrument_First_Price_Date",
                "Previous_PriceClose",
                "Current_PriceClose",
                "Price_Return_From_Close",
            ],
        ],
        on=["Instrument", "Date"],
        how="left",
    )
    merged["Return_Error"] = merged["Price_Return_From_Close"] - merged["TotalReturn"]
    merged["Absolute_Return_Error"] = merged["Return_Error"].abs()
    merged["Price_Return_Available"] = merged["Price_Return_From_Close"].notna()
    merged["Ignore_For_Validation"] = (
        merged["Current_PriceClose"].notna()
        & merged["Previous_PriceClose"].isna()
        & (merged["Date"] == merged["Instrument_First_Price_Date"])
    )
    merged["Outlier_Return_Flag"] = merged["TotalReturn"].map(_is_outlier_return)

    mismatch_events = merged.loc[
        merged["Outlier_Return_Flag"]
        & merged["Price_Return_Available"]
        & ~merged["Ignore_For_Validation"]
        & (merged["Absolute_Return_Error"] > float(PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS)),
        ["Instrument", "Date"],
    ].copy()

    removed_mismatch_count = 0
    if not mismatch_events.empty:
        for instrument, instrument_dates in mismatch_events.groupby("Instrument"):
            return_column = ("TotalReturn", str(instrument).strip())
            detail_rows = merged.loc[
                merged["Instrument"].astype("string").str.strip().eq(str(instrument).strip())
                & merged["Date"].isin(pd.to_datetime(instrument_dates["Date"], errors="coerce"))
            ].copy()
            for row in detail_rows.itertuples(index=False):
                if not pd.notna(row.Price_Return_From_Close):
                    continue
                matching_index = repaired_market_data.index == pd.Timestamp(row.Date)
                if not matching_index.any():
                    continue
                existing_non_na = repaired_market_data.loc[matching_index, return_column].notna()
                removed_mismatch_count += int(existing_non_na.sum())
                repaired_market_data.loc[matching_index, return_column] = np.nan

    return repaired_market_data.sort_index(), {
        **stock_day_filter_counts,
        PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY: int(removed_mismatch_count),
    }
