from __future__ import annotations

from math import prod
from typing import Any, cast
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar.
    def tqdm(iterable, *args, **kwargs):
        return iterable

import pandas as pd

from ..core.pipeline_config import (
    ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    MAIN_REGRESSION_BHAR_WINDOW,
    POST_ANNOUNCEMENT_MISSING_RETURN_FILL_VALUE,
    PRE_ANNOUNCEMENT_WINDOW_LENGTH,
)
from .market_data_fetch import extract_total_return_history

PRE_ANNOUNCEMENT_START_DAY = -int(PRE_ANNOUNCEMENT_WINDOW_LENGTH)
RETURN_WINDOW_END_DAY = 90
EVENT_WINDOW_END_DAY = 1
PEAD_WINDOW_START_DAY = 2
ANNOUNCEMENT_DAY_OFFSET = -PRE_ANNOUNCEMENT_START_DAY
ZERO_FILL_RETURN_TREATMENT = "zero_fill"
COMPLETE_CASE_RETURN_TREATMENT = "complete_case"
TERMINAL_LOSS_RETURN_TREATMENT = "terminal_loss"
MISSING_RETURN_TREATMENTS = {
    ZERO_FILL_RETURN_TREATMENT,
    COMPLETE_CASE_RETURN_TREATMENT,
    TERMINAL_LOSS_RETURN_TREATMENT,
}
POST_ANNOUNCEMENT_BHAR_WINDOWS = tuple(
    dict.fromkeys(
        (MAIN_REGRESSION_BHAR_WINDOW, *ALTERNATIVE_REGRESSION_BHAR_WINDOWS)
    )
)


def build_abnormal_returns_for_earnings_events(
    earnings_events: pd.DataFrame,
    stock_universe: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    *,
    missing_return_treatment: str = ZERO_FILL_RETURN_TREATMENT,
    failure_records: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    if missing_return_treatment not in MISSING_RETURN_TREATMENTS:
        supported = ", ".join(sorted(MISSING_RETURN_TREATMENTS))
        raise ValueError(
            f"Unsupported missing-return treatment {missing_return_treatment!r}. "
            f"Expected one of: {supported}."
        )

    stock_returns = _prepare_returns_frame(extract_total_return_history(market_data))
    benchmark_returns = _prepare_returns_frame(benchmark_returns)
    benchmark_portfolios = build_benchmark_portfolio_map(stock_universe)
    retire_dates = build_retire_date_map(stock_universe)
    output_columns = build_abnormal_return_output_columns(earnings_events.columns)

    if earnings_events.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, Any]] = []
    for _, event in tqdm(
        earnings_events.iterrows(),
        total=earnings_events.shape[0],
        desc="Calculating abnormal returns",
    ):
        try:
            event_rows = build_abnormal_return_rows(
                event=event,
                benchmark_portfolios=benchmark_portfolios,
                retire_dates=retire_dates,
                stock_returns=stock_returns,
                benchmark_returns=benchmark_returns,
                missing_return_treatment=missing_return_treatment,
            )
        except (KeyError, ValueError) as error:
            if failure_records is not None:
                failure_records.append(
                    {
                        "Event_ID": build_event_id(event),
                        "Instrument": str(event.get("Instrument", "")),
                        "Ann_Date": str(event.get("Ann_Date", "")),
                        "Error_Type": type(error).__name__,
                        "Reason": str(error),
                    }
                )
            continue
        records.extend(event_rows)

    return pd.DataFrame(records).reindex(columns=output_columns)


def build_benchmark_portfolio_map(stock_universe: pd.DataFrame) -> dict[str, str]:
    required_columns = {"Instrument", "Benchmark_Portfolio"}
    missing_columns = required_columns.difference(stock_universe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"stock_universe.csv is missing required columns: {missing}.")

    benchmark_portfolio_frame = (
        stock_universe.loc[:, ["Instrument", "Benchmark_Portfolio"]]
        .dropna(subset=["Instrument", "Benchmark_Portfolio"])
        .drop_duplicates(subset=["Instrument"])
    )

    return {
        instrument: portfolio
        for instrument, portfolio in zip(
            benchmark_portfolio_frame["Instrument"].astype(str).tolist(),
            benchmark_portfolio_frame["Benchmark_Portfolio"].astype(str).tolist(),
            strict=False,
        )
    }


def build_retire_date_map(stock_universe: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if "Instrument" not in stock_universe.columns or "Retire_Date" not in stock_universe.columns:
        return {}

    retire_frame = stock_universe.loc[:, ["Instrument", "Retire_Date"]].copy()
    retire_frame["Retire_Date"] = pd.to_datetime(
        retire_frame["Retire_Date"], errors="coerce"
    ).dt.tz_localize(None)
    retire_frame = retire_frame.dropna(subset=["Instrument", "Retire_Date"])
    retire_frame = retire_frame.sort_values(["Instrument", "Retire_Date"])
    retire_frame = retire_frame.drop_duplicates(subset=["Instrument"], keep="first")
    return {
        str(instrument): retire_date
        for instrument, retire_date in zip(
            retire_frame["Instrument"].astype(str).tolist(),
            retire_frame["Retire_Date"].tolist(),
            strict=False,
        )
    }


def build_abnormal_return_output_columns(
    base_columns: pd.Index | list[str],
) -> list[str]:
    base_columns = list(base_columns)
    metadata_columns = [
        "Event_ID",
        "Benchmark_Portfolio",
        "Announcement_Trading_Date",
        "Window_End_Trading_Date",
        *(
            f"BHAR_{int(day_start)}_{int(day_end)}"
            for day_start, day_end in POST_ANNOUNCEMENT_BHAR_WINDOWS
        ),
        "Return_Treatment",
        "Missing_Pre_Announcement_Return_Day_Count",
        "Missing_Announcement_Window_Return_Day_Count",
        "Missing_Post_Announcement_Return_Day_Count",
        "First_Missing_Relative_Day",
        "Last_Observed_Relative_Day",
        "Has_Interior_Missing_Return",
        "Interior_Missing_Return_Day_Count",
        "Has_Terminal_Missing_Return",
        "Terminal_Missing_Return_Day_Count",
        "Terminal_Loss_Applied",
        "Retire_Date_In_Event_Window",
        "Has_Post_Retirement_Missing_Return",
        "Post_Retirement_Missing_Return_Day_Count",
        "Relative_Day",
        "Trading_Date",
        "Window_Label",
        "Raw_Security_Return",
        "Security_Return_Was_Imputed",
        "Security_Return",
        "Benchmark_Return",
        "Abnormal_Return",
    ]
    return base_columns + metadata_columns


def build_abnormal_return_rows(
    event: pd.Series,
    benchmark_portfolios: dict[str, str],
    retire_dates: dict[str, pd.Timestamp],
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    *,
    missing_return_treatment: str,
) -> list[dict[str, Any]]:
    row: dict[str, Any] = {str(key): value for key, value in event.to_dict().items()}
    instrument = str(event["Instrument"])

    if instrument not in benchmark_portfolios:
        raise KeyError(
            f"Missing benchmark portfolio for instrument {instrument} in stock_universe.csv."
        )

    event_id = build_event_id(event)
    benchmark_portfolio = benchmark_portfolios[instrument]
    row["Event_ID"] = event_id
    row["Benchmark_Portfolio"] = benchmark_portfolio

    announcement_date = require_announcement_date(event)
    retire_date = retire_dates.get(instrument)
    if benchmark_portfolio not in benchmark_returns.columns:
        raise KeyError(
            f"Missing benchmark returns for portfolio {benchmark_portfolio} "
            f"for event {event_id}."
        )
    benchmark_series = pd.to_numeric(
        benchmark_returns[benchmark_portfolio], errors="coerce"
    )
    benchmark_trading_calendar = pd.DatetimeIndex(
        benchmark_series.loc[benchmark_series.notna()].index
    ).sort_values()
    window_dates = locate_event_window_dates(
        announcement_date=announcement_date,
        trading_calendar=benchmark_trading_calendar,
        event_id=event_id,
    )
    if stock_returns.index.max() < window_dates[-1]:
        raise ValueError(
            f"Stock-return data end before the event window for {event_id}; "
            "terminal missing returns cannot be classified safely."
        )

    raw_stock_window = extract_window_returns(
        returns_frame=stock_returns,
        column_name=instrument,
        window_dates=window_dates,
        series_label=f"stock returns for {instrument}",
        event_id=event_id,
        # Preserve missing pre-announcement and announcement-window returns rather
        # than excluding the event. The treatment rules below apply only from day
        # 2 onward; BHAR(0,1) remains missing when either announcement return is
        # unavailable, while post-announcement BHAR can still be constructed.
        allowed_missing_positions=list(range(len(window_dates))),
    )
    benchmark_window = extract_window_returns(
        returns_frame=benchmark_returns,
        column_name=benchmark_portfolio,
        window_dates=window_dates,
        series_label=f"benchmark returns for {benchmark_portfolio}",
        event_id=event_id,
    )
    stock_window, missing_metadata = apply_missing_return_treatment(
        raw_stock_window=raw_stock_window,
        stock_return_history=stock_returns[instrument],
        window_dates=window_dates,
        missing_return_treatment=missing_return_treatment,
        retire_date=retire_date,
    )

    return finalize_abnormal_return_rows(
        row=row,
        window_dates=window_dates,
        raw_stock_window=raw_stock_window,
        stock_window=stock_window,
        benchmark_window=benchmark_window,
        missing_metadata=missing_metadata,
    )


def apply_missing_return_treatment(
    *,
    raw_stock_window: pd.Series,
    stock_return_history: pd.Series,
    window_dates: pd.DatetimeIndex,
    missing_return_treatment: str,
    retire_date: pd.Timestamp | None,
) -> tuple[pd.Series, dict[str, object]]:
    """Apply a post-announcement missing-return rule without changing days <= 1."""
    treated = raw_stock_window.copy().astype("float64")
    post_start_position = ANNOUNCEMENT_DAY_OFFSET + PEAD_WINDOW_START_DAY
    pre_window = raw_stock_window.iloc[:ANNOUNCEMENT_DAY_OFFSET]
    announcement_window = raw_stock_window.iloc[
        ANNOUNCEMENT_DAY_OFFSET:post_start_position
    ]
    post_window = raw_stock_window.iloc[post_start_position:]
    missing_pre_count = int(pre_window.isna().sum())
    missing_announcement_count = int(announcement_window.isna().sum())
    missing_post_positions = [
        int(position) for position in post_window.index[post_window.isna()].tolist()
    ]

    numeric_history = pd.to_numeric(stock_return_history, errors="coerce")
    observed_history = numeric_history.dropna()
    last_observed_history_date = (
        pd.Timestamp(observed_history.index.max()) if not observed_history.empty else None
    )
    terminal_positions: list[int] = []
    interior_positions: list[int] = []
    for position in missing_post_positions:
        missing_date = window_dates[position]
        has_later_observation = bool(
            last_observed_history_date is not None
            and last_observed_history_date > missing_date
        )
        if has_later_observation:
            interior_positions.append(position)
        else:
            terminal_positions.append(position)

    terminal_loss_applied = False
    if missing_return_treatment == ZERO_FILL_RETURN_TREATMENT:
        treated.iloc[missing_post_positions] = POST_ANNOUNCEMENT_MISSING_RETURN_FILL_VALUE
    elif missing_return_treatment == TERMINAL_LOSS_RETURN_TREATMENT:
        treated.iloc[interior_positions] = POST_ANNOUNCEMENT_MISSING_RETURN_FILL_VALUE
        treated.iloc[terminal_positions] = POST_ANNOUNCEMENT_MISSING_RETURN_FILL_VALUE
        if terminal_positions:
            treated.iloc[min(terminal_positions)] = -100.0
            terminal_loss_applied = True
    elif missing_return_treatment != COMPLETE_CASE_RETURN_TREATMENT:
        raise ValueError(f"Unsupported missing-return treatment: {missing_return_treatment}.")

    relative_days = list(range(PRE_ANNOUNCEMENT_START_DAY, RETURN_WINDOW_END_DAY + 1))
    observed_positions = raw_stock_window.index[raw_stock_window.notna()].tolist()
    last_observed_relative_day = (
        relative_days[int(max(observed_positions))] if observed_positions else None
    )
    first_missing_relative_day = (
        relative_days[min(missing_post_positions)] if missing_post_positions else None
    )
    retire_date_in_window = bool(
        retire_date is not None
        and window_dates[0] <= retire_date.normalize() <= window_dates[-1]
    )
    post_retirement_positions = [
        position
        for position in missing_post_positions
        if retire_date is not None and window_dates[position] >= retire_date.normalize()
    ]
    metadata: dict[str, object] = {
        "Return_Treatment": missing_return_treatment,
        "Missing_Pre_Announcement_Return_Day_Count": missing_pre_count,
        "Missing_Announcement_Window_Return_Day_Count": missing_announcement_count,
        "Missing_Post_Announcement_Return_Day_Count": len(missing_post_positions),
        "First_Missing_Relative_Day": first_missing_relative_day,
        "Last_Observed_Relative_Day": last_observed_relative_day,
        "Has_Interior_Missing_Return": bool(interior_positions),
        "Interior_Missing_Return_Day_Count": len(interior_positions),
        "Has_Terminal_Missing_Return": bool(terminal_positions),
        "Terminal_Missing_Return_Day_Count": len(terminal_positions),
        "Terminal_Loss_Applied": terminal_loss_applied,
        "Retire_Date_In_Event_Window": retire_date_in_window,
        "Has_Post_Retirement_Missing_Return": bool(post_retirement_positions),
        "Post_Retirement_Missing_Return_Day_Count": len(post_retirement_positions),
    }
    return treated, metadata


def get_stock_trading_calendar(
    stock_returns: pd.DataFrame,
    instrument: str,
    event_id: str,
) -> pd.DatetimeIndex:
    if stock_returns.empty:
        raise ValueError("Stock returns frame is empty.")

    if instrument not in stock_returns.columns:
        raise KeyError(
            f"Missing stock returns for {instrument} in returns frame for event {event_id}."
        )

    stock_series = pd.to_numeric(stock_returns[instrument], errors="coerce")
    stock_trading_dates = stock_series.loc[stock_series.notna()].index
    stock_trading_calendar = pd.DatetimeIndex(stock_trading_dates).sort_values()

    if stock_trading_calendar.empty:
        raise ValueError(
            f"Stock trading calendar is empty for {instrument} in event {event_id}."
        )

    return stock_trading_calendar


def finalize_abnormal_return_rows(
    row: dict[str, Any],
    window_dates: pd.DatetimeIndex,
    raw_stock_window: pd.Series,
    stock_window: pd.Series,
    benchmark_window: pd.Series,
    missing_metadata: dict[str, object],
) -> list[dict[str, Any]]:
    abnormal_window = cast(pd.Series, stock_window - benchmark_window)
    announcement_trading_date = format_timestamp(
        window_dates[ANNOUNCEMENT_DAY_OFFSET]
    )
    window_end_trading_date = format_timestamp(window_dates[-1])

    # Save only the configured post-announcement benchmark-relative BHARs.
    post_announcement_stock_window = cast(
        pd.Series,
        stock_window.iloc[ANNOUNCEMENT_DAY_OFFSET + PEAD_WINDOW_START_DAY :],
    )
    post_announcement_benchmark_window = cast(
        pd.Series,
        benchmark_window.iloc[ANNOUNCEMENT_DAY_OFFSET + PEAD_WINDOW_START_DAY :],
    )
    configured_post_announcement_bhars: dict[str, float] = {}
    for day_start, day_end in POST_ANNOUNCEMENT_BHAR_WINDOWS:
        if int(day_start) != PEAD_WINDOW_START_DAY:
            continue
        observation_count = int(day_end) - int(day_start) + 1
        configured_post_announcement_bhars[f"BHAR_{int(day_start)}_{int(day_end)}"] = (
            calculate_benchmark_relative_buy_and_hold_return_if_complete(
                security_returns=cast(
                    pd.Series, post_announcement_stock_window.iloc[:observation_count]
                ),
                benchmark_returns=cast(
                    pd.Series, post_announcement_benchmark_window.iloc[:observation_count]
                ),
            )
        )
    rows: list[dict[str, Any]] = []
    relative_days = range(PRE_ANNOUNCEMENT_START_DAY, RETURN_WINDOW_END_DAY + 1)
    for offset, day in enumerate(relative_days):
        daily_row = row.copy()
        daily_row["Announcement_Trading_Date"] = announcement_trading_date
        daily_row["Window_End_Trading_Date"] = window_end_trading_date
        daily_row.update(configured_post_announcement_bhars)
        daily_row.update(missing_metadata)
        daily_row["Relative_Day"] = day
        daily_row["Trading_Date"] = format_timestamp(window_dates[offset])
        daily_row["Window_Label"] = classify_window(day)
        daily_row["Raw_Security_Return"] = float(raw_stock_window.iloc[offset])
        daily_row["Security_Return_Was_Imputed"] = bool(
            pd.isna(raw_stock_window.iloc[offset]) and pd.notna(stock_window.iloc[offset])
        )
        daily_row["Security_Return"] = float(stock_window.iloc[offset])
        daily_row["Benchmark_Return"] = float(benchmark_window.iloc[offset])
        daily_row["Abnormal_Return"] = float(abnormal_window.iloc[offset])
        rows.append(daily_row)

    return rows


def classify_window(relative_day: int) -> str:
    if relative_day < 0:
        return "Pre_Announcement_Window"
    if relative_day <= EVENT_WINDOW_END_DAY:
        return "Event_Window"
    return "PEAD_Window"


def build_event_id(event: pd.Series) -> str:
    instrument = str(event["Instrument"])
    ann_date = str(event["Ann_Date"])
    report_frequency = str(event.get("Report_Frequency", "NA"))
    return f"{instrument}|{ann_date}|{report_frequency}"


def require_announcement_date(event: pd.Series) -> pd.Timestamp:
    announcement_date = pd.to_datetime(
        cast(Any, event["Ann_Date"]), errors="coerce"
    )
    if pd.isna(announcement_date):
        raise ValueError(
            "Invalid or missing announcement date for "
            f"event {build_event_id(event)}."
        )

    if getattr(announcement_date, "tzinfo", None) is not None:
        announcement_date = announcement_date.tz_localize(None)

    return announcement_date.normalize()


def locate_event_window_dates(
    announcement_date: pd.Timestamp,
    trading_calendar: pd.DatetimeIndex,
    event_id: str,
) -> pd.DatetimeIndex:
    if trading_calendar.empty:
        raise ValueError("Trading calendar is empty.")

    start_pos = trading_calendar.searchsorted(announcement_date, side="left")
    window_start_pos = start_pos + PRE_ANNOUNCEMENT_START_DAY
    window_end_pos = start_pos + RETURN_WINDOW_END_DAY + 1

    if window_start_pos < 0:
        raise ValueError(
            f"Need at least {-PRE_ANNOUNCEMENT_START_DAY} pre-announcement trading days "
            f"for event {event_id}, but the market-data history starts too late."
        )

    window_dates = trading_calendar[window_start_pos:window_end_pos]
    expected_window_length = RETURN_WINDOW_END_DAY - PRE_ANNOUNCEMENT_START_DAY + 1

    if len(window_dates) != expected_window_length:
        raise ValueError(
            f"Expected {expected_window_length} trading days for event {event_id}, "
            f"but found {len(window_dates)}."
        )

    return window_dates


def locate_event_window_dates_with_retirement_fallback(
    *,
    announcement_date: pd.Timestamp,
    stock_trading_calendar: pd.DatetimeIndex,
    benchmark_trading_calendar: pd.DatetimeIndex,
    retire_date: pd.Timestamp | None,
    event_id: str,
) -> pd.DatetimeIndex:
    try:
        return locate_event_window_dates(
            announcement_date=announcement_date,
            trading_calendar=stock_trading_calendar,
            event_id=event_id,
        )
    except ValueError:
        if retire_date is None:
            raise

    fallback_window_dates = locate_event_window_dates(
        announcement_date=announcement_date,
        trading_calendar=benchmark_trading_calendar,
        event_id=event_id,
    )
    if retire_date < fallback_window_dates[0] or retire_date > fallback_window_dates[-1]:
        raise ValueError(
            f"Retire date {retire_date.date()} is outside the event window for {event_id}."
        )

    return fallback_window_dates


def extract_window_returns(
    returns_frame: pd.DataFrame,
    column_name: str,
    window_dates: pd.DatetimeIndex,
    series_label: str,
    event_id: str,
    fill_missing_with_value: float | None = None,
    allowed_missing_positions: list[int] | None = None,
) -> pd.Series:
    if returns_frame.empty:
        raise ValueError(f"Cannot read {series_label}: returns frame is empty.")

    if column_name not in returns_frame.columns:
        raise KeyError(
            f"Missing {series_label} in returns frame for event {event_id}."
        )

    values = cast(
        pd.Series,
        returns_frame[column_name].reindex(window_dates).reset_index(drop=True),
    )
    numeric_values = cast(pd.Series, pd.to_numeric(values, errors="coerce"))

    if numeric_values.isna().any():
        if allowed_missing_positions is not None:
            allowed_missing_positions_set = {int(position) for position in allowed_missing_positions}
            actual_missing_positions = numeric_values.index[numeric_values.isna()].tolist()
            disallowed_missing_positions = [
                int(position)
                for position in actual_missing_positions
                if int(position) not in allowed_missing_positions_set
            ]
            if disallowed_missing_positions:
                raise ValueError(
                    f"Missing values in {series_label} for event {event_id} "
                    f"at relative days {disallowed_missing_positions}."
                )
            return numeric_values.astype("float64")
        if fill_missing_with_value is not None:
            return numeric_values.fillna(float(fill_missing_with_value)).astype("float64")

        missing_days = numeric_values.index[numeric_values.isna()].tolist()
        raise ValueError(
            f"Missing values in {series_label} for event {event_id} "
            f"at relative days {missing_days}."
        )

    return numeric_values.astype("float64")


# def calculate_buy_and_hold_return(returns: pd.Series) -> float:
#     gross_return_multiplier = calculate_gross_return_multiplier(returns)
#     compounded_return = gross_return_multiplier - 1.0
#     return compounded_return * 100.0


def calculate_benchmark_relative_buy_and_hold_return(
    security_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    security_gross_return = calculate_gross_return_multiplier(security_returns)
    benchmark_gross_return = calculate_gross_return_multiplier(benchmark_returns)

    return (security_gross_return - benchmark_gross_return) * 100.0


def calculate_benchmark_relative_buy_and_hold_return_if_complete(
    security_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    security_numeric = cast(pd.Series, pd.to_numeric(security_returns, errors="coerce"))
    benchmark_numeric = cast(pd.Series, pd.to_numeric(benchmark_returns, errors="coerce"))
    if security_numeric.isna().any() or benchmark_numeric.isna().any():
        return float("nan")
    return calculate_benchmark_relative_buy_and_hold_return(
        security_returns=security_numeric,
        benchmark_returns=benchmark_numeric,
    )


def calculate_gross_return_multiplier(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Cannot calculate BHAR on an empty return series.")

    numeric_returns = cast(pd.Series, pd.to_numeric(returns, errors="coerce"))
    if numeric_returns.isna().any():
        raise ValueError("Cannot calculate BHAR with missing return values.")

    gross_return_values: list[float] = [
        1.0 + (float(value) / 100.0) for value in numeric_returns.tolist()
    ]
    return prod(gross_return_values)


def _prepare_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("Returns frame is empty.")

    out = returns.copy()
    out.index = pd.to_datetime(out.index, errors="coerce", utc=True).tz_localize(None)
    out = out.loc[out.index.notna()]
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]

    if out.empty:
        raise ValueError("Returns frame has no valid dated rows after cleaning.")

    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def format_timestamp(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")
