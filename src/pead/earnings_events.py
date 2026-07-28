import datetime as dt
import time

import pandas as pd
import lseg.data as ld
from tqdm import tqdm

from ..core.pipeline_config import (
    DETAILED_EPS_ESTIMATE_FIELD,
    EARNINGS_RELEASE_FREQUENCIES,
    EARNINGS_REQUEST_INSTRUMENT_LIMIT,
    FORECAST_LOOKBACK_DAYS,
    FORECAST_PERIOD_TEMPLATES_BY_FREQUENCY,
    MIN_ANALYST_FORECASTS_FOR_SUE,
    SLEEP_BTWN_PULLS,
    SUE_EARNINGS_FREQUENCIES,
)


def is_partial_earnings_collection_error(error: Exception) -> bool:
    message = str(error).lower()
    return "unable to collect data for the field" in message


def robust_get_data(
    universe,
    fields,
    parameters=None,
    header_type=ld.HeaderType.NAME,
    max_retries=2,
    base_sleep=1.0,
) -> pd.DataFrame:
    last_error = None

    for attempt in range(max_retries):
        try:
            return ld.get_data(
                universe=universe,
                fields=fields,
                parameters=parameters,
                header_type=header_type,
            )
        except Exception as error:
            if is_partial_earnings_collection_error(error):
                raise
            last_error = error
            sleep_s = base_sleep * (2**attempt)
            print(f"[retry {attempt + 1}/{max_retries}] earnings get_data failed: {error}")
            print(f"Sleeping {sleep_s:.1f}s before retry...")
            time.sleep(sleep_s)

    raise last_error  # type: ignore[misc]


def chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def batched_get_data(
    universe,
    fields,
    parameters=None,
    header_type=ld.HeaderType.NAME,
    max_instruments_per_request: int = EARNINGS_REQUEST_INSTRUMENT_LIMIT,
    failure_stats: dict[str, int] | None = None,
    failure_records: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    universe_list = list(universe)
    fields_list = list(fields)

    if not universe_list:
        return pd.DataFrame()

    field_count = max(len(fields_list), 1)
    batch_size = max(max_instruments_per_request, 1)

    batches = chunk_list(universe_list, batch_size)
    frames: list[pd.DataFrame] = []

    for batch_index, universe_batch in enumerate(batches, start=1):
        print(
            "Downloading earnings batch "
            f"{batch_index}/{len(batches)} "
            f"({len(universe_batch)} instruments, "
            f"{len(universe_batch) * field_count} requested fields)"
        )
        batch_frame = fetch_earnings_batch_with_fallback(
            universe_batch=universe_batch,
            fields=fields_list,
            parameters=parameters,
            header_type=header_type,
            failure_stats=failure_stats,
            failure_records=failure_records,
        )
        if batch_frame is not None and not batch_frame.empty:
            frames.append(batch_frame)
        time.sleep(SLEEP_BTWN_PULLS)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_earnings_batch_with_fallback(
    universe_batch,
    fields,
    parameters=None,
    header_type=ld.HeaderType.NAME,
    failure_stats: dict[str, int] | None = None,
    failure_records: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    universe_list = list(universe_batch)
    fields_list = list(fields)

    try:
        return robust_get_data(
            universe=universe_list,
            fields=fields_list,
            parameters=parameters,
            header_type=header_type,
        )
    except Exception as error:
        if not is_partial_earnings_collection_error(error):
            raise

        print(
            "Skipping failed earnings batch "
            f"({len(universe_list)} instruments) instead of splitting."
        )
        reason = str(error).splitlines()[0]
        if failure_stats is not None:
            failure_stats["failed_batches"] = failure_stats.get("failed_batches", 0) + 1
            failure_stats["failed_instruments"] = failure_stats.get("failed_instruments", 0) + len(universe_list)
        if failure_records is not None:
            for instrument in universe_list:
                failure_records.append(
                    {
                        "Instrument": str(instrument),
                        "Reason": reason,
                    }
                )
        print(f"Reason: {reason}")
        return pd.DataFrame()


def calculate_sue_for_universe(
    stock_universe,
    price_history,
    year,
    sample_size,
    currency,
    year_context,
):
    earnings_events = fetch_earnings_events(
        stock_universe=stock_universe,
        year=year,
        currency=currency,
        frequencies=SUE_EARNINGS_FREQUENCIES,
    )

    earnings_events = clean_earnings_events(earnings_events, year, sample_size)
    full_earnings_events = add_pre_announcement_forecast_medians(
        earnings_events,
        year,
        currency,
        sample_size,
        apply_filters=False,
    )
    full_earnings_events = add_lagged_prices(
        full_earnings_events,
        price_history,
        sample_size,
        apply_filter=False,
    )
    full_earnings_events = calculate_sue(full_earnings_events)

    earnings_events = build_sue_filtered_event_sample(
        full_earnings_events,
        min_analyst_forecasts=MIN_ANALYST_FORECASTS_FOR_SUE,
        sample_size=sample_size,
    )

    return earnings_events, full_earnings_events, sample_size


def build_sue_filtered_event_sample(
    full_earnings_events: pd.DataFrame,
    *,
    min_analyst_forecasts: int,
    sample_size: dict | None = None,
    sample_size_key_suffix: str = "",
) -> pd.DataFrame:
    earnings_events = full_earnings_events.copy()
    applied_sample_size = sample_size if sample_size is not None else {}
    suffix = str(sample_size_key_suffix)
    earnings_events = apply_earnings_filter(
        earnings_events,
        earnings_events["Forecast_Median"].notna(),
        f"Earnings events with 90-day forecast median{suffix}",
        applied_sample_size,
    )
    earnings_events = apply_earnings_filter(
        earnings_events,
        earnings_events["Forecast_Analyst_Count"] >= int(min_analyst_forecasts),
        f"Earnings events with enough analyst forecasts{suffix}",
        applied_sample_size,
    )
    earnings_events = apply_earnings_filter(
        earnings_events,
        earnings_events["Price_Lag_5"].notna() & (earnings_events["Price_Lag_5"] > 0),
        f"Earnings events with positive lagged price{suffix}",
        applied_sample_size,
    )
    return earnings_events


def fetch_earnings_events(
    stock_universe: pd.DataFrame,
    year: int,
    currency: str,
    frequencies: dict[str, str] | None = None,
    failure_stats: dict[str, int] | None = None,
    failure_records: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    selected_frequencies = frequencies or EARNINGS_RELEASE_FREQUENCIES
    events_by_frequency = [
        fetch_earnings_events_for_frequency(
            stock_universe,
            year,
            frequency,
            label,
            currency,
            failure_stats=failure_stats,
            failure_records=failure_records,
        )
        for frequency, label in selected_frequencies.items()
    ]

    events = pd.concat(events_by_frequency, ignore_index=True)
    return drop_duplicate_earnings_events(events)


def fetch_earnings_events_for_frequency(
    stock_universe: pd.DataFrame,
    year: int,
    frequency: str,
    report_label: str,
    currency: str,
    failure_stats: dict[str, int] | None = None,
    failure_records: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    field_map = build_earnings_field_map(year, frequency, currency)

    events = batched_get_data(
        universe=stock_universe["Instrument"].tolist(),
        fields=list(field_map.keys()),
        header_type=ld.HeaderType.NAME,
        failure_stats=failure_stats,
        failure_records=failure_records,
    )

    events = rename_earnings_columns(events, field_map)
    events["Report_Frequency"] = frequency
    events["Report_Type"] = report_label

    return events


def build_earnings_field_map(
    year: int,
    frequency: str,
    currency: str,
) -> dict[str, str]:
    formation_date = dt.datetime(year, 7, 1).strftime("%Y-%m-%d")
    end_window = dt.datetime(year + 1, 7, 1).strftime("%Y-%m-%d")

    return {
        (
            f"TR.EPSActValue(SDate={formation_date},"
            f"EDate={end_window},Frq={frequency},Curn={currency})"
        ): "Actual_EPS",
        "TR.EPSActValue.fperiod": "fperiod",
        "TR.EPSActValue.periodenddate": "periodenddate",
        (
            f"TR.EPSActReportDate(SDate={formation_date},"
            f"EDate={end_window},Frq={frequency})"
        ): "Ann_Date",
    }


def rename_earnings_columns(
    events: pd.DataFrame,
    field_map: dict[str, str],
) -> pd.DataFrame:
    rename_map = {field.upper(): name for field, name in field_map.items()}

    return events.rename(
        columns=lambda column: rename_map.get(str(column).upper(), column)
    )


def drop_duplicate_earnings_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.drop_duplicates(
        subset=["Instrument", "Ann_Date", "Actual_EPS", "Report_Frequency"],
        keep="first",
    ).reset_index(drop=True)


def clean_earnings_events(
    events: pd.DataFrame,
    year: int,
    sample_size: dict,
) -> pd.DataFrame:
    formation_date = pd.Timestamp(dt.datetime(year, 7, 1))
    end_window = pd.Timestamp(dt.datetime(year + 1, 7, 1))

    out = events.copy()
    out["Ann_Date"] = pd.to_datetime(
        out["Ann_Date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    out["Actual_EPS"] = pd.to_numeric(out["Actual_EPS"], errors="coerce")

    out = apply_earnings_filter(
        out,
        (out["Ann_Date"] >= formation_date) & (out["Ann_Date"] < end_window),
        "All earnings announcements in event window",
        sample_size,
    )

    out = apply_earnings_filter(
        out,
        out["Actual_EPS"].notna(),
        "Earnings events with valid actual EPS",
        sample_size,
    )

    return out


def add_pre_announcement_forecast_medians(
    events: pd.DataFrame,
    year: int,
    currency: str,
    sample_size: dict,
    apply_filters: bool = True,
) -> pd.DataFrame:
    out = events.copy()

    if out.empty:
        out["Forecast_Median"] = pd.NA
        out["Forecast_Analyst_Count"] = pd.Series(dtype="int64")
        out["Forecast_Window_Start"] = pd.NA
        out["Forecast_Window_End"] = pd.NA
        out["Forecast_Error"] = pd.NA
        sample_size["Earnings events with 90-day forecast median"] = 0
        sample_size["Earnings events with enough analyst forecasts"] = 0
        return out

    forecast_summaries = build_forecast_summaries_for_events(
        out,
        currency=currency,
        desc=f"Downloading analyst forecasts {year}",
    )
    out = pd.concat([out, forecast_summaries], axis=1)

    if apply_filters:
        out = apply_earnings_filter(
            out,
            out["Forecast_Median"].notna(),
            "Earnings events with 90-day forecast median",
            sample_size,
        )

        out = apply_earnings_filter(
            out,
            out["Forecast_Analyst_Count"] >= MIN_ANALYST_FORECASTS_FOR_SUE,
            "Earnings events with enough analyst forecasts",
            sample_size,
        )

    return out


def build_forecast_summaries_for_events(
    events: pd.DataFrame,
    currency: str,
    desc: str,
) -> pd.DataFrame:
    summaries = [
        build_forecast_summary_for_event(event, currency)
        for _, event in tqdm(
            events.iterrows(),
            total=len(events),
            desc=desc,
        )
    ]

    return pd.DataFrame(summaries, index=events.index)


def build_forecast_summary_for_event(
    event: pd.Series,
    currency: str,
) -> pd.Series:
    forecasts, forecast_period, error = fetch_forecasts_for_event(event, currency)

    summary = {
        "Forecast_Median": pd.NA,
        "Forecast_Analyst_Count": 0,
        "Forecast_Period": forecast_period,
        "Forecast_Window_Start": forecast_window_start(event["Ann_Date"]),
        "Forecast_Window_End": forecast_window_end(event["Ann_Date"]),
        "Forecast_Error": error,
    }

    if forecasts.empty:
        return pd.Series(summary)

    latest_forecasts = keep_latest_forecast_per_analyst(forecasts)

    if latest_forecasts.empty:
        return pd.Series(summary)

    summary["Forecast_Median"] = latest_forecasts["Forecast_EPS"].median()
    summary["Forecast_Analyst_Count"] = len(latest_forecasts)

    return pd.Series(summary)


def fetch_forecasts_for_event(
    event: pd.Series,
    currency: str,
) -> tuple[pd.DataFrame, str | None, str | None]:
    candidate_periods = infer_absolute_forecast_periods(event)

    if not candidate_periods:
        frequency = event.get("Report_Frequency", "<missing>")
        return (
            pd.DataFrame(),
            None,
            f"No configured forecast period mapping for {frequency}",
        )

    field_map = build_forecast_detail_field_map()
    period_errors: list[str] = []

    for period in candidate_periods:
        try:
            forecasts = ld.get_data(
                universe=[event["Instrument"]],
                fields=list(field_map.keys()),
                parameters={
                    "Period": period,
                    "Frq": event["Report_Frequency"],
                    "Curn": currency,
                },
                header_type=ld.HeaderType.NAME,
            )
            time.sleep(SLEEP_BTWN_PULLS)
        except Exception as error:
            period_errors.append(f"{period}: {str(error).splitlines()[0]}")
            continue

        if forecasts is None or forecasts.empty:
            continue

        forecasts = rename_earnings_columns(forecasts, field_map)
        forecasts = clean_forecast_detail(forecasts, event)
        if forecasts.empty:
            continue

        return forecasts, period, None

    error_message = None
    if period_errors and len(period_errors) == len(candidate_periods):
        error_message = " | ".join(period_errors)

    return pd.DataFrame(), candidate_periods[0], error_message


def infer_absolute_forecast_periods(event: pd.Series) -> list[str]:
    ann_year = pd.Timestamp(event["Ann_Date"]).year
    frequency = str(event["Report_Frequency"])
    templates = FORECAST_PERIOD_TEMPLATES_BY_FREQUENCY.get(frequency, ())
    format_context = {
        "announcement_year": int(ann_year),
        "announcement_year_minus_1": int(ann_year - 1),
        "announcement_year_plus_1": int(ann_year + 1),
    }

    candidate_periods: list[str] = []
    for template in templates:
        if not template:
            continue
        period = str(template).format(**format_context).strip()
        if period and period not in candidate_periods:
            candidate_periods.append(period)

    return candidate_periods


def build_forecast_detail_field_map() -> dict[str, str]:
    estimate_field = f"{DETAILED_EPS_ESTIMATE_FIELD}()"

    return {
        estimate_field: "Forecast_EPS",
        f"{estimate_field}.Date": "Forecast_Date",
        f"{estimate_field}.BrokerName": "Broker_Name",
        f"{estimate_field}.AnalystName": "Analyst_Name",
        f"{estimate_field}.AnalystCode": "Analyst_Code",
        f"{estimate_field}.FPeriod": "Forecast_FPeriod",
    }


def clean_forecast_detail(
    forecasts: pd.DataFrame,
    event: pd.Series,
) -> pd.DataFrame:
    required_columns = [
        "Forecast_EPS",
        "Forecast_Date",
        "Broker_Name",
        "Analyst_Name",
        "Analyst_Code",
    ]
    out = forecasts.copy()

    if out.empty:
        return out

    for column in required_columns:
        if column not in out.columns:
            out[column] = pd.NA

    if "Forecast_FPeriod" not in out.columns:
        out["Forecast_FPeriod"] = pd.NA

    out["Forecast_Date"] = pd.to_datetime(
        out["Forecast_Date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    out["Forecast_EPS"] = pd.to_numeric(out["Forecast_EPS"], errors="coerce")
    out["Forecast_FPeriod"] = out["Forecast_FPeriod"].astype("string")
    out["Forecast_Window_Start"] = forecast_window_start(event["Ann_Date"])
    out["Forecast_Window_End"] = forecast_window_end(event["Ann_Date"])

    start_date = pd.Timestamp(out["Forecast_Window_Start"].iloc[0])
    end_date = pd.Timestamp(out["Forecast_Window_End"].iloc[0])

    out = out[
        (out["Forecast_Date"] >= start_date)
        & (out["Forecast_Date"] < end_date)
    ].copy()
    out = out.dropna(subset=["Forecast_EPS", "Forecast_Date"])
    out["Analyst_Key"] = build_analyst_key(out)
    out = out[out["Analyst_Key"].notna()]

    return out


def build_analyst_key(forecasts: pd.DataFrame) -> pd.Series:
    analyst_code = forecasts["Analyst_Code"].astype("string").str.strip()
    broker_name = forecasts["Broker_Name"].astype("string").str.strip()
    analyst_name = forecasts["Analyst_Name"].astype("string").str.strip()

    analyst_key = analyst_code.mask(
        analyst_code.isna() | (analyst_code == ""),
        broker_name.fillna("") + "|" + analyst_name.fillna(""),
    )
    analyst_key = analyst_key.mask(analyst_key == "|", pd.NA)

    return analyst_key


def keep_latest_forecast_per_analyst(forecasts: pd.DataFrame) -> pd.DataFrame:
    return (
        forecasts.sort_values(["Analyst_Key", "Forecast_Date"])
        .drop_duplicates(subset=["Analyst_Key"], keep="last")
        .reset_index(drop=True)
    )


def forecast_window_start(ann_date) -> str:
    return (
        pd.Timestamp(ann_date).normalize()
        - pd.Timedelta(days=FORECAST_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")


def forecast_window_end(ann_date) -> str:
    return pd.Timestamp(ann_date).normalize().strftime("%Y-%m-%d")


def add_lagged_prices(
    events: pd.DataFrame,
    price_history: pd.DataFrame,
    sample_size: dict,
    apply_filter: bool = True,
) -> pd.DataFrame:
    out = events.copy()
    out["Price_Lag_5"] = out.apply(
        lambda row: get_lagged_price(row, price_history),
        axis=1,
    )

    if apply_filter:
        out = apply_earnings_filter(
            out,
            out["Price_Lag_5"].notna() & (out["Price_Lag_5"] > 0),
            "Earnings events with positive lagged price",
            sample_size,
        )

    return out


def apply_earnings_filter(
    events: pd.DataFrame,
    condition,
    label: str,
    sample_size: dict,
) -> pd.DataFrame:
    filtered = events.loc[condition].copy()
    sample_size[label] = len(filtered)
    return filtered


def get_lagged_price(row: pd.Series, price_history: pd.DataFrame):
    ticker = row["Instrument"]
    ann_date = row["Ann_Date"]

    if ticker not in price_history.columns:
        return pd.NA

    series = price_history[ticker].dropna()

    try:
        idx_pos = series.index.get_indexer([ann_date], method="pad")[0]
    except Exception:
        return pd.NA

    if idx_pos >= 5:
        return series.iloc[idx_pos - 5]

    return pd.NA


def calculate_sue(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["Actual_EPS"] = pd.to_numeric(out["Actual_EPS"], errors="coerce")
    out["Forecast_Median"] = pd.to_numeric(out["Forecast_Median"], errors="coerce")
    out["Price_Lag_5"] = pd.to_numeric(out["Price_Lag_5"], errors="coerce")
    out["SUE"] = (
        out["Actual_EPS"] - out["Forecast_Median"]
    ) / out["Price_Lag_5"]
    return out
