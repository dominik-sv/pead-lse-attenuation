"""Daily market-cap inputs for fixed-constituent benchmark portfolios."""

from __future__ import annotations

import json
import time

import pandas as pd

try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - local cached-data workflows do not need LSEG.
    ld = None

from ..core.pipeline_config import TARGETED_RETURN_BATCH_SIZE
from ..utils.pandas_utils import chunk_list
from .market_data_fetch import require_lseg, robust_get_data


MARKET_CAP_FIELD = "TR.CompanyMarketCap(Scale=6)"
MARKET_CAP_DATE_FIELD = "TR.CompanyMarketCap.Date"
DAILY_MARKET_CAP_WEIGHTING_METHOD = (
    "preceding_trading_day_market_cap_with_june_formation_anchor"
)


def _normalize_label(value: object) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _find_column(
    columns: pd.Index,
    *,
    include: str,
    exclude: set[object] | None = None,
) -> object:
    excluded = exclude or set()
    normalized_include = _normalize_label(include)
    matches = [
        column
        for column in columns
        if column not in excluded and normalized_include in _normalize_label(column)
    ]
    if not matches:
        raise ValueError(
            f"Could not identify a column containing {include!r}. "
            f"Returned columns: {list(columns)}"
        )
    return matches[0]


def _reshape_market_cap_history(
    raw: pd.DataFrame | None,
    *,
    requested_instruments: list[str],
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Instrument", "MarketCap"])

    out = raw.copy()
    instrument_column = _find_column(out.columns, include="INSTRUMENT")
    date_column = _find_column(
        out.columns,
        include="DATE",
        exclude={instrument_column},
    )
    market_cap_column = _find_column(
        out.columns,
        include="COMPANYMARKETCAP",
        exclude={instrument_column, date_column},
    )
    out = out.rename(
        columns={
            instrument_column: "Instrument",
            date_column: "Date",
            market_cap_column: "MarketCap",
        }
    )
    out = out.loc[:, ["Date", "Instrument", "MarketCap"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Instrument"] = out["Instrument"].astype("string").str.strip()
    out["MarketCap"] = pd.to_numeric(out["MarketCap"], errors="coerce")
    out = out.dropna(subset=["Date", "Instrument"])
    out = out.loc[out["Instrument"].isin(requested_instruments)].copy()
    return out.drop_duplicates(["Date", "Instrument"], keep="last")


def _request_market_cap_history(
    instruments: list[str],
    *,
    start: str,
    end: str,
    currency: str,
) -> pd.DataFrame:
    require_lseg()
    if ld is None:  # Keeps static type checkers and cached workflows explicit.
        raise ModuleNotFoundError("lseg.data is required to request daily market caps.")
    raw = robust_get_data(
        universe=instruments,
        fields=[MARKET_CAP_DATE_FIELD, MARKET_CAP_FIELD],
        parameters={"SDate": start, "EDate": end, "Frq": "D", "Curn": currency},
        header_type=ld.HeaderType.NAME,
        max_retries=2,
        base_sleep=1.0,
    )
    return _reshape_market_cap_history(raw, requested_instruments=instruments)


def _request_batch_with_fallback(
    instruments: list[str],
    *,
    start: str,
    end: str,
    currency: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    batch_history = pd.DataFrame(columns=["Date", "Instrument", "MarketCap"])
    try:
        batch_history = _request_market_cap_history(
            instruments, start=start, end=end, currency=currency
        )
        returned = set(batch_history["Instrument"].dropna().astype(str))
        instruments_to_retry = [
            instrument for instrument in instruments if instrument not in returned
        ]
        if not instruments_to_retry:
            return batch_history, []
    except Exception as batch_error:
        instruments_to_retry = list(instruments)
        if len(instruments) > 1:
            print(
                f"Batch request failed after retries ({batch_error}). "
                "Retrying its instruments individually."
            )

    parts: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for instrument in instruments_to_retry:
        try:
            part = _request_market_cap_history(
                [instrument], start=start, end=end, currency=currency
            )
            if part.empty:
                raise ValueError(f"No market-cap rows returned for {instrument}")
            parts.append(part)
        except Exception as error:
            failures.append({"instrument": instrument, "error": str(error)})

    parts.insert(0, batch_history)
    return pd.concat(parts, ignore_index=True), failures


def _chunk_cache_is_valid(
    path,
    *,
    expected_instruments: set[str],
    start: str,
    end: str,
    currency: str,
) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        cached = pd.read_csv(path, usecols=["Date", "Instrument", "MarketCap"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_metadata = {
        "instruments": sorted(expected_instruments),
        "start": start,
        "end": end,
        "field": MARKET_CAP_FIELD,
        "date_field": MARKET_CAP_DATE_FIELD,
        "currency": currency,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return False
    cached_instruments = set(cached["Instrument"].dropna().astype(str).str.strip())
    unavailable = set(metadata.get("unavailable_instruments", []))
    return cached_instruments.union(unavailable) == expected_instruments


def download_daily_market_cap_history(
    *,
    benchmark_constituents: pd.DataFrame,
    return_history: pd.DataFrame,
    year_context,
    currency: str,
    batch_size: int = TARGETED_RETURN_BATCH_SIZE,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    """Download/cache constituent caps, including a short pre-window lead-in."""
    instruments = sorted(
        set(benchmark_constituents["Instrument"].dropna().astype("string").str.strip())
    )
    if not instruments:
        raise ValueError(f"No benchmark constituents are available for {year_context.year}.")
    dates = pd.DatetimeIndex(pd.to_datetime(return_history.index, errors="coerce")).dropna()
    if dates.empty:
        raise ValueError(f"No benchmark return dates are available for {year_context.year}.")

    start = (dates.min() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    cache_dir = year_context.cache_dir / "daily_market_cap_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for batch_number, batch in enumerate(chunk_list(instruments, batch_size), start=1):
        batch = list(batch)
        chunk_path = cache_dir / f"batch_{batch_number:04d}.csv"
        if _chunk_cache_is_valid(
            chunk_path,
            expected_instruments=set(batch),
            start=start,
            end=end,
            currency=currency,
        ):
            continue
        history, batch_failures = _request_batch_with_fallback(
            batch, start=start, end=end, currency=currency
        )
        history.to_csv(chunk_path, index=False)
        chunk_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "instruments": sorted(batch),
                    "start": start,
                    "end": end,
                    "field": MARKET_CAP_FIELD,
                    "date_field": MARKET_CAP_DATE_FIELD,
                    "currency": currency,
                    "unavailable_instruments": sorted(
                        failure["instrument"] for failure in batch_failures
                    ),
                    "failures": batch_failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    failures: list[dict[str, str]] = []
    for metadata_path in sorted(cache_dir.glob("batch_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failures.extend(metadata.get("failures", []))
    failure_path = year_context.cache_dir / "daily_market_cap_request_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return load_cached_daily_market_caps(year_context)


def load_cached_daily_market_caps(year_context) -> pd.DataFrame:
    chunk_paths = sorted((year_context.cache_dir / "daily_market_cap_chunks").glob("batch_*.csv"))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No daily market-cap chunks found for formation year {year_context.year}."
        )
    market_caps = pd.concat((pd.read_csv(path) for path in chunk_paths), ignore_index=True)
    market_caps["Date"] = pd.to_datetime(market_caps["Date"], errors="coerce")
    market_caps["Instrument"] = market_caps["Instrument"].astype("string").str.strip()
    market_caps["MarketCap"] = pd.to_numeric(market_caps["MarketCap"], errors="coerce")
    return market_caps.dropna(subset=["Date", "Instrument"]).drop_duplicates(
        ["Date", "Instrument"], keep="last"
    )


def complete_daily_market_cap_panel(
    *,
    benchmark_constituents: pd.DataFrame,
    return_history: pd.DataFrame,
    price_history: pd.DataFrame,
    observed_market_caps: pd.DataFrame,
    year_context,
) -> pd.DataFrame:
    """Fill unavailable daily caps from price-scaled or fixed formation caps."""
    instruments = pd.Index(
        benchmark_constituents["Instrument"].dropna().astype("string").str.strip().unique(),
        name="Instrument",
    )
    dates = pd.DatetimeIndex(pd.to_datetime(return_history.index, errors="coerce")).dropna()
    dates = dates.unique().sort_values()
    preceding_dates = observed_market_caps.loc[
        observed_market_caps["Date"].lt(dates.min()), "Date"
    ]
    if not preceding_dates.empty:
        dates = pd.DatetimeIndex([preceding_dates.max(), *dates.tolist()], name="Date")
    full_index = pd.MultiIndex.from_product([dates, instruments], names=["Date", "Instrument"])

    observed = (
        observed_market_caps.set_index(["Date", "Instrument"])["MarketCap"]
        .reindex(full_index)
        .astype("float64")
        .where(lambda values: values.gt(0))
    )
    constituent_reference = (
        benchmark_constituents.drop_duplicates("Instrument", keep="first")
        .set_index("Instrument")
        .reindex(instruments)
    )
    formation_market_cap = pd.to_numeric(
        constituent_reference["Market_Cap_Current"], errors="coerce"
    ).where(lambda values: values.gt(0))
    if formation_market_cap.isna().any():
        raise ValueError(
            "Formation-date market capitalization is unavailable for "
            f"{formation_market_cap.isna().sum()} benchmark constituent(s) in {year_context.year}."
        )

    prices = price_history.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[prices.index.notna()].sort_index()
    prices.columns = prices.columns.astype("string").str.strip()
    prices = prices.reindex(index=dates, columns=instruments)
    prices = prices.apply(pd.to_numeric, errors="coerce").where(lambda values: values.gt(0))
    formation_date = pd.Timestamp(year_context.formation_date)
    formation_price_source = pd.to_numeric(
        constituent_reference.get("Price"), errors="coerce"
    ).where(lambda values: values.gt(0))
    historical_prices = prices.loc[prices.index <= formation_date]
    if not historical_prices.empty:
        formation_price_source = formation_price_source.combine_first(
            historical_prices.ffill().iloc[-1]
        )
    formation_prices = formation_price_source.where(lambda values: values.gt(0))

    formation_cap_panel = pd.Series(
        full_index.get_level_values("Instrument").map(formation_market_cap),
        index=full_index,
        dtype="float64",
    )
    formation_price_panel = pd.Series(
        full_index.get_level_values("Instrument").map(formation_prices),
        index=full_index,
        dtype="float64",
    )
    price_scaled = formation_cap_panel.mul(
        prices.stack(future_stack=True).reindex(full_index).div(formation_price_panel)
    ).where(lambda values: values.gt(0))
    completed = observed.combine_first(price_scaled).combine_first(formation_cap_panel)
    if completed.isna().any() or completed.le(0).any():
        raise ValueError(f"Market-cap fallback failed to complete the panel for {year_context.year}.")

    method = pd.Series("observed_market_cap", index=full_index, dtype="string")
    method.loc[observed.isna() & price_scaled.notna()] = "formation_cap_scaled_by_price_ratio"
    method.loc[observed.isna() & price_scaled.isna()] = "formation_market_cap_constant"
    completed_panel = pd.DataFrame(
        {"MarketCap": completed, "MarketCapMethod": method}
    ).reset_index()
    completed_panel.to_csv(year_context.cache_dir / "daily_market_caps_completed.csv", index=False)
    (year_context.cache_dir / "daily_market_cap_completion_summary.json").write_text(
        json.dumps(
            {
                "formation_year": year_context.year,
                "total_constituent_date_rows": int(len(completed_panel)),
                "remaining_missing_market_caps": int(completed_panel["MarketCap"].isna().sum()),
                "market_cap_method_counts": {
                    str(key): int(value)
                    for key, value in completed_panel["MarketCapMethod"].value_counts().items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return completed_panel


def build_daily_value_weighted_benchmark_returns(
    *,
    benchmark_constituents: pd.DataFrame,
    return_history: pd.DataFrame,
    completed_market_caps: pd.DataFrame,
    formation_date: str,
    portfolio_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return preceding-day-cap-weighted returns for fixed June constituents."""
    returns = return_history.copy()
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    returns = returns.loc[returns.index.notna()].sort_index()
    returns.columns = returns.columns.astype("string").str.strip()
    returns_long = (
        returns.rename_axis("Date")
        .reset_index()
        .melt(id_vars="Date", var_name="Instrument", value_name="TotalReturn")
    )
    returns_long["TotalReturn"] = pd.to_numeric(returns_long["TotalReturn"], errors="coerce")
    returns_long = returns_long.dropna(subset=["Date", "Instrument", "TotalReturn"])

    cap_wide = completed_market_caps.pivot(index="Date", columns="Instrument", values="MarketCap")
    cap_wide.columns = cap_wide.columns.astype("string").str.strip()
    preceding_dates = cap_wide.index[cap_wide.index < returns.index.min()]
    weighting_calendar = returns.index
    if len(preceding_dates):
        weighting_calendar = pd.DatetimeIndex([preceding_dates.max(), *returns.index.tolist()])
    lagged_caps = cap_wide.reindex(weighting_calendar).shift(1).reindex(returns.index)

    july_dates = returns.index[returns.index > pd.Timestamp(formation_date)]
    if not len(july_dates):
        raise ValueError(f"No post-formation trading date is available after {formation_date}.")
    formation_caps = (
        completed_market_caps.loc[
            completed_market_caps["Date"].le(pd.Timestamp(formation_date))
        ]
        .sort_values("Date")
        .drop_duplicates("Instrument", keep="last")
        .set_index("Instrument")["MarketCap"]
    )
    lagged_caps.loc[july_dates.min()] = formation_caps.reindex(lagged_caps.columns)
    lagged_caps = (
        lagged_caps.rename_axis(index="Date", columns="Instrument")
        .stack(future_stack=True)
        .rename("LaggedMarketCap")
        .reset_index()
    )

    portfolio_assignment = benchmark_constituents.loc[
        :, ["Instrument", "Benchmark_Portfolio"]
    ].drop_duplicates("Instrument")
    panel = returns_long.merge(
        portfolio_assignment, on="Instrument", how="inner", validate="many_to_one"
    ).merge(
        lagged_caps, on=["Date", "Instrument"], how="left", validate="one_to_one"
    )
    eligible = panel.loc[panel["LaggedMarketCap"].gt(0)].copy()
    eligible["WeightedReturn"] = eligible["TotalReturn"] * eligible["LaggedMarketCap"]
    grouped = eligible.groupby(["Date", "Benchmark_Portfolio"], observed=True)
    portfolio_returns = (
        grouped["WeightedReturn"].sum(min_count=1).div(grouped["LaggedMarketCap"].sum()).unstack()
    )
    unexpected = set(portfolio_returns.columns).difference(portfolio_labels)
    if unexpected:
        raise ValueError(f"Unexpected benchmark portfolio label(s): {sorted(unexpected)}.")
    portfolio_returns = portfolio_returns.reindex(index=returns.index, columns=list(portfolio_labels))
    portfolio_returns.index.name = "Date"
    coverage = grouped.agg(
        constituent_count=("Instrument", "nunique"),
        lagged_market_cap_sum=("LaggedMarketCap", "sum"),
    ).reset_index()
    return portfolio_returns, coverage
