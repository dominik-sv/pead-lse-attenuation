from __future__ import annotations

"""Download missing prior-announcement market-cap observations from Datastream.

The benchmark daily-market-cap cache covers benchmark constituents only, whereas
the regression event sample can include additional securities. This script finds
events in every configured PEAD sample for which that cache has no positive
market cap on the preceding observed trading day, requests the announcement
date's preceding nine calendar days, and stores the results in a supplemental
cache read by the regression suite.
"""

import json
from pathlib import Path
import sys
import time

import pandas as pd


PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.regression_suite import (  # noqa: E402
    SUPPLEMENTAL_PRE_ANNOUNCEMENT_MARKET_CAP_FILENAME,
)
from src.analysis.time_varying_analysis import (  # noqa: E402
    FIRM_IDENTIFIER_COLUMN,
    FORMATION_YEAR_COLUMN,
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
)
from src.core.pipeline_config import CURRENCY, SLEEP_BTWN_PULLS, TARGETED_RETURN_BATCH_SIZE  # noqa: E402
from src.core.pead_sample_variants import PEAD_EVENT_SAMPLE_VARIANTS  # noqa: E402
from src.core.project_paths import DATA_DIR  # noqa: E402
from src.pead.market_data_fetch import require_lseg, robust_get_data  # noqa: E402
from src.utils.pandas_utils import chunk_list  # noqa: E402

try:  # Import separately so importing this module remains local-only.
    import lseg.data as ld
except ImportError:  # pragma: no cover - checked before the first request.
    ld = None


MARKET_CAP_FIELD = "TR.CompanyMarketCap(Scale=6)"
MARKET_CAP_DATE_FIELD = "TR.CompanyMarketCap.Date"
REQUEST_LEAD_DAYS = 9
REQUEST_BATCH_SIZE = TARGETED_RETURN_BATCH_SIZE
SUPPLEMENTAL_METHOD = "supplemental_datastream"


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
        raise ValueError(f"Could not find a Datastream column containing {include!r}.")
    return matches[0]


def _reshape_market_cap_response(
    raw: pd.DataFrame | None,
    *,
    requested_instruments: list[str],
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Instrument", "MarketCap", "MarketCapMethod"])

    instrument_column = _find_column(raw.columns, include="INSTRUMENT")
    date_column = _find_column(raw.columns, include="DATE", exclude={instrument_column})
    market_cap_column = _find_column(
        raw.columns,
        include="COMPANYMARKETCAP",
        exclude={instrument_column, date_column},
    )
    result = raw.rename(
        columns={
            instrument_column: "Instrument",
            date_column: "Date",
            market_cap_column: "MarketCap",
        }
    )[["Date", "Instrument", "MarketCap"]].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    result["Instrument"] = result["Instrument"].astype("string").str.strip()
    result["MarketCap"] = pd.to_numeric(result["MarketCap"], errors="coerce")
    result = result.loc[
        result["Date"].notna()
        & result["Instrument"].isin(requested_instruments)
        & result["MarketCap"].gt(0)
    ].copy()
    result["MarketCapMethod"] = SUPPLEMENTAL_METHOD
    return result.drop_duplicates(["Date", "Instrument"], keep="last")


def _request_market_caps(instruments: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = robust_get_data(
        universe=instruments,
        fields=[MARKET_CAP_DATE_FIELD, MARKET_CAP_FIELD],
        parameters={
            "SDate": start.strftime("%Y-%m-%d"),
            "EDate": end.strftime("%Y-%m-%d"),
            "Frq": "D",
            "Curn": CURRENCY,
        },
        header_type=ld.HeaderType.NAME,
        max_retries=2,
        base_sleep=1.0,
    )
    return _reshape_market_cap_response(raw, requested_instruments=instruments)


def _load_available_market_caps(formation_year: int, instruments: set[str]) -> pd.DataFrame:
    cache_dir = DATA_DIR / "yearly" / str(formation_year) / "_cache"
    cache_paths = [
        cache_dir / "daily_market_caps_completed.csv",
        cache_dir / SUPPLEMENTAL_PRE_ANNOUNCEMENT_MARKET_CAP_FILENAME,
    ]
    if not cache_paths[0].exists():
        raise FileNotFoundError(f"Missing daily market-cap cache: {cache_paths[0]}")
    chunks: list[pd.DataFrame] = []
    for cache_priority, cache_path in enumerate(cache_paths):
        if not cache_path.exists():
            continue
        for chunk in pd.read_csv(
            cache_path,
            usecols=["Date", "Instrument", "MarketCap"],
            chunksize=250_000,
        ):
            chunk = chunk.loc[chunk["Instrument"].astype(str).isin(instruments)].copy()
            if not chunk.empty:
                chunk["_Cache_Priority"] = cache_priority
                chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["Date", "Instrument", "MarketCap"])

    cached = pd.concat(chunks, ignore_index=True)
    cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
    cached["Instrument"] = cached["Instrument"].astype("string").str.strip()
    cached["MarketCap"] = pd.to_numeric(cached["MarketCap"], errors="coerce")
    return (
        cached.loc[cached["Date"].notna() & cached["MarketCap"].gt(0)]
        .sort_values(["Instrument", "Date", "_Cache_Priority"], kind="stable")
        .drop_duplicates(["Date", "Instrument"], keep="last")
    )


def _events_missing_base_market_cap(events: pd.DataFrame, base_market_caps: pd.DataFrame) -> pd.DataFrame:
    missing_rows: list[dict[str, object]] = []
    for instrument, instrument_events in events.groupby(FIRM_IDENTIFIER_COLUMN, sort=False):
        available_dates = (
            base_market_caps.loc[
                base_market_caps["Instrument"].astype(str).eq(str(instrument)), "Date"
            ]
            .sort_values()
            .drop_duplicates()
            .to_numpy(dtype="datetime64[ns]")
        )
        for event in instrument_events.itertuples(index=False):
            announcement_date = pd.Timestamp(event.Ann_Date).normalize()
            position = available_dates.searchsorted(announcement_date.to_datetime64(), side="left") - 1
            if position >= 0:
                continue
            missing_rows.append(
                {
                    FORMATION_YEAR_COLUMN: int(getattr(event, FORMATION_YEAR_COLUMN)),
                    FIRM_IDENTIFIER_COLUMN: str(getattr(event, FIRM_IDENTIFIER_COLUMN)),
                    "Ann_Date": announcement_date,
                    "Request_Start": announcement_date - pd.Timedelta(days=REQUEST_LEAD_DAYS),
                    "Request_End": announcement_date - pd.Timedelta(days=1),
                }
            )
    return pd.DataFrame(missing_rows)


def _upsert_supplemental_cache(formation_year: int, downloaded: pd.DataFrame) -> Path:
    cache_dir = DATA_DIR / "yearly" / str(formation_year) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / SUPPLEMENTAL_PRE_ANNOUNCEMENT_MARKET_CAP_FILENAME
    existing = (
        pd.read_csv(destination)
        if destination.exists()
        else pd.DataFrame(columns=["Date", "Instrument", "MarketCap", "MarketCapMethod"])
    )
    if existing.empty:
        combined = downloaded.copy()
    elif downloaded.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, downloaded], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.normalize()
    combined["Instrument"] = combined["Instrument"].astype("string").str.strip()
    combined["MarketCap"] = pd.to_numeric(combined["MarketCap"], errors="coerce")
    combined = combined.loc[
        combined["Date"].notna() & combined["Instrument"].notna() & combined["MarketCap"].gt(0)
    ].drop_duplicates(["Date", "Instrument"], keep="last")
    combined.sort_values(["Instrument", "Date"]).to_csv(destination, index=False)
    return destination


def main() -> None:
    require_lseg()
    if ld is None:
        raise ModuleNotFoundError("lseg.data must be importable to request supplemental market caps.")

    ld.open_session()
    try:
        event_level = pd.concat(
            [
                collapse_to_event_level(
                    load_abnormal_returns_with_groups(
                        DATA_DIR,
                        abnormal_returns_filename=sample.abnormal_returns_filename,
                    )
                )
                for sample in PEAD_EVENT_SAMPLE_VARIANTS
            ],
            ignore_index=True,
        )
        event_level["Ann_Date"] = pd.to_datetime(event_level["Ann_Date"], errors="coerce").dt.normalize()
        event_level[FIRM_IDENTIFIER_COLUMN] = event_level[FIRM_IDENTIFIER_COLUMN].astype("string").str.strip()
        event_level[FORMATION_YEAR_COLUMN] = pd.to_numeric(
            event_level[FORMATION_YEAR_COLUMN], errors="coerce"
        )
        events = event_level.dropna(
            subset=[FORMATION_YEAR_COLUMN, FIRM_IDENTIFIER_COLUMN, "Ann_Date"]
        ).copy()
        events[FORMATION_YEAR_COLUMN] = events[FORMATION_YEAR_COLUMN].astype(int)
        events = events.drop_duplicates(
            [FORMATION_YEAR_COLUMN, FIRM_IDENTIFIER_COLUMN, "Ann_Date"]
        )

        missing_by_year: list[pd.DataFrame] = []
        for formation_year, year_events in events.groupby(FORMATION_YEAR_COLUMN, sort=True):
            instruments = set(year_events[FIRM_IDENTIFIER_COLUMN].astype(str))
            available_market_caps = _load_available_market_caps(int(formation_year), instruments)
            missing_by_year.append(_events_missing_base_market_cap(year_events, available_market_caps))

        missing_events = pd.concat(missing_by_year, ignore_index=True)
        if missing_events.empty:
            print("All regression events already have a positive pre-announcement market-cap record.")
            return

        audit_path = DATA_DIR / "pre_announcement_market_cap_request_audit.csv"
        missing_events.sort_values([FORMATION_YEAR_COLUMN, FIRM_IDENTIFIER_COLUMN, "Ann_Date"]).to_csv(
            audit_path, index=False
        )
        print(
            f"Found {len(missing_events):,} earnings events without a prior-trading-day market-cap record "
            f"in the available daily caches. Request plan saved to {audit_path}."
        )

        total_downloaded_rows = 0
        for formation_year, year_missing in missing_events.groupby(FORMATION_YEAR_COLUMN, sort=True):
            downloaded_parts: list[pd.DataFrame] = []
            request_groups = year_missing.groupby(["Request_Start", "Request_End"], sort=True)
            for (request_start, request_end), requests in request_groups:
                instruments = sorted(requests[FIRM_IDENTIFIER_COLUMN].astype(str).unique().tolist())
                for instrument_batch in chunk_list(instruments, REQUEST_BATCH_SIZE):
                    downloaded_parts.append(
                        _request_market_caps(list(instrument_batch), request_start, request_end)
                    )
                    time.sleep(SLEEP_BTWN_PULLS)
            downloaded = pd.concat(downloaded_parts, ignore_index=True)
            destination = _upsert_supplemental_cache(int(formation_year), downloaded)
            total_downloaded_rows += len(downloaded)
            print(
                f"{int(formation_year)}: saved {len(downloaded):,} positive market-cap observations to {destination}."
            )

        summary_path = DATA_DIR / "pre_announcement_market_cap_request_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "missing_event_count": int(len(missing_events)),
                    "requested_instrument_count": int(missing_events[FIRM_IDENTIFIER_COLUMN].nunique()),
                    "downloaded_positive_market_cap_row_count": int(total_downloaded_rows),
                    "request_lead_calendar_days": REQUEST_LEAD_DAYS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Supplemental request summary saved to {summary_path}.")
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
