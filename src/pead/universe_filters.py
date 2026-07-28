from __future__ import annotations

import time

import pandas as pd
try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - local cached-data workflows do not need LSEG.
    ld = None
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..core.pipeline_config import (
    MARKET_CAP_SIZE_SPLIT_PERCENTILE,
    ORDINARY_SHARE_TYPES,
    SLEEP_BTWN_PULLS,
    UNIVERSE_SOURCE,
    UNIVERSE_ENRICHMENT_BATCH_SIZE,
)
from .market_cap_splits import (
    assign_market_cap_size_split_from_breakpoint,
    build_market_cap_size_split_metadata,
    validate_market_cap_size_split_percentile,
)
from ..utils.pandas_utils import chunk_list

RAW_HISTORICAL_CANDIDATES_LABEL = "Raw historical candidates"
TARGET_EXCHANGE_LABEL = "Target exchange"
PRIMARY_LISTINGS_LABEL = "Primary listings"
ACTIVE_OVERLAP_LABEL = "Active through formation window / valid historical overlap"
ENRICHMENT_SUCCESS_LABEL = "Enrichment success"
USABLE_IDENTIFIER_LABEL = "Usable identifier"
HISTORICAL_ORDINARY_COMMON_SHARE_CANDIDATES_LABEL = (
    "Historical ordinary/common share candidates"
)
ORDINARY_SHARES_LABEL = "Ordinary shares"
NON_MISSING_DATA_LABEL = "Non-missing book-to-market inputs"
POSITIVE_BOOK_TO_MARKET_LABEL = "Positive book-to-market last fiscal year"
REPORTED_ORDINARY_COMMON_SHARES_LABEL = "Ordinary/common shares"
REPORTED_REQUIRED_ACCOUNTING_AND_MARKET_DATA_LABEL = (
    "Required accounting and market data available"
)

CANDIDATE_AUDIT_COLUMNS = [
    "Instrument",
    "RIC",
    "Historical_RIC",
    "Archived_RIC",
    "Ticker",
    "Name",
    "Instrument_Type",
    "Security_Type",
    "Exchange_Name",
    "Exchange_Code",
    "Exchange_Local_Code",
    "Listing_Status",
    "Asset_State",
    "Retire_Date",
    "Primary_Listing_Flag",
    "Identifier_Source",
    "Candidate_Fetch_Method",
    "Universe_Window_Start",
    "Universe_Window_End",
]

RAW_DISCOVERY_COLUMN_MAP = {
    "RIC": "Archived_RIC",
    "DocumentTitle": "Name",
    "CommonName": "Name",
    "ExchangeName": "Exchange_Name",
    "ExchangeCode": "Exchange_Local_Code",
    "ExchangeMarketIdCode": "Exchange_Code",
    "AssetState": "Asset_State",
    "ListingStatus": "Listing_Status",
    "IsPrimaryIssueRIC": "Primary_Listing_Flag",
    "InstrumentType": "Instrument_Type",
    "SecurityType": "Security_Type",
    "RetireDate": "Retire_Date",
}

LSEG_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0, 20.0)


def require_lseg() -> None:
    if ld is None:
        raise ModuleNotFoundError(
            "lseg.data is required for remote universe enrichment, but it is not "
            "installed in the active Python runtime. Use cached local inputs or run "
            "this script in an environment with the LSEG SDK available."
        )


def concat_meaningful_frames(
    frames: list[pd.DataFrame],
    *,
    ignore_index: bool = False,
) -> pd.DataFrame:
    """Concatenate frames after excluding empty or all-NA inputs."""
    schema_frame = next((frame for frame in frames if frame is not None), None)
    meaningful_frames = [
        frame
        for frame in frames
        if frame is not None and not frame.empty and not frame.isna().all().all()
    ]

    if not meaningful_frames:
        if schema_frame is None:
            return pd.DataFrame()
        return schema_frame.iloc[0:0].copy()

    return pd.concat(meaningful_frames, ignore_index=ignore_index)


def is_transient_lseg_connection_error(error: Exception) -> bool:
    message = str(error)
    transient_markers = (
        "[WinError 10054]",
        "forcibly closed by the remote host",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "backend error. 400 bad request",
        "requested universes:",
        "remoteprotocolerror",
        "session is not opened",
    )
    message_lower = message.lower()
    return any(marker.lower() in message_lower for marker in transient_markers)


def is_lseg_batch_data_error(error: Exception) -> bool:
    message_lower = str(error).lower()
    return (
        "unable to collect data for the field" in message_lower
        and "requested universes:" in message_lower
    )


def lseg_get_data_with_retry(
    *,
    universe,
    fields,
    parameters: dict,
    header_type,
) -> pd.DataFrame | None:
    require_lseg()
    attempts = len(LSEG_RETRY_DELAYS_SECONDS) + 1
    last_error = None

    for attempt_index in range(attempts):
        try:
            return ld.get_data(
                universe=universe,
                fields=fields,
                parameters=parameters,
                header_type=header_type,
            )
        except Exception as error:
            last_error = error
            is_last_attempt = attempt_index == attempts - 1
            if is_last_attempt or not is_transient_lseg_connection_error(error):
                raise
            try:
                ld.open_session()
            except Exception:
                pass
            time.sleep(LSEG_RETRY_DELAYS_SECONDS[attempt_index])

    raise last_error


def lseg_get_data_with_batch_fallback(
    *,
    universe,
    fields,
    parameters: dict,
    header_type,
) -> pd.DataFrame | None:
    try:
        return lseg_get_data_with_retry(
            universe=universe,
            fields=fields,
            parameters=parameters,
            header_type=header_type,
        )
    except Exception as error:
        universe_list = list(universe)
        if not is_lseg_batch_data_error(error):
            raise
        if len(universe_list) <= 1:
            failed_instrument = universe_list[0] if universe_list else "<empty>"
            print(
                "Skipping LSEG enrichment for instrument "
                f"{failed_instrument!r}: {error}"
            )
            return None

        midpoint = len(universe_list) // 2
        left = lseg_get_data_with_batch_fallback(
            universe=universe_list[:midpoint],
            fields=fields,
            parameters=parameters,
            header_type=header_type,
        )
        right = lseg_get_data_with_batch_fallback(
            universe=universe_list[midpoint:],
            fields=fields,
            parameters=parameters,
            header_type=header_type,
        )
        return concat_meaningful_frames([left, right], ignore_index=True)


def build_french_benchmark_reference_universe_for_year(
    year_context,
    currency: str,
    exchange_definitions: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    reference_universe_parts = []
    exchange_diagnostics = []
    total_requests = len(exchange_definitions) * 2
    with tqdm(
        total=total_requests,
        desc=f"French benchmark requests {year_context.year}",
    ) as progress_bar:
        for _, exchange_code in exchange_definitions:
            historical_candidates = fetch_historical_exchange_candidates(
                year_context=year_context,
                currency=currency,
                exchange=exchange_code,
            )
            normalized_candidates = normalize_historical_candidates(
                historical_candidates,
                year_context=year_context,
            )
            progress_bar.update(1)

            enriched_candidates = enrich_candidates_with_universe_fields(
                normalized_candidates,
                year_context=year_context,
                currency=currency,
            )
            progress_bar.update(1)

            eligible_reference_universe = build_french_benchmark_reference_universe(
                df=enriched_candidates,
                year_context=year_context,
                exchange=exchange_code,
            )

            if eligible_reference_universe.empty:
                enriched_candidates = backfill_french_benchmark_reference_fields(
                    candidates=enriched_candidates,
                    year_context=year_context,
                    currency=currency,
                )
                eligible_reference_universe = build_french_benchmark_reference_universe(
                    df=enriched_candidates,
                    year_context=year_context,
                    exchange=exchange_code,
                )

            if not eligible_reference_universe.empty:
                reference_universe_parts.append(eligible_reference_universe)

            exchange_diagnostics.append(
                build_single_exchange_french_reference_universe_diagnostic(
                    df=enriched_candidates,
                    year_context=year_context,
                    exchange=exchange_code,
                )
            )

    if not reference_universe_parts:
        raise RuntimeError(
            "French benchmark reference universe is empty across all configured "
            f"European exchanges. {'; '.join(exchange_diagnostics)}"
        )

    combined = pd.concat(reference_universe_parts, ignore_index=True)
    combined = (
        combined.sort_values(
            ["Instrument", "Identifier_Source", "Retire_Date", "Exchange_Code"]
        )
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )
    return combined


def build_single_exchange_french_reference_universe_diagnostic(
    df: pd.DataFrame,
    year_context,
    exchange: str,
) -> str:
    filtered_candidates, _ = filter_to_common_stock_candidates(
        df=df,
        year_context=year_context,
        exchange=exchange,
        sample_size=None,
    )
    if filtered_candidates.empty:
        return f"{exchange}: common-stock candidates=0, benchmark-eligible=0"

    current_market_cap = pd.to_numeric(
        filtered_candidates["Market_Cap_Current"],
        errors="coerce",
    )
    book_to_market = pd.to_numeric(
        filtered_candidates["BM_French"],
        errors="coerce",
    )
    eligible = (
        current_market_cap.notna()
        & book_to_market.notna()
        & book_to_market.gt(0)
    )
    return (
        f"{exchange}: common-stock candidates={len(filtered_candidates)}, "
        f"benchmark-eligible={int(eligible.sum())}, "
        f"BM_French non-missing={int(book_to_market.notna().sum())}"
    )


def fetch_and_parse_exchange_universe(
    year_context,
    currency: str,
    exchange: str,
    progress_bar=None,
) -> pd.DataFrame:
    historical_candidates = fetch_historical_exchange_candidates(
        year_context=year_context,
        currency=currency,
        exchange=exchange,
    )
    if progress_bar is not None:
        progress_bar.update(1)
    normalized_candidates = normalize_historical_candidates(
        historical_candidates,
        year_context=year_context,
    )
    enriched_candidates = enrich_candidates_with_universe_fields(
        normalized_candidates,
        year_context=year_context,
        currency=currency,
    )
    if progress_bar is not None:
        progress_bar.update(1)
    return parse_universe_columns(enriched_candidates)


def fetch_historical_exchange_candidates(
    year_context,
    currency: str,
    exchange: str,
) -> pd.DataFrame:
    """
    Fetch a broad exchange candidate list around formation date.

    This project currently uses one explicit interim source only:
    a broad Workspace equity snapshot that removes the previous active-only
    restriction. We do not treat discovery search as a historical-membership
    solution because it does not provide a clean point-in-time exchange universe.
    The fetch layer remains isolated so a proper historical population endpoint
    can replace this implementation later.
    """
    try:
        candidates = fetch_candidates_from_workspace_snapshot(
            year_context=year_context,
            currency=currency,
            exchange=exchange,
        )
    except Exception as error:
        raise RuntimeError(
            "Historical exchange candidate fetch failed for "
            f"{exchange} on {year_context.formation_date} "
            f"using the interim Workspace snapshot source."
        ) from error

    if candidates is None or candidates.empty:
        return empty_historical_candidate_frame()

    return candidates


def fetch_candidates_from_workspace_snapshot(
    year_context,
    currency: str,
    exchange: str,
) -> pd.DataFrame:
    """
    Fallback broad Workspace universe fetch.

    This intentionally removes the prior active-only restriction. It is still
    not a definitive historical-membership endpoint, so the resulting candidate
    set should be viewed as an auditable interim approximation until a dedicated
    historical exchange-population API is wired in.
    """
    universe = [
        f"""SCREEN(
            U(IN(Equity(active or inactive,public,primary))/*UNV:Public*/),
            IN(TR.ExchangeMarketIdCode,"{exchange}"),
            CURN="{currency}"
        )"""
    ]

    fields = [
        "TR.RIC",
        "TR.ExchangeTicker",
        "TR.CommonName",
        "TR.InstrumentType",
        "TR.ExchangeName",
        "TR.ExchangeMarketIdCode",
        "TR.RetireDate",
    ]

    out = lseg_get_data_with_retry(
        universe=universe,
        fields=fields,
        parameters={"Curn": currency},
        header_type=ld.HeaderType.NAME,
    )

    if out is None or out.empty:
        return empty_historical_candidate_frame()

    rename_map = {
        "TR.RIC": "Archived_RIC",
        "TR.ExchangeTicker": "Ticker",
        "TR.CommonName": "Name",
        "TR.InstrumentType": "Instrument_Type",
        "TR.ExchangeName": "Exchange_Name",
        "TR.ExchangeMarketIdCode": "Exchange_Code",
        "TR.RetireDate": "Retire_Date",
    }

    out = rename_lseg_columns(out, rename_map)
    out["Candidate_Fetch_Method"] = "workspace_exchange_snapshot"
    return out


def empty_historical_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_AUDIT_COLUMNS)


def normalize_historical_candidates(
    candidates: pd.DataFrame,
    year_context,
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        out = empty_historical_candidate_frame()
        out["Universe_Window_Start"] = pd.Series(dtype="string")
        out["Universe_Window_End"] = pd.Series(dtype="string")
        return out

    out = candidates.copy()
    out = out.rename(columns=RAW_DISCOVERY_COLUMN_MAP)

    for column in [
        "Instrument",
        "RIC",
        "Historical_RIC",
        "Archived_RIC",
        "Ticker",
        "Name",
        "Instrument_Type",
        "Security_Type",
        "Exchange_Name",
        "Exchange_Code",
        "Exchange_Local_Code",
        "Listing_Status",
        "Asset_State",
        "Identifier_Source",
        "Candidate_Fetch_Method",
    ]:
        if column not in out.columns:
            out[column] = pd.NA

    if "Retire_Date" not in out.columns:
        out["Retire_Date"] = pd.NaT
    if "Primary_Listing_Flag" not in out.columns:
        out["Primary_Listing_Flag"] = pd.NA

    out["Universe_Window_Start"] = year_context.universe_window_start
    out["Universe_Window_End"] = year_context.universe_window_end

    out["Archived_RIC"] = coalesce_text_columns(
        out,
        ["Archived_RIC", "RIC", "Instrument"],
    )
    out["Historical_RIC"] = coalesce_text_columns(
        out,
        ["Historical_RIC", "RIC", "Archived_RIC"],
    )
    out["RIC"] = coalesce_text_columns(
        out,
        ["RIC", "Historical_RIC", "Archived_RIC", "Instrument"],
    )
    out["Instrument"] = choose_best_instrument_identifier(out)
    out["Identifier_Source"] = build_identifier_source(out)
    out["Primary_Listing_Flag"] = parse_boolean_column(out["Primary_Listing_Flag"])
    out["Retire_Date"] = parse_datetime_column(out["Retire_Date"])

    for column in (
        "Instrument",
        "RIC",
        "Historical_RIC",
        "Archived_RIC",
        "Ticker",
        "Name",
        "Instrument_Type",
        "Security_Type",
        "Exchange_Name",
        "Exchange_Code",
        "Exchange_Local_Code",
        "Listing_Status",
        "Asset_State",
        "Identifier_Source",
        "Candidate_Fetch_Method",
        "Universe_Window_Start",
        "Universe_Window_End",
    ):
        out[column] = out[column].astype("string").str.strip()

    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out.drop_duplicates(
        subset=["Instrument", "Historical_RIC", "Archived_RIC", "Name"],
        keep="first",
    ).reset_index(drop=True)

    return ensure_universe_output_columns(out)


def enrich_candidates_with_universe_fields(
    candidates: pd.DataFrame,
    year_context,
    currency: str,
    fiscal_anchor_date: str | None = None,
    include_price_to_book: bool = True,
) -> pd.DataFrame:
    out = ensure_universe_output_columns(candidates.copy())

    instrument_list = (
        out["Instrument"]
        .dropna()
        .astype("string")
        .str.strip()
    )
    instrument_list = instrument_list[instrument_list != ""].drop_duplicates().tolist()

    if not instrument_list:
        return ensure_universe_output_columns(out)

    field_map = build_universe_field_map(
        year_context,
        fiscal_anchor_date=fiscal_anchor_date,
        include_price_to_book=include_price_to_book,
    )
    enrichment_batches = []
    instrument_batches = list(
        chunk_list(instrument_list, UNIVERSE_ENRICHMENT_BATCH_SIZE)
    )
    for batch_index, instrument_batch in enumerate(instrument_batches):
        batch_enrichment = lseg_get_data_with_batch_fallback(
            universe=instrument_batch,
            fields=list(field_map.keys()),
            parameters={"Curn": currency},
            header_type=ld.HeaderType.NAME,
        )
        if batch_enrichment is not None and not batch_enrichment.empty:
            enrichment_batches.append(batch_enrichment)

        is_last_batch = batch_index == len(instrument_batches) - 1
        if not is_last_batch and SLEEP_BTWN_PULLS > 0:
            time.sleep(SLEEP_BTWN_PULLS)

    enrichment = concat_meaningful_frames(
        enrichment_batches,
        ignore_index=True,
    )

    if enrichment is None or enrichment.empty:
        return ensure_universe_output_columns(out)

    enrichment = enrichment.copy()
    if "Instrument" in enrichment.columns:
        enrichment = enrichment.rename(columns={"Instrument": "Query_Instrument"})
    else:
        enrichment["Query_Instrument"] = pd.Series(instrument_list, dtype="string")

    enrichment = rename_lseg_columns(enrichment, field_map)
    enrichment = parse_universe_columns(enrichment)

    enrichment = enrichment.drop_duplicates(subset=["Query_Instrument"], keep="first")

    merged = out.merge(
        enrichment,
        left_on="Instrument",
        right_on="Query_Instrument",
        how="left",
        suffixes=("", "_enriched"),
    )

    for column in [
        "RIC",
        "Ticker",
        "Name",
        "Instrument_Type",
        "Exchange_Name",
        "Exchange_Code",
        "Price",
        "Market_Cap_Current",
        "Market_Cap_Last_Fiscal_Year_End",
        "BM_French",
        "BM",
        "Announcement_Date",
    ]:
        enriched_column = f"{column}_enriched"
        if enriched_column not in merged.columns:
            continue

        if column in merged.columns:
            merged[column] = merged[column].where(
                merged[column].notna(),
                merged[enriched_column],
            )
        else:
            merged[column] = merged[enriched_column]

    if "Historical_RIC_From_LSEG" in merged.columns:
        merged["Historical_RIC"] = merged["Historical_RIC"].where(
            merged["Historical_RIC"].notna() & (merged["Historical_RIC"] != ""),
            merged["Historical_RIC_From_LSEG"],
        )

    merged["Archived_RIC"] = coalesce_text_columns(
        merged,
        ["Archived_RIC", "RIC", "Query_Instrument"],
    )
    merged["RIC"] = coalesce_text_columns(
        merged,
        ["RIC", "Historical_RIC", "Archived_RIC", "Query_Instrument"],
    )
    merged["Instrument"] = choose_best_instrument_identifier(merged)
    merged["Identifier_Source"] = build_identifier_source(merged)

    drop_columns = [
        column
        for column in merged.columns
        if column.endswith("_enriched")
        or column in {"Query_Instrument", "Historical_RIC_From_LSEG"}
    ]
    merged = merged.drop(columns=drop_columns, errors="ignore")

    return ensure_universe_output_columns(merged)


def backfill_french_benchmark_reference_fields(
    candidates: pd.DataFrame,
    year_context,
    currency: str,
) -> pd.DataFrame:
    """
    Backfill benchmark fundamentals using the June formation date as the anchor.

    The strict December 31 point-in-time query can legitimately come back sparse
    for the following year's benchmark build because annual fundamentals may not
    yet be fully available at that exact date. Re-querying the same fields as of
    the June formation date preserves the benchmark design while avoiding a
    spurious empty reference universe.
    """
    out = ensure_universe_output_columns(candidates.copy())
    fallback = enrich_candidates_with_universe_fields(
        candidates=out,
        year_context=year_context,
        currency=currency,
        fiscal_anchor_date=year_context.formation_date,
    )

    for column in (
        "BM_French",
        "BM",
        "Announcement_Date",
    ):
        out[column] = out[column].where(out[column].notna(), fallback[column])

    return ensure_universe_output_columns(out)


def build_universe_field_map(
    year_context,
    fiscal_anchor_date: str | None = None,
    include_price_to_book: bool = True,
) -> dict[str, str]:
    """
    LSEG fields used to enrich the historically constructed candidate list.

    The final universe remains keyed by a single downstream `Instrument`
    column, while historical and archived identifiers are retained separately
    for auditability.
    """
    fiscal_anchor_date = fiscal_anchor_date or year_context.last_fiscal_year_end

    field_map = {
        "TR.RIC": "RIC",
        f"TR.RIC(SDate={year_context.formation_date})": "Historical_RIC_From_LSEG",
        "TR.ExchangeTicker": "Ticker",
        "TR.CommonName": "Name",
        "TR.InstrumentType": "Instrument_Type",
        "TR.ExchangeName": "Exchange_Name",
        "TR.ExchangeMarketIdCode": "Exchange_Code",
        f"TR.PriceClose(SDate={year_context.formation_date})": "Price",
        (
            f"TR.CompanyMarketCap(SDate={year_context.formation_date},Scale=6)"
        ): "Market_Cap_Current",
        (
            f"TR.CompanyMarketCap(SDate={fiscal_anchor_date},Scale=6)"
        ): "Market_Cap_Last_Fiscal_Year_End",
        (
            f"TR.F.OriginalAnnouncementDate("
            f"SDate={fiscal_anchor_date},Period=FY0)"
        ): "Announcement_Date",
    }

    return field_map


def build_french_reference_universe_diagnostics(
    df: pd.DataFrame,
    year_context,
    exchange_definitions: tuple[tuple[str, str], ...],
) -> str:
    if df.empty:
        return "Combined candidate set is empty before benchmark eligibility filters."

    total_candidates = int(len(df))
    numeric_columns = (
        "Market_Cap_Current",
        "BM_French",
    )
    numeric_counts = []
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        numeric_counts.append(f"{column} non-missing={int(values.notna().sum())}")

    exchange_summaries = []
    for _, exchange_code in exchange_definitions:
        filtered_candidates, _ = filter_to_common_stock_candidates(
            df=df,
            year_context=year_context,
            exchange=exchange_code,
            sample_size=None,
        )
        if filtered_candidates.empty:
            continue

        current_market_cap = pd.to_numeric(
            filtered_candidates["Market_Cap_Current"],
            errors="coerce",
        )
        book_to_market = pd.to_numeric(
            filtered_candidates["BM_French"],
            errors="coerce",
        )
        eligible = (
            current_market_cap.notna()
            & book_to_market.notna()
            & book_to_market.gt(0)
        )
        exchange_summaries.append(
            f"{exchange_code}: common-stock candidates={len(filtered_candidates)}, "
            f"benchmark-eligible={int(eligible.sum())}"
        )

    if not exchange_summaries:
        exchange_summary = "No exchange retained any common-stock candidates."
    else:
        exchange_summary = "; ".join(exchange_summaries)

    return (
        f"Combined candidates={total_candidates}; "
        f"{', '.join(numeric_counts)}; "
        f"{exchange_summary}"
    )


def rename_lseg_columns(
    frame: pd.DataFrame,
    field_map: dict[str, str],
) -> pd.DataFrame:
    rename_map = {field.upper(): name for field, name in field_map.items()}
    return frame.rename(columns=lambda column: rename_map.get(str(column).upper(), column))


def ensure_universe_output_columns(stock_universe: pd.DataFrame) -> pd.DataFrame:
    out = stock_universe.copy()

    required_columns = [
        *CANDIDATE_AUDIT_COLUMNS,
        "Price",
        "Market_Cap_Current",
        "Market_Cap_Last_Fiscal_Year_End",
        "BM_French",
        "BM",
        "Announcement_Date",
    ]

    for column in required_columns:
        if column not in out.columns:
            out[column] = pd.NA

    return out


def parse_universe_columns(stock_universe: pd.DataFrame) -> pd.DataFrame:
    out = ensure_universe_output_columns(stock_universe)

    for column in (
        "Announcement_Date",
        "Retire_Date",
    ):
        out[column] = parse_datetime_column(out[column])

    for column in (
        "Price",
        "Market_Cap_Current",
        "Market_Cap_Last_Fiscal_Year_End",
        "BM_French",
        "BM",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in (
        "Instrument",
        "RIC",
        "Historical_RIC",
        "Archived_RIC",
        "Ticker",
        "Name",
        "Instrument_Type",
        "Security_Type",
        "Exchange_Name",
        "Exchange_Code",
        "Exchange_Local_Code",
        "Listing_Status",
        "Asset_State",
        "Identifier_Source",
        "Candidate_Fetch_Method",
        "Universe_Window_Start",
        "Universe_Window_End",
    ):
        out[column] = out[column].astype("string").str.strip()

    if "Primary_Listing_Flag" not in out.columns:
        out["Primary_Listing_Flag"] = pd.NA
    out["Primary_Listing_Flag"] = parse_boolean_column(out["Primary_Listing_Flag"])

    return out


def parse_datetime_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def parse_boolean_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    cleaned = series.astype("string").str.strip().str.lower()
    mapped = cleaned.map(
        {
            "true": True,
            "t": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "f": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    return mapped.astype("boolean")


def coalesce_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="string")

    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].astype("string").str.strip()
        valid = values.notna() & (values != "")
        result = result.where(result.notna() & (result != ""), values.where(valid))

    return result


def choose_best_instrument_identifier(df: pd.DataFrame) -> pd.Series:
    return coalesce_text_columns(
        df,
        ["Instrument", "Historical_RIC", "Archived_RIC", "RIC"],
    )


def build_identifier_source(df: pd.DataFrame) -> pd.Series:
    source = pd.Series(pd.NA, index=df.index, dtype="string")

    for column_name, label in (
        ("Historical_RIC", "historical_ric"),
        ("Archived_RIC", "archived_ric"),
        ("RIC", "current_ric"),
        ("Instrument", "input_instrument"),
    ):
        if column_name not in df.columns:
            continue
        values = df[column_name].astype("string").str.strip()
        valid = values.notna() & (values != "")
        source = source.where(source.notna() & (source != ""), pd.Series(label, index=df.index).where(valid))

    return source


def apply_filter(
    df: pd.DataFrame,
    condition,
    label: str,
    sample_size: dict,
) -> pd.DataFrame:
    filtered = df.loc[condition].copy()
    sample_size[label] = len(filtered)
    return filtered


def filter_to_common_stock_candidates(
    df: pd.DataFrame,
    year_context,
    exchange: str,
    sample_size: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    universe_window_start_ts = pd.Timestamp(year_context.universe_window_start)

    df = ensure_universe_output_columns(df)

    if sample_size is None:
        sample_size = {RAW_HISTORICAL_CANDIDATES_LABEL: len(df)}

    required_columns = [
        "Instrument",
        "Instrument_Type",
        "Price",
        "Market_Cap_Current",
        "BM_French",
        "Announcement_Date",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Universe snapshot is missing columns: {missing_columns}")

    exchange_code = df["Exchange_Code"].astype("string").str.strip().str.upper()
    local_exchange_code = df["Exchange_Local_Code"].astype("string").str.strip().str.upper()
    target_exchange = str(exchange).strip().upper()
    df = apply_filter(
        df.assign(
            Exchange_Code=exchange_code,
            Exchange_Local_Code=local_exchange_code,
        ),
        (exchange_code == target_exchange) | (local_exchange_code == target_exchange),
        TARGET_EXCHANGE_LABEL,
        sample_size,
    )

    primary_flag = df["Primary_Listing_Flag"]
    df = apply_filter(
        df,
        primary_flag.fillna(True),
        PRIMARY_LISTINGS_LABEL,
        sample_size,
    )

    enrichment_success = df[
        [
            "RIC",
            "Historical_RIC",
            "Archived_RIC",
            "Instrument_Type",
            "Price",
            "Market_Cap_Current",
        ]
    ].notna().any(axis=1)
    df = apply_filter(
        df,
        enrichment_success,
        ENRICHMENT_SUCCESS_LABEL,
        sample_size,
    )

    df = apply_filter(
        df,
        df["Instrument"].notna() & (df["Instrument"] != ""),
        USABLE_IDENTIFIER_LABEL,
        sample_size,
    )

    df = apply_filter(
        df,
        df["Instrument_Type"].isin(ORDINARY_SHARE_TYPES),
        ORDINARY_SHARES_LABEL,
        sample_size,
    )

    df = apply_filter(
        df,
        build_historical_security_definition_mask(df),
        HISTORICAL_ORDINARY_COMMON_SHARE_CANDIDATES_LABEL,
        sample_size,
    )

    return df, sample_size


def build_market_cap_size_split_reference_universe(
    df: pd.DataFrame,
    year_context,
    exchange: str,
) -> pd.DataFrame:
    """
    Build the reference universe used only for the yearly market-cap breakpoint.

    This universe deliberately applies only the minimal stock-definition filters
    plus the requirement that current market cap exists. The full PEAD sample
    screens are applied later and do not affect the breakpoint itself.
    """
    reference_universe, _ = filter_to_common_stock_candidates(
        df=df,
        year_context=year_context,
        exchange=exchange,
        sample_size=None,
    )
    reference_market_caps = pd.to_numeric(
        reference_universe["Market_Cap_Current"],
        errors="coerce",
    )
    reference_universe = reference_universe.loc[reference_market_caps.notna()].copy()
    reference_universe["Market_Cap_Current"] = reference_market_caps.loc[
        reference_universe.index
    ]
    reference_universe = (
        reference_universe.sort_values(["Instrument", "Identifier_Source", "Retire_Date"])
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )
    return reference_universe


def build_french_benchmark_reference_universe(
    df: pd.DataFrame,
    year_context,
    exchange: str,
) -> pd.DataFrame:
    reference_universe, _ = filter_to_common_stock_candidates(
        df=df,
        year_context=year_context,
        exchange=exchange,
        sample_size=None,
    )

    for column in (
        "Market_Cap_Current",
        "BM_French",
    ):
        reference_universe[column] = pd.to_numeric(
            reference_universe[column], errors="coerce"
        )

    reference_universe = reference_universe.loc[
        reference_universe["Market_Cap_Current"].notna()
        & reference_universe["BM_French"].notna()
        & reference_universe["BM_French"].gt(0)
    ].copy()

    reference_universe = (
        reference_universe.sort_values(
            ["Instrument", "Identifier_Source", "Retire_Date", "Exchange_Code"]
        )
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )
    return reference_universe


def filter_universe(
    df: pd.DataFrame,
    year_context,
    exchange: str,
    market_cap_threshold: float,
    stock_price_threshold: float,
    sample_size: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    df, sample_size = filter_to_common_stock_candidates(
        df=df,
        year_context=year_context,
        exchange=exchange,
        sample_size=sample_size,
    )

    df = apply_filter(
        df,
        df[
            [
                "Instrument",
                "Price",
                "Market_Cap_Current",
                "Market_Cap_Last_Fiscal_Year_End",
                "BM_French",
            ]
        ]
        .notna()
        .all(axis=1),
        NON_MISSING_DATA_LABEL,
        sample_size,
    )

    df = apply_filter(
        df,
        df["Market_Cap_Current"] >= market_cap_threshold,
        "Market cap >= threshold",
        sample_size,
    )

    df = apply_filter(
        df,
        df["Price"] >= stock_price_threshold,
        "Price >= threshold",
        sample_size,
    )

    df = apply_filter(
        df,
        df["BM_French"] > 0,
        POSITIVE_BOOK_TO_MARKET_LABEL,
        sample_size,
    )

    df = (
        df.sort_values(["Instrument", "Identifier_Source", "Retire_Date"])
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )
    sample_size["Unique security identifiers"] = len(df)

    return df, sample_size


def valid_year_specific_candidates_label(exchange: str) -> str:
    return f"Valid year-specific {exchange} candidates"


def build_reported_sample_size(raw_sample_size: dict, exchange: str) -> dict:
    reported_sample_size = {}

    count_mappings = [
        (RAW_HISTORICAL_CANDIDATES_LABEL, RAW_HISTORICAL_CANDIDATES_LABEL),
        (REPORTED_ORDINARY_COMMON_SHARES_LABEL, ORDINARY_SHARES_LABEL),
        (
            REPORTED_REQUIRED_ACCOUNTING_AND_MARKET_DATA_LABEL,
            NON_MISSING_DATA_LABEL,
        ),
        ("Market cap >= threshold", "Market cap >= threshold"),
        ("Price >= threshold", "Price >= threshold"),
        ("Final exchange validation (XLON)", "Final exchange validation (XLON)"),
        (POSITIVE_BOOK_TO_MARKET_LABEL, POSITIVE_BOOK_TO_MARKET_LABEL),
        ("Unique security identifiers", "Unique security identifiers"),
    ]

    for reported_label, raw_label in count_mappings:
        if raw_label in raw_sample_size:
            reported_sample_size[reported_label] = raw_sample_size[raw_label]

    for metadata_label in [
        "Universe source",
        "Formation date",
        "Universe window start",
        "Universe window end",
        "Market cap size split percentile",
        "Market cap size split breakpoint",
        "Market cap size split breakpoint unit",
        "Market cap decile breakpoints",
        "Market cap size split reference universe count",
        "Microcap count",
        "All-but-microcap count",
        "French benchmark formation date",
        "French benchmark reference universe count",
        "French benchmark exchanges",
        "French benchmark size breakpoints",
        "French benchmark big stock market cap share",
        "French benchmark big stock count",
        "French benchmark big stock market cap floor",
        "French benchmark B/M breakpoints",
    ]:
        if metadata_label in raw_sample_size:
            reported_sample_size[metadata_label] = raw_sample_size[metadata_label]

    return reported_sample_size


def build_historical_security_definition_mask(df: pd.DataFrame) -> pd.Series:
    common_name = (
        df["Name"]
        .astype("string")
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    padded_name = " " + common_name.fillna("") + " "

    excluded_expressions = [
        " FUND ",
        " TRUST ",
        "NIL PAID",
        "STOCK UNIT",
        "ANNUITY UNIT",
        "UNIT £",
        "UNIT TRUST",
        " UNITS",
        " ZDP ",
        "REIT",
        "POST RED",
        "DEPOSITARY",
        " RECEIPT",
        "INTERIM SHARES",
        "REEDEMABLE",
        "PREFERENCE",
        "INVESTMENT TRUST",
        " ADR",
        "FULLY PAID",
        "PARTLY PAID",
        " BDR",
        " NRDF",
        "DEFERRED",
    ]

    exclude_mask = pd.Series(False, index=df.index)
    for expression in excluded_expressions:
        exclude_mask = exclude_mask | padded_name.str.contains(expression, regex=False, na=False)

    return ~exclude_mask


def add_holding_columns(
    stock_universe: pd.DataFrame,
    year_context,
) -> pd.DataFrame:
    out = stock_universe.copy()

    out["Formation_Year"] = year_context.year
    out["Formation_Date"] = year_context.formation_date
    out["Universe_Window_Start"] = year_context.universe_window_start
    out["Universe_Window_End"] = year_context.universe_window_end
    out["Market_Data_Window_Start"] = year_context.market_data_window_start
    out["Market_Data_Window_End"] = year_context.market_data_window_end

    return out
