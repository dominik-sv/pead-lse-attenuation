from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.pipeline_config import (
    MARKET_CAP_SIZE_SPLIT_PERCENTILE,
    ORDINARY_SHARE_TYPES,
    UNIVERSE_SOURCE,
)
from .market_cap_splits import (
    assign_market_cap_size_split_from_breakpoint,
    build_market_cap_size_split_metadata,
    validate_market_cap_size_split_percentile,
)
from .universe_filters import (
    ENRICHMENT_SUCCESS_LABEL,
    HISTORICAL_ORDINARY_COMMON_SHARE_CANDIDATES_LABEL,
    ORDINARY_SHARES_LABEL,
    RAW_HISTORICAL_CANDIDATES_LABEL,
    add_holding_columns,
    apply_filter,
    build_historical_security_definition_mask,
    build_reported_sample_size,
    enrich_candidates_with_universe_fields,
    ensure_universe_output_columns,
    parse_universe_columns,
)


REQUIRED_ENRICHED_UNIVERSE_COLUMNS = {
    "Instrument",
    "Name",
    "Instrument_Type",
    "Price",
    "Market_Cap_Current",
    "Market_Cap_Last_Fiscal_Year_End",
    "BM_French",
    "Announcement_Date",
}

RIC_DEDUPLICATION_LABEL = "Deduplicated by RIC after enrichment"
POSITIVE_MARKET_CAP_LABEL = "Positive market cap"
NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL = "Non-missing book-to-market inputs"
POSITIVE_BOOK_TO_MARKET_LABEL = "Positive book-to-market last fiscal year"
EXCHANGE_CODE_XLON_OR_MISSING_LABEL = "Exchange code XLON or missing"
EXCHANGE_NAME_LSE_OR_MISSING_LABEL = "Exchange name London Stock Exchange or missing"
CURRENT_RIC_ENDS_WITH_DOT_L_LABEL = "Current RIC ends with .L"
BOOK_EQUITY_COLUMN = "Book_Equity_Last_Fiscal_Year"
BOOK_EQUITY_DATADATE_COLUMN = "Book_Equity_Datadate"
BOOK_EQUITY_SOURCE_FIELD_COLUMN = "Book_Equity_Source_Field"
COMPUSTAT_OPTIONAL_COLUMNS = (
    "RIC",
    "Historical_RIC",
    "Archived_RIC",
    "Ticker",
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
    "Price",
    "Market_Cap_Current",
    "Market_Cap_Last_Fiscal_Year_End",
    "BM_French",
    "BM",
    "Announcement_Date",
    "validated_lseg_identifier",
    "compustat_sedol_identifier",
    BOOK_EQUITY_COLUMN,
    BOOK_EQUITY_DATADATE_COLUMN,
    BOOK_EQUITY_SOURCE_FIELD_COLUMN,
    "Compustat_CEQ",
    "Compustat_SEQ",
    "Compustat_TEQ",
)


def _deduplicate_by_instrument(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["Instrument", "Identifier_Source", "Retire_Date"])
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )


def _apply_conservative_xlon_venue_filter(
    df: pd.DataFrame,
    sample_size: dict,
) -> pd.DataFrame:
    exchange_code = df["Exchange_Code"].astype("string").str.strip().str.upper()
    exchange_name = df["Exchange_Name"].astype("string").str.strip().str.upper()
    current_ric = df["RIC"].astype("string").str.strip().str.upper()

    df = apply_filter(
        df,
        exchange_code.eq("XLON") | exchange_code.eq("") | exchange_code.isna(),
        EXCHANGE_CODE_XLON_OR_MISSING_LABEL,
        sample_size,
    )

    exchange_name = df["Exchange_Name"].astype("string").str.strip().str.upper()
    df = apply_filter(
        df,
        exchange_name.eq("LONDON STOCK EXCHANGE") | exchange_name.eq("") | exchange_name.isna(),
        EXCHANGE_NAME_LSE_OR_MISSING_LABEL,
        sample_size,
    )

    current_ric = df["RIC"].astype("string").str.strip().str.upper()
    df = apply_filter(
        df,
        current_ric.str.contains(r"\.L(?:\^.*)?$", regex=True, na=False),
        CURRENT_RIC_ENDS_WITH_DOT_L_LABEL,
        sample_size,
    )

    return df


def _deduplicate_one_row_per_lseg_identifier(
    df: pd.DataFrame,
    sample_size: dict | None = None,
) -> pd.DataFrame:
    ordered = df.copy()
    gvkey = (
        ordered.get("gvkey", pd.Series(pd.NA, index=ordered.index))
        .astype("string")
        .str.strip()
    )
    has_gvkey = gvkey.notna() & gvkey.ne("")

    with_gvkey = ordered.loc[has_gvkey].copy()
    if not with_gvkey.empty:
        with_gvkey["gvkey"] = gvkey.loc[has_gvkey]
        with_gvkey["_primary_listing_rank"] = (
            with_gvkey.get(
                "Primary_Listing_Flag",
                pd.Series(pd.NA, index=with_gvkey.index),
            )
            .fillna(False)
            .astype(bool)
            .astype(int)
        )
        with_gvkey["_market_cap_rank"] = pd.to_numeric(
            with_gvkey.get(
                "Market_Cap_Current",
                pd.Series(pd.NA, index=with_gvkey.index),
            ),
            errors="coerce",
        )

        duplicate_group_sizes = with_gvkey.groupby("gvkey").size()
        duplicated_gvkeys = duplicate_group_sizes[duplicate_group_sizes > 1]

        has_primary_in_group = with_gvkey.groupby("gvkey")[
            "_primary_listing_rank"
        ].transform("max")
        primary_stage = with_gvkey.loc[
            (has_primary_in_group == 0)
            | with_gvkey["_primary_listing_rank"].eq(has_primary_in_group)
        ].copy()

        max_market_cap = primary_stage.groupby("gvkey")["_market_cap_rank"].transform(
            "max"
        )
        market_cap_stage = primary_stage.loc[
            primary_stage["_market_cap_rank"].eq(max_market_cap)
        ].copy()

        deduplicated_with_lseg_identifier = (
            market_cap_stage.sort_values(
                [
                    "gvkey",
                    "_primary_listing_rank",
                    "_market_cap_rank",
                    "Instrument",
                    "Identifier_Source",
                    "Retire_Date",
                ],
                ascending=[True, False, False, True, True, True],
                na_position="last",
            )
            .drop_duplicates(subset=["lseg_identifier"], keep="first")
            .reset_index(drop=True)
        )

        if sample_size is not None:
            sample_size["Firm-level duplicate rows before deduplication"] = int(
                duplicated_gvkeys.sum()
            )
            sample_size["Firm-level duplicate gvkeys before deduplication"] = int(
                len(duplicated_gvkeys)
            )
            sample_size["Listings removed by primary listing preference"] = int(
                len(with_gvkey) - len(primary_stage)
            )
            sample_size["Listings removed by market-cap tie-break"] = int(
                len(primary_stage) - len(market_cap_stage)
            )
            sample_size["Listings removed by deterministic tie-break"] = int(
                len(market_cap_stage) - len(deduplicated_with_lseg_identifier)
            )

        with_gvkey = deduplicated_with_lseg_identifier
    elif sample_size is not None:
        sample_size["Firm-level duplicate rows before deduplication"] = 0
        sample_size["Firm-level duplicate gvkeys before deduplication"] = 0
        sample_size["Listings removed by primary listing preference"] = 0
        sample_size["Listings removed by market-cap tie-break"] = 0
        sample_size["Listings removed by deterministic tie-break"] = 0

    without_gvkey = ordered.loc[~has_gvkey].drop_duplicates(
        subset=["Instrument"],
        keep="first",
    )

    return (
        pd.concat([with_gvkey, without_gvkey], ignore_index=True)
        .drop(columns=["_primary_listing_rank", "_market_cap_rank"], errors="ignore")
        .reset_index(drop=True)
    )


def _add_universe_context_metadata(sample_size: dict, year_context) -> None:
    sample_size["Universe source"] = UNIVERSE_SOURCE
    sample_size["Universe window start"] = year_context.universe_window_start
    sample_size["Universe window end"] = year_context.universe_window_end
    sample_size["Formation date"] = year_context.formation_date


def _get_string_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return frame.get(column, pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()


def _get_bool_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame.get(column, pd.Series(pd.NA, index=frame.index))
    if pd.api.types.is_bool_dtype(values):
        return values.astype("boolean")
    return values.astype("string").str.strip().map(
        {
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).astype("boolean")


def _assign_book_to_market_from_compustat_equity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    market_cap_last_fy_end = pd.to_numeric(
        out.get("Market_Cap_Last_Fiscal_Year_End"),
        errors="coerce",
    )
    book_equity = pd.to_numeric(out.get(BOOK_EQUITY_COLUMN), errors="coerce")
    valid_mask = (
        market_cap_last_fy_end.notna()
        & market_cap_last_fy_end.gt(0)
        & book_equity.notna()
        & book_equity.gt(0)
    )
    computed_book_to_market = pd.Series(pd.NA, index=out.index, dtype="Float64")
    computed_book_to_market.loc[valid_mask] = (
        book_equity.loc[valid_mask] / market_cap_last_fy_end.loc[valid_mask]
    )

    existing_book_to_market = pd.to_numeric(
        out.get("BM_French"),
        errors="coerce",
    )
    use_compustat_mask = computed_book_to_market.notna()
    if existing_book_to_market is None:
        out["BM_French"] = computed_book_to_market
        out["BM"] = computed_book_to_market
        return out

    out["BM_French"] = existing_book_to_market.where(
        ~use_compustat_mask,
        computed_book_to_market,
    )
    out["BM"] = out["BM_French"]
    return out


def _compute_book_to_market_ratio(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("BM_French"), errors="coerce")


def _filter_reference_universe(
    df: pd.DataFrame,
    *,
    require_positive_book_to_market: bool,
) -> pd.DataFrame:
    reference_universe, _ = filter_to_gbp_common_stock_candidates(
        df=df,
        sample_size=None,
    )
    reference_universe["Market_Cap_Current"] = pd.to_numeric(
        reference_universe["Market_Cap_Current"],
        errors="coerce",
    )
    valid_mask = (
        reference_universe["Market_Cap_Current"].notna()
        & reference_universe["Market_Cap_Current"].gt(0)
    )
    if require_positive_book_to_market:
        reference_universe["_book_to_market_ratio"] = _compute_book_to_market_ratio(
            reference_universe
        )
        valid_mask = (
            valid_mask
            & reference_universe["_book_to_market_ratio"].notna()
            & reference_universe["_book_to_market_ratio"].gt(0)
        )

    filtered = reference_universe.loc[valid_mask].copy()
    filtered = filtered.drop(columns=["_book_to_market_ratio"], errors="ignore")
    return filtered


def load_gbp_constituent_candidates(year_context) -> pd.DataFrame:
    constituents_path = year_context.gbp_constituents_path
    if not constituents_path.exists():
        raise FileNotFoundError(
            "GBP constituent inputs are missing. "
            "Run scripts/00_build_gbp_stock_universes.py first. "
            f"Missing: {constituents_path}"
        )

    constituents = pd.read_csv(constituents_path, dtype=str)
    required_columns = {"lseg_identifier", "conm", "formation_year"}
    missing_columns = required_columns.difference(constituents.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "GBP constituent file is missing required columns: "
            f"{missing}. File: {constituents_path}"
        )

    out = pd.DataFrame(
        {
            "Instrument": constituents["lseg_identifier"].astype("string").str.strip(),
            "RIC": _get_string_column(constituents, "RIC"),
            "Historical_RIC": _get_string_column(constituents, "Historical_RIC"),
            "Archived_RIC": _get_string_column(constituents, "Archived_RIC"),
            "Ticker": _get_string_column(constituents, "Ticker"),
            "Name": constituents["conm"].astype("string").str.strip(),
            "Instrument_Type": _get_string_column(constituents, "Instrument_Type"),
            "Security_Type": _get_string_column(constituents, "Security_Type"),
            "Exchange_Name": _get_string_column(constituents, "Exchange_Name"),
            "Exchange_Code": _get_string_column(constituents, "Exchange_Code"),
            "Exchange_Local_Code": _get_string_column(constituents, "Exchange_Local_Code"),
            "Listing_Status": _get_string_column(constituents, "Listing_Status"),
            "Asset_State": _get_string_column(constituents, "Asset_State"),
            "Retire_Date": pd.to_datetime(
                constituents.get("Retire_Date", pd.Series(pd.NaT, index=constituents.index)),
                errors="coerce",
            ),
            "Primary_Listing_Flag": _get_bool_column(constituents, "Primary_Listing_Flag"),
            "Identifier_Source": _get_string_column(constituents, "Identifier_Source").fillna(
                "gbp_lseg_identifier"
            ),
            "Candidate_Fetch_Method": _get_string_column(
                constituents,
                "Candidate_Fetch_Method",
            ).fillna("00_gbp_stock_universe"),
            "Universe_Window_Start": pd.Series(
                year_context.universe_window_start,
                index=constituents.index,
                dtype="string",
            ),
            "Universe_Window_End": pd.Series(
                year_context.universe_window_end,
                index=constituents.index,
                dtype="string",
            ),
            "formation_year": constituents["formation_year"].astype("string").str.strip(),
            "junedate": constituents.get("junedate", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
            "fic": constituents.get("fic", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
            "gvkey": constituents.get("gvkey", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
            "iid": constituents.get("iid", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
            "sedol": constituents.get("sedol", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
            "lseg_identifier": constituents["lseg_identifier"].astype("string").str.strip(),
            "datadate": constituents.get("datadate", pd.Series(pd.NA, index=constituents.index)).astype("string").str.strip(),
        }
    )
    for column in COMPUSTAT_OPTIONAL_COLUMNS:
        if column in out.columns or column not in constituents.columns:
            continue
        out[column] = constituents[column]
    out = out.loc[out["Instrument"].notna() & (out["Instrument"] != "")].copy()
    return ensure_universe_output_columns(out.reset_index(drop=True))


def build_enriched_gbp_universe_for_year(
    year_context,
    currency: str,
) -> pd.DataFrame:
    candidates = load_gbp_constituent_candidates(year_context)
    enriched = enrich_candidates_with_universe_fields(
        candidates=candidates,
        year_context=year_context,
        currency=currency,
        include_price_to_book=False,
    )
    enriched = parse_universe_columns(enriched)
    enriched = _assign_book_to_market_from_compustat_equity(enriched)
    enriched = (
        enriched.sort_values(
            [
                "lseg_identifier",
                "Primary_Listing_Flag",
                "Market_Cap_Current",
                "Instrument",
                "Identifier_Source",
                "Retire_Date",
            ],
            ascending=[True, False, False, True, True, True],
            na_position="last",
        )
        .drop_duplicates(subset=["lseg_identifier"], keep="first")
        .reset_index(drop=True)
    )
    return ensure_universe_output_columns(enriched)


def read_enriched_gbp_universe_cache(path: str | Path) -> pd.DataFrame:
    enriched = pd.read_csv(path)
    return parse_universe_columns(enriched)


def enriched_gbp_universe_cache_has_expected_columns(path: str | Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False

    return REQUIRED_ENRICHED_UNIVERSE_COLUMNS.issubset(columns)


def filter_to_gbp_common_stock_candidates(
    df: pd.DataFrame,
    sample_size: dict | None = None,
    apply_conservative_xlon_filter: bool = False,
) -> tuple[pd.DataFrame, dict]:
    out = ensure_universe_output_columns(df.copy())

    if sample_size is None:
        sample_size = {RAW_HISTORICAL_CANDIDATES_LABEL: int(len(out))}

    missing_columns = REQUIRED_ENRICHED_UNIVERSE_COLUMNS.difference(out.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"GBP universe cache is missing required columns: {missing}")

    enrichment_success = out[
        [
            "RIC",
            "Historical_RIC",
            "Archived_RIC",
            "Instrument_Type",
            "Price",
            "Market_Cap_Current",
        ]
    ].notna().any(axis=1)
    out = apply_filter(
        out,
        enrichment_success,
        ENRICHMENT_SUCCESS_LABEL,
        sample_size,
    )

    if apply_conservative_xlon_filter:
        out = _apply_conservative_xlon_venue_filter(out, sample_size)

    out = apply_filter(
        out,
        out["Instrument_Type"].isin(ORDINARY_SHARE_TYPES),
        ORDINARY_SHARES_LABEL,
        sample_size,
    )

    out = apply_filter(
        out,
        build_historical_security_definition_mask(out),
        HISTORICAL_ORDINARY_COMMON_SHARE_CANDIDATES_LABEL,
        sample_size,
    )

    return out, sample_size


def build_market_cap_size_split_reference_universe_from_gbp(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return _filter_reference_universe(
        df,
        require_positive_book_to_market=False,
    )


def build_market_cap_size_split_reference_universe_from_cleaned_gbp(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = ensure_universe_output_columns(df.copy())
    out["Market_Cap_Current"] = pd.to_numeric(
        out["Market_Cap_Current"],
        errors="coerce",
    )
    out = out.loc[
        out["Market_Cap_Current"].notna()
        & out["Market_Cap_Current"].gt(0)
    ].copy()
    out = (
        out.sort_values(["Instrument", "Identifier_Source", "Retire_Date"])
        .drop_duplicates(subset=["Instrument"], keep="first")
        .reset_index(drop=True)
    )
    return out


def build_benchmark_reference_universe_from_gbp(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return _filter_reference_universe(
        df,
        require_positive_book_to_market=True,
    )


def filter_final_gbp_universe(
    df: pd.DataFrame,
    market_cap_threshold: float,
    stock_price_threshold: float,
    sample_size: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    out, sample_size = filter_to_gbp_common_stock_candidates(
        df=df,
        sample_size=sample_size,
        apply_conservative_xlon_filter=True,
    )

    out = apply_filter(
        out,
        out[
            [
                "Instrument",
                "Price",
                "Market_Cap_Current",
                "Market_Cap_Last_Fiscal_Year_End",
                BOOK_EQUITY_COLUMN,
            ]
        ]
        .notna()
        .all(axis=1),
        NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL,
        sample_size,
    )

    out = apply_filter(
        out,
        pd.to_numeric(out["Market_Cap_Current"], errors="coerce") > 0,
        POSITIVE_MARKET_CAP_LABEL,
        sample_size,
    )

    out = apply_filter(
        out,
        _compute_book_to_market_ratio(out) > 0,
        POSITIVE_BOOK_TO_MARKET_LABEL,
        sample_size,
    )

    sample_size["Rows before firm-level deduplication"] = int(len(out))
    out = _deduplicate_one_row_per_lseg_identifier(out, sample_size=sample_size)
    sample_size["Unique firms (gvkey where available)"] = int(len(out))

    out = apply_filter(
        out,
        pd.to_numeric(out["Market_Cap_Current"], errors="coerce") >= market_cap_threshold,
        "Market cap >= threshold",
        sample_size,
    )

    out = apply_filter(
        out,
        pd.to_numeric(out["Price"], errors="coerce") >= stock_price_threshold,
        "Price >= threshold",
        sample_size,
    )

    return out, sample_size


def filter_analysis_gbp_universe_from_cleaned_sample(
    df: pd.DataFrame,
    market_cap_threshold: float,
    stock_price_threshold: float,
    sample_size: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    out = ensure_universe_output_columns(df.copy())

    if sample_size is None:
        sample_size = {}

    out["Market_Cap_Current"] = pd.to_numeric(
        out["Market_Cap_Current"],
        errors="coerce",
    )
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    out = apply_filter(
        out,
        out["Market_Cap_Current"] >= market_cap_threshold,
        "Market cap >= threshold",
        sample_size,
    )

    out = apply_filter(
        out,
        out["Price"] >= stock_price_threshold,
        "Price >= threshold",
        sample_size,
    )

    return out, sample_size


def build_base_universe_from_enriched_gbp_universe(
    enriched_universe: pd.DataFrame,
    year_context,
    market_cap_threshold: float,
    stock_price_threshold: float,
    market_cap_size_split_percentile: float = MARKET_CAP_SIZE_SPLIT_PERCENTILE,
) -> tuple[pd.DataFrame, dict]:
    size_split_reference_universe = build_market_cap_size_split_reference_universe_from_gbp(
        enriched_universe
    )
    if size_split_reference_universe.empty:
        raise RuntimeError(
            "Market-cap size split reference universe is empty after the GBP "
            "ordinary-share screen and requiring non-missing current market cap."
        )

    normalized_split_percentile = validate_market_cap_size_split_percentile(
        market_cap_size_split_percentile
    )
    reference_market_caps = pd.to_numeric(
        size_split_reference_universe["Market_Cap_Current"],
        errors="coerce",
    )
    size_split_breakpoint = float(
        reference_market_caps.quantile(normalized_split_percentile)
    )
    size_split_metadata = build_market_cap_size_split_metadata(
        market_caps=reference_market_caps,
        split_percentile=normalized_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )

    stock_universe, sample_size = filter_final_gbp_universe(
        df=enriched_universe,
        market_cap_threshold=market_cap_threshold,
        stock_price_threshold=stock_price_threshold,
    )

    if stock_universe.empty:
        stock_universe = add_holding_columns(stock_universe, year_context)
        sample_size["Market cap size split skipped"] = (
            "Universe empty after GBP constituent enrichment and downstream filters"
        )
        sample_size.update(size_split_metadata)
        _add_universe_context_metadata(sample_size, year_context)
        return stock_universe, sample_size

    stock_universe = assign_market_cap_size_split_from_breakpoint(
        stock_universe,
        split_percentile=market_cap_size_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )
    sample_size.update(size_split_metadata)
    stock_universe = add_holding_columns(stock_universe, year_context)
    _add_universe_context_metadata(sample_size, year_context)

    return stock_universe, sample_size


def build_base_universe_from_cleaned_gbp_universe(
    cleaned_universe: pd.DataFrame,
    year_context,
    market_cap_threshold: float,
    stock_price_threshold: float,
    market_cap_size_split_percentile: float = MARKET_CAP_SIZE_SPLIT_PERCENTILE,
) -> tuple[pd.DataFrame, dict]:
    size_split_reference_universe = (
        build_market_cap_size_split_reference_universe_from_cleaned_gbp(
            cleaned_universe
        )
    )
    if size_split_reference_universe.empty:
        raise RuntimeError(
            "Market-cap size split reference universe is empty after loading the "
            "post-cleaning GBP universe and requiring non-missing current market cap."
        )

    normalized_split_percentile = validate_market_cap_size_split_percentile(
        market_cap_size_split_percentile
    )
    reference_market_caps = pd.to_numeric(
        size_split_reference_universe["Market_Cap_Current"],
        errors="coerce",
    )
    size_split_breakpoint = float(
        reference_market_caps.quantile(normalized_split_percentile)
    )
    size_split_metadata = build_market_cap_size_split_metadata(
        market_caps=reference_market_caps,
        split_percentile=normalized_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )

    stock_universe, sample_size = filter_analysis_gbp_universe_from_cleaned_sample(
        df=cleaned_universe,
        market_cap_threshold=market_cap_threshold,
        stock_price_threshold=stock_price_threshold,
    )

    if stock_universe.empty:
        stock_universe = add_holding_columns(stock_universe, year_context)
        sample_size["Market cap size split skipped"] = (
            "Universe empty after analysis market-cap and price threshold filters"
        )
        sample_size.update(size_split_metadata)
        _add_universe_context_metadata(sample_size, year_context)
        return stock_universe, sample_size

    stock_universe = assign_market_cap_size_split_from_breakpoint(
        stock_universe,
        split_percentile=market_cap_size_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )
    sample_size.update(size_split_metadata)
    stock_universe = add_holding_columns(stock_universe, year_context)
    _add_universe_context_metadata(sample_size, year_context)

    return stock_universe, sample_size


def build_base_universe_from_benchmark_constituents(
    benchmark_constituents: pd.DataFrame,
    year_context,
    market_cap_threshold: float,
    stock_price_threshold: float,
    market_cap_size_split_percentile: float = MARKET_CAP_SIZE_SPLIT_PERCENTILE,
) -> tuple[pd.DataFrame, dict]:
    if benchmark_constituents.empty:
        raise RuntimeError(
            "Benchmark constituent sample is empty before final stock-universe "
            "filters were applied."
        )

    size_split_reference_universe = benchmark_constituents.copy()
    if size_split_reference_universe.empty:
        raise RuntimeError(
            "Market-cap size split reference universe is empty after benchmark "
            "portfolio construction."
        )

    normalized_split_percentile = validate_market_cap_size_split_percentile(
        market_cap_size_split_percentile
    )
    reference_market_caps = pd.to_numeric(
        size_split_reference_universe["Market_Cap_Current"],
        errors="coerce",
    )
    size_split_breakpoint = float(
        reference_market_caps.quantile(normalized_split_percentile)
    )
    size_split_metadata = build_market_cap_size_split_metadata(
        market_caps=reference_market_caps,
        split_percentile=normalized_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )

    stock_universe = benchmark_constituents.copy()
    sample_size: dict = {}

    stock_universe = apply_filter(
        stock_universe,
        stock_universe[
            [
                "Instrument",
                "Price",
                "Market_Cap_Current",
                "Market_Cap_Last_Fiscal_Year_End",
                BOOK_EQUITY_COLUMN,
            ]
        ]
        .notna()
        .all(axis=1),
        NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL,
        sample_size,
    )

    if stock_universe.empty:
        stock_universe = add_holding_columns(stock_universe, year_context)
        sample_size["Market cap size split skipped"] = (
            "Universe empty after benchmark-eligible sample and downstream filters"
        )
        sample_size.update(size_split_metadata)
        _add_universe_context_metadata(sample_size, year_context)
        return stock_universe, sample_size

    stock_universe = apply_filter(
        stock_universe,
        pd.to_numeric(stock_universe["Market_Cap_Current"], errors="coerce")
        >= market_cap_threshold,
        "Market cap >= threshold",
        sample_size,
    )

    stock_universe = apply_filter(
        stock_universe,
        pd.to_numeric(stock_universe["Price"], errors="coerce") >= stock_price_threshold,
        "Price >= threshold",
        sample_size,
    )

    if stock_universe.empty:
        stock_universe = add_holding_columns(stock_universe, year_context)
        sample_size["Market cap size split skipped"] = (
            "Universe empty after benchmark-eligible sample and threshold filters"
        )
        sample_size.update(size_split_metadata)
        _add_universe_context_metadata(sample_size, year_context)
        return stock_universe, sample_size

    stock_universe = assign_market_cap_size_split_from_breakpoint(
        stock_universe,
        split_percentile=market_cap_size_split_percentile,
        breakpoint_value=size_split_breakpoint,
    )
    sample_size.update(size_split_metadata)
    stock_universe = add_holding_columns(stock_universe, year_context)
    _add_universe_context_metadata(sample_size, year_context)

    return stock_universe, sample_size


def build_reported_gbp_sample_size(raw_sample_size: dict, exchange: str) -> dict:
    reported_sample_size = build_reported_sample_size(raw_sample_size, exchange=exchange)
    for metadata_label in [
        EXCHANGE_CODE_XLON_OR_MISSING_LABEL,
        EXCHANGE_NAME_LSE_OR_MISSING_LABEL,
        CURRENT_RIC_ENDS_WITH_DOT_L_LABEL,
        POSITIVE_MARKET_CAP_LABEL,
        NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL,
        POSITIVE_BOOK_TO_MARKET_LABEL,
        "Rows before firm-level deduplication",
        "Firm-level duplicate rows before deduplication",
        "Firm-level duplicate gvkeys before deduplication",
        "Listings removed by primary listing preference",
        "Listings removed by market-cap tie-break",
        "Listings removed by deterministic tie-break",
        "Unique firms (gvkey where available)",
        "Benchmark universe source",
        "Benchmark return window start",
        "Benchmark return window end",
        "Benchmark weighting method",
        "Benchmark return request batches",
        "Benchmark constituent count",
    ]:
        if metadata_label in raw_sample_size:
            reported_sample_size[metadata_label] = raw_sample_size[metadata_label]
    return reported_sample_size
