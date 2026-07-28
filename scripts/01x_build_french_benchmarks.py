from pathlib import Path
import sys

import lseg.data as ld
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import (
    ANALYSIS_EXCHANGE,
    BASE_PIPELINE_VERSION,
    CURRENCY,
    FORMATION_YEARS,
    HIGH_VOLATILITY_STD_THRESHOLD_PCT,
    IMPLAUSIBLE_SAME_SIGN_SHARE_THRESHOLD,
    LOW_VOLATILITY_STD_THRESHOLD_PCT,
    UNIVERSE_SOURCE,
    ZERO_RETURN_SHARE_THRESHOLD,
)
from src.core.year_context import build_year_context
from src.pead.french_benchmarks import (
    STANDARD_PORTFOLIO_LABELS,
    build_benchmark_portfolio_constituents,
    compute_french_benchmark_breakpoints,
)
from src.pead.gbp_benchmark_builder import (
    NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL,
    POSITIVE_BOOK_TO_MARKET_LABEL,
    POSITIVE_MARKET_CAP_LABEL,
    _compute_book_to_market_ratio,
    _deduplicate_one_row_per_lseg_identifier,
    build_reported_gbp_sample_size,
    filter_to_gbp_common_stock_candidates,
    build_benchmark_reference_universe_from_gbp,
    build_enriched_gbp_universe_for_year,
    enriched_gbp_universe_cache_has_expected_columns,
    read_enriched_gbp_universe_cache,
)
from src.pead.market_data_fetch import (
    download_market_data_for_instruments,
    extract_price_history,
    extract_total_return_history,
    market_data_file_has_expected_columns,
    return_window_cache_has_expected_columns,
    save_window_cache,
    subset_market_data_to_instruments,
    wide_history_to_long,
)
from src.pead.market_data_repairs import (
    IDENTICAL_PRICE_REMOVAL_KEY,
    LOW_COVERAGE_DATE_REMOVAL_KEY,
    NONPOSITIVE_PRICE_REMOVAL_KEY,
    PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY,
    REVERSAL_OUTLIER_REMOVAL_KEY,
    TRAILING_ZERO_RETURN_REMOVAL_KEY,
    apply_market_data_repairs,
)
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.core.pipeline_state import (
    has_pipeline_version,
    write_stage_completion,
)
from src.core.yearly_data_io import load_sample_size, merge_and_save_sample_size
from src.pead.daily_market_cap_benchmarks import (
    DAILY_MARKET_CAP_WEIGHTING_METHOD,
    build_daily_value_weighted_benchmark_returns,
    complete_daily_market_cap_panel,
    download_daily_market_cap_history,
)
from src.utils.io_utils import load_json, save_json

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR
BENCHMARK_WEIGHTING_METHOD = DAILY_MARKET_CAP_WEIGHTING_METHOD
BENCHMARK_EXCHANGE_DEFINITIONS = (("Great Britain", ANALYSIS_EXCHANGE),)
REQUIRED_GBP_UNIVERSE_SCREEN_SAMPLE_SIZE_KEYS = [
    "Raw historical candidates",
    "Enrichment success",
    "Exchange code XLON or missing",
    "Exchange name London Stock Exchange or missing",
    "Current RIC ends with .L",
    "Historical ordinary/common share candidates",
    "Ordinary shares",
    "Non-missing book-to-market inputs",
    "Positive market cap",
    "Positive book-to-market last fiscal year",
    "Rows before firm-level deduplication",
    "Unique firms (gvkey where available)",
    "Benchmark branch: universe before bottom 20% price filter",
    "Benchmark branch: universe after bottom 20% price filter",
    "Benchmark branch: Stock-level return filter: implausibility (>98% same sign)",
    "Benchmark branch: Stock-level return filter: few observations (>95% daily zero)",
    "Benchmark branch: Stock-level return filter: high volatility (sd > 40%)",
    "Benchmark branch: Stock-level return filter: low volatility (sd < 1e-4)",
    f"Benchmark branch: {TRAILING_ZERO_RETURN_REMOVAL_KEY}",
    f"Benchmark branch: {IDENTICAL_PRICE_REMOVAL_KEY}",
    f"Benchmark branch: {REVERSAL_OUTLIER_REMOVAL_KEY}",
    f"Benchmark branch: {LOW_COVERAGE_DATE_REMOVAL_KEY}",
    f"Benchmark branch: {NONPOSITIVE_PRICE_REMOVAL_KEY}",
    f"Benchmark branch: {PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY}",
]

class SkipBenchmarkYear(RuntimeError):
    """Raised when a formation year should be skipped without aborting the batch."""


def load_or_build_enriched_gbp_universe(year_context) -> pd.DataFrame:
    cache_path = year_context.enriched_gbp_universe_path
    if enriched_gbp_universe_cache_has_expected_columns(cache_path):
        return read_enriched_gbp_universe_cache(cache_path)

    enriched_universe = build_enriched_gbp_universe_for_year(
        year_context=year_context,
        currency=CURRENCY,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_universe.to_csv(cache_path, index=False)
    return enriched_universe


def has_expected_benchmark_constituent_columns(path: Path) -> bool:
    required_columns = {
        "Instrument",
        "Benchmark_Portfolio",
        "Benchmark_Weight",
        "Market_Cap_Current",
    }
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False

    return required_columns.issubset(columns)


def enriched_universe_cache_is_conservatively_screened(path: Path) -> bool:
    if not enriched_gbp_universe_cache_has_expected_columns(path):
        return False

    try:
        enriched_universe = read_enriched_gbp_universe_cache(path)
    except Exception:
        return False

    screened_universe, _ = filter_to_gbp_common_stock_candidates(
        df=enriched_universe,
        apply_conservative_xlon_filter=True,
    )
    return int(len(screened_universe)) == int(len(enriched_universe))


def build_market_data_identifier_options(
    benchmark_constituents: pd.DataFrame,
) -> dict[str, list[str]]:
    identifier_options: dict[str, list[str]] = {}
    identifier_columns = [
        "Instrument",
        "RIC",
        "Historical_RIC",
        "Archived_RIC",
    ]

    available_columns = [
        column for column in identifier_columns if column in benchmark_constituents.columns
    ]
    if not available_columns:
        return identifier_options

    for row in benchmark_constituents.loc[:, available_columns].itertuples(index=False):
        ordered_candidates: list[str] = []
        for value in row:
            if pd.isna(value):
                continue
            candidate = str(value).strip()
            if candidate and candidate not in ordered_candidates:
                ordered_candidates.append(candidate)
        if ordered_candidates:
            identifier_options[ordered_candidates[0]] = ordered_candidates

    return identifier_options


def is_benchmark_year_complete(year_context) -> bool:
    if not year_context.benchmark_complete_path.exists():
        return False

    if not has_pipeline_version(year_context.benchmark_complete_path, BASE_PIPELINE_VERSION):
        return False

    if not all(path.exists() for path in year_context.benchmark_output_paths):
        return False

    if not enriched_universe_cache_is_conservatively_screened(
        year_context.enriched_gbp_universe_path
    ):
        return False

    if not has_expected_benchmark_constituent_columns(
        year_context.benchmark_constituents_path
    ):
        return False

    if not return_window_cache_has_expected_columns(
        year_context.benchmark_return_windows_path
    ):
        return False
    if not enriched_gbp_universe_cache_has_expected_columns(
        year_context.shared_post_cleaning_universe_path
    ):
        return False
    if not market_data_file_has_expected_columns(year_context.shared_market_data_path):
        return False
    if not market_data_file_has_expected_columns(year_context.benchmark_market_data_path):
        return False
    if not (
        (year_context.cache_dir / "daily_market_caps_completed.csv").exists()
        and (year_context.cache_dir / "daily_market_cap_completion_summary.json").exists()
        and (year_context.cache_dir / "daily_market_cap_benchmark_coverage.csv").exists()
    ):
        return False

    sample_size = load_sample_size(year_context)
    if any(
        key not in sample_size for key in REQUIRED_GBP_UNIVERSE_SCREEN_SAMPLE_SIZE_KEYS
    ):
        return False

    benchmark_breakpoints = load_json(
        year_context.benchmark_breakpoints_path,
        default={},
    )
    required_breakpoint_keys = {
        "formation_date",
        "reference_universe_count",
        "french_benchmark_exchanges",
        "size_breakpoints",
        "big_stock_market_cap_share",
        "big_stock_count",
        "big_stock_market_cap_floor",
        "bm_breakpoints",
        "benchmark_universe_source",
        "benchmark_return_window_start",
        "benchmark_return_window_end",
        "benchmark_weighting_method",
        "benchmark_return_request_batches",
        "benchmark_constituent_count",
    }
    if not required_breakpoint_keys.issubset(benchmark_breakpoints):
        return False
    if benchmark_breakpoints["benchmark_weighting_method"] != BENCHMARK_WEIGHTING_METHOD:
        return False

    try:
        benchmark_returns = pd.read_csv(
            year_context.benchmark_returns_path,
            index_col=0,
            parse_dates=True,
        )
    except Exception:
        return False

    return (
        not benchmark_returns.empty
        and list(benchmark_returns.columns) == list(STANDARD_PORTFOLIO_LABELS)
    )


def mark_benchmark_year_complete(year_context) -> None:
    write_stage_completion(
        path=year_context.benchmark_complete_path,
        year=year_context.year,
        stage="self_constructed_french_benchmark_preprocessing",
        pipeline_version=BASE_PIPELINE_VERSION,
        outputs=year_context.benchmark_output_paths,
        extra_fields={
            "benchmark_universe_source": UNIVERSE_SOURCE,
            "benchmark_return_window_start": year_context.market_data_window_start,
            "benchmark_return_window_end": year_context.market_data_window_end,
            "benchmark_weighting_method": BENCHMARK_WEIGHTING_METHOD,
        },
    )


def record_gbp_universe_screen_sample_size(
    year_context,
    enriched_universe: pd.DataFrame,
) -> pd.DataFrame:
    screened_universe, raw_sample_size = filter_to_gbp_common_stock_candidates(
        df=enriched_universe,
        apply_conservative_xlon_filter=True,
    )
    screened_universe = screened_universe.loc[
        screened_universe[
            [
                "Instrument",
                "Price",
                "Market_Cap_Current",
                "Market_Cap_Last_Fiscal_Year_End",
                "Book_Equity_Last_Fiscal_Year",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()
    raw_sample_size[NON_MISSING_BOOK_TO_MARKET_INPUTS_LABEL] = int(
        len(screened_universe)
    )
    screened_universe = screened_universe.loc[
        pd.to_numeric(screened_universe["Market_Cap_Current"], errors="coerce") > 0
    ].copy()
    raw_sample_size[POSITIVE_MARKET_CAP_LABEL] = int(len(screened_universe))
    screened_universe = screened_universe.loc[
        _compute_book_to_market_ratio(screened_universe) > 0
    ].copy()
    raw_sample_size[POSITIVE_BOOK_TO_MARKET_LABEL] = int(len(screened_universe))
    raw_sample_size["Rows before firm-level deduplication"] = int(len(screened_universe))
    screened_universe = _deduplicate_one_row_per_lseg_identifier(
        screened_universe,
        sample_size=raw_sample_size,
    )
    raw_sample_size["Unique firms (gvkey where available)"] = int(len(screened_universe))
    reported_sample_size = build_reported_gbp_sample_size(
        raw_sample_size,
        exchange=ANALYSIS_EXCHANGE,
    )
    merge_and_save_sample_size(
        year_context,
        {
            **raw_sample_size,
            **reported_sample_size,
        },
    )
    return screened_universe.reset_index(drop=True)


def prefix_sample_size_keys(sample_size: dict[str, int], prefix: str) -> dict[str, int]:
    return {f"{prefix}{key}": value for key, value in sample_size.items()}


def apply_stock_level_return_filters(
    universe: pd.DataFrame,
    market_data: pd.DataFrame,
    year_context,
    sample_size_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    return_history = extract_total_return_history(market_data)
    if return_history.empty:
        raise RuntimeError(
            "Downloaded market data contains no return history after trimming "
            f"for formation year {year_context.year}."
        )

    return_history = return_history.loc[:, ~return_history.columns.duplicated()].copy()
    return_history.columns = return_history.columns.astype("string").str.strip()

    available_instruments = {
        instrument
        for instrument in return_history.columns.tolist()
        if pd.notna(instrument) and str(instrument).strip()
    }
    filtered_universe = universe.loc[
        universe["Instrument"].astype("string").isin(available_instruments)
    ].copy()

    if filtered_universe.empty:
        raise RuntimeError(
            "No universe rows have matched return histories after download for "
            f"formation year {year_context.year}."
        )

    sample_size: dict[str, int] = {}

    def _apply_filter(
        rows: pd.DataFrame,
        keep_instruments: pd.Index,
        sample_size_key: str,
    ) -> pd.DataFrame:
        out = rows.loc[
            rows["Instrument"].astype("string").isin(set(keep_instruments.astype(str)))
        ].copy()
        sample_size[f"{sample_size_prefix}{sample_size_key}"] = int(
            out["Instrument"].astype("string").nunique()
        )
        return out

    def _universe_instruments(frame: pd.DataFrame) -> pd.Index:
        return pd.Index(frame["Instrument"].astype("string").dropna().unique())

    working_universe = filtered_universe.copy()
    working_returns = return_history.loc[:, _universe_instruments(working_universe)].copy()

    observed_nonzero = working_returns.where(working_returns.ne(0))
    positive_share = observed_nonzero.gt(0).sum(axis=0).div(observed_nonzero.notna().sum(axis=0))
    negative_share = observed_nonzero.lt(0).sum(axis=0).div(observed_nonzero.notna().sum(axis=0))
    plausible_mask = ~(
        positive_share.fillna(0).gt(IMPLAUSIBLE_SAME_SIGN_SHARE_THRESHOLD)
        | negative_share.fillna(0).gt(IMPLAUSIBLE_SAME_SIGN_SHARE_THRESHOLD)
    )
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[plausible_mask],
        "Stock-level return filter: implausibility (>98% same sign)",
    )

    working_returns = return_history.loc[:, _universe_instruments(working_universe)].copy()
    zero_share = working_returns.eq(0).sum(axis=0).div(working_returns.notna().sum(axis=0))
    enough_nonzero_mask = ~zero_share.fillna(1).gt(ZERO_RETURN_SHARE_THRESHOLD)
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[enough_nonzero_mask],
        "Stock-level return filter: few observations (>95% daily zero)",
    )

    working_returns = return_history.loc[:, _universe_instruments(working_universe)].copy()
    return_std = working_returns.std(axis=0, skipna=True)
    not_too_volatile_mask = ~return_std.fillna(float("inf")).gt(
        HIGH_VOLATILITY_STD_THRESHOLD_PCT
    )
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[not_too_volatile_mask],
        "Stock-level return filter: high volatility (sd > 40%)",
    )

    working_returns = return_history.loc[:, _universe_instruments(working_universe)].copy()
    not_too_flat_mask = ~return_std.reindex(working_returns.columns).fillna(0).lt(
        LOW_VOLATILITY_STD_THRESHOLD_PCT
    )
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[not_too_flat_mask],
        "Stock-level return filter: low volatility (sd < 1e-4)",
    )

    if working_universe.empty:
        raise SkipBenchmarkYear(
            "All universe rows were removed by stock-level return filters for "
            f"formation year {year_context.year}."
        )

    kept_instruments = working_universe["Instrument"].astype("string").tolist()
    filtered_market_data = subset_market_data_to_instruments(market_data, kept_instruments)
    return (
        working_universe.reset_index(drop=True),
        filtered_market_data,
        sample_size,
    )


def apply_bottom_price_filter_for_benchmark(
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    sample_size = {
        "Benchmark branch: universe before bottom 20% price filter": int(len(universe))
    }
    if universe.empty:
        sample_size["Benchmark branch: universe after bottom 20% price filter"] = 0
        return universe.copy(), sample_size

    working = universe.copy()
    working["Price"] = pd.to_numeric(working["Price"], errors="coerce")
    working["Instrument"] = working["Instrument"].astype("string").str.strip()
    working = working.sort_values(["Price", "Instrument"], ascending=[True, True], na_position="last")
    drop_count = int(np.floor(0.20 * len(working)))
    if drop_count > 0:
        working = working.iloc[drop_count:].copy()
    sample_size["Benchmark branch: universe after bottom 20% price filter"] = int(len(working))
    return working.reset_index(drop=True), sample_size


def build_benchmark_year(year: int) -> None:
    year_context = build_year_context(year, DATA_DIR)
    year_context.year_dir.mkdir(parents=True, exist_ok=True)

    if is_benchmark_year_complete(year_context):
        print(
            f"\n=== Skipping formation year {year}: benchmark data already complete ==="
        )
        return

    print(f"\n=== Building benchmark inputs for formation year {year} ===")

    enriched_universe = load_or_build_enriched_gbp_universe(year_context)
    enriched_universe = record_gbp_universe_screen_sample_size(
        year_context,
        enriched_universe,
    )
    year_context.enriched_gbp_universe_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_universe.to_csv(year_context.enriched_gbp_universe_path, index=False)
    enriched_universe.to_csv(year_context.shared_post_cleaning_universe_path, index=False)
    shared_market_data, shared_request_batches = download_market_data_for_instruments(
        instruments=enriched_universe["Instrument"].astype(str).tolist(),
        start=year_context.market_data_window_start,
        end=year_context.market_data_window_end,
        cache_path=year_context.shared_market_data_path,
        currency=CURRENCY,
        identifier_options_by_instrument=build_market_data_identifier_options(
            enriched_universe
        ),
        desc=f"Downloading shared market data {year}",
    )
    shared_market_data.to_csv(year_context.shared_market_data_path, index=True)

    benchmark_universe = enriched_universe.copy()
    benchmark_market_data = subset_market_data_to_instruments(
        market_data=shared_market_data,
        instruments=benchmark_universe["Instrument"].astype(str).tolist(),
    )
    benchmark_universe, benchmark_market_data, stock_level_filter_counts = apply_stock_level_return_filters(
        universe=benchmark_universe,
        market_data=benchmark_market_data,
        year_context=year_context,
        sample_size_prefix="Benchmark branch: ",
    )
    merge_and_save_sample_size(year_context, stock_level_filter_counts)
    benchmark_market_data, market_data_repair_counts = apply_market_data_repairs(
        benchmark_market_data
    )
    benchmark_market_data = subset_market_data_to_instruments(
        market_data=benchmark_market_data,
        instruments=benchmark_universe["Instrument"].astype(str).tolist(),
    )
    merge_and_save_sample_size(
        year_context,
        prefix_sample_size_keys(market_data_repair_counts, "Benchmark branch: "),
    )
    benchmark_universe, benchmark_price_filter_counts = apply_bottom_price_filter_for_benchmark(
        benchmark_universe
    )
    merge_and_save_sample_size(year_context, benchmark_price_filter_counts)
    benchmark_market_data = subset_market_data_to_instruments(
        market_data=benchmark_market_data,
        instruments=benchmark_universe["Instrument"].astype(str).tolist(),
    )
    benchmark_market_data.to_csv(year_context.benchmark_market_data_path, index=True)

    reference_universe = build_benchmark_reference_universe_from_gbp(benchmark_universe)
    if reference_universe.empty:
        raise RuntimeError(
            "Benchmark reference universe is empty after the GBP constituent "
            f"enrichment and benchmark-eligibility filters for formation year {year}."
        )

    benchmark_breakpoints = compute_french_benchmark_breakpoints(
        reference_universe=reference_universe,
        formation_date=year_context.formation_date,
        exchange_definitions=BENCHMARK_EXCHANGE_DEFINITIONS,
    )
    benchmark_constituents = build_benchmark_portfolio_constituents(
        reference_universe=reference_universe,
        breakpoints=benchmark_breakpoints,
    )
    if benchmark_constituents.empty:
        raise RuntimeError(
            "Benchmark constituent set is empty after portfolio assignment for "
            f"formation year {year}."
        )
    benchmark_market_data = subset_market_data_to_instruments(
        market_data=benchmark_market_data,
        instruments=benchmark_constituents["Instrument"].astype(str).tolist(),
    )
    return_history = extract_total_return_history(benchmark_market_data)
    price_history = extract_price_history(benchmark_market_data)
    return_windows = wide_history_to_long(
        history=return_history,
        value_column="TotalReturn",
    )
    save_window_cache(return_windows, year_context.benchmark_return_windows_path)
    observed_market_caps = download_daily_market_cap_history(
        benchmark_constituents=benchmark_constituents,
        return_history=return_history,
        year_context=year_context,
        currency=CURRENCY,
    )
    completed_market_caps = complete_daily_market_cap_panel(
        benchmark_constituents=benchmark_constituents,
        return_history=return_history,
        price_history=price_history,
        observed_market_caps=observed_market_caps,
        year_context=year_context,
    )
    benchmark_returns, benchmark_coverage = build_daily_value_weighted_benchmark_returns(
        benchmark_constituents=benchmark_constituents,
        return_history=return_history,
        completed_market_caps=completed_market_caps,
        formation_date=year_context.formation_date,
        portfolio_labels=STANDARD_PORTFOLIO_LABELS,
    )
    benchmark_returns = benchmark_returns.loc[
        (benchmark_returns.index >= pd.Timestamp(year_context.market_data_window_start))
        & (benchmark_returns.index <= pd.Timestamp(year_context.market_data_window_end))
    ].copy()
    benchmark_returns.index.name = "Date"
    benchmark_coverage.to_csv(
        year_context.cache_dir / "daily_market_cap_benchmark_coverage.csv",
        index=False,
    )

    benchmark_breakpoints.update(
        {
            "benchmark_universe_source": UNIVERSE_SOURCE,
            "benchmark_return_window_start": year_context.market_data_window_start,
            "benchmark_return_window_end": year_context.market_data_window_end,
            "benchmark_weighting_method": BENCHMARK_WEIGHTING_METHOD,
            "benchmark_return_request_batches": int(shared_request_batches),
            "benchmark_constituent_count": int(len(benchmark_constituents)),
        }
    )

    benchmark_constituents.to_csv(year_context.benchmark_constituents_path, index=False)
    benchmark_returns.to_csv(year_context.benchmark_returns_path, index=True)
    save_json(benchmark_breakpoints, year_context.benchmark_breakpoints_path)
    mark_benchmark_year_complete(year_context)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pending_years = []
    for year in FORMATION_YEARS:
        year_context = build_year_context(year, DATA_DIR)
        if is_benchmark_year_complete(year_context):
            print(
                f"\n=== Skipping formation year {year}: benchmark data already complete ==="
            )
            continue
        pending_years.append(year)

    if pending_years:
        ld.open_session()
    try:
        for year in pending_years:
            try:
                build_benchmark_year(year)
            except SkipBenchmarkYear as exc:
                print(f"\n=== Skipping formation year {year}: {exc} ===")
    finally:
        if pending_years:
            ld.close_session()


if __name__ == "__main__":
    main()
