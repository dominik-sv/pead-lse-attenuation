from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import (
    ANALYSIS_EXCHANGE,
    BASE_PIPELINE_VERSION,
    FORMATION_YEARS,
    HIGH_VOLATILITY_STD_THRESHOLD_PCT,
    IMPLAUSIBLE_SAME_SIGN_SHARE_THRESHOLD,
    LOW_VOLATILITY_STD_THRESHOLD_PCT,
    MARKET_CAP_SIZE_SPLIT_PERCENTILE,
    MARKET_CAP_THRESHOLD,
    STOCK_PRICE_THRESHOLD,
    ZERO_RETURN_SHARE_THRESHOLD,
)
from src.core.year_context import build_year_context
from src.pead.french_benchmarks import (
    assign_french_benchmark_portfolios,
    build_french_benchmark_sample_metadata,
)
from src.pead.gbp_benchmark_builder import (
    build_base_universe_from_cleaned_gbp_universe,
    build_reported_gbp_sample_size,
    enriched_gbp_universe_cache_has_expected_columns,
)
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.core.pipeline_state import has_pipeline_version, write_stage_completion
from src.tooling.aggregate_sample_size_all_years import rebuild_yearly_json_aggregate
from src.pead.market_cap_splits import (
    MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN,
    MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN,
    MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN,
    MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN,
)
from src.pead.market_data_fetch import (
    extract_total_return_history,
    market_data_file_has_expected_columns,
    read_market_data_file,
    subset_market_data_to_instruments,
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
from src.utils.io_utils import load_json
from src.core.yearly_data_io import load_sample_size, merge_and_save_sample_size

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR
REQUIRED_SIZE_SPLIT_COLUMNS = [
    MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN,
    MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN,
    MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN,
    MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN,
]
REQUIRED_BENCHMARK_ASSIGNMENT_COLUMNS = [
    "BM_French",
    "BM",
    "Benchmark_Size_Group",
    "Benchmark_BM_Group",
    "Benchmark_Size_Q",
    "Benchmark_BM_Q",
    "Benchmark_Portfolio",
]
REQUIRED_SIZE_SPLIT_SAMPLE_SIZE_KEYS = [
    "Raw historical candidates",
    "Ordinary/common shares",
    "Required accounting and market data available",
    "Market cap >= threshold",
    "Price >= threshold",
    "Market cap size split percentile",
    "Market cap size split breakpoint",
    "Market cap decile breakpoints",
    "Market cap size split reference universe count",
    "Microcap count",
    "All-but-microcap count",
    "Positive book-to-market last fiscal year",
    "French benchmark formation date",
    "French benchmark reference universe count",
    "French benchmark exchanges",
    "French benchmark sort",
    "French benchmark portfolio labels",
    "French benchmark big stock market cap share",
    "French benchmark big stock count",
    "French benchmark size breakpoint",
    "French benchmark big stock market cap floor",
    "French benchmark B/M percentiles",
    "French benchmark B/M breakpoints",
    "Benchmark universe source",
    "Benchmark return window start",
    "Benchmark return window end",
    "Benchmark weighting method",
    "Benchmark return request batches",
    "Benchmark constituent count",
    "Universe source",
    "Formation date",
    "Universe window start",
    "Universe window end",
    "Analysis branch: Stock-level return filter: implausibility (>98% same sign)",
    "Analysis branch: Stock-level return filter: few observations (>95% daily zero)",
    "Analysis branch: Stock-level return filter: high volatility (sd > 40%)",
    "Analysis branch: Stock-level return filter: low volatility (sd < 1e-4)",
    f"Analysis branch: {TRAILING_ZERO_RETURN_REMOVAL_KEY}",
    f"Analysis branch: {IDENTICAL_PRICE_REMOVAL_KEY}",
    f"Analysis branch: {REVERSAL_OUTLIER_REMOVAL_KEY}",
    f"Analysis branch: {LOW_COVERAGE_DATE_REMOVAL_KEY}",
    f"Analysis branch: {NONPOSITIVE_PRICE_REMOVAL_KEY}",
    f"Analysis branch: {PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY}",
]

def format_sample_size_audit(sample_size: dict) -> str:
    if not sample_size:
        return "No sample-size audit trail was recorded."

    ordered_pairs = list(sample_size.items())
    zero_step = next(
        (
            (label, value)
            for label, value in ordered_pairs
            if isinstance(value, int) and value == 0
        ),
        None,
    )
    tail_pairs = ordered_pairs[-6:]
    tail_summary = ", ".join(f"{label}={value}" for label, value in tail_pairs)

    if zero_step is None:
        return f"Latest sample-size checkpoints: {tail_summary}"

    zero_label, _ = zero_step
    return (
        f"First zero-count filter: {zero_label}. "
        f"Latest sample-size checkpoints: {tail_summary}"
    )


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


def load_benchmark_inputs(year_context) -> dict:
    missing_paths = [
        path
        for path in (
            year_context.benchmark_breakpoints_path,
            year_context.benchmark_returns_path,
            year_context.benchmark_complete_path,
            year_context.enriched_gbp_universe_path,
            year_context.shared_post_cleaning_universe_path,
            year_context.shared_market_data_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Benchmark inputs are missing. Run scripts/01x_build_french_benchmarks.py first. "
            f"Missing: {missing}"
        )

    if not enriched_gbp_universe_cache_has_expected_columns(
        year_context.enriched_gbp_universe_path
    ):
        raise RuntimeError(
            "Cached enriched GBP universe is missing or stale. "
            "Run scripts/01x_build_french_benchmarks.py first."
        )
    if not has_expected_benchmark_constituent_columns(
        year_context.benchmark_constituents_path
    ):
        raise RuntimeError(
            "Cached benchmark constituent universe is missing or stale. "
            "Run scripts/01x_build_french_benchmarks.py first."
        )
    if not market_data_file_has_expected_columns(year_context.shared_market_data_path):
        raise RuntimeError(
            "Cached shared market data is missing or stale. "
            "Run scripts/01x_build_french_benchmarks.py first."
        )

    return load_json(
        year_context.benchmark_breakpoints_path,
        default={},
    )


def is_base_year_complete(year_context) -> bool:
    if not year_context.base_complete_path.exists():
        return False

    if not has_pipeline_version(year_context.base_complete_path, BASE_PIPELINE_VERSION):
        return False

    if not year_context.benchmark_complete_path.exists():
        return False

    if not all(path.exists() for path in year_context.base_output_paths):
        return False

    if not year_context.stock_universe_path.exists():
        return False

    if not year_context.market_data_path.exists():
        return False

    if not year_context.benchmark_returns_path.exists():
        return False
    if not year_context.benchmark_breakpoints_path.exists():
        return False
    if not market_data_file_has_expected_columns(year_context.market_data_path):
        return False
    if not enriched_gbp_universe_cache_has_expected_columns(
        year_context.enriched_gbp_universe_path
    ):
        return False

    try:
        stock_universe = pd.read_csv(year_context.stock_universe_path)
    except Exception:
        return False

    sample_size = load_sample_size(year_context)
    if any(
        column not in stock_universe.columns for column in REQUIRED_SIZE_SPLIT_COLUMNS
    ):
        return False
    if any(
        column not in stock_universe.columns
        for column in REQUIRED_BENCHMARK_ASSIGNMENT_COLUMNS
    ):
        return False
    if any(key not in sample_size for key in REQUIRED_SIZE_SPLIT_SAMPLE_SIZE_KEYS):
        return False

    saved_split_percentile = sample_size.get("Market cap size split percentile")
    if saved_split_percentile is None:
        return False
    if (
        abs(float(saved_split_percentile) - float(MARKET_CAP_SIZE_SPLIT_PERCENTILE))
        > 1e-12
    ):
        return False

    return True


def mark_base_year_complete(year_context) -> None:
    write_stage_completion(
        path=year_context.base_complete_path,
        year=year_context.year,
        stage="base_universe_and_market_data",
        pipeline_version=BASE_PIPELINE_VERSION,
        outputs=year_context.base_output_paths,
        extra_fields={
            "market_cap_size_split_percentile": MARKET_CAP_SIZE_SPLIT_PERCENTILE,
            "analysis_exchange": ANALYSIS_EXCHANGE,
            "benchmark_dependency": {
                "path": year_context.benchmark_complete_path.name,
            },
            "enriched_gbp_universe_dependency": {
                "path": year_context.enriched_gbp_universe_path.name,
            },
        },
    )


def load_completed_base_sample_size(year_context) -> dict:
    return load_sample_size(year_context)


def save_base_outputs(
    year_context,
    stock_universe: pd.DataFrame,
    market_data: pd.DataFrame,
    sample_size: dict,
) -> None:
    stock_universe.to_csv(year_context.stock_universe_path, index=False)
    market_data.to_csv(year_context.market_data_path, index=True)
    merge_and_save_sample_size(year_context, sample_size)


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

    def _apply_filter(rows: pd.DataFrame, keep_instruments: pd.Index, key: str) -> pd.DataFrame:
        out = rows.loc[
            rows["Instrument"].astype("string").isin(set(keep_instruments.astype(str)))
        ].copy()
        sample_size[f"{sample_size_prefix}{key}"] = int(
            out["Instrument"].astype("string").nunique()
        )
        return out

    def _instruments(frame: pd.DataFrame) -> pd.Index:
        return pd.Index(frame["Instrument"].astype("string").dropna().unique())

    working_universe = filtered_universe.copy()
    working_returns = return_history.loc[:, _instruments(working_universe)].copy()
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

    working_returns = return_history.loc[:, _instruments(working_universe)].copy()
    zero_share = working_returns.eq(0).sum(axis=0).div(working_returns.notna().sum(axis=0))
    enough_nonzero_mask = ~zero_share.fillna(1).gt(ZERO_RETURN_SHARE_THRESHOLD)
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[enough_nonzero_mask],
        "Stock-level return filter: few observations (>95% daily zero)",
    )

    working_returns = return_history.loc[:, _instruments(working_universe)].copy()
    return_std = working_returns.std(axis=0, skipna=True)
    not_too_volatile_mask = ~return_std.fillna(float("inf")).gt(
        HIGH_VOLATILITY_STD_THRESHOLD_PCT
    )
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[not_too_volatile_mask],
        "Stock-level return filter: high volatility (sd > 40%)",
    )

    working_returns = return_history.loc[:, _instruments(working_universe)].copy()
    not_too_flat_mask = ~return_std.reindex(working_returns.columns).fillna(0).lt(
        LOW_VOLATILITY_STD_THRESHOLD_PCT
    )
    working_universe = _apply_filter(
        working_universe,
        working_returns.columns[not_too_flat_mask],
        "Stock-level return filter: low volatility (sd < 1e-4)",
    )

    kept_instruments = working_universe["Instrument"].astype("string").tolist()
    filtered_market_data = subset_market_data_to_instruments(market_data, kept_instruments)
    return working_universe.reset_index(drop=True), filtered_market_data, sample_size


def build_base_year(year: int) -> dict:
    year_context = build_year_context(year, DATA_DIR)
    year_context.year_dir.mkdir(parents=True, exist_ok=True)

    if is_base_year_complete(year_context):
        print(f"\n=== Skipping formation year {year}: base data already complete ===")
        return load_completed_base_sample_size(year_context)

    benchmark_breakpoints = load_benchmark_inputs(year_context)
    shared_universe = pd.read_csv(year_context.shared_post_cleaning_universe_path)
    shared_market_data = read_market_data_file(year_context.shared_market_data_path)

    print(f"\n=== Building universe for formation year {year} ===")

    shared_universe, shared_market_data, stock_level_filter_counts = apply_stock_level_return_filters(
        universe=shared_universe,
        market_data=shared_market_data,
        year_context=year_context,
        sample_size_prefix="Analysis branch: ",
    )
    analysis_market_data, market_data_repair_counts = apply_market_data_repairs(shared_market_data)
    merge_and_save_sample_size(
        year_context,
        {
            **stock_level_filter_counts,
            **prefix_sample_size_keys(market_data_repair_counts, "Analysis branch: "),
        },
    )
    shared_universe = shared_universe.loc[
        shared_universe["Instrument"].astype("string").isin(
            analysis_market_data.columns.get_level_values(-1).astype("string")
            if isinstance(analysis_market_data.columns, pd.MultiIndex)
            else pd.Index(analysis_market_data.columns).astype("string")
        )
    ].copy()

    stock_universe, sample_size = build_base_universe_from_cleaned_gbp_universe(
        cleaned_universe=shared_universe,
        year_context=year_context,
        market_cap_threshold=MARKET_CAP_THRESHOLD,
        stock_price_threshold=STOCK_PRICE_THRESHOLD,
        market_cap_size_split_percentile=MARKET_CAP_SIZE_SPLIT_PERCENTILE,
    )

    if stock_universe.empty:
        raise RuntimeError(
            "No stocks left after universe filters for formation year "
            f"{year}. {format_sample_size_audit(sample_size)}"
        )

    sample_size.update(build_french_benchmark_sample_metadata(benchmark_breakpoints))
    reported_sample_size = build_reported_gbp_sample_size(
        sample_size,
        exchange=ANALYSIS_EXCHANGE,
    )

    stock_universe = assign_french_benchmark_portfolios(
        stock_universe,
        benchmark_breakpoints,
    )
    print(f"Universe after filters: {len(stock_universe)} stocks")

    # Reuse the broader benchmark-universe market-data cache from stage 01 and
    # subset it to the final study universe for downstream BHAR processing.
    market_data = subset_market_data_to_instruments(
        market_data=analysis_market_data,
        instruments=stock_universe["Instrument"].astype(str).tolist(),
    )

    save_base_outputs(
        year_context=year_context,
        stock_universe=stock_universe,
        market_data=market_data,
        sample_size=reported_sample_size,
    )
    mark_base_year_complete(year_context)

    return reported_sample_size


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pending_years = []

    for year in FORMATION_YEARS:
        year_context = build_year_context(year, DATA_DIR)

        if is_base_year_complete(year_context):
            print(f"\n=== Skipping formation year {year}: base data already complete ===")
            continue

        pending_years.append(year)

    for year in pending_years:
        build_base_year(year)

    rebuild_yearly_json_aggregate(
        yearly_filename="sample_size.json",
        output_filename="sample_size_all_years.json",
    )


if __name__ == "__main__":
    main()
