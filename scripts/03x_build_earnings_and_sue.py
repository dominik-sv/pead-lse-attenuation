from pathlib import Path
import sys

import lseg.data as ld
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import (
    CURRENCY,
    FORECAST_PERIOD_TEMPLATES_BY_FREQUENCY,
    FORMATION_YEARS,
    SUE_BASE_PIPELINE_VERSION,
    SUE_EARNINGS_FREQUENCIES,
)
from src.core.pead_sample_variants import (
    MAIN_PEAD_SAMPLE,
    PEAD_EVENT_SAMPLE_VARIANTS,
)
from src.core.year_context import build_year_context
from src.pead.earnings_events import (
    build_sue_filtered_event_sample,
    calculate_sue_for_universe,
)
from src.pead.market_data_fetch import (
    extract_price_history,
    price_window_cache_has_expected_columns,
    read_market_data_file,
    save_window_cache,
    wide_history_to_long,
)
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.core.pipeline_state import has_pipeline_version, load_completion_state, write_stage_completion
from src.tooling.aggregate_sample_size_all_years import rebuild_sample_size_all_years
from src.pead.sue_groups import drop_sue_group_columns, normalize_sue_group_columns
from src.core.yearly_data_io import load_csv_if_exists, load_sample_size, save_sample_size

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR


def build_sue_restrictions_metadata() -> dict:
    return {
        "min_analyst_forecasts": int(MAIN_PEAD_SAMPLE.min_analyst_forecasts),
        "event_sample_variants": [
            {
                "key": variant.key,
                "min_analyst_forecasts": int(variant.min_analyst_forecasts),
                "earnings_events_filename": variant.earnings_events_filename,
            }
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ],
        "report_frequencies": sorted(SUE_EARNINGS_FREQUENCIES.keys()),
        "forecast_period_templates_by_frequency": {
            frequency: list(FORECAST_PERIOD_TEMPLATES_BY_FREQUENCY.get(frequency, ()))
            for frequency in sorted(SUE_EARNINGS_FREQUENCIES.keys())
        },
    }


def has_required_sue_restrictions_metadata(completion_state: dict) -> bool:
    restrictions = completion_state.get("saved_event_restrictions")
    if not isinstance(restrictions, dict):
        return False

    min_analyst_forecasts = restrictions.get("min_analyst_forecasts")
    report_frequencies = restrictions.get("report_frequencies")
    forecast_period_templates = restrictions.get(
        "forecast_period_templates_by_frequency"
    )
    event_sample_variants = restrictions.get("event_sample_variants")
    return isinstance(min_analyst_forecasts, int) and isinstance(
        report_frequencies, list
    ) and isinstance(forecast_period_templates, dict) and isinstance(
        event_sample_variants, list
    )


def has_matching_sue_restrictions(completion_state: dict) -> bool:
    if not has_required_sue_restrictions_metadata(completion_state):
        return False

    saved_restrictions = completion_state.get("saved_event_restrictions")
    if not isinstance(saved_restrictions, dict):
        return False

    current_restrictions = build_sue_restrictions_metadata()

    saved_min_analyst_forecasts = saved_restrictions.get("min_analyst_forecasts")
    if saved_min_analyst_forecasts != current_restrictions["min_analyst_forecasts"]:
        return False

    saved_report_frequencies = {
        str(frequency).strip()
        for frequency in saved_restrictions.get("report_frequencies", [])
    }
    current_report_frequencies = set(current_restrictions["report_frequencies"])
    if not current_report_frequencies.issubset(saved_report_frequencies):
        return False

    saved_templates = saved_restrictions.get("forecast_period_templates_by_frequency")
    if not isinstance(saved_templates, dict):
        return False

    current_templates = current_restrictions["forecast_period_templates_by_frequency"]
    for frequency, expected_templates in current_templates.items():
        saved_frequency_templates = saved_templates.get(frequency)
        if list(saved_frequency_templates) != list(expected_templates):
            return False

    if saved_restrictions.get("event_sample_variants") != current_restrictions["event_sample_variants"]:
        return False

    return True


def load_existing_earnings_events(year_context) -> pd.DataFrame | None:
    return load_csv_if_exists(
        year_context.earnings_events_path,
        normalizer=normalize_sue_group_columns,
    )


def load_existing_full_earnings_events(year_context) -> pd.DataFrame | None:
    return load_csv_if_exists(year_context.earnings_events_full_path)


def load_existing_variant_earnings_events(year_context, variant) -> pd.DataFrame | None:
    return load_csv_if_exists(
        variant.earnings_events_path(year_context.year_dir),
        normalizer=normalize_sue_group_columns,
    )


def can_reuse_existing_sue_events(earnings_events: pd.DataFrame) -> bool:
    required_columns = {"Instrument", "Ann_Date", "SUE"}
    return required_columns.issubset(earnings_events.columns)


def has_complete_sue_outputs(year_context) -> bool:
    required_paths = [
        year_context.earnings_events_full_path,
        year_context.sample_size_path,
        year_context.price_windows_path,
        *[
            variant.earnings_events_path(year_context.year_dir)
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ],
    ]
    if not all(path.exists() for path in required_paths):
        return False

    if not price_window_cache_has_expected_columns(year_context.price_windows_path):
        return False

    required_columns = {"Instrument", "Ann_Date", "SUE"}
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        earnings_events = load_existing_variant_earnings_events(year_context, variant)
        if earnings_events is None or not required_columns.issubset(earnings_events.columns):
            return False
    return True


def is_sue_year_complete(year_context) -> bool:
    if not year_context.sue_complete_path.exists():
        return False

    completion_state = load_completion_state(year_context.sue_complete_path)
    if not has_pipeline_version(year_context.sue_complete_path, SUE_BASE_PIPELINE_VERSION):
        return False
    if not has_matching_sue_restrictions(completion_state):
        return False

    return has_complete_sue_outputs(year_context)


def mark_sue_year_complete(year_context) -> None:
    output_paths = [
        *[
            variant.earnings_events_path(year_context.year_dir)
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ],
        year_context.earnings_events_full_path,
        year_context.sample_size_path,
        year_context.price_windows_path,
    ]
    write_stage_completion(
        path=year_context.sue_complete_path,
        year=year_context.year,
        stage="earnings_forecasts_and_sue",
        pipeline_version=SUE_BASE_PIPELINE_VERSION,
        outputs=output_paths,
        extra_fields={
            "saved_event_restrictions": build_sue_restrictions_metadata(),
        },
    )


def load_completed_earnings_sample_size(year_context) -> dict:
    return load_sample_size(year_context)


def load_base_inputs(year_context) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    missing_paths = [
        path
        for path in (
            year_context.stock_universe_path,
            year_context.sample_size_path,
            year_context.market_data_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Base files are missing. Run scripts/02_build_universe_and_market_data.py first. "
            f"Missing: {missing}"
        )

    stock_universe = pd.read_csv(year_context.stock_universe_path)
    sample_size = load_sample_size(year_context)
    market_data = read_market_data_file(year_context.market_data_path)

    return stock_universe, sample_size, market_data


def save_sue_outputs(
    year_context,
    full_earnings_events: pd.DataFrame,
    sample_size: dict,
) -> None:
    full_base_events = drop_sue_group_columns(full_earnings_events)
    full_base_events.to_csv(year_context.earnings_events_full_path, index=False)
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        variant_events = build_sue_filtered_event_sample(
            full_earnings_events,
            min_analyst_forecasts=variant.min_analyst_forecasts,
            sample_size=sample_size,
            sample_size_key_suffix=variant.sample_size_suffix,
        )
        base_events = drop_sue_group_columns(variant_events)
        base_events.to_csv(
            variant.earnings_events_path(year_context.year_dir),
            index=False,
        )
    save_sample_size(year_context, sample_size)


def needs_remote_sue_build(year_context) -> bool:
    if is_sue_year_complete(year_context):
        return False
    main_earnings_events = load_existing_earnings_events(year_context)
    full_earnings_events = load_existing_full_earnings_events(year_context)
    if (
        main_earnings_events is not None
        and full_earnings_events is not None
        and can_reuse_existing_sue_events(main_earnings_events)
    ):
        return False
    return not has_complete_sue_outputs(year_context)


def sanitize_existing_sue_outputs(year_context) -> dict:
    earnings_events = load_existing_earnings_events(year_context)
    if earnings_events is None:
        raise FileNotFoundError(
            f"Cannot sanitize missing earnings events for {year_context.year}."
        )
    full_earnings_events = load_existing_full_earnings_events(year_context)
    if full_earnings_events is None:
        raise FileNotFoundError(
            f"Cannot sanitize missing full earnings events for {year_context.year}."
        )

    sample_size = load_completed_earnings_sample_size(year_context)
    save_sue_outputs(
        year_context=year_context,
        full_earnings_events=full_earnings_events,
        sample_size=sample_size,
    )
    mark_sue_year_complete(year_context)
    return sample_size


def build_sue_year(year: int) -> dict:
    year_context = build_year_context(year, DATA_DIR)
    year_context.year_dir.mkdir(parents=True, exist_ok=True)

    if is_sue_year_complete(year_context):
        print(f"\n=== Skipping formation year {year}: base SUE data already complete ===")
        return load_completed_earnings_sample_size(year_context)

    if has_complete_sue_outputs(year_context):
        print(
            f"\n=== Reusing existing formation year {year}: base SUE outputs already available ==="
        )
        return sanitize_existing_sue_outputs(year_context)

    print(f"\n=== Building earnings forecasts and SUE for formation year {year} ===")

    stock_universe, sample_size, market_data = load_base_inputs(year_context)
    main_earnings_events = load_existing_earnings_events(year_context)
    full_earnings_events = load_existing_full_earnings_events(year_context)
    price_history = extract_price_history(market_data)
    has_valid_price_window_cache = price_window_cache_has_expected_columns(
        year_context.price_windows_path
    )
    if has_valid_price_window_cache:
        print(f"Reusing existing price_windows.csv for {year}.")
    else:
        price_window_cache = wide_history_to_long(
            history=price_history,
            value_column="PriceClose",
        )
        save_window_cache(price_window_cache, year_context.price_windows_path)
        has_valid_price_window_cache = True
    sample_size["Lagged price request batches"] = 0

    if (
        main_earnings_events is None
        or full_earnings_events is None
        or not can_reuse_existing_sue_events(main_earnings_events)
        or not has_valid_price_window_cache
    ):
        _, full_earnings_events, sample_size = calculate_sue_for_universe(
            stock_universe=stock_universe,
            price_history=price_history,
            year=year,
            sample_size=sample_size,
            currency=CURRENCY,
            year_context=year_context,
        )
    else:
        print(
            f"Reusing existing earnings_events.csv for {year} without refetching remote data."
        )

    save_sue_outputs(
        year_context=year_context,
        full_earnings_events=full_earnings_events,
        sample_size=sample_size,
    )
    mark_sue_year_complete(year_context)

    main_variant_events = load_existing_variant_earnings_events(
        year_context,
        MAIN_PEAD_SAMPLE,
    )
    main_variant_count = 0 if main_variant_events is None else len(main_variant_events)
    print(f"Earnings events with SUE for {year}: {main_variant_count}")

    return sample_size


def main() -> None:
    base_data_dir = DATA_DIR
    base_data_dir.mkdir(parents=True, exist_ok=True)

    pending_years = []

    for year in FORMATION_YEARS:
        year_context = build_year_context(year, DATA_DIR)

        if is_sue_year_complete(year_context):
            print(f"\n=== Skipping formation year {year}: base SUE data already complete ===")
            continue

        pending_years.append(year)

    remote_pending_years = [
        year
        for year in pending_years
        if needs_remote_sue_build(build_year_context(year, DATA_DIR))
    ]

    if remote_pending_years:
        ld.open_session()

    for year in pending_years:
        build_sue_year(year)

    rebuild_sample_size_all_years()


if __name__ == "__main__":
    main()
