from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import (
    FORMATION_YEARS,
    SUE_COMPUTATION_GROUP_COUNT,
    SUE_GROUPS_PIPELINE_VERSION,
)
from src.core.pead_sample_variants import MAIN_PEAD_SAMPLE, PEAD_EVENT_SAMPLE_VARIANTS
from src.core.year_context import build_year_context
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.core.pipeline_state import has_pipeline_version, load_completion_state, write_stage_completion
from src.pead.sue_groups import (
    SUE_GROUP_COLUMN,
    assign_prior_year_sue_groups,
    normalize_sue_group_columns,
)
from src.core.yearly_data_io import load_csv_if_exists

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR
SELECTED_YEARS_ENV_VAR = "BACHELOR_THESIS_SELECTED_YEARS"


def configured_formation_years() -> list[int]:
    override = os.environ.get(SELECTED_YEARS_ENV_VAR, "").strip()
    if not override:
        return list(FORMATION_YEARS)
    return [int(value.strip()) for value in override.split(",") if value.strip()]


def build_sue_group_restrictions_metadata() -> dict:
    return {
        "sue_computation_group_count": int(SUE_COMPUTATION_GROUP_COUNT),
        "event_sample_variants": [
            {
                "key": variant.key,
                "earnings_events_filename": variant.earnings_events_filename,
                "min_analyst_forecasts": int(variant.min_analyst_forecasts),
            }
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ],
    }


def has_matching_sue_group_restrictions(completion_state: dict) -> bool:
    restrictions = completion_state.get("saved_group_restrictions")
    return isinstance(restrictions, dict) and restrictions == (
        build_sue_group_restrictions_metadata()
    )


def load_sue_base_dependency(year_context, required: bool = True) -> dict | None:
    if not year_context.sue_complete_path.exists():
        if required:
            raise FileNotFoundError(
                "Base SUE inputs are missing. "
                "Run scripts/04_build_earnings_and_sue.py first."
            )
        return None

    completion_state = load_completion_state(year_context.sue_complete_path)
    restrictions = completion_state.get("saved_event_restrictions")
    if not isinstance(restrictions, dict):
        raise RuntimeError(
            "Cannot verify SUE-group dependencies because the base SUE completion "
            f"metadata is incomplete for {year_context.year}. "
            "Run scripts/04_build_earnings_and_sue.py once to refresh it."
        )

    return {
        "year": int(year_context.year),
        "pipeline_version": completion_state.get("pipeline_version"),
        "completed_at_utc": completion_state.get("completed_at_utc"),
        "saved_event_restrictions": restrictions,
    }


def load_existing_earnings_events(year_context) -> pd.DataFrame:
    return load_existing_variant_earnings_events(
        year_context,
        MAIN_PEAD_SAMPLE,
    )


def load_existing_variant_earnings_events(year_context, variant) -> pd.DataFrame:
    earnings_events = load_csv_if_exists(
        variant.earnings_events_path(year_context.year_dir),
        normalizer=normalize_sue_group_columns,
    )
    if earnings_events is None:
        raise FileNotFoundError(
            "Base SUE event file is missing. "
            "Run scripts/04_build_earnings_and_sue.py first. "
            f"Missing: {variant.earnings_events_path(year_context.year_dir)}"
        )
    required_columns = {"Instrument", "Ann_Date", "SUE"}
    missing_columns = required_columns.difference(earnings_events.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"{variant.earnings_events_filename} is missing required base SUE columns. "
            f"Run scripts/04_build_earnings_and_sue.py again. Missing: {missing}"
        )

    return earnings_events


def load_prior_year_earnings_events(year: int, variant) -> pd.DataFrame | None:
    prior_context = build_year_context(year - 1, DATA_DIR)

    if not variant.earnings_events_path(prior_context.year_dir).exists():
        return None

    return load_existing_variant_earnings_events(prior_context, variant)


def has_complete_sue_group_outputs(year_context) -> bool:
    required_columns = {"Instrument", "Ann_Date", "SUE", SUE_GROUP_COLUMN}
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        earnings_events = load_csv_if_exists(
            variant.earnings_events_path(year_context.year_dir),
            normalizer=normalize_sue_group_columns,
        )
        if earnings_events is None or not required_columns.issubset(earnings_events.columns):
            return False
    return True


def is_sue_group_year_complete(year_context) -> bool:
    if not year_context.sue_groups_complete_path.exists():
        return False

    completion_state = load_completion_state(year_context.sue_groups_complete_path)
    if not has_pipeline_version(
        year_context.sue_groups_complete_path, SUE_GROUPS_PIPELINE_VERSION
    ):
        return False
    if not has_matching_sue_group_restrictions(completion_state):
        return False

    current_dependency = load_sue_base_dependency(year_context, required=True)
    prior_dependency = load_sue_base_dependency(
        build_year_context(year_context.year - 1, DATA_DIR),
        required=False,
    )

    if completion_state.get("current_sue_dependency") != current_dependency:
        return False
    if completion_state.get("prior_sue_dependency") != prior_dependency:
        return False

    return has_complete_sue_group_outputs(year_context)


def mark_sue_group_year_complete(year_context) -> None:
    write_stage_completion(
        path=year_context.sue_groups_complete_path,
        year=year_context.year,
        stage="earnings_sue_groups",
        pipeline_version=SUE_GROUPS_PIPELINE_VERSION,
        outputs=[
            variant.earnings_events_path(year_context.year_dir)
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ],
        extra_fields={
            "saved_group_restrictions": build_sue_group_restrictions_metadata(),
            "current_sue_dependency": load_sue_base_dependency(
                year_context, required=True
            ),
            "prior_sue_dependency": load_sue_base_dependency(
                build_year_context(year_context.year - 1, DATA_DIR),
                required=False,
            ),
        },
    )


def save_sue_group_outputs(year_context, grouped_events_by_variant: dict[str, pd.DataFrame]) -> None:
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        grouped_events = grouped_events_by_variant[variant.key]
        grouped_events.to_csv(
            variant.earnings_events_path(year_context.year_dir),
            index=False,
        )


def build_sue_groups_year(year: int) -> None:
    year_context = build_year_context(year, DATA_DIR)
    year_context.year_dir.mkdir(parents=True, exist_ok=True)

    if is_sue_group_year_complete(year_context):
        print(f"\n=== Skipping formation year {year}: SUE groups already complete ===")
        return

    print(f"\n=== Building SUE groups for formation year {year} ===")

    grouped_events_by_variant: dict[str, pd.DataFrame] = {}
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        current_events = load_existing_variant_earnings_events(year_context, variant)
        prior_year_events = load_prior_year_earnings_events(year, variant)
        grouped_events_by_variant[variant.key] = assign_prior_year_sue_groups(
            current_events=current_events,
            prior_year_events=prior_year_events,
            group_count=SUE_COMPUTATION_GROUP_COUNT,
        )

    save_sue_group_outputs(year_context, grouped_events_by_variant)
    mark_sue_group_year_complete(year_context)

    print(
        f"Earnings events with SUE groups for {year}: "
        f"{len(grouped_events_by_variant['main'])}"
    )


def main() -> None:
    for year in configured_formation_years():
        build_sue_groups_year(year)


if __name__ == "__main__":
    main()


