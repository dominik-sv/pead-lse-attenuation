from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pead.abnormal_returns import (
    COMPLETE_CASE_RETURN_TREATMENT,
    TERMINAL_LOSS_RETURN_TREATMENT,
    ZERO_FILL_RETURN_TREATMENT,
    build_abnormal_returns_for_earnings_events,
    PRE_ANNOUNCEMENT_START_DAY,
    RETURN_WINDOW_END_DAY,
)
from src.core.pipeline_config import (
    ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    ABNORMAL_RETURNS_PIPELINE_VERSION,
    FORMATION_YEARS,
    MAIN_REGRESSION_BHAR_WINDOW,
)
from src.core.pead_sample_variants import (
    MAIN_PEAD_SAMPLE,
    PEAD_EVENT_SAMPLE_VARIANTS,
)
from src.core.year_context import build_year_context
from src.core.pipeline_state import has_pipeline_version, load_completion_state, write_stage_completion
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.pead.market_data_fetch import (
    extract_total_return_history,
    read_market_data_file,
    return_window_cache_has_expected_columns,
    save_window_cache,
    wide_history_to_long,
)
from src.tooling.aggregate_sample_size_all_years import rebuild_sample_size_all_years
from src.pead.sue_groups import SUE_GROUP_COLUMN, normalize_sue_group_columns
from src.utils.io_utils import load_json, save_json
from src.core.yearly_data_io import (
    load_csv_if_exists,
    load_sample_size as load_year_sample_size,
    save_sample_size as save_year_sample_size,
)

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR
SELECTED_YEARS_ENV_VAR = "BACHELOR_THESIS_SELECTED_YEARS"
SAMPLE_SIZE_BHAR_COLUMNS = tuple(
    f"BHAR_{int(day_start)}_{int(day_end)}"
    for day_start, day_end in (
        MAIN_REGRESSION_BHAR_WINDOW,
        *ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    )
)


def configured_formation_years() -> list[int]:
    override = os.environ.get(SELECTED_YEARS_ENV_VAR, "").strip()
    if not override:
        return list(FORMATION_YEARS)
    return [int(value.strip()) for value in override.split(",") if value.strip()]


def current_bhar_restrictions() -> dict:
    return {
        "event_sample_variants": [
            {
                "key": variant.key,
                "min_analyst_forecasts": int(variant.min_analyst_forecasts),
                "earnings_events_filename": variant.earnings_events_filename,
                "abnormal_returns_filename": variant.abnormal_returns_filename,
                "abnormal_returns_drop_missing_filename": (
                    variant.abnormal_returns_drop_missing_filename
                ),
                "abnormal_returns_terminal_loss_filename": (
                    variant.abnormal_returns_terminal_loss_filename
                ),
            }
            for variant in PEAD_EVENT_SAMPLE_VARIANTS
        ]
    }


def has_matching_bhar_restrictions(completion_state: dict) -> bool:
    restrictions = completion_state.get("analysis_restrictions")
    if not isinstance(restrictions, dict):
        return False

    return restrictions == current_bhar_restrictions()


def load_saved_sue_group_dependency(year_context) -> dict:
    completion_state = load_completion_state(year_context.sue_groups_complete_path)
    restrictions = completion_state.get("saved_group_restrictions")
    if not isinstance(restrictions, dict):
        raise RuntimeError(
            "Cannot verify grouped SUE inputs because the SUE-group completion metadata "
            f"is missing for {year_context.year}. "
            "Run scripts/05_build_sue_groups.py once to refresh it."
        )

    return {
        "year": int(year_context.year),
        "pipeline_version": completion_state.get("pipeline_version"),
        "completed_at_utc": completion_state.get("completed_at_utc"),
        "saved_group_restrictions": restrictions,
    }


def has_matching_sue_group_dependency(
    completion_state: dict,
    year_context,
) -> bool:
    dependency = completion_state.get("sue_group_dependency")
    if not isinstance(dependency, dict):
        return False

    return dependency == load_saved_sue_group_dependency(year_context)


def validate_bhar_subset_restrictions_for_all_years() -> None:
    for year in configured_formation_years():
        year_context = build_year_context(year, DATA_DIR)
        load_saved_sue_group_dependency(year_context)
        for variant in PEAD_EVENT_SAMPLE_VARIANTS:
            load_earnings_events(year_context, variant=variant)


def load_earnings_events(year_context, *, variant=MAIN_PEAD_SAMPLE) -> pd.DataFrame:
    earnings_events = load_csv_if_exists(
        variant.earnings_events_path(year_context.year_dir),
        normalizer=normalize_sue_group_columns,
    )
    if earnings_events is None:
        raise FileNotFoundError(
            "Grouped SUE inputs are missing. Run scripts/04_build_earnings_and_sue.py "
            "and scripts/05_build_sue_groups.py first. "
            f"Missing: {variant.earnings_events_path(year_context.year_dir)}"
        )
    required_columns = {"Instrument", "Ann_Date", "SUE", SUE_GROUP_COLUMN}
    missing_columns = required_columns.difference(earnings_events.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        if missing_columns == {SUE_GROUP_COLUMN}:
            rerun_message = "Run scripts/05_build_sue_groups.py again."
        else:
            rerun_message = (
                "Run scripts/04_build_earnings_and_sue.py and "
                "scripts/05_build_sue_groups.py again."
            )
        raise KeyError(
            f"{variant.earnings_events_filename} is missing required SUE columns. "
            f"{rerun_message} Missing: {missing}"
        )

    return earnings_events


def load_base_inputs(year_context) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_paths = [
        path
        for path in (
            year_context.stock_universe_path,
            year_context.market_data_path,
            year_context.benchmark_returns_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Base files are missing. Run scripts/01_build_french_benchmarks.py and "
            "scripts/02_build_universe_and_market_data.py first. "
            f"Missing: {missing}"
        )

    stock_universe = pd.read_csv(year_context.stock_universe_path)
    market_data = read_market_data_file(year_context.market_data_path)
    benchmark_returns = pd.read_csv(
        year_context.benchmark_returns_path,
        index_col=0,
        parse_dates=True,
    )

    return stock_universe, market_data, benchmark_returns


def load_sample_size(year_context) -> dict:
    return load_year_sample_size(year_context)


def save_sample_size(year_context, sample_size: dict) -> None:
    save_year_sample_size(year_context, sample_size)


def can_reuse_existing_abnormal_returns(abnormal_returns: pd.DataFrame) -> bool:
    required_columns = {
        "Event_ID",
        "Relative_Day",
        "Trading_Date",
        "Abnormal_Return",
        *SAMPLE_SIZE_BHAR_COLUMNS,
    }
    if not required_columns.issubset(abnormal_returns.columns):
        return False

    relative_days = pd.to_numeric(
        abnormal_returns["Relative_Day"], errors="coerce"
    ).dropna()
    if relative_days.empty:
        return False

    return (
        int(relative_days.min()) <= PRE_ANNOUNCEMENT_START_DAY
        and int(relative_days.max()) >= RETURN_WINDOW_END_DAY
    )


def can_reuse_existing_alternative_abnormal_returns_outputs(year_context, *, variant) -> bool:
    alternative_paths = (
        variant.abnormal_returns_drop_missing_path(year_context.year_dir),
        variant.abnormal_returns_terminal_loss_path(year_context.year_dir),
    )
    if not all(path.exists() for path in alternative_paths):
        return False

    return all(
        can_reuse_existing_abnormal_returns(pd.read_csv(path))
        for path in alternative_paths
    )


def build_missing_return_fill_summary(
    abnormal_returns: pd.DataFrame,
) -> dict[str, object]:
    if abnormal_returns.empty or "Event_ID" not in abnormal_returns.columns:
        event_level = pd.DataFrame()
    else:
        event_level = abnormal_returns.drop_duplicates(subset=["Event_ID"]).copy()

    missing_pre_counts = pd.to_numeric(
        event_level.get(
            "Missing_Pre_Announcement_Return_Day_Count",
            pd.Series(0, index=event_level.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    missing_announcement_counts = pd.to_numeric(
        event_level.get(
            "Missing_Announcement_Window_Return_Day_Count",
            pd.Series(0, index=event_level.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)

    missing_counts = pd.to_numeric(
        event_level.get(
            "Missing_Post_Announcement_Return_Day_Count",
            pd.Series(0, index=event_level.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    affected = event_level.loc[missing_counts.gt(0)].copy()
    affected_missing_counts = missing_counts.loc[affected.index]
    retirement_mask = affected.get(
        "Has_Post_Retirement_Missing_Return", pd.Series(False, index=affected.index)
    ).fillna(False).astype(bool)
    post_retirement_missing_counts = pd.to_numeric(
        affected.get(
            "Post_Retirement_Missing_Return_Day_Count",
            pd.Series(0, index=affected.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    interior_mask = affected.get(
        "Has_Interior_Missing_Return", pd.Series(False, index=affected.index)
    ).fillna(False).astype(bool)
    interior_missing_counts = pd.to_numeric(
        affected.get(
            "Interior_Missing_Return_Day_Count",
            pd.Series(0, index=affected.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    terminal_mask = affected.get(
        "Has_Terminal_Missing_Return", pd.Series(False, index=affected.index)
    ).fillna(False).astype(bool)
    terminal_missing_counts = pd.to_numeric(
        affected.get(
            "Terminal_Missing_Return_Day_Count",
            pd.Series(0, index=affected.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    if int(interior_missing_counts.sum() + terminal_missing_counts.sum()) != int(
        affected_missing_counts.sum()
    ):
        raise ValueError(
            "Interior and terminal missing-return day counts do not partition all "
            "missing post-announcement return days."
        )
    if int(post_retirement_missing_counts.sum()) > int(affected_missing_counts.sum()):
        raise ValueError(
            "Post-retirement missing-return days exceed all missing post-announcement days."
        )

    affected_records: list[dict[str, object]] = []
    diagnostic_columns = [
        "Event_ID",
        "Instrument",
        "Ann_Date",
        "Missing_Post_Announcement_Return_Day_Count",
        "First_Missing_Relative_Day",
        "Last_Observed_Relative_Day",
        "Has_Interior_Missing_Return",
        "Interior_Missing_Return_Day_Count",
        "Has_Terminal_Missing_Return",
        "Terminal_Missing_Return_Day_Count",
        "Retire_Date_In_Event_Window",
        "Has_Post_Retirement_Missing_Return",
        "Post_Retirement_Missing_Return_Day_Count",
    ]
    def json_safe_scalar(value):
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return value.item() if hasattr(value, "item") else value

    for _, record in affected.loc[
        :, [column for column in diagnostic_columns if column in affected.columns]
    ].iterrows():
        affected_records.append(
            {
                str(key): json_safe_scalar(value)
                for key, value in record.to_dict().items()
            }
        )

    return {
        "events_with_missing_pre_announcement_returns": int(
            missing_pre_counts.gt(0).sum()
        ),
        "missing_pre_announcement_return_days": int(missing_pre_counts.sum()),
        "events_with_missing_announcement_window_returns": int(
            missing_announcement_counts.gt(0).sum()
        ),
        "missing_announcement_window_return_days": int(
            missing_announcement_counts.sum()
        ),
        "events_with_zero_filled_post_announcement_returns": int(len(affected)),
        "zero_filled_post_announcement_return_days": int(affected_missing_counts.sum()),
        "events_with_zero_filled_post_retirement_returns": int(retirement_mask.sum()),
        "zero_filled_post_retirement_return_days": int(
            post_retirement_missing_counts.sum()
        ),
        "events_with_interior_missing_post_announcement_returns": int(interior_mask.sum()),
        "interior_missing_post_announcement_return_days": int(
            interior_missing_counts.sum()
        ),
        "events_with_terminal_missing_post_announcement_returns": int(terminal_mask.sum()),
        "terminal_missing_post_announcement_return_days": int(
            terminal_missing_counts.sum()
        ),
        "affected_events": affected_records,
    }


def save_missing_return_fill_summary(year_context, fill_summary: dict[str, object], *, variant) -> None:
    save_json(fill_summary, variant.missing_return_fill_summary_path(year_context.year_dir))


def load_missing_return_fill_summary(year_context, *, variant) -> dict[str, object]:
    default_summary = build_missing_return_fill_summary(pd.DataFrame())
    saved_summary = load_json(
        variant.missing_return_fill_summary_path(year_context.year_dir),
        default=default_summary,
    )
    if not isinstance(saved_summary, dict):
        return default_summary

    affected_events = saved_summary.get("affected_events", [])
    if not isinstance(affected_events, list):
        affected_events = []

    return {
        "events_with_missing_pre_announcement_returns": int(
            saved_summary.get("events_with_missing_pre_announcement_returns", 0)
        ),
        "missing_pre_announcement_return_days": int(
            saved_summary.get("missing_pre_announcement_return_days", 0)
        ),
        "events_with_missing_announcement_window_returns": int(
            saved_summary.get("events_with_missing_announcement_window_returns", 0)
        ),
        "missing_announcement_window_return_days": int(
            saved_summary.get("missing_announcement_window_return_days", 0)
        ),
        "events_with_zero_filled_post_announcement_returns": int(
            saved_summary.get(
                "events_with_zero_filled_post_announcement_returns",
                len(affected_events),
            )
        ),
        "zero_filled_post_announcement_return_days": int(
            saved_summary.get("zero_filled_post_announcement_return_days", 0)
        ),
        "events_with_zero_filled_post_retirement_returns": int(
            saved_summary.get("events_with_zero_filled_post_retirement_returns", 0)
        ),
        "zero_filled_post_retirement_return_days": int(
            saved_summary.get("zero_filled_post_retirement_return_days", 0)
        ),
        "events_with_interior_missing_post_announcement_returns": int(
            saved_summary.get("events_with_interior_missing_post_announcement_returns", 0)
        ),
        "interior_missing_post_announcement_return_days": int(
            saved_summary.get("interior_missing_post_announcement_return_days", 0)
        ),
        "events_with_terminal_missing_post_announcement_returns": int(
            saved_summary.get("events_with_terminal_missing_post_announcement_returns", 0)
        ),
        "terminal_missing_post_announcement_return_days": int(
            saved_summary.get("terminal_missing_post_announcement_return_days", 0)
        ),
        "affected_events": affected_events,
    }


def update_abnormal_returns_sample_size(
    sample_size: dict,
    kept_event_count: int,
    dropped_events: pd.DataFrame,
    fill_summary: dict[str, object],
    sample_size_suffix: str = "",
    zero_fill_abnormal_returns: pd.DataFrame | None = None,
    complete_case_abnormal_returns: pd.DataFrame | None = None,
    terminal_loss_abnormal_returns: pd.DataFrame | None = None,
) -> dict:
    # A count-only DataFrame has rows but no columns, so DataFrame.empty is True.
    # Use the index length to preserve the actual number of excluded events.
    dropped_event_count = int(dropped_events.shape[0])
    selected_event_count = int(kept_event_count) + dropped_event_count
    suffix = str(sample_size_suffix)
    sample_size[f"Earnings events with valid SUE and positive lagged price{suffix}"] = int(
        selected_event_count
    )
    sample_size[f"Earnings events excluded before missing-return treatment{suffix}"] = int(
        dropped_event_count
    )
    sample_size[f"Dropped earnings events with incomplete stock return window{suffix}"] = (
        dropped_event_count
    )
    sample_size[f"Earnings events retained for BHAR construction{suffix}"] = int(kept_event_count)
    sample_size[f"Earnings events with missing pre-announcement stock returns{suffix}"] = int(
        fill_summary["events_with_missing_pre_announcement_returns"]
    )
    sample_size[f"Pre-announcement stock return days missing{suffix}"] = int(
        fill_summary["missing_pre_announcement_return_days"]
    )
    sample_size[f"Earnings events with missing announcement-window stock returns{suffix}"] = int(
        fill_summary["events_with_missing_announcement_window_returns"]
    )
    sample_size[f"Announcement-window stock return days missing{suffix}"] = int(
        fill_summary["missing_announcement_window_return_days"]
    )
    sample_size[f"Earnings events with zero-filled post-announcement stock returns{suffix}"] = (
        int(fill_summary["events_with_zero_filled_post_announcement_returns"])
    )
    sample_size[f"Post-announcement stock return days filled with 0{suffix}"] = int(
        fill_summary["zero_filled_post_announcement_return_days"]
    )
    sample_size[f"Earnings events with zero-filled post-retirement stock returns{suffix}"] = (
        int(fill_summary["events_with_zero_filled_post_retirement_returns"])
    )
    sample_size[f"Post-retirement stock return days filled with 0{suffix}"] = int(
        fill_summary["zero_filled_post_retirement_return_days"]
    )
    sample_size[f"Earnings events with interior missing post-announcement returns{suffix}"] = int(
        fill_summary["events_with_interior_missing_post_announcement_returns"]
    )
    sample_size[f"Interior post-announcement stock return days missing{suffix}"] = int(
        fill_summary["interior_missing_post_announcement_return_days"]
    )
    sample_size[f"Earnings events with terminal missing post-announcement returns{suffix}"] = int(
        fill_summary["events_with_terminal_missing_post_announcement_returns"]
    )
    sample_size[f"Terminal post-announcement stock return days missing{suffix}"] = int(
        fill_summary["terminal_missing_post_announcement_return_days"]
    )

    treatment_frames = {
        "Zero-fill": zero_fill_abnormal_returns,
        "Complete-case": complete_case_abnormal_returns,
        "Terminal-loss": terminal_loss_abnormal_returns,
    }
    for treatment_label, treatment_frame in treatment_frames.items():
        if treatment_frame is None or treatment_frame.empty:
            treatment_events = pd.DataFrame()
        else:
            treatment_events = treatment_frame.drop_duplicates(subset=["Event_ID"])
        for column in SAMPLE_SIZE_BHAR_COLUMNS:
            valid_count = (
                0
                if column not in treatment_events.columns
                else int(pd.to_numeric(treatment_events[column], errors="coerce").notna().sum())
            )
            sample_size[f"{treatment_label} events with non-missing {column}{suffix}"] = valid_count

    complete_case_counts = [
        int(sample_size[f"Complete-case events with non-missing {column}{suffix}"])
        for column in SAMPLE_SIZE_BHAR_COLUMNS
    ]
    if any(
        later_count > earlier_count
        for earlier_count, later_count in zip(
            complete_case_counts, complete_case_counts[1:], strict=False
        )
    ):
        raise ValueError(
            "Complete-case event counts increase as the BHAR window becomes longer."
        )

    longest_bhar_column = SAMPLE_SIZE_BHAR_COLUMNS[-1]
    for treatment_label in ("Zero-fill", "Terminal-loss"):
        valid_longest_bhar_count = int(
            sample_size[
                f"{treatment_label} events with non-missing {longest_bhar_column}{suffix}"
            ]
        )
        if valid_longest_bhar_count != int(kept_event_count):
            raise ValueError(
                f"{treatment_label} {longest_bhar_column} count {valid_longest_bhar_count} does not "
                f"match the {kept_event_count} retained events."
            )

    complete_longest_bhar_key = (
        f"Complete-case events with non-missing {longest_bhar_column}{suffix}"
    )
    sample_size[f"Earnings events with complete stock return window{suffix}"] = int(
        sample_size.get(complete_longest_bhar_key, 0)
    )

    if terminal_loss_abnormal_returns is None or terminal_loss_abnormal_returns.empty:
        terminal_events = pd.DataFrame()
    else:
        terminal_events = terminal_loss_abnormal_returns.drop_duplicates(subset=["Event_ID"])
    terminal_applied = terminal_events.get(
        "Terminal_Loss_Applied", pd.Series(False, index=terminal_events.index)
    ).fillna(False).astype(bool)
    terminal_loss_event_count = int(terminal_applied.sum())
    sample_size[f"Earnings events assigned a terminal -100% return{suffix}"] = terminal_loss_event_count
    sample_size[f"Stock return days assigned -100% under terminal-loss treatment{suffix}"] = (
        terminal_loss_event_count
    )
    return sample_size


def sync_sample_size_with_existing_abnormal_returns(year_context) -> None:
    sample_size = load_sample_size(year_context)
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        abnormal_returns_path = variant.abnormal_returns_path(year_context.year_dir)
        if not abnormal_returns_path.exists():
            continue

        earnings_events = load_earnings_events(year_context, variant=variant)
        sample_size[
            f"BHAR events selected before return-window checks{variant.sample_size_suffix}"
        ] = int(len(earnings_events))
        abnormal_returns = pd.read_csv(abnormal_returns_path)
        if abnormal_returns.empty or "Event_ID" not in abnormal_returns.columns:
            continue

        kept_event_count = int(abnormal_returns["Event_ID"].nunique())
        fill_summary = load_missing_return_fill_summary(year_context, variant=variant)
        complete_case_abnormal_returns = pd.read_csv(
            variant.abnormal_returns_drop_missing_path(year_context.year_dir)
        )
        terminal_loss_abnormal_returns = pd.read_csv(
            variant.abnormal_returns_terminal_loss_path(year_context.year_dir)
        )
        dropped_events = pd.DataFrame(
            index=range(max(len(earnings_events) - kept_event_count, 0))
        )
        sample_size = update_abnormal_returns_sample_size(
            sample_size=sample_size,
            kept_event_count=kept_event_count,
            dropped_events=dropped_events,
            fill_summary=fill_summary,
            sample_size_suffix=variant.sample_size_suffix,
            zero_fill_abnormal_returns=abnormal_returns,
            complete_case_abnormal_returns=complete_case_abnormal_returns,
            terminal_loss_abnormal_returns=terminal_loss_abnormal_returns,
        )
    save_sample_size(year_context, sample_size)


def has_complete_abnormal_returns_outputs(year_context) -> bool:
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        required_paths = (
            variant.abnormal_returns_path(year_context.year_dir),
            variant.abnormal_returns_drop_missing_path(year_context.year_dir),
            variant.abnormal_returns_terminal_loss_path(year_context.year_dir),
            variant.missing_return_fill_summary_path(year_context.year_dir),
            variant.abnormal_return_failures_path(year_context.year_dir),
        )
        if not all(path.exists() for path in required_paths):
            return False

        abnormal_returns = pd.read_csv(
            variant.abnormal_returns_path(year_context.year_dir)
        )
        if not can_reuse_existing_abnormal_returns(abnormal_returns):
            return False
        if not can_reuse_existing_alternative_abnormal_returns_outputs(
            year_context,
            variant=variant,
        ):
            return False

    return return_window_cache_has_expected_columns(year_context.return_windows_path)


def build_local_return_window_cache(
    return_history: pd.DataFrame,
    year_context,
) -> None:
    if return_window_cache_has_expected_columns(year_context.return_windows_path):
        print(f"Reusing existing return_windows.csv for {year_context.year}.")
        return
    return_window_cache = wide_history_to_long(
        history=return_history,
        value_column="TotalReturn",
    )
    save_window_cache(return_window_cache, year_context.return_windows_path)


def validate_return_treatment_outputs(
    zero_fill: pd.DataFrame,
    complete_case: pd.DataFrame,
    terminal_loss: pd.DataFrame,
) -> None:
    outputs = {
        ZERO_FILL_RETURN_TREATMENT: zero_fill,
        COMPLETE_CASE_RETURN_TREATMENT: complete_case,
        TERMINAL_LOSS_RETURN_TREATMENT: terminal_loss,
    }
    key_columns = ["Event_ID", "Relative_Day"]
    expected_keys: pd.MultiIndex | None = None
    for treatment, frame in outputs.items():
        missing_columns = set(key_columns).difference(frame.columns)
        if missing_columns:
            raise KeyError(
                f"{treatment} abnormal-return output is missing keys: "
                f"{sorted(missing_columns)}."
            )
        if frame.duplicated(subset=key_columns).any():
            raise ValueError(
                f"{treatment} abnormal-return output has duplicate event-day rows."
            )
        observed_treatments = set(frame["Return_Treatment"].dropna().astype(str).unique())
        if observed_treatments and observed_treatments != {treatment}:
            raise ValueError(
                f"{treatment} output contains unexpected treatment labels: "
                f"{sorted(observed_treatments)}."
            )
        keys = pd.MultiIndex.from_frame(frame.loc[:, key_columns])
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise ValueError(
                "Missing-return treatments produced different ordered event-day samples."
            )

    for treatment, frame in (
        (ZERO_FILL_RETURN_TREATMENT, zero_fill),
        (TERMINAL_LOSS_RETURN_TREATMENT, terminal_loss),
    ):
        post_mask = pd.to_numeric(frame["Relative_Day"], errors="coerce").ge(2)
        post_returns = pd.to_numeric(
            frame.loc[post_mask, "Security_Return"], errors="coerce"
        )
        if post_returns.isna().any():
            raise ValueError(
                f"{treatment} output still contains missing post-announcement returns."
            )

    complete_security = pd.to_numeric(complete_case["Security_Return"], errors="coerce")
    complete_raw = pd.to_numeric(complete_case["Raw_Security_Return"], errors="coerce")
    values_match = complete_security.eq(complete_raw) | (
        complete_security.isna() & complete_raw.isna()
    )
    if not values_match.all():
        raise ValueError("Complete-case treatment changed observed or missing stock returns.")

    pre_or_event_mask = pd.to_numeric(
        zero_fill["Relative_Day"], errors="coerce"
    ).le(1)
    if zero_fill.loc[pre_or_event_mask, "Security_Return_Was_Imputed"].fillna(False).astype(bool).any():
        raise ValueError("Zero-fill treatment imputed a return on or before relative day 1.")


def can_reuse_current_abnormal_returns_outputs(year_context) -> bool:
    if not has_complete_abnormal_returns_outputs(year_context):
        return False

    if not year_context.earnings_complete_path.exists():
        return False

    completion_state = load_completion_state(year_context.earnings_complete_path)
    return (
        completion_state.get("pipeline_version")
        == ABNORMAL_RETURNS_PIPELINE_VERSION
        and has_matching_bhar_restrictions(completion_state)
        and has_matching_sue_group_dependency(completion_state, year_context)
    )


def is_abnormal_returns_year_complete(year_context) -> bool:
    if not year_context.earnings_complete_path.exists():
        return False

    completion_state = load_completion_state(year_context.earnings_complete_path)
    if not has_pipeline_version(
        year_context.earnings_complete_path, ABNORMAL_RETURNS_PIPELINE_VERSION
    ):
        return False
    if not has_matching_bhar_restrictions(completion_state):
        return False
    if not has_matching_sue_group_dependency(completion_state, year_context):
        return False

    return has_complete_abnormal_returns_outputs(year_context)


def mark_abnormal_returns_year_complete(year_context) -> None:
    output_paths = [
        path
        for variant in PEAD_EVENT_SAMPLE_VARIANTS
        for path in (
            variant.abnormal_returns_path(year_context.year_dir),
            variant.abnormal_returns_drop_missing_path(year_context.year_dir),
            variant.abnormal_returns_terminal_loss_path(year_context.year_dir),
            variant.missing_return_fill_summary_path(year_context.year_dir),
            variant.abnormal_return_failures_path(year_context.year_dir),
        )
    ]
    output_paths.append(year_context.return_windows_path)
    write_stage_completion(
        path=year_context.earnings_complete_path,
        year=year_context.year,
        stage="earnings_abnormal_returns",
        pipeline_version=ABNORMAL_RETURNS_PIPELINE_VERSION,
        outputs=output_paths,
        extra_fields={
            "analysis_restrictions": current_bhar_restrictions(),
            "sue_group_dependency": load_saved_sue_group_dependency(year_context),
        },
    )


def build_abnormal_returns_year(year: int) -> None:
    year_context = build_year_context(year, DATA_DIR)
    year_context.year_dir.mkdir(parents=True, exist_ok=True)

    if is_abnormal_returns_year_complete(year_context):
        sync_sample_size_with_existing_abnormal_returns(year_context)
        print(
            f"\n=== Skipping formation year {year}: abnormal returns already complete ==="
        )
        return

    if can_reuse_current_abnormal_returns_outputs(year_context):
        sync_sample_size_with_existing_abnormal_returns(year_context)
        print(
            f"\n=== Reusing existing formation year {year}: abnormal returns already available ==="
        )
        mark_abnormal_returns_year_complete(year_context)
        return

    print(f"\n=== Building abnormal returns for formation year {year} ===")

    stock_universe, market_data, benchmark_returns = load_base_inputs(year_context)
    sample_size = load_sample_size(year_context)

    return_history = extract_total_return_history(market_data)
    build_local_return_window_cache(
        return_history=return_history,
        year_context=year_context,
    )
    sample_size["Return request batches"] = 0
    save_sample_size(year_context, sample_size)

    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        earnings_events = load_earnings_events(year_context, variant=variant)
        sample_size[
            f"BHAR events selected before return-window checks{variant.sample_size_suffix}"
        ] = int(len(earnings_events))
        save_sample_size(year_context, sample_size)

        failure_records: list[dict[str, str]] = []
        abnormal_returns = build_abnormal_returns_for_earnings_events(
            earnings_events=earnings_events,
            stock_universe=stock_universe,
            market_data=return_history,
            benchmark_returns=benchmark_returns,
            missing_return_treatment=ZERO_FILL_RETURN_TREATMENT,
            failure_records=failure_records,
        )
        abnormal_returns_drop_missing = build_abnormal_returns_for_earnings_events(
            earnings_events=earnings_events,
            stock_universe=stock_universe,
            market_data=return_history,
            benchmark_returns=benchmark_returns,
            missing_return_treatment=COMPLETE_CASE_RETURN_TREATMENT,
        )
        abnormal_returns_terminal_loss = build_abnormal_returns_for_earnings_events(
            earnings_events=earnings_events,
            stock_universe=stock_universe,
            market_data=return_history,
            benchmark_returns=benchmark_returns,
            missing_return_treatment=TERMINAL_LOSS_RETURN_TREATMENT,
        )
        validate_return_treatment_outputs(
            zero_fill=abnormal_returns,
            complete_case=abnormal_returns_drop_missing,
            terminal_loss=abnormal_returns_terminal_loss,
        )
        save_json(
            {
                "failed_event_count": len(failure_records),
                "failed_events": failure_records,
            },
            variant.abnormal_return_failures_path(year_context.year_dir),
        )

        fill_summary = build_missing_return_fill_summary(abnormal_returns)
        save_missing_return_fill_summary(year_context, fill_summary, variant=variant)
        kept_event_count = (
            0
            if abnormal_returns.empty or "Event_ID" not in abnormal_returns.columns
            else int(abnormal_returns["Event_ID"].nunique())
        )
        dropped_events = pd.DataFrame(
            index=range(max(len(earnings_events) - kept_event_count, 0))
        )
        sample_size = update_abnormal_returns_sample_size(
            sample_size=sample_size,
            kept_event_count=kept_event_count,
            dropped_events=dropped_events,
            fill_summary=fill_summary,
            sample_size_suffix=variant.sample_size_suffix,
            zero_fill_abnormal_returns=abnormal_returns,
            complete_case_abnormal_returns=abnormal_returns_drop_missing,
            terminal_loss_abnormal_returns=abnormal_returns_terminal_loss,
        )
        save_sample_size(year_context, sample_size)

        abnormal_returns.to_csv(
            variant.abnormal_returns_path(year_context.year_dir),
            index=False,
        )
        abnormal_returns_drop_missing.to_csv(
            variant.abnormal_returns_drop_missing_path(year_context.year_dir),
            index=False,
        )
        abnormal_returns_terminal_loss.to_csv(
            variant.abnormal_returns_terminal_loss_path(year_context.year_dir),
            index=False,
        )
    mark_abnormal_returns_year_complete(year_context)

    main_abnormal_returns = pd.read_csv(MAIN_PEAD_SAMPLE.abnormal_returns_path(year_context.year_dir))
    print(f"Abnormal return rows for {year}: {len(main_abnormal_returns)}")


def main() -> None:
    validate_bhar_subset_restrictions_for_all_years()

    for year in configured_formation_years():
        build_abnormal_returns_year(year)

    rebuild_sample_size_all_years()


if __name__ == "__main__":
    main()
