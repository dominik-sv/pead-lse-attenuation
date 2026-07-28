from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import patsy
from pandas.api.types import (
    is_bool_dtype,
    is_extension_array_dtype,
    is_numeric_dtype,
)
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ..core.pipeline_config import (
    MAIN_REGRESSION_BHAR_WINDOW,
    MARKET_CAP_ANALYSIS_SPLIT_PERCENTILE,
    UNBIASEDNESS_ANNOUNCEMENT_WINDOW,
    UNBIASEDNESS_FULL_WINDOW,
    SUE_COMPUTATION_GROUP_COUNT,
    SUE_PLOT_GROUP_COUNT,
    COLOR_PALETTE
)
from ..pead.market_cap_splits import (
    MARKET_CAP_DECILE_BREAKPOINT_PERCENTILES,
    MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN,
    MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN,
    MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN,
    MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN,
    validate_market_cap_size_split_percentile,
)
from ..pead.sue_groups import (
    SUE_GROUP_COLUMN,
    SUE_PLOT_GROUP_COLUMN,
    add_plot_group_column,
    normalize_sue_group_columns,
)
from ..core.project_paths import resolve_yearly_data_dir
from .bhar_outlier_policy import (
    apply_outlier_policy,
    load_outlier_policy,
    resolve_outlier_policy_path,
)
from ..utils.io_utils import load_json


PEAD_COLUMN = f"BHAR_{int(MAIN_REGRESSION_BHAR_WINDOW[0])}_{int(MAIN_REGRESSION_BHAR_WINDOW[1])}"
ANNOUNCEMENT_WINDOW_COLUMN = (
    f"BHAR_{int(UNBIASEDNESS_ANNOUNCEMENT_WINDOW[0])}_{int(UNBIASEDNESS_ANNOUNCEMENT_WINDOW[1])}"
)
FULL_WINDOW_COLUMN = f"BHAR_{int(UNBIASEDNESS_FULL_WINDOW[0])}_{int(UNBIASEDNESS_FULL_WINDOW[1])}"
FORMATION_YEAR_COLUMN = "Formation_Year"
TIME_PERIOD_COLUMN = "Time_Period"
TIME_PERIOD_START_COLUMN = "Time_Period_Start"
TIME_PERIOD_END_COLUMN = "Time_Period_End"
ANNOUNCEMENT_YEAR_COLUMN = "Announcement_Calendar_Year"
ANNOUNCEMENT_QUARTER_COLUMN = "Announcement_Calendar_Quarter"
FIRM_IDENTIFIER_COLUMN = "Instrument"
ANALYSIS_MARKET_CAP_SPLIT_PERCENTILE_COLUMN = "Market_Cap_Analysis_Split_Percentile"
ANALYSIS_MARKET_CAP_SPLIT_BREAKPOINT_COLUMN = "Market_Cap_Analysis_Split_Breakpoint"
ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN = "Market_Cap_Analysis_Split_Group"

CLUSTER_LABELS = {
    "none": "No clustering",
    "heteroskedasticity_robust": "Heteroskedasticity-robust (HC1)",
    "firm": "Clustered by firm",
    "quarter": "Clustered by quarter",
    "firm_quarter": "Clustered by firm and quarter",
}
ABSORBED_FIXED_EFFECT_TERMS = (
    f"C({ANNOUNCEMENT_QUARTER_COLUMN})",
    f"C({FIRM_IDENTIFIER_COLUMN})",
)

ABNORMAL_RETURN_NUMERIC_COLUMNS = [
    "SUE",
    PEAD_COLUMN,
    ANNOUNCEMENT_WINDOW_COLUMN,
    FULL_WINDOW_COLUMN,
    SUE_GROUP_COLUMN,
    "Relative_Day",
    "Abnormal_Return",
    "Security_Return",
    "Benchmark_Return",
    "Forecast_Analyst_Count",
    "Price_Lag_5",
]

UNIVERSE_NUMERIC_COLUMNS = [
    "Price",
    "Market_Cap_Current",
    "Market_Cap_Last_Fiscal_Year_End",
    "BM_French",
    "BM",
    "Size_Q",
    "BM_Q",
    MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN,
    MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN,
    MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN,
]

UNIVERSE_SNAPSHOT_COLUMNS = [
    FORMATION_YEAR_COLUMN,
    FIRM_IDENTIFIER_COLUMN,
    "Price",
    "Market_Cap_Current",
    "Formation_Date",
    "Market_Cap_Last_Fiscal_Year_End",
    "BM_French",
    "BM",
    "Size_Q",
    "BM_Q",
    MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN,
    MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN,
    MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN,
    MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN,
]

DEFAULT_ANALYSIS_SAMPLE_PRESET = "default"
EXTENDED_ANALYSIS_SAMPLE_PRESET = "extended"
DIAGNOSTIC_ANALYSIS_SAMPLE_PRESET = "diagnostic"
DEFAULT_ANALYSIS_REPORT_FREQUENCIES = ("FY",)
ANALYSIS_SAMPLE_PRESETS = {
    DEFAULT_ANALYSIS_SAMPLE_PRESET,
    EXTENDED_ANALYSIS_SAMPLE_PRESET,
    DIAGNOSTIC_ANALYSIS_SAMPLE_PRESET,
}


@dataclass(frozen=True)
class AnalysisSample:
    preset: str = DEFAULT_ANALYSIS_SAMPLE_PRESET
    report_frequencies: tuple[str, ...] | None = None
    min_analyst_forecasts: int | None = None
    formation_years: tuple[int, ...] | None = None
    announcement_years: tuple[int, ...] | None = None


def _normalize_sample_values(
    values: Iterable[int | str] | None,
    *,
    label: str,
) -> tuple[int, ...] | None:
    if values is None:
        return None

    normalized: list[int] = []
    for value in values:
        if pd.isna(value):
            raise ValueError(f"{label} cannot contain missing values.")
        normalized.append(int(value))

    return tuple(sorted(dict.fromkeys(normalized)))


def _normalize_frequency_values(
    values: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None

    normalized: list[str] = []
    for value in values:
        if pd.isna(value):
            raise ValueError("report_frequencies cannot contain missing values.")
        frequency = str(value).strip().upper()
        if not frequency:
            raise ValueError("report_frequencies cannot contain blank values.")
        normalized.append(frequency)

    return tuple(dict.fromkeys(normalized))


def build_analysis_sample(
    sample: AnalysisSample | str | None = None,
    *,
    preset: str | None = None,
    report_frequencies: Iterable[str] | None = None,
    min_analyst_forecasts: int | None = None,
    formation_years: Iterable[int | str] | None = None,
    announcement_years: Iterable[int | str] | None = None,
) -> AnalysisSample:
    if isinstance(sample, AnalysisSample):
        if any(
            value is not None
            for value in (
                preset,
                report_frequencies,
                min_analyst_forecasts,
                formation_years,
                announcement_years,
            )
        ):
            raise ValueError(
                "Cannot combine an AnalysisSample instance with additional sample overrides."
            )
        normalized_sample = sample
    else:
        resolved_preset = preset if preset is not None else sample
        normalized_sample = AnalysisSample(
            preset=str(resolved_preset or DEFAULT_ANALYSIS_SAMPLE_PRESET).strip().lower(),
            report_frequencies=_normalize_frequency_values(report_frequencies),
            min_analyst_forecasts=(
                None if min_analyst_forecasts is None else int(min_analyst_forecasts)
            ),
            formation_years=_normalize_sample_values(
                formation_years,
                label="formation_years",
            ),
            announcement_years=_normalize_sample_values(
                announcement_years,
                label="announcement_years",
            ),
        )

    if normalized_sample.preset not in ANALYSIS_SAMPLE_PRESETS:
        raise ValueError(
            "Unsupported analysis sample preset "
            f"{normalized_sample.preset!r}. "
            f"Choose from {sorted(ANALYSIS_SAMPLE_PRESETS)}."
        )

    if (
        normalized_sample.min_analyst_forecasts is not None
        and normalized_sample.min_analyst_forecasts < 0
    ):
        raise ValueError("min_analyst_forecasts must be non-negative.")

    return normalized_sample


def _require_sample_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    reason: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Cannot apply analysis sample restriction for {reason}. "
            f"Missing columns: {sorted(missing)}."
        )


def _resolve_report_frequencies(sample: AnalysisSample) -> tuple[str, ...] | None:
    if sample.report_frequencies is not None:
        return sample.report_frequencies
    if sample.preset == DEFAULT_ANALYSIS_SAMPLE_PRESET:
        return DEFAULT_ANALYSIS_REPORT_FREQUENCIES
    if sample.preset in {
        EXTENDED_ANALYSIS_SAMPLE_PRESET,
        DIAGNOSTIC_ANALYSIS_SAMPLE_PRESET,
    }:
        return None
    raise ValueError(f"Unsupported analysis sample preset {sample.preset!r}.")


def apply_analysis_sample_filters(
    frame: pd.DataFrame,
    sample: AnalysisSample | str | None = None,
    *,
    preset: str | None = None,
    report_frequencies: Iterable[str] | None = None,
    min_analyst_forecasts: int | None = None,
    formation_years: Iterable[int | str] | None = None,
    announcement_years: Iterable[int | str] | None = None,
) -> pd.DataFrame:
    resolved_sample = build_analysis_sample(
        sample,
        preset=preset,
        report_frequencies=report_frequencies,
        min_analyst_forecasts=min_analyst_forecasts,
        formation_years=formation_years,
        announcement_years=announcement_years,
    )

    filtered = frame.copy()

    selected_frequencies = _resolve_report_frequencies(resolved_sample)
    if selected_frequencies is not None:
        _require_sample_columns(
            filtered,
            ["Report_Frequency"],
            reason="report frequency filter",
        )
        normalized_frequencies = (
            filtered["Report_Frequency"].astype("string").str.strip().str.upper()
        )
        filtered = filtered.loc[normalized_frequencies.isin(selected_frequencies)].copy()

    if resolved_sample.min_analyst_forecasts is not None:
        _require_sample_columns(
            filtered,
            ["Forecast_Analyst_Count"],
            reason="minimum analyst coverage filter",
        )
        analyst_counts = pd.to_numeric(
            filtered["Forecast_Analyst_Count"], errors="coerce"
        )
        filtered = filtered.loc[
            analyst_counts >= resolved_sample.min_analyst_forecasts
        ].copy()

    if resolved_sample.formation_years is not None:
        _require_sample_columns(
            filtered,
            [FORMATION_YEAR_COLUMN],
            reason="formation-year filter",
        )
        formation_year_values = pd.to_numeric(
            filtered[FORMATION_YEAR_COLUMN], errors="coerce"
        )
        filtered = filtered.loc[
            formation_year_values.isin(resolved_sample.formation_years)
        ].copy()

    if resolved_sample.announcement_years is not None:
        announcement_year_column = ANNOUNCEMENT_YEAR_COLUMN
        if announcement_year_column not in filtered.columns:
            _require_sample_columns(
                filtered,
                ["Ann_Date"],
                reason="announcement-year filter",
            )
            ann_dates = pd.to_datetime(filtered["Ann_Date"], errors="coerce")
            filtered[announcement_year_column] = ann_dates.dt.year.astype("Int64")

        announcement_year_values = pd.to_numeric(
            filtered[announcement_year_column], errors="coerce"
        )
        filtered = filtered.loc[
            announcement_year_values.isin(resolved_sample.announcement_years)
        ].copy()

    return filtered


def market_cap_analysis_breakpoint_key(percentile: float) -> str:
    normalized_percentile = validate_market_cap_size_split_percentile(percentile)
    return f"{int(round(normalized_percentile * 100))}th percentile"


def format_market_cap_analysis_group_labels(split_percentile: float) -> list[str]:
    normalized_percentile = validate_market_cap_size_split_percentile(split_percentile)
    bottom_percent = int(round(normalized_percentile * 100))
    top_percent = 100 - bottom_percent
    return [
        f"Top {top_percent}% by market cap",
        f"Bottom {bottom_percent}% by market cap",
    ]


def load_market_cap_analysis_breakpoints(
    data_dir: Path,
    split_percentile: float = MARKET_CAP_ANALYSIS_SPLIT_PERCENTILE,
) -> pd.DataFrame:
    normalized_percentile = validate_market_cap_size_split_percentile(split_percentile)
    aggregate_path = data_dir / "sample_size_all_years.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Missing {aggregate_path}. Run scripts/02_build_universe_and_market_data.py first."
        )

    aggregate = load_json(aggregate_path, default={})
    breakpoint_key = market_cap_analysis_breakpoint_key(normalized_percentile)
    recorded_percentiles = {
        round(float(percentile), 10) for percentile in MARKET_CAP_DECILE_BREAKPOINT_PERCENTILES
    }

    rows = []
    for year_label, sample_size in sorted(aggregate.items()):
        formation_year = int(year_label)
        saved_percentile = sample_size.get("Market cap size split percentile")
        saved_breakpoint = sample_size.get("Market cap size split breakpoint")

        if (
            saved_percentile is not None
            and abs(float(saved_percentile) - normalized_percentile) <= 1e-12
            and saved_breakpoint is not None
        ):
            breakpoint_value = float(saved_breakpoint)
        else:
            decile_breakpoints = sample_size.get("Market cap decile breakpoints")
            if not isinstance(decile_breakpoints, dict):
                raise KeyError(
                    "Market-cap decile breakpoints are missing from "
                    f"{aggregate_path}. Rerun scripts/02_build_universe_and_market_data.py "
                    "to populate the saved yearly breakpoint ladder."
                )
            if round(normalized_percentile, 10) not in recorded_percentiles:
                raise ValueError(
                    "The section-10 market-cap analysis split percentile must match one of "
                    "the recorded deciles: 0.10, 0.20, ..., 0.90."
                )
            if breakpoint_key not in decile_breakpoints:
                raise KeyError(
                    f"Missing saved breakpoint {breakpoint_key!r} for formation year "
                    f"{formation_year}. Rerun scripts/02_build_universe_and_market_data.py "
                    "to repopulate the yearly breakpoint metadata."
                )
            breakpoint_value = float(decile_breakpoints[breakpoint_key])

        rows.append(
            {
                FORMATION_YEAR_COLUMN: formation_year,
                ANALYSIS_MARKET_CAP_SPLIT_PERCENTILE_COLUMN: normalized_percentile,
                ANALYSIS_MARKET_CAP_SPLIT_BREAKPOINT_COLUMN: breakpoint_value,
            }
        )

    return pd.DataFrame(rows)


def apply_market_cap_analysis_split(
    frame: pd.DataFrame,
    data_dir: Path,
    split_percentile: float = MARKET_CAP_ANALYSIS_SPLIT_PERCENTILE,
    market_cap_column: str = "Market_Cap_Current",
) -> pd.DataFrame:
    if FORMATION_YEAR_COLUMN not in frame.columns:
        raise KeyError(
            f"Cannot assign the market-cap analysis split without {FORMATION_YEAR_COLUMN!r}."
        )
    if market_cap_column not in frame.columns:
        raise KeyError(
            f"Cannot assign the market-cap analysis split without {market_cap_column!r}."
        )

    breakpoints = load_market_cap_analysis_breakpoints(
        data_dir=data_dir,
        split_percentile=split_percentile,
    )
    plot_group_order = format_market_cap_analysis_group_labels(split_percentile)

    out = frame.copy()
    out = out.merge(
        breakpoints,
        on=FORMATION_YEAR_COLUMN,
        how="left",
        validate="m:1",
    )

    market_caps = pd.to_numeric(out[market_cap_column], errors="coerce")
    breakpoint_values = pd.to_numeric(
        out[ANALYSIS_MARKET_CAP_SPLIT_BREAKPOINT_COLUMN], errors="coerce"
    )
    valid = market_caps.notna() & breakpoint_values.notna()
    is_bottom_group = valid & market_caps.le(breakpoint_values)

    group_values = pd.Series(pd.NA, index=out.index, dtype="object")
    group_values.loc[valid & ~is_bottom_group] = plot_group_order[0]
    group_values.loc[is_bottom_group] = plot_group_order[1]

    out[ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN] = pd.Categorical(
        group_values,
        categories=plot_group_order,
        ordered=True,
    )
    return out


def _load_yearly_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    yearly_files = sorted(
        resolve_yearly_data_dir(data_dir).glob(f"[0-9][0-9][0-9][0-9]/{filename}")
    )

    if not yearly_files:
        raise FileNotFoundError(
            f"No yearly {filename} files found under data/<year>/. "
            "Run the upstream data-building scripts first."
        )

    yearly_frames = []
    for path in yearly_files:
        formation_year = int(path.parent.name)
        frame = pd.read_csv(path)
        frame[FORMATION_YEAR_COLUMN] = formation_year
        yearly_frames.append(frame)

    return pd.concat(yearly_frames, ignore_index=True)


def load_abnormal_returns_with_groups(
    data_dir: Path,
    sample: AnalysisSample | str | None = None,
    apply_saved_outlier_policy: bool = True,
    outlier_policy_path: Path | None = None,
    abnormal_returns_filename: str = "earnings_abnormal_returns.csv",
    **sample_overrides,
) -> pd.DataFrame:
    abnormal_returns = normalize_sue_group_columns(
        _load_yearly_csv(data_dir, abnormal_returns_filename)
    )

    if SUE_GROUP_COLUMN not in abnormal_returns.columns:
        raise KeyError(
            "Saved abnormal-return files are missing SUE group assignments. "
            "Run scripts/04_build_earnings_and_sue.py, scripts/05_build_sue_groups.py, "
            "and then scripts/06_build_abnormal_returns.py first."
        )

    for column in ABNORMAL_RETURN_NUMERIC_COLUMNS:
        if column in abnormal_returns.columns:
            abnormal_returns[column] = pd.to_numeric(abnormal_returns[column], errors="coerce")

    abnormal_returns = apply_analysis_sample_filters(
        abnormal_returns,
        sample=sample,
        **sample_overrides,
    )
    if apply_saved_outlier_policy:
        resolved_policy_path = (
            Path(outlier_policy_path)
            if outlier_policy_path is not None
            else resolve_outlier_policy_path(data_dir)
        )
        policy = load_outlier_policy(resolved_policy_path)
        abnormal_returns = apply_outlier_policy(abnormal_returns, policy=policy)
    return abnormal_returns


def load_event_level_sample(
    data_dir: Path,
    sample: AnalysisSample | str | None = None,
    apply_saved_outlier_policy: bool = True,
    outlier_policy_path: Path | None = None,
    **sample_overrides,
) -> pd.DataFrame:
    abnormal_returns = load_abnormal_returns_with_groups(
        data_dir,
        sample=sample,
        apply_saved_outlier_policy=apply_saved_outlier_policy,
        outlier_policy_path=outlier_policy_path,
        **sample_overrides,
    )
    return collapse_to_event_level(abnormal_returns)


def load_stock_universe_snapshots(data_dir: Path) -> pd.DataFrame:
    stock_universe = _load_yearly_csv(data_dir, "stock_universe.csv")

    for column in UNIVERSE_NUMERIC_COLUMNS:
        if column in stock_universe.columns:
            stock_universe[column] = pd.to_numeric(stock_universe[column], errors="coerce")

    if "Announcement_Date" in stock_universe.columns:
        stock_universe["Announcement_Date"] = pd.to_datetime(
            stock_universe["Announcement_Date"], errors="coerce"
        )

    if "BM" not in stock_universe.columns and "BM_French" in stock_universe.columns:
        stock_universe["BM"] = pd.to_numeric(
            stock_universe["BM_French"], errors="coerce"
        )

    return stock_universe


def ensure_event_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if "Event_ID" in frame.columns:
        result = frame.copy()
        result["Event_ID"] = result["Event_ID"].astype(str)
        return result

    id_columns = [FIRM_IDENTIFIER_COLUMN, "Ann_Date"]
    if "Report_Frequency" in frame.columns:
        id_columns.append("Report_Frequency")

    missing_id_columns = [column for column in id_columns if column not in frame.columns]
    if missing_id_columns:
        raise KeyError(
            f"Cannot build event identifiers. Missing columns: {sorted(missing_id_columns)}."
        )

    result = frame.copy()
    result["Event_ID"] = result[id_columns].astype(str).agg("|".join, axis=1)
    return result


def _ensure_bhar_window_column(
    frame: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    if column_name in frame.columns:
        return frame

    match = re.fullmatch(r"BHAR_(\d+)_(\d+)", column_name)
    if match is None:
        return frame

    required_columns = {"Relative_Day", "Security_Return", "Benchmark_Return"}
    if not required_columns.issubset(frame.columns):
        return frame

    day_start = int(match.group(1))
    day_end = int(match.group(2))
    expected_observations = day_end - day_start + 1
    if expected_observations <= 0:
        return frame

    event_group_columns = [
        column for column in [FORMATION_YEAR_COLUMN, "Event_ID"] if column in frame.columns
    ]
    if not event_group_columns:
        return frame

    path_frame = frame.loc[frame["Relative_Day"].between(day_start, day_end)].copy()
    if path_frame.empty:
        return frame

    path_frame["Security_Return"] = pd.to_numeric(path_frame["Security_Return"], errors="coerce")
    path_frame["Benchmark_Return"] = pd.to_numeric(
        path_frame["Benchmark_Return"], errors="coerce"
    )
    path_frame = path_frame.dropna(subset=["Security_Return", "Benchmark_Return"]).copy()
    if path_frame.empty:
        return frame

    path_frame = path_frame.sort_values(event_group_columns + ["Relative_Day"])
    path_frame["Security_Gross"] = 1.0 + path_frame["Security_Return"] / 100.0
    path_frame["Benchmark_Gross"] = 1.0 + path_frame["Benchmark_Return"] / 100.0
    path_frame["Security_Cumulative_Gross"] = (
        path_frame.groupby(event_group_columns)["Security_Gross"].cumprod()
    )
    path_frame["Benchmark_Cumulative_Gross"] = (
        path_frame.groupby(event_group_columns)["Benchmark_Gross"].cumprod()
    )
    path_frame[column_name] = (
        path_frame["Security_Cumulative_Gross"] - path_frame["Benchmark_Cumulative_Gross"]
    ) * 100.0

    window_values = (
        path_frame.groupby(event_group_columns)
        .agg(
            _observation_count=("Relative_Day", "nunique"),
            _window_bhar=(column_name, "last"),
        )
        .reset_index()
    )
    window_values[column_name] = np.where(
        window_values["_observation_count"] == expected_observations,
        window_values["_window_bhar"],
        np.nan,
    )
    window_values = window_values[event_group_columns + [column_name]]

    return frame.merge(window_values, on=event_group_columns, how="left")


def collapse_to_event_level(
    abnormal_returns: pd.DataFrame,
    *,
    additional_bhar_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    event_level = normalize_sue_group_columns(ensure_event_ids(abnormal_returns))
    event_level = _ensure_bhar_window_column(event_level, ANNOUNCEMENT_WINDOW_COLUMN)
    event_level = _ensure_bhar_window_column(event_level, FULL_WINDOW_COLUMN)
    for column_name in additional_bhar_columns:
        event_level = _ensure_bhar_window_column(event_level, column_name)

    sort_columns = [
        column
        for column in [FORMATION_YEAR_COLUMN, "Event_ID", "Relative_Day"]
        if column in event_level.columns
    ]
    if sort_columns:
        event_level = event_level.sort_values(sort_columns)

    dedupe_columns = [
        column for column in [FORMATION_YEAR_COLUMN, "Event_ID"] if column in event_level.columns
    ]
    if not dedupe_columns:
        dedupe_columns = ["Event_ID"]

    event_level = event_level.drop_duplicates(subset=dedupe_columns, keep="first").copy()

    if "Ann_Date" in event_level.columns:
        event_level["Ann_Date"] = pd.to_datetime(event_level["Ann_Date"], errors="coerce")
        event_level[ANNOUNCEMENT_YEAR_COLUMN] = event_level["Ann_Date"].dt.year.astype("Int64")
        event_level[ANNOUNCEMENT_QUARTER_COLUMN] = (
            event_level["Ann_Date"].dt.to_period("Q").astype(str)
        )

    if FIRM_IDENTIFIER_COLUMN in event_level.columns:
        event_level[FIRM_IDENTIFIER_COLUMN] = event_level[FIRM_IDENTIFIER_COLUMN].astype(str)

    numeric_columns = [
        "SUE",
        PEAD_COLUMN,
        ANNOUNCEMENT_WINDOW_COLUMN,
        FULL_WINDOW_COLUMN,
        SUE_GROUP_COLUMN,
        "Forecast_Analyst_Count",
        "Price_Lag_5",
    ]
    for column in numeric_columns:
        if column in event_level.columns:
            event_level[column] = pd.to_numeric(event_level[column], errors="coerce")

    if SUE_GROUP_COLUMN in event_level.columns:
        event_level[SUE_GROUP_COLUMN] = event_level[SUE_GROUP_COLUMN].astype("Int64")

    event_level["PEAD"] = pd.to_numeric(event_level[PEAD_COLUMN], errors="coerce")
    return event_level


def attach_universe_snapshot(
    event_level: pd.DataFrame, stock_universe: pd.DataFrame
) -> pd.DataFrame:
    merge_columns = [FORMATION_YEAR_COLUMN, FIRM_IDENTIFIER_COLUMN]
    missing_merge_columns = [column for column in merge_columns if column not in stock_universe.columns]
    if missing_merge_columns:
        raise KeyError(
            "Stock-universe snapshot is missing merge columns: "
            f"{sorted(missing_merge_columns)}."
        )

    snapshot_columns = [
        column
        for column in UNIVERSE_SNAPSHOT_COLUMNS
        if column in stock_universe.columns
    ]
    snapshot = stock_universe[snapshot_columns].drop_duplicates(merge_columns).copy()

    merged = event_level.merge(
        snapshot,
        on=merge_columns,
        how="left",
        validate="m:1",
    )

    if "BM" not in merged.columns and "BM_French" in merged.columns:
        merged["BM"] = pd.to_numeric(merged["BM_French"], errors="coerce")

    return merged


def build_time_periods(
    available_years: Iterable[int],
    period_length: int = 10,
    explicit_periods: list[tuple[int, int]] | None = None,
) -> list[dict[str, int | str]]:
    years = sorted(int(year) for year in available_years)
    if not years:
        raise ValueError("No available years were provided.")

    periods: list[dict[str, int | str]] = []

    if explicit_periods is not None:
        for start_year, end_year in explicit_periods:
            if end_year < start_year:
                raise ValueError(f"Invalid time period ({start_year}, {end_year}).")
            periods.append(
                {
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "label": f"{int(start_year)}-{int(end_year)}",
                }
            )
        return periods

    start_year = years[0]
    end_year = years[-1]
    current_start = start_year

    while current_start <= end_year:
        current_end = min(current_start + period_length - 1, end_year)
        periods.append(
            {
                "start_year": current_start,
                "end_year": current_end,
                "label": f"{current_start}-{current_end}",
            }
        )
        current_start += period_length

    return periods


def assign_time_periods(
    frame: pd.DataFrame,
    periods: list[dict[str, int | str]],
    year_column: str = FORMATION_YEAR_COLUMN,
    label_column: str = TIME_PERIOD_COLUMN,
) -> pd.DataFrame:
    result = frame.copy()
    years = pd.to_numeric(result[year_column], errors="coerce")

    result[TIME_PERIOD_START_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result[TIME_PERIOD_END_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result[label_column] = pd.Series(pd.NA, index=result.index, dtype="object")

    for period in periods:
        start_year = int(period["start_year"])
        end_year = int(period["end_year"])
        label = str(period["label"])
        mask = years.between(start_year, end_year)
        result.loc[mask, TIME_PERIOD_START_COLUMN] = start_year
        result.loc[mask, TIME_PERIOD_END_COLUMN] = end_year
        result.loc[mask, label_column] = label

    result[label_column] = pd.Categorical(
        result[label_column],
        categories=[str(period["label"]) for period in periods],
        ordered=True,
    )

    return result


def percentile_limits(
    values: Iterable[float] | pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[float, float]:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return (np.nan, np.nan)

    lower_bound = float(series.quantile(lower))
    upper_bound = float(series.quantile(upper))

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        return (np.nan, np.nan)

    if lower_bound == upper_bound:
        padding = max(abs(lower_bound) * 0.05, 1e-9)
        return (lower_bound - padding, upper_bound + padding)

    return (lower_bound, upper_bound)


def build_period_axes(
    period_count: int,
    ncols: int = 2,
    subplot_width: float = 7.0,
    subplot_height: float = 4.5,
) -> tuple[object, list[object]]:
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(period_count / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7, 3.5),
        squeeze=False,
    )

    flat_axes = axes.ravel().tolist()
    for axis in flat_axes[period_count:]:
        axis.set_visible(False)

    return fig, flat_axes[:period_count]


def prepare_bhar_path_events(
    abnormal_returns: pd.DataFrame,
    day_start: int,
    day_end: int,
) -> pd.DataFrame:
    path_events = normalize_sue_group_columns(ensure_event_ids(abnormal_returns))
    path_events = path_events.dropna(
        subset=[SUE_GROUP_COLUMN, "Relative_Day", "Security_Return", "Benchmark_Return"]
    ).copy()
    path_events = path_events.loc[path_events["Relative_Day"].between(day_start, day_end)].copy()

    path_events[SUE_GROUP_COLUMN] = pd.to_numeric(
        path_events[SUE_GROUP_COLUMN], errors="coerce"
    ).astype(int)
    path_events = add_plot_group_column(
        path_events,
        group_column=SUE_GROUP_COLUMN,
        plot_group_column=SUE_PLOT_GROUP_COLUMN,
        computation_group_count=SUE_COMPUTATION_GROUP_COUNT,
        plot_group_count=SUE_PLOT_GROUP_COUNT,
    )
    path_events[SUE_PLOT_GROUP_COLUMN] = path_events[SUE_PLOT_GROUP_COLUMN].astype(int)

    event_group_columns = [
        column for column in [FORMATION_YEAR_COLUMN, "Event_ID"] if column in path_events.columns
    ]
    if len(event_group_columns) < 2:
        raise KeyError("Need Formation_Year and Event_ID columns to build BHAR paths.")

    path_events = path_events.sort_values(event_group_columns + ["Relative_Day"])
    path_events["Security_Gross"] = 1.0 + path_events["Security_Return"] / 100.0
    path_events["Benchmark_Gross"] = 1.0 + path_events["Benchmark_Return"] / 100.0
    path_events["Security_Cumulative_Gross"] = (
        path_events.groupby(event_group_columns)["Security_Gross"].cumprod()
    )
    path_events["Benchmark_Cumulative_Gross"] = (
        path_events.groupby(event_group_columns)["Benchmark_Gross"].cumprod()
    )
    path_events["Cumulative_BHAR"] = (
        path_events["Security_Cumulative_Gross"]
        - path_events["Benchmark_Cumulative_Gross"]
    ) * 100.0

    return path_events


def summarize_plot_group_paths(
    path_events: pd.DataFrame,
    group_column: str = TIME_PERIOD_COLUMN,
) -> pd.DataFrame:
    required_columns = [group_column, SUE_PLOT_GROUP_COLUMN, "Relative_Day", "Cumulative_BHAR"]
    missing_columns = [column for column in required_columns if column not in path_events.columns]
    if missing_columns:
        raise KeyError(f"Path summary is missing columns: {sorted(missing_columns)}.")

    return (
        path_events.groupby([group_column, SUE_PLOT_GROUP_COLUMN, "Relative_Day"])[
            "Cumulative_BHAR"
        ]
        .mean()
        .reset_index()
    )


def prepare_regression_frame(
    event_level: pd.DataFrame,
    regressor_column: str = SUE_GROUP_COLUMN,
) -> pd.DataFrame:
    regression_df = event_level[
        [
            FIRM_IDENTIFIER_COLUMN,
            ANNOUNCEMENT_YEAR_COLUMN,
            ANNOUNCEMENT_QUARTER_COLUMN,
            regressor_column,
            "PEAD",
        ]
    ].dropna().copy()
    regression_df[ANNOUNCEMENT_YEAR_COLUMN] = regression_df[ANNOUNCEMENT_YEAR_COLUMN].astype(int)
    regression_df[ANNOUNCEMENT_QUARTER_COLUMN] = regression_df[
        ANNOUNCEMENT_QUARTER_COLUMN
    ].astype(str)
    regression_df[FIRM_IDENTIFIER_COLUMN] = regression_df[FIRM_IDENTIFIER_COLUMN].astype(str)
    return regression_df


def build_cluster_groups(regression_df: pd.DataFrame, cluster_spec: str):
    if cluster_spec in {"none", "heteroskedasticity_robust"}:
        return None
    if cluster_spec == "firm":
        return pd.Categorical(regression_df[FIRM_IDENTIFIER_COLUMN]).codes.astype(int)
    if cluster_spec == "quarter":
        return pd.Categorical(regression_df[ANNOUNCEMENT_QUARTER_COLUMN]).codes.astype(int)
    if cluster_spec == "firm_quarter":
        firm_codes = pd.Categorical(regression_df[FIRM_IDENTIFIER_COLUMN]).codes.astype(int)
        quarter_codes = pd.Categorical(regression_df[ANNOUNCEMENT_QUARTER_COLUMN]).codes.astype(
            int
        )
        return np.column_stack([firm_codes, quarter_codes])
    raise ValueError(f"Unsupported cluster spec: {cluster_spec}")


def build_cluster_count_diagnostics(
    regression_df: pd.DataFrame,
    cluster_spec: str,
) -> dict[str, int | None]:
    """Return counts for the clustering dimensions used in a fitted sample."""
    return {
        "Firm_Cluster_Count": (
            int(regression_df[FIRM_IDENTIFIER_COLUMN].nunique())
            if cluster_spec in {"firm", "firm_quarter"}
            else None
        ),
        "Quarter_Cluster_Count": (
            int(regression_df[ANNOUNCEMENT_QUARTER_COLUMN].nunique())
            if cluster_spec in {"quarter", "firm_quarter"}
            else None
        ),
    }


def _referenced_formula_columns(
    formula: str,
    available_columns: Sequence[str],
) -> list[str]:
    referenced_columns: list[str] = []
    for column in available_columns:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        if re.search(pattern, formula):
            referenced_columns.append(column)
    return referenced_columns


def _split_top_level_terms(expression: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    depth = 0

    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1

        if character == "+" and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
            continue

        current.append(character)

    trailing = "".join(current).strip()
    if trailing:
        terms.append(trailing)
    return terms


def _build_absorbed_formula_components(
    formula: str,
    absorb_terms: tuple[str, ...] = ABSORBED_FIXED_EFFECT_TERMS,
) -> tuple[str, list[str]]:
    if "~" not in formula:
        raise ValueError(f"Invalid formula {formula!r}.")

    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    rhs_terms = _split_top_level_terms(rhs)

    retained_terms = [term for term in rhs_terms if term not in absorb_terms]
    absorbed_terms = [term for term in rhs_terms if term in absorb_terms]

    reduced_rhs = " + ".join(retained_terms) if retained_terms else "1"
    return f"{lhs} ~ {reduced_rhs}", absorbed_terms


def _formula_has_absorbed_fixed_effects(
    formula: str,
    absorb_terms: tuple[str, ...] = ABSORBED_FIXED_EFFECT_TERMS,
) -> bool:
    _, absorbed_terms = _build_absorbed_formula_components(formula, absorb_terms=absorb_terms)
    return bool(absorbed_terms)


def _first_scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return np.nan
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return np.nan
    return scalar if np.isfinite(scalar) else np.nan


class _AbsorbedResultAdapter:
    def __init__(self, result, exog_columns: list[str], fittedvalues, residuals):
        self._result = result
        parameter_names = list(exog_columns)
        params = pd.Series(
            np.asarray(result.params).reshape(-1),
            index=parameter_names,
            dtype=float,
        )
        covariance = pd.DataFrame(
            np.asarray(result.cov),
            index=parameter_names,
            columns=parameter_names,
            dtype=float,
        )
        std_errors = pd.Series(
            np.asarray(result.std_errors).reshape(-1),
            index=parameter_names,
            dtype=float,
        )
        tstats = pd.Series(
            np.asarray(result.tstats).reshape(-1),
            index=parameter_names,
            dtype=float,
        )
        pvalues = pd.Series(
            np.asarray(result.pvalues).reshape(-1),
            index=parameter_names,
            dtype=float,
        )
        confidence_intervals = np.asarray(result.conf_int())
        confidence_intervals = pd.DataFrame(
            confidence_intervals,
            index=parameter_names,
            columns=[0, 1],
            dtype=float,
        )

        self.params = params
        self.cov = covariance
        self.bse = std_errors
        self.tvalues = tstats
        self.pvalues = pvalues
        self._conf_int = confidence_intervals
        self.fittedvalues = pd.Series(
            np.asarray(fittedvalues).reshape(-1),
            index=fittedvalues.index if hasattr(fittedvalues, "index") else None,
            dtype=float,
        )
        self.resid = pd.Series(
            np.asarray(residuals).reshape(-1),
            index=residuals.index if hasattr(residuals, "index") else None,
            dtype=float,
        )
        self.nobs = int(getattr(result, "nobs", len(self.fittedvalues)))
        self.rsquared = float(getattr(result, "rsquared", np.nan))
        default_df_model = len(parameter_names) - 1 if "Intercept" in parameter_names else len(parameter_names)
        self.df_model = float(getattr(result, "df_model", default_df_model))
        self.df_resid = float(getattr(result, "df_resid", self.nobs - len(parameter_names)))
        if self.nobs > len(parameter_names) and not np.isnan(self.rsquared):
            numerator_df = max(self.nobs - 1, 1)
            denominator_df = max(self.nobs - len(parameter_names), 1)
            self.rsquared_adj = 1 - (1 - self.rsquared) * (numerator_df / denominator_df)
        else:
            self.rsquared_adj = np.nan

        f_statistic = getattr(result, "f_statistic", None)
        if f_statistic is not None:
            self.fvalue = _first_scalar(getattr(f_statistic, "stat", np.nan))
            self.f_pvalue = _first_scalar(getattr(f_statistic, "pval", np.nan))
        else:
            self.fvalue = _first_scalar(getattr(result, "fvalue", np.nan))
            self.f_pvalue = _first_scalar(getattr(result, "f_pvalue", np.nan))

    def conf_int(self) -> pd.DataFrame:
        return self._conf_int.copy()

    def wald_test(self, *args, **kwargs):
        """Delegate joint-coefficient tests to the underlying AbsorbingLS result."""
        return self._result.wald_test(*args, **kwargs)


def _fit_absorbed_formula_model(
    regression_df: pd.DataFrame,
    formula: str,
    cluster_spec: str,
):
    reduced_formula, absorbed_terms = _build_absorbed_formula_components(formula)
    if not absorbed_terms:
        return None

    try:
        from linearmodels.iv.absorbing import AbsorbingLS
    except Exception as exc:
        raise RuntimeError(
            "Absorbed fixed-effects estimation is required for this formula, "
            "but `linearmodels` could not be imported in the active environment."
        ) from exc

    y_matrix, x_matrix = patsy.dmatrices(
        reduced_formula,
        data=regression_df,
        return_type="dataframe",
        NA_action="drop",
    )
    regression_df = regression_df.loc[x_matrix.index].copy()
    y_matrix = y_matrix.loc[regression_df.index]
    x_matrix = x_matrix.loc[regression_df.index]

    absorb_columns: dict[str, pd.Series] = {}
    if f"C({ANNOUNCEMENT_QUARTER_COLUMN})" in absorbed_terms:
        absorb_columns[ANNOUNCEMENT_QUARTER_COLUMN] = regression_df[
            ANNOUNCEMENT_QUARTER_COLUMN
        ].astype("category")
    if f"C({FIRM_IDENTIFIER_COLUMN})" in absorbed_terms:
        absorb_columns[FIRM_IDENTIFIER_COLUMN] = regression_df[
            FIRM_IDENTIFIER_COLUMN
        ].astype("category")

    absorb_df = pd.DataFrame(absorb_columns, index=regression_df.index)
    model = AbsorbingLS(
        dependent=y_matrix.iloc[:, 0],
        exog=x_matrix,
        absorb=absorb_df,
    )

    fit_kwargs: dict[str, object] = {"debiased": True}
    groups = build_cluster_groups(regression_df, cluster_spec)
    if cluster_spec == "heteroskedasticity_robust":
        fit_kwargs["cov_type"] = "robust"
    elif groups is None:
        fit_kwargs["cov_type"] = "unadjusted"
    else:
        fit_kwargs["cov_type"] = "clustered"
        fit_kwargs["clusters"] = groups

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="covariance of constraints does not have full rank.*"
        )
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        absorbed_result = model.fit(**fit_kwargs)

    fittedvalues = absorbed_result.fitted_values
    residuals = absorbed_result.resids
    adapted_result = _AbsorbedResultAdapter(
        absorbed_result,
        list(x_matrix.columns),
        fittedvalues,
        residuals,
    )
    return regression_df, adapted_result


def fit_event_level_model(
    event_level: pd.DataFrame,
    formula: str,
    model_label: str,
    cluster_spec: str,
    regressor_column: str = SUE_GROUP_COLUMN,
) -> dict[str, object]:
    regression_df = prepare_regression_frame(
        event_level,
        regressor_column=regressor_column,
    )
    return fit_formula_model(
        regression_df,
        formula=formula,
        model_label=model_label,
        cluster_spec=cluster_spec,
    )


def fit_formula_model(
    frame: pd.DataFrame,
    formula: str,
    model_label: str,
    cluster_spec: str,
) -> dict[str, object]:
    regression_df = frame.copy()

    # Patsy/statsmodels can trip over pandas extension dtypes (`Int64`, `boolean`,
    # nullable strings/categoricals) because missing values are represented as
    # `pd.NA`, whose truth value is ambiguous. Normalize those columns to plain
    # numpy/object-backed arrays so missing values become `np.nan`.
    for column in regression_df.columns:
        series = regression_df[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            regression_df[column] = series.astype(object).where(series.notna(), np.nan)
            continue
        if is_numeric_dtype(series):
            regression_df[column] = pd.Series(
                pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan),
                index=series.index,
            )
            continue
        if is_bool_dtype(series):
            regression_df[column] = series.astype(object).where(series.notna(), np.nan)
            continue
        if is_extension_array_dtype(series):
            regression_df[column] = series.astype(object).where(series.notna(), np.nan)
            continue
        if series.dtype == object:
            regression_df[column] = series.mask(pd.isna(series), np.nan)

    referenced_columns = _referenced_formula_columns(formula, regression_df.columns)
    if referenced_columns:
        # Drop rows on the actual formula variables before Patsy expands the
        # categoricals. Otherwise it can retain category levels that appear only
        # in rows later dropped for missing data, creating all-zero dummy columns
        # and a rank-deficient fixed-effects design.
        regression_df = regression_df.dropna(subset=referenced_columns).copy()

    if ANNOUNCEMENT_YEAR_COLUMN in regression_df.columns:
        regression_df[ANNOUNCEMENT_YEAR_COLUMN] = pd.Series(
            pd.to_numeric(
                regression_df[ANNOUNCEMENT_YEAR_COLUMN], errors="coerce"
            ).to_numpy(dtype=float, na_value=np.nan),
            index=regression_df.index,
        )
        regression_df = regression_df.dropna(subset=[ANNOUNCEMENT_YEAR_COLUMN]).copy()
        regression_df[ANNOUNCEMENT_YEAR_COLUMN] = regression_df[ANNOUNCEMENT_YEAR_COLUMN].astype(int)

    if ANNOUNCEMENT_QUARTER_COLUMN in regression_df.columns:
        regression_df[ANNOUNCEMENT_QUARTER_COLUMN] = regression_df[
            ANNOUNCEMENT_QUARTER_COLUMN
        ].astype(object)
        regression_df = regression_df.dropna(subset=[ANNOUNCEMENT_QUARTER_COLUMN]).copy()
        regression_df[ANNOUNCEMENT_QUARTER_COLUMN] = regression_df[
            ANNOUNCEMENT_QUARTER_COLUMN
        ].astype(str)

    if FIRM_IDENTIFIER_COLUMN in regression_df.columns:
        regression_df[FIRM_IDENTIFIER_COLUMN] = regression_df[FIRM_IDENTIFIER_COLUMN].astype(str)

    if _formula_has_absorbed_fixed_effects(formula):
        # Firm/quarter FE specifications should be estimated via the absorbed
        # path. Falling back to explicit dummy expansion would silently change
        # the computational method for the same model family.
        absorbed_fit = _fit_absorbed_formula_model(
            regression_df,
            formula,
            cluster_spec,
        )
        regression_df, result = absorbed_fit
    else:
        model = smf.ols(formula, data=regression_df, missing="drop")
        regression_df = regression_df.loc[model.data.row_labels].copy()
        groups = build_cluster_groups(regression_df, cluster_spec)

        fit_kwargs = {}
        if cluster_spec == "heteroskedasticity_robust":
            fit_kwargs = {"cov_type": "HC1"}
        elif groups is not None:
            fit_kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="covariance of constraints does not have full rank.*"
            )
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            result = model.fit(**fit_kwargs)

    analysis_df = regression_df.copy()
    analysis_df["fitted"] = result.fittedvalues
    analysis_df["residual"] = result.resid
    cluster_counts = build_cluster_count_diagnostics(regression_df, cluster_spec)

    return {
        "label": model_label,
        "formula": formula,
        "cluster_label": CLUSTER_LABELS[cluster_spec],
        "cluster_counts": cluster_counts,
        "data": analysis_df,
        "result": result,
    }


def fit_simple_ols(frame: pd.DataFrame, x_col: str, y_col: str) -> dict[str, object]:
    """Fit the unbiasedness regression with firm--announcement-quarter clustering."""
    required_columns = [
        x_col,
        y_col,
        FIRM_IDENTIFIER_COLUMN,
        ANNOUNCEMENT_QUARTER_COLUMN,
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            "The unbiasedness regression requires firm and announcement-quarter "
            f"identifiers for two-way clustering; missing: {missing_columns}."
        )

    regression_df = frame[required_columns].dropna().copy()

    if len(regression_df) < 3:
        raise ValueError(f"Need at least 3 observations to fit {y_col} on {x_col}.")

    regression_df[x_col] = regression_df[x_col].astype(float)
    regression_df[y_col] = regression_df[y_col].astype(float)
    regression_df[FIRM_IDENTIFIER_COLUMN] = regression_df[FIRM_IDENTIFIER_COLUMN].astype(str)
    regression_df[ANNOUNCEMENT_QUARTER_COLUMN] = regression_df[
        ANNOUNCEMENT_QUARTER_COLUMN
    ].astype(str)

    x = sm.add_constant(regression_df[[x_col]], has_constant="add")
    y = regression_df[y_col]
    cluster_groups = build_cluster_groups(regression_df, "firm_quarter")
    result = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": cluster_groups,
            "use_correction": True,
            "df_correction": True,
        },
    )

    regression_df["fitted"] = result.fittedvalues
    regression_df["residual"] = result.resid

    confidence_intervals = result.conf_int()
    coefficient_table = pd.DataFrame(
        {
            "Term": ["Intercept", x_col],
            "Coefficient": [result.params["const"], result.params[x_col]],
            "Std_Error": [result.bse["const"], result.bse[x_col]],
            "t_stat": [result.tvalues["const"], result.tvalues[x_col]],
            "p_value": [result.pvalues["const"], result.pvalues[x_col]],
            "CI_95_Low": [
                confidence_intervals.loc["const", 0],
                confidence_intervals.loc[x_col, 0],
            ],
            "CI_95_High": [
                confidence_intervals.loc["const", 1],
                confidence_intervals.loc[x_col, 1],
            ],
        }
    )

    model_table = pd.Series(
        {
            "Dependent_Variable": y_col,
            "Predictor": x_col,
            "Std_Error_Treatment": CLUSTER_LABELS["firm_quarter"],
            **build_cluster_count_diagnostics(regression_df, "firm_quarter"),
            "N": int(result.nobs),
            "R": np.sign(result.params[x_col]) * np.sqrt(result.rsquared),
            "R_Squared": result.rsquared,
            "Residual_Std_Error": np.sqrt(result.mse_resid),
            "Slope_p_value": result.pvalues[x_col],
        },
        name="Model_Summary",
    )

    return {
        "data": regression_df,
        "coefficients": coefficient_table,
        "model": model_table,
        "degrees_of_freedom": int(result.df_resid),
        "result": result,
    }


def build_unbiasedness_tests(
    model_output: dict[str, object],
    intercept_null: float = 0.0,
    slope_null: float = 1.0,
) -> pd.Series:
    coefficient_table = model_output["coefficients"].set_index("Term")
    intercept_row = coefficient_table.loc["Intercept"]
    slope_term = coefficient_table.index[coefficient_table.index != "Intercept"][0]
    slope_row = coefficient_table.loc[slope_term]

    intercept_t = (intercept_row["Coefficient"] - intercept_null) / intercept_row["Std_Error"]
    # The fitted model uses two-way clustered covariance.  Use its cluster-robust
    # standard errors and the corresponding asymptotic normal reference distribution
    # for the restrictions alpha = 0 and beta = 1.
    intercept_p = 2 * stats.norm.sf(abs(intercept_t))

    slope_t = (slope_row["Coefficient"] - slope_null) / slope_row["Std_Error"]
    slope_p = 2 * stats.norm.sf(abs(slope_t))
    slope_greater_than_null_p = stats.norm.sf(slope_t)

    return pd.Series(
        {
            "Intercept_Null": intercept_null,
            "Intercept_t_stat": intercept_t,
            "Intercept_p_value": intercept_p,
            "Slope_Null": slope_null,
            "Slope_t_stat": slope_t,
            "Slope_p_value": slope_p,
            "Slope_greater_than_null_p_value": slope_greater_than_null_p,
        }
    )


def safe_scalar(value) -> float:
    if value is None:
        return np.nan
    array = np.asarray(value)
    if array.size == 0:
        return np.nan
    return float(array.reshape(-1)[0])


def build_coefficient_table(result) -> pd.DataFrame:
    confidence_intervals = result.conf_int()
    return pd.DataFrame(
        {
            "Term": result.params.index,
            "Coefficient": result.params.to_numpy(dtype=float),
            "Std_Error": result.bse.to_numpy(dtype=float),
            "t_stat": result.tvalues.to_numpy(dtype=float),
            "p_value": result.pvalues.to_numpy(dtype=float),
            "CI_95_Low": confidence_intervals.iloc[:, 0].to_numpy(dtype=float),
            "CI_95_High": confidence_intervals.iloc[:, 1].to_numpy(dtype=float),
        }
    )


def summarize_by_group(
    event_level: pd.DataFrame,
    group_column: str = SUE_GROUP_COLUMN,
    value_column: str = "PEAD",
) -> pd.DataFrame:
    summary = (
        event_level.dropna(subset=[group_column, value_column])
        .groupby(group_column)[value_column]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .rename(
            columns={
                "count": "Event_Count",
                "mean": "PEAD_Mean",
                "std": "PEAD_Std",
                "median": "PEAD_Median",
                "min": "PEAD_Min",
                "max": "PEAD_Max",
            }
        )
    )
    summary["PEAD_SE"] = summary["PEAD_Std"] / np.sqrt(summary["Event_Count"])
    return summary.reset_index()


def build_regressor_result_table(
    model_output: dict[str, object],
    regressor_column: str = SUE_GROUP_COLUMN,
    regressor_label: str = "SUE decile",
) -> pd.DataFrame:
    result = model_output["result"]
    confidence_intervals = result.conf_int()

    return pd.DataFrame(
        [
            {
                "Model": model_output["label"],
                "Std_Error_Treatment": model_output["cluster_label"],
                "Regressor": regressor_label,
                "Regressor_Column": regressor_column,
                "Regressor_Coefficient": result.params[regressor_column],
                "Regressor_Std_Error": result.bse[regressor_column],
                "Regressor_t_Statistic": result.tvalues[regressor_column],
                "Regressor_p_value": result.pvalues[regressor_column],
                "CI_95_Low": confidence_intervals.loc[regressor_column, 0],
                "CI_95_High": confidence_intervals.loc[regressor_column, 1],
            }
        ]
    )


def build_model_diagnostics_table(model_output: dict[str, object]) -> pd.DataFrame:
    result = model_output["result"]
    diagnostics = pd.Series(
        {
            "Model": model_output["label"],
            "Formula": model_output["formula"],
            "Std_Error_Treatment": model_output["cluster_label"],
            "Firm_Cluster_Count": model_output["cluster_counts"]["Firm_Cluster_Count"],
            "Quarter_Cluster_Count": model_output["cluster_counts"]["Quarter_Cluster_Count"],
            "R_Squared": result.rsquared,
            "Adjusted_R_Squared": result.rsquared_adj,
            "Sample_Size": int(result.nobs),
            "DF_Model": safe_scalar(result.df_model),
            "DF_Residual": safe_scalar(result.df_resid),
            "F_Statistic": safe_scalar(result.fvalue),
            "F_p_value": safe_scalar(result.f_pvalue),
        },
        name="Value",
    )
    return diagnostics.to_frame()


def build_comparison_report(
    model_outputs: list[dict[str, object]],
    regressor_column: str = SUE_GROUP_COLUMN,
    regressor_label: str = "SUE decile",
) -> pd.DataFrame:
    rows = []
    for model_output in model_outputs:
        result = model_output["result"]
        rows.append(
            {
                "Model": model_output["label"],
                "Std_Error_Treatment": model_output["cluster_label"],
                "Regressor": regressor_label,
                "Regressor_Column": regressor_column,
                "Regressor_Coefficient": result.params[regressor_column],
                "Regressor_Std_Error": result.bse[regressor_column],
                "Regressor_p_value": result.pvalues[regressor_column],
                "R_Squared": result.rsquared,
                "Adjusted_R_Squared": result.rsquared_adj,
                "Sample_Size": int(result.nobs),
                "DF_Model": safe_scalar(result.df_model),
                "DF_Residual": safe_scalar(result.df_resid),
                "F_Statistic": safe_scalar(result.fvalue),
                "F_p_value": safe_scalar(result.f_pvalue),
            }
        )
    return pd.DataFrame(rows)


def run_pooled_residual_autocorrelation(
    model_output: dict[str, object],
    year_column: str = ANNOUNCEMENT_YEAR_COLUMN,
    firm_column: str = FIRM_IDENTIFIER_COLUMN,
) -> dict[str, object]:
    residual_df = model_output["data"][[firm_column, year_column, "residual"]].sort_values(
        [firm_column, year_column]
    ).copy()

    residual_df["Residual_Lag"] = residual_df.groupby(firm_column)["residual"].shift(1)
    residual_df["Year_Lag"] = residual_df.groupby(firm_column)[year_column].shift(1)

    lag_pairs = residual_df.loc[
        (residual_df[year_column] - residual_df["Year_Lag"]) == 1,
        [firm_column, year_column, "residual", "Residual_Lag"],
    ].dropna().copy()

    if lag_pairs.empty:
        raise ValueError("No adjacent-year residual pairs are available for the autocorrelation regression.")

    result = smf.ols("residual ~ Residual_Lag", data=lag_pairs).fit()
    plot_result = smf.ols("Residual_Lag ~ residual", data=lag_pairs).fit()

    return {
        "pairs": lag_pairs,
        "result": result,
        "plot_result": plot_result,
    }


def build_autocorrelation_summary(output: dict[str, object]) -> pd.DataFrame:
    result = output["result"]
    confidence_intervals = result.conf_int()

    return pd.DataFrame(
        [
            {
                "Model": "Pooled residual autocorrelation",
                "Lag_Coefficient": result.params["Residual_Lag"],
                "Lag_Std_Error": result.bse["Residual_Lag"],
                "Lag_t_Statistic": result.tvalues["Residual_Lag"],
                "Lag_p_value": result.pvalues["Residual_Lag"],
                "CI_95_Low": confidence_intervals.loc["Residual_Lag", 0],
                "CI_95_High": confidence_intervals.loc["Residual_Lag", 1],
                "Sample_Size": int(result.nobs),
                "DF_Model": safe_scalar(result.df_model),
                "DF_Residual": safe_scalar(result.df_resid),
                "R_Squared": result.rsquared,
                "Adjusted_R_Squared": result.rsquared_adj,
                "F_Statistic": safe_scalar(result.fvalue),
                "F_p_value": safe_scalar(result.f_pvalue),
            }
        ]
    )


def plot_autocorrelation_pairs(
    output: dict[str, object],
    color_value: float = 0.65,
    output_manager=None,
    output_name: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    pairs = output["pairs"]
    plot_result = output["plot_result"]

    x_grid = np.linspace(pairs["residual"].min(), pairs["residual"].max(), 200)
    fitted_line = plot_result.predict(pd.DataFrame({"residual": x_grid}))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(
        pairs["residual"],
        pairs["Residual_Lag"],
        alpha=0.45,
        s=24,
        color=plt.get_cmap(COLOR_PALETTE)(color_value),
        edgecolor="none",
    )
    ax.plot(x_grid, fitted_line, color="black", linewidth=2, label="OLS line")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"$\hat{u}_t$")
    ax.set_ylabel(r"$\hat{u}_{t-1}$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_manager is not None and output_name is not None:
        output_manager.save_figure(fig, output_name)
    plt.show()


def run_pesaran_cd_test(
    analysis_df: pd.DataFrame,
    year_column: str = ANNOUNCEMENT_YEAR_COLUMN,
    firm_column: str = FIRM_IDENTIFIER_COLUMN,
) -> pd.Series:
    panel_residuals = (
        analysis_df[[year_column, firm_column, "residual"]]
        .dropna(subset=["residual"])
        .groupby([year_column, firm_column], as_index=False)["residual"]
        .mean()
    )

    pivot = panel_residuals.pivot(
        index=year_column,
        columns=firm_column,
        values="residual",
    ).sort_index()

    values = pivot.to_numpy(dtype=float)
    n_units = values.shape[1]

    if n_units < 2:
        raise ValueError("Need at least two firms for the Pesaran CD test.")

    cd_sum = 0.0
    pair_count = 0
    overlaps = []
    correlations = []

    for i in range(n_units - 1):
        left = values[:, i]
        left_mask = ~np.isnan(left)

        for j in range(i + 1, n_units):
            right = values[:, j]
            overlap_mask = left_mask & ~np.isnan(right)
            overlap_count = int(overlap_mask.sum())

            if overlap_count < 2:
                continue

            rho = np.corrcoef(left[overlap_mask], right[overlap_mask])[0, 1]
            if np.isnan(rho):
                continue

            cd_sum += np.sqrt(overlap_count) * rho
            pair_count += 1
            overlaps.append(overlap_count)
            correlations.append(rho)

    cd_statistic = np.sqrt(2.0 / (n_units * (n_units - 1))) * cd_sum
    p_value = 2 * (1 - stats.norm.cdf(abs(cd_statistic)))

    return pd.Series(
        {
            "Pesaran_CD_Statistic": cd_statistic,
            "Pesaran_CD_p_value": p_value,
            "Firm_Count": n_units,
            "Firm_Pair_Count": pair_count,
            "Average_Overlap_Years": float(np.mean(overlaps)) if overlaps else np.nan,
            "Average_Pairwise_Correlation": float(np.mean(correlations)) if correlations else np.nan,
            "Reject_Cross_Sectional_Independence_5pct": bool(p_value < 0.05),
        },
        name="Pesaran_CD_Test",
    )


def plot_residual_diagnostics(
    model_output: dict[str, object],
    regressor_column: str = SUE_GROUP_COLUMN,
    regressor_label: str = "SUE decile",
    color_value: float = 0.7,
    output_manager=None,
    output_name: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    analysis_df = model_output["data"].copy()

    firm_order = (
        analysis_df.groupby(FIRM_IDENTIFIER_COLUMN)["residual"]
        .mean()
        .sort_values()
        .index
    )
    analysis_df["Firm_Index"] = (
        pd.Categorical(
            analysis_df[FIRM_IDENTIFIER_COLUMN],
            categories=firm_order,
            ordered=True,
        ).codes
        + 1
    )

    year_values = sorted(analysis_df[ANNOUNCEMENT_YEAR_COLUMN].unique().tolist())
    residual_by_year = [
        analysis_df.loc[analysis_df[ANNOUNCEMENT_YEAR_COLUMN] == year, "residual"].to_numpy()
        for year in year_values
    ]

    color = plt.get_cmap(COLOR_PALETTE)(color_value)

    fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))

    axes[0, 0].scatter(
        analysis_df["Firm_Index"],
        analysis_df["residual"],
        alpha=0.35,
        s=18,
        color=color,
        edgecolor="none",
    )
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set_xlabel("Firm (ordered by mean residual)")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].set_xticks([])
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].boxplot(residual_by_year, tick_labels=year_values, showfliers=False)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_xlabel("Announcement year")
    axes[0, 1].set_ylabel("Residual")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].scatter(
        analysis_df[regressor_column],
        analysis_df["residual"],
        alpha=0.35,
        s=18,
        color=color,
        edgecolor="none",
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_xlabel(regressor_label)
    axes[1, 0].set_ylabel("Residual")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].scatter(
        analysis_df["fitted"],
        analysis_df["residual"],
        alpha=0.35,
        s=18,
        color=color,
        edgecolor="none",
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_xlabel("Fitted value")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].grid(alpha=0.2)

    fig.tight_layout()
    if output_manager is not None and output_name is not None:
        output_manager.save_figure(fig, output_name)
    plt.show()


def run_breusch_pagan_test(model_output: dict[str, object]) -> pd.Series:
    from statsmodels.stats.diagnostic import het_breuschpagan

    result = model_output["result"]
    lm_stat, lm_p_value, f_stat, f_p_value = het_breuschpagan(
        result.resid,
        result.model.exog,
    )

    return pd.Series(
        {
            "LM_Statistic": lm_stat,
            "LM_p_value": lm_p_value,
            "F_Statistic": f_stat,
            "F_p_value": f_p_value,
            "Reject_Homoskedasticity_5pct": bool(lm_p_value < 0.05),
        },
        name="Breusch_Pagan_Test",
    )


def summarize_period_slopes(
    result,
    base_term: str,
    period_labels: list[str],
    period_column: str = TIME_PERIOD_COLUMN,
) -> tuple[pd.DataFrame, list[str]]:
    parameter_names = list(result.params.index)
    if base_term not in parameter_names:
        raise KeyError(f"Base term {base_term!r} is not present in the model.")

    base_index = parameter_names.index(base_term)
    interaction_terms: list[str] = []
    rows: list[dict[str, object]] = []

    for position, label in enumerate(period_labels):
        matching_terms = [
            name
            for name in parameter_names
            if name != base_term and base_term in name and period_column in name and label in name
        ]
        interaction_term = matching_terms[0] if matching_terms else None
        if interaction_term is not None and interaction_term not in interaction_terms:
            interaction_terms.append(interaction_term)

        contrast = np.zeros(len(parameter_names))
        contrast[base_index] = 1.0
        if interaction_term is not None:
            contrast[parameter_names.index(interaction_term)] = 1.0

        test_result = result.t_test(contrast)
        confidence_interval = np.asarray(test_result.conf_int()).reshape(-1, 2)[0]
        rows.append(
            {
                "Time_Period": label,
                "Slope": safe_scalar(test_result.effect),
                "Std_Error": safe_scalar(test_result.sd),
                "t_stat": safe_scalar(test_result.tvalue),
                "p_value": safe_scalar(test_result.pvalue),
                "CI_95_Low": float(confidence_interval[0]),
                "CI_95_High": float(confidence_interval[1]),
                "Interaction_Term": interaction_term if interaction_term is not None else "(baseline)",
                "Is_Baseline_Period": interaction_term is None,
                "Period_Order": position,
            }
        )

    return pd.DataFrame(rows), interaction_terms


def build_joint_zero_test(result, term_names: list[str]) -> pd.Series:
    """Test selected non-intercept coefficients against zero jointly."""
    if not term_names:
        return pd.Series(
            {
                "Restriction_Count": 0,
                "F_Statistic": np.nan,
                "p_value": np.nan,
                "DF_Numerator": np.nan,
                "DF_Denominator": np.nan,
            }
        )

    parameter_names = list(result.params.index)
    intercept_terms = {"Intercept", "const"}
    requested_intercepts = [term for term in term_names if str(term) in intercept_terms]
    if requested_intercepts:
        raise ValueError(
            "Joint F-tests must not include the intercept: "
            + ", ".join(map(str, requested_intercepts))
        )
    restriction_matrix = np.zeros((len(term_names), len(parameter_names)))
    for row_index, term_name in enumerate(term_names):
        restriction_matrix[row_index, parameter_names.index(term_name)] = 1.0

    test_result = result.f_test(restriction_matrix)
    return pd.Series(
        {
            "Restriction_Count": len(term_names),
            "F_Statistic": safe_scalar(test_result.fvalue),
            "p_value": safe_scalar(test_result.pvalue),
            "DF_Numerator": safe_scalar(getattr(test_result, "df_num", np.nan)),
            "DF_Denominator": safe_scalar(getattr(test_result, "df_denom", np.nan)),
        }
    )
