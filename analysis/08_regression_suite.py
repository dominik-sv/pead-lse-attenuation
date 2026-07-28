from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import traceback
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

try:
    from IPython.display import display
except ImportError:
    def display(obj) -> None:
        print(obj)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar.
    tqdm = None


PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.regression_suite import (  # noqa: E402
    RegressionSpec,
    build_chunked_regression_latex_document,
    build_combined_regression_latex_document_from_summaries,
    build_regression_suite_dataset,
    PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN,
    plan_regression_suite_run,
    run_regression_suite,
)
from src.analysis.time_varying_analysis import (  # noqa: E402
    ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN,
    ANALYSIS_MARKET_CAP_SPLIT_BREAKPOINT_COLUMN,
    ANALYSIS_MARKET_CAP_SPLIT_PERCENTILE_COLUMN,
    ANNOUNCEMENT_QUARTER_COLUMN,
    ANNOUNCEMENT_WINDOW_COLUMN,
    FULL_WINDOW_COLUMN,
    FIRM_IDENTIFIER_COLUMN,
    FORMATION_YEAR_COLUMN,
    TIME_PERIOD_COLUMN,
    apply_market_cap_analysis_split,
    build_unbiasedness_tests,
    fit_formula_model,
    fit_simple_ols,
)
from src.pead.sue_groups import (  # noqa: E402
    SUE_GROUP_COLUMN,
    assign_prior_year_sue_groups,
)
from _analysis_shared import AnalysisOutputManager  # noqa: E402
from src.core.pipeline_config import (  # noqa: E402
    ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS,
    MAIN_REGRESSION_BHAR_WINDOW,
    MARKET_CAP_THRESHOLD as LSE_MARKET_CAP_FILTER_THRESHOLD,
    STOCK_PRICE_THRESHOLD as LSE_PRICE_FILTER_THRESHOLD,
    SUE_COMPUTATION_GROUP_COUNT,
    SUE_PLOT_GROUP_COUNT,
)
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR  # noqa: E402
from src.core.pead_sample_variants import (  # noqa: E402
    MAIN_PEAD_SAMPLE,
    MIN1_PEAD_SAMPLE,
)


DATA_DIR = PROJECT_DATA_DIR
OUTPUTS = AnalysisOutputManager(__file__)
OUTPUT_DIR = OUTPUTS.get_output_dir()

TIME_PERIOD_BUCKETS: tuple[tuple[int, int], ...] = (
    (1991, 2002),
    (2003, 2013),
    (2014, 2024),
)
# Center the ordinal time variable at the midpoint of the sample so that
# level SUE effects in linear-trend specifications are interpreted in 2008.
TIME_TREND_REFERENCE_YEAR = 2008
EXPECTED_TIME_PERIOD_LABELS = {
    f"{start_year}-{end_year}"
    for start_year, end_year in TIME_PERIOD_BUCKETS
}
MIDDLE_TIME_PERIOD_LABEL = f"{TIME_PERIOD_BUCKETS[1][0]}-{TIME_PERIOD_BUCKETS[1][1]}"
LATE_TIME_PERIOD_LABEL = f"{TIME_PERIOD_BUCKETS[2][0]}-{TIME_PERIOD_BUCKETS[2][1]}"

def format_bhar_column(day_start: int, day_end: int) -> str:
    return f"BHAR_{int(day_start)}_{int(day_end)}"


def format_bhar_label(day_start: int, day_end: int) -> str:
    return f"BHAR({int(day_start)},{int(day_end)})"


def format_bhar_window_key(day_start: int, day_end: int) -> str:
    return f"bhar_{int(day_start)}_{int(day_end)}"


def format_sue_split_key(group_count: int) -> str:
    return f"q{int(group_count)}"


def describe_sue_group_count(group_count: int) -> str:
    normalized_count = int(group_count)
    if normalized_count == 10:
        return "SUE decile"
    if normalized_count == 5:
        return "SUEQ"
    if normalized_count == 2:
        return "SUE 2-bin group"
    return f"SUE {normalized_count}-bin group"


MAIN_DEPENDENT_VARIABLE = format_bhar_column(*MAIN_REGRESSION_BHAR_WINDOW)
MAIN_DEPENDENT_VARIABLE_LABEL = format_bhar_label(*MAIN_REGRESSION_BHAR_WINDOW)
H3_SIZE_TIME_SPEC_KEY = "added_variables_firm_average_log_market_cap_time_interactions"
FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN = "Firm_Average_Log_Pre_Announcement_Market_Cap"
CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN = (
    "Centered_Firm_Average_Log_Pre_Announcement_Market_Cap"
)
FIRM_SIZE_MIDDLE_PERIOD_INTERACTION_COLUMN = "Firm_Size_x_Middle_Period"
FIRM_SIZE_LATE_PERIOD_INTERACTION_COLUMN = "Firm_Size_x_Late_Period"
YEAR_ORDINAL_COLUMN = "Year_Ordinal"
H3_SIZE_LINEAR_TIME_TREND_SPEC_KEY = (
    "added_variables_firm_average_log_market_cap_linear_time_trend_interactions"
)
H2_LINEAR_TIME_TREND_SPEC_KEY = "time_variation_linear_year_trend"
RAW_MAIN_REGRESSION_GROUP_COLUMN = f"Regression_SUE_Group_{SUE_COMPUTATION_GROUP_COUNT}"
MAIN_REGRESSION_GROUP_COLUMN = "SUEQ"
ALTERNATIVE_REGRESSION_GROUP_COLUMNS = {
    int(group_count): f"Regression_SUE_Group_{int(group_count)}"
    for group_count in ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS
}
MAIN_REGRESSION_GROUP_LABEL = (
    "SUEQ"
    if SUE_COMPUTATION_GROUP_COUNT == 5
    else "SUE decile"
    if SUE_COMPUTATION_GROUP_COUNT == 10
    else f"SUE {SUE_COMPUTATION_GROUP_COUNT}-bin group"
)
MAIN_FORMULA = (
    f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} + C({ANNOUNCEMENT_QUARTER_COLUMN}) "
    f"+ C({FIRM_IDENTIFIER_COLUMN})"
)
FIRM_AND_QUARTER_FIXED_EFFECT_TERMS = (
    f"C({ANNOUNCEMENT_QUARTER_COLUMN})",
    f"C({FIRM_IDENTIFIER_COLUMN})",
)
METHODOLOGY_CONTINUATION_INTERVALS: tuple[tuple[int, int], ...] = ((21, 40), (41, 60))
METHODOLOGY_CONTINUATION_SPEC_KEYS = {
    interval: f"methodology_continuation_{day_start}_{day_end}"
    for interval in METHODOLOGY_CONTINUATION_INTERVALS
    for day_start, day_end in [interval]
}
H1_TEST_ALPHA = 0.05
CONTINUATION_TEST_ALPHA = 0.10

REGRESSION_SPECS = [
    RegressionSpec(
        key="main_regression",
        family="main_regression",
        label=f"Main regression: {MAIN_DEPENDENT_VARIABLE_LABEL} on SUEQ with firm and quarter fixed effects",
        formula=MAIN_FORMULA,
        cluster_spec="firm_quarter",
    ),
    RegressionSpec(
        key="technical_no_fixed_effects",
        family="technical_fixed_effect_alternatives",
        label="Technical alternative: no fixed effects",
        formula=f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN}",
        cluster_spec="firm_quarter",
    ),
    RegressionSpec(
        key="technical_quarter_fixed_effects_only",
        family="technical_fixed_effect_alternatives",
        label="Technical alternative: quarter fixed effects only",
        formula=f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} + C({ANNOUNCEMENT_QUARTER_COLUMN})",
        cluster_spec="firm_quarter",
    ),
    RegressionSpec(
        key="technical_no_clustering",
        family="technical_clustering_alternatives",
        label="Technical alternative: no clustering",
        formula=MAIN_FORMULA,
        cluster_spec="none",
    ),
    RegressionSpec(
        key="technical_firm_clustering",
        family="technical_clustering_alternatives",
        label="Technical alternative: clustered by firm",
        formula=MAIN_FORMULA,
        cluster_spec="firm",
    ),
    RegressionSpec(
        key=H3_SIZE_TIME_SPEC_KEY,
        family="added_variable_alternatives",
        label=(
            "H3: firm-average log pre-announcement market cap with "
            "period-specific level and SUEQ interactions"
        ),
        formula=(
            f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:C({TIME_PERIOD_COLUMN}) "
            f"+ {FIRM_SIZE_MIDDLE_PERIOD_INTERACTION_COLUMN} "
            f"+ {FIRM_SIZE_LATE_PERIOD_INTERACTION_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN}:C({TIME_PERIOD_COLUMN}) "
            f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
        ),
        cluster_spec="firm_quarter",
        row_filter_query=f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} > 0",
        notes=(
            "This otherwise identical H3 specification additionally includes the centered "
            "firm-size interactions with the middle and late period indicators. The early "
            "period is the omitted reference category, and the time-invariant firm-size main "
            "effect is absorbed by firm fixed effects."
        ),
    ),
    RegressionSpec(
        key=H3_SIZE_LINEAR_TIME_TREND_SPEC_KEY,
        family="added_variable_alternatives",
        label=(
            "H3: firm-average log pre-announcement market cap with "
            "linear year-trend SUEQ interactions"
        ),
        formula=(
            f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} "
            f"+ {YEAR_ORDINAL_COLUMN}:{CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{YEAR_ORDINAL_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{YEAR_ORDINAL_COLUMN}:{CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN} "
            f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
        ),
        cluster_spec="firm_quarter",
        row_filter_query=f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} > 0",
        notes=(
            "The continuous time variable t is Year_Ordinal, which increases by one "
            f"for each formation year and equals zero in {TIME_TREND_REFERENCE_YEAR}. "
            "This specification replaces the period indicators in H3 with linear "
            "year-trend interactions."
        ),
    ),
    RegressionSpec(
        key=H2_LINEAR_TIME_TREND_SPEC_KEY,
        family="time_variation",
        label="Time variation: linear year trend in the SUEQ slope",
        formula=(
            f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} "
            f"+ {MAIN_REGRESSION_GROUP_COLUMN}:{YEAR_ORDINAL_COLUMN} "
            f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
        ),
        cluster_spec="firm_quarter",
        notes=(
            "The continuous time variable t is Year_Ordinal, which increases by one "
            f"for each formation year and equals zero in {TIME_TREND_REFERENCE_YEAR}."
        ),
    ),
]


def build_methodology_continuation_specs() -> list[RegressionSpec]:
    """Return the prespecified non-overlapping drift-interval regressions.

    Section 4b of the methodology uses these estimates for the sequential,
    one-sided duration rule. They are included in the baseline suite and in
    the robustness suites so the selected horizon can be checked under each
    alternative sample, return treatment, and SUE classification.
    """
    return [
        RegressionSpec(
            key=METHODOLOGY_CONTINUATION_SPEC_KEYS[(day_start, day_end)],
            family="methodology_drift_duration",
            label=(
                "Methodology continuation test: "
                f"{format_bhar_label(day_start, day_end)} on {MAIN_REGRESSION_GROUP_LABEL}"
            ),
            formula=(
                f"{format_bhar_column(day_start, day_end)} ~ {MAIN_REGRESSION_GROUP_COLUMN} "
                f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
            ),
            cluster_spec="firm_quarter",
        )
        for day_start, day_end in METHODOLOGY_CONTINUATION_INTERVALS
    ]


for alt_day_start, alt_day_end in ALTERNATIVE_REGRESSION_BHAR_WINDOWS:
    alt_bhar_column = format_bhar_column(alt_day_start, alt_day_end)
    REGRESSION_SPECS.append(
        RegressionSpec(
            key=f"variable_spec_{format_bhar_window_key(alt_day_start, alt_day_end)}",
            family="variable_specification_alternatives",
            label=f"Variable specification: {format_bhar_label(alt_day_start, alt_day_end)}",
            formula=(
                f"{alt_bhar_column} ~ {MAIN_REGRESSION_GROUP_COLUMN} + C({ANNOUNCEMENT_QUARTER_COLUMN}) "
                f"+ C({FIRM_IDENTIFIER_COLUMN})"
            ),
            cluster_spec="firm_quarter",
            notes=(
                "This model runs only if the prepared regression dataset includes "
                f"{alt_bhar_column}."
            ),
        )
    )

for alternative_group_count in ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS:
    alternative_group_count = int(alternative_group_count)
    alternative_group_column = ALTERNATIVE_REGRESSION_GROUP_COLUMNS[alternative_group_count]
    REGRESSION_SPECS.append(
        RegressionSpec(
            key=f"variable_spec_{format_sue_split_key(alternative_group_count)}",
            family="variable_specification_alternatives",
            label=(
                "Variable specification: "
                f"{describe_sue_group_count(alternative_group_count)} instead of {MAIN_REGRESSION_GROUP_LABEL.lower()}"
            ),
            formula=(
                f"{MAIN_DEPENDENT_VARIABLE} ~ {alternative_group_column} + C({ANNOUNCEMENT_QUARTER_COLUMN}) "
                f"+ C({FIRM_IDENTIFIER_COLUMN})"
            ),
            cluster_spec="firm_quarter",
            notes=(
                f"This specification uses the configured {alternative_group_count}-group prior-year SUE split "
                "instead of the main regression split."
            ),
        )
    )

DATASET_CACHE_FILENAME = "analysis_dataset_cache.pkl"
DATASET_CACHE_METADATA_FILENAME = "analysis_dataset_cache_metadata.json"
DATASET_CACHE_VERSION = 9
REGRESSION_SUITE_OUTPUT_VERSION = 6
DATASET_BUILD_PARAMETERS = {
    "explicit_time_periods": [list(period) for period in TIME_PERIOD_BUCKETS],
    "analyst_following_cutoffs": [10, 5, 3],
    "additional_bhar_windows": [list(interval) for interval in METHODOLOGY_CONTINUATION_INTERVALS],
}

BASELINE_SUITE_LABEL = "baseline_sample"
SPLIT_SAMPLE_PARENT_LABEL = "split_sample_50_50"
SPLIT_SAMPLE_WINDOW_SELECTION_LABEL = "split_sample_50pct_window_selection"
SPLIT_SAMPLE_ANALYSIS_LABEL = "split_sample_50pct_main_analysis"
SPLIT_SAMPLE_RANDOM_SEED = 20260715
SPLIT_SAMPLE_WINDOW_SELECTION_SHARE = 0.50
SPLIT_SAMPLE_ASSIGNMENT_COLUMN = "Split_Sample_Group"
SPLIT_SAMPLE_STRATUM_COLUMN = "Split_Sample_Formation_Year_Stratum"
SUE_DECILE_FULL_SUITE_LABEL = "sue_decile_full_suite"
SUE_TWO_BIN_FULL_SUITE_LABEL = "sue_2_bin_full_suite"
HETEROSKEDASTICITY_ROBUST_SUITE_LABEL = "heteroskedasticity_robust_se"
NO_FIXED_EFFECTS_FULL_SUITE_LABEL = "no_fixed_effects_full_suite"
FIRM_QUARTER_FIXED_EFFECTS_FULL_SUITE_LABEL = "firm_quarter_fixed_effects_full_suite"
MARKET_CAP_PRICE_FILTER_LABEL = "market_cap_ge_5m_price_ge_1"
POST_2000_BASELINE_LABEL = "year_2000_plus"
POST_2000_MARKET_CAP_PRICE_FILTER_LABEL = "year_2000_plus_market_cap_ge_5m_price_ge_1"
COMPLETE_CASE_RETURN_LABEL = "missing_returns_complete_case"
TERMINAL_LOSS_RETURN_LABEL = "missing_returns_terminal_loss"
BASELINE_RETURN_FILENAME = MAIN_PEAD_SAMPLE.abnormal_returns_filename
COMPLETE_CASE_RETURN_FILENAME = MAIN_PEAD_SAMPLE.abnormal_returns_drop_missing_filename
TERMINAL_LOSS_RETURN_FILENAME = MAIN_PEAD_SAMPLE.abnormal_returns_terminal_loss_filename
MIN1_BASELINE_RETURN_FILENAME = MIN1_PEAD_SAMPLE.abnormal_returns_filename
MIN1_COMPLETE_CASE_RETURN_FILENAME = MIN1_PEAD_SAMPLE.abnormal_returns_drop_missing_filename
MIN1_TERMINAL_LOSS_RETURN_FILENAME = MIN1_PEAD_SAMPLE.abnormal_returns_terminal_loss_filename
MARKET_CAP_FILTER_THRESHOLD = 5
PRICE_FILTER_THRESHOLD = 1.0
MIN_SAMPLE_YEAR = 2000

NON_ORIGINAL_EXCLUDED_FAMILIES = {
    "technical_fixed_effect_alternatives",
    "technical_clustering_alternatives",
}

# Disabled outputs retained in the source so they can be restored if needed.
DISABLED_MAIN_OUTPUT_FAMILIES = {
    "technical_fixed_effect_alternatives",
    "technical_clustering_alternatives",
}

# These suites are retained in the source but are not scheduled for estimation.
DISABLED_SUITE_KEYS = {
    "heteroskedasticity_robust_se",
    "split_sample_50pct_window_selection",
    "split_sample_50pct_main_analysis",
    "bhar_2_60_main",
    "grid_search",
    "year_2000_plus",
    "year_2000_plus_market_cap_ge_5m_price_ge_1",
    "log_bhar_2_20_main",
    "variant_grid_search",
}

ALL_CONFIGURED_REGRESSION_BHAR_WINDOWS = (
    MAIN_REGRESSION_BHAR_WINDOW,
    *ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    *METHODOLOGY_CONTINUATION_INTERVALS,
)
REGRESSION_BHAR_COLUMNS = tuple(
    dict.fromkeys(format_bhar_column(day_start, day_end) for day_start, day_end in ALL_CONFIGURED_REGRESSION_BHAR_WINDOWS)
)
LOG_BHAR20_MAIN_LABEL = f"log_{format_bhar_window_key(*MAIN_REGRESSION_BHAR_WINDOW)}_main"
LOG_BHAR20_MAIN_COLUMN = f"{MAIN_DEPENDENT_VARIABLE}_Signed_Log1p"
LOG_BHAR20_MAIN_LABEL_TEXT = f"SymLog {MAIN_DEPENDENT_VARIABLE_LABEL}"
LOG_BHAR_COLUMN_MAP = {
    bhar_column: f"{bhar_column}_Signed_Log1p"
    for bhar_column in REGRESSION_BHAR_COLUMNS
}

WINSORIZED_BHAR20_MAIN_LABEL = f"winsorized_{format_bhar_window_key(*MAIN_REGRESSION_BHAR_WINDOW)}_main"
WINSORIZED_BHAR20_MAIN_COLUMN = f"{MAIN_DEPENDENT_VARIABLE}_Winsorized_5_95"
WINSORIZED_BHAR20_MAIN_LABEL_TEXT = f"Winsorized {MAIN_DEPENDENT_VARIABLE_LABEL}"
WINSORIZED_BHAR_COLUMN_MAP = {
    bhar_column: f"{bhar_column}_Winsorized_5_95"
    for bhar_column in REGRESSION_BHAR_COLUMNS
}

BHAR_2_60_MAIN_COLUMN = "BHAR_2_60"
BHAR_2_60_MAIN_LABEL = format_bhar_window_key(2, 60) + "_main"
BHAR_2_60_MAIN_LABEL_TEXT = format_bhar_label(2, 60)

GRID_SEARCH_LABEL = "grid_search"
GRID_SEARCH_TITLE = "Grid Search: Market-Cap Breakpoints"
GRID_REGRESSIONS_PER_TABLE = 6
GRID_TABLES_PER_PAGE = 2
VARIANT_GRID_SEARCH_LABEL = "variant_grid_search"
VARIANT_GRID_SEARCH_TITLE = "Grid Search: BHAR / Transform / SUE Variant Grid"
GRID_BASE_SAMPLES = (
    {"key": "base", "label": "Analyst >= 3 sample", "apply_market_cap_price_filter": False},
)
GRID_MARKET_CAP_BREAKPOINTS = (
    {"key": "p10", "label": "10th percentile", "percentile": 0.10, "column": "Grid_MCap_Split_10"},
    {"key": "p20", "label": "20th percentile", "percentile": 0.20, "column": "Grid_MCap_Split_20"},
    {"key": "p30", "label": "30th percentile", "percentile": 0.30, "column": "Grid_MCap_Split_30"},
    {"key": "p40", "label": "40th percentile", "percentile": 0.40, "column": "Grid_MCap_Split_40"},
    {"key": "p50", "label": "50th percentile", "percentile": 0.50, "column": "Grid_MCap_Split_50"},
    {"key": "p60", "label": "60th percentile", "percentile": 0.60, "column": "Grid_MCap_Split_60"},
    {"key": "p70", "label": "70th percentile", "percentile": 0.70, "column": "Grid_MCap_Split_70"},
    {"key": "p80", "label": "80th percentile", "percentile": 0.80, "column": "Grid_MCap_Split_80"},
    {"key": "p90", "label": "90th percentile", "percentile": 0.90, "column": "Grid_MCap_Split_90"},
)
VARIANT_GRID_BHAR_CONFIGS = tuple(
    {
        "key": f"b{int(day_end)}",
        "source_column": format_bhar_column(day_start, day_end),
        "label": format_bhar_label(day_start, day_end),
    }
    for day_start, day_end in (MAIN_REGRESSION_BHAR_WINDOW, *ALTERNATIVE_REGRESSION_BHAR_WINDOWS)
)
VARIANT_GRID_SUE_SPLITS = tuple(
    {
        "key": format_sue_split_key(group_count),
        "label": f"Q{int(group_count)}",
        "group_count": int(group_count),
        "column": f"Variant_Grid_SUE_Group_{int(group_count)}",
    }
    for group_count in ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS
)
VARIANT_GRID_WINSOR_OPTIONS = (
    {"key": "raw", "label": "Raw BHAR", "apply_winsorized": False},
    {"key": "win", "label": "Winsorized BHAR", "apply_winsorized": True},
)
VARIANT_GRID_LOG_OPTIONS = (
    {"key": "lin", "label": "Linear BHAR", "apply_symlog": False},
    {"key": "sym", "label": "SymLog BHAR", "apply_symlog": True},
)

TIME_VARIATION_SPEC_KEY = "time_variation_three_periods"
TIME_VARIATION_RESULTS_TEX_PATH = (
    PROJECT_ROOT / "thesis2" / "02_Mainmatter" / "05a_time_variation_results.tex"
)


@dataclass(frozen=True)
class SuiteJob:
    key: str
    output_dir: Path
    specs: list[RegressionSpec]
    dataset_builder: Callable[[pd.DataFrame], pd.DataFrame]
    dataset_recipe: dict[str, object]
    source_dataset_key: str = BASELINE_SUITE_LABEL
    table_mode: str = "family"
    display_outputs: bool = False
    title: str = ""
    suite_parameters: dict[str, object] | None = None
    post_run_callback: Callable[[Path, pd.DataFrame, dict[str, object]], None] | None = None
    pre_filter_audit_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None


@dataclass(frozen=True)
class RegressionSampleConfig:
    key: str
    label: str
    min_analyst_count: int
    baseline_return_filename: str
    complete_case_return_filename: str
    terminal_loss_return_filename: str
    output_dir: Path


MAIN_REGRESSION_SAMPLE = RegressionSampleConfig(
    key="main",
    label="Analyst >= 3 sample",
    min_analyst_count=int(MAIN_PEAD_SAMPLE.min_analyst_forecasts),
    baseline_return_filename=BASELINE_RETURN_FILENAME,
    complete_case_return_filename=COMPLETE_CASE_RETURN_FILENAME,
    terminal_loss_return_filename=TERMINAL_LOSS_RETURN_FILENAME,
    output_dir=OUTPUT_DIR,
)

MIN1_REGRESSION_SAMPLE = RegressionSampleConfig(
    key="min1",
    label="Analyst >= 1 sample",
    min_analyst_count=int(MIN1_PEAD_SAMPLE.min_analyst_forecasts),
    baseline_return_filename=MIN1_BASELINE_RETURN_FILENAME,
    complete_case_return_filename=MIN1_COMPLETE_CASE_RETURN_FILENAME,
    terminal_loss_return_filename=MIN1_TERMINAL_LOSS_RETURN_FILENAME,
    output_dir=OUTPUT_DIR / "min1",
)

REGRESSION_SAMPLE_CONFIGS = (
    MAIN_REGRESSION_SAMPLE,
    MIN1_REGRESSION_SAMPLE,
)

MIN1_ENABLED_SUITE_KEYS = {
    BASELINE_SUITE_LABEL,
}


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _collect_dataset_dependency_signatures(
    data_dir: Path,
    abnormal_returns_filename: str,
) -> list[dict[str, object]]:
    dependency_paths = sorted(data_dir.rglob(abnormal_returns_filename))
    dependency_paths.extend(sorted(data_dir.rglob("stock_universe.csv")))
    dependency_paths.extend(sorted(data_dir.rglob("daily_market_caps_completed.csv")))

    aggregate_path = data_dir / "sample_size_all_years.json"
    if aggregate_path.exists():
        dependency_paths.append(aggregate_path)

    source_paths = [
        PROJECT_ROOT / "analysis" / "11_regression_suite.py",
        PROJECT_ROOT / "src" / "analysis" / "regression_suite.py",
        PROJECT_ROOT / "src" / "analysis" / "time_varying_analysis.py",
        PROJECT_ROOT / "src" / "pead" / "sue_groups.py",
        PROJECT_ROOT / "src" / "analysis" / "bhar_outlier_policy.py",
    ]
    dependency_paths.extend(path for path in source_paths if path.exists())

    return [_path_signature(path) for path in dependency_paths]


def _dataset_cache_metadata(
    output_dir: Path,
    data_dir: Path,
    *,
    abnormal_returns_filename: str,
) -> dict[str, object]:
    return {
        "cache_version": DATASET_CACHE_VERSION,
        "build_parameters": DATASET_BUILD_PARAMETERS,
        "abnormal_returns_filename": abnormal_returns_filename,
        "dependencies": _collect_dataset_dependency_signatures(
            data_dir, abnormal_returns_filename
        ),
        "output_dir": str(output_dir.resolve()),
    }


def load_or_build_regression_suite_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    abnormal_returns_filename: str = BASELINE_RETURN_FILENAME,
    cache_key: str = BASELINE_SUITE_LABEL,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_key == BASELINE_SUITE_LABEL:
        cache_path = output_dir / DATASET_CACHE_FILENAME
        metadata_path = output_dir / DATASET_CACHE_METADATA_FILENAME
    else:
        cache_path = output_dir / f"analysis_dataset_cache_{cache_key}.pkl"
        metadata_path = output_dir / f"analysis_dataset_cache_metadata_{cache_key}.json"
    current_metadata = _dataset_cache_metadata(
        output_dir,
        data_dir,
        abnormal_returns_filename=abnormal_returns_filename,
    )

    if cache_path.exists() and metadata_path.exists():
        try:
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved_metadata = None
        if saved_metadata == current_metadata:
            dataset = pd.read_pickle(cache_path)
            dataset = add_regression_suite_sue_columns(dataset)
            return dataset

    dataset = build_regression_suite_dataset(
        data_dir,
        explicit_time_periods=TIME_PERIOD_BUCKETS,
        analyst_following_cutoffs=tuple(DATASET_BUILD_PARAMETERS["analyst_following_cutoffs"]),
        abnormal_returns_filename=abnormal_returns_filename,
        additional_bhar_windows=METHODOLOGY_CONTINUATION_INTERVALS,
    )
    dataset = add_regression_suite_sue_columns(dataset)
    try:
        dataset.to_pickle(cache_path)
        metadata_path.write_text(
            json.dumps(current_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        # The cache only avoids rebuilding the prepared regression dataset on a
        # later run. A cache-write failure (for example, a transient Windows
        # file-handle issue) must not prevent the empirical suite itself from
        # using the already prepared in-memory dataset.
        print(
            "Warning: could not save the regression-suite dataset cache "
            f"at {cache_path}: {exc}. Continuing without updating this cache."
        )
    return dataset


def drop_first_formation_year(dataset: pd.DataFrame) -> pd.DataFrame:
    formation_years = pd.to_numeric(dataset[FORMATION_YEAR_COLUMN], errors="coerce")
    valid_years = formation_years.dropna()
    if valid_years.empty:
        return dataset.copy()

    first_year = int(valid_years.min())
    return dataset.loc[formation_years.ne(first_year)].copy()


def compute_base_dataset_fingerprint(data: pd.DataFrame) -> str:
    digest = sha256()
    digest.update(str(data.shape).encode("utf-8"))
    digest.update("||".join(f"{column}:{dtype}" for column, dtype in data.dtypes.items()).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _save_table_to_dir(output_dir: Path, table: pd.DataFrame | pd.Series, name: str) -> Path:
    path = output_dir / f"{name}.csv"
    if isinstance(table, pd.Series):
        table = table.to_frame(name=table.name if table.name is not None else "Value")
        table.to_csv(path)
    else:
        table.to_csv(path, index=False)
    return path


def _save_latex_to_dir(output_dir: Path, name: str, content: str) -> Path:
    path = output_dir / f"{name}.tex"
    path.write_text(content, encoding="utf-8")
    return path


def build_dataset_column_overview(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in data.columns:
        series = data[column]
        rows.append(
            {
                "Column": column,
                "Dtype": str(series.dtype),
                "Non_Missing": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Unique_Values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("Column").reset_index(drop=True)


def _post_announcement_window_missing_mask(
    dataset: pd.DataFrame,
    *,
    window_start: int = 2,
    window_end: int = 20,
) -> pd.Series:
    """Identify events with an unavailable stock return in the specified BHAR window."""
    missing_count = pd.to_numeric(
        dataset["Missing_Post_Announcement_Return_Day_Count"], errors="coerce"
    )
    first_missing_day = pd.to_numeric(
        dataset["First_Missing_Relative_Day"], errors="coerce"
    )
    return (
        missing_count.gt(0)
        & first_missing_day.ge(window_start)
        & first_missing_day.le(window_end)
    )


def build_appendix_b_1_3_missing_return_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize the baseline missing-return diagnostics reported in Appendix B.1.3."""
    required_columns = {
        "Missing_Post_Announcement_Return_Day_Count",
        "First_Missing_Relative_Day",
        "Has_Terminal_Missing_Return",
    }
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise KeyError(
            "Cannot build the Appendix B.1.3 missing-return summary; missing columns: "
            f"{missing_columns}."
        )

    event_count = len(dataset)
    missing_in_window = _post_announcement_window_missing_mask(dataset)
    terminal_loss_in_window = (
        missing_in_window
        & dataset["Has_Terminal_Missing_Return"].fillna(False).astype(bool)
    )

    rows = [
        ("Baseline events", event_count),
        ("Events with a missing stock return in BHAR[2,20]", int(missing_in_window.sum())),
        (
            "Events excluded under the complete-case BHAR[2,20] treatment",
            int(missing_in_window.sum()),
        ),
        ("Events receiving a terminal loss in BHAR[2,20]", int(terminal_loss_in_window.sum())),
    ]
    summary = pd.DataFrame(rows, columns=["Metric", "Event_Count"])
    summary["Sample_Share_Percent"] = np.where(
        event_count > 0,
        100.0 * summary["Event_Count"] / event_count,
        np.nan,
    )
    return summary


def build_appendix_b_1_4_winsorization_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the percentage-point distribution table reported in Appendix B.1.4."""
    raw_column = MAIN_DEPENDENT_VARIABLE
    winsorized_column = WINSORIZED_BHAR20_MAIN_COLUMN
    missing_columns = [
        column for column in (raw_column, winsorized_column) if column not in dataset.columns
    ]
    if missing_columns:
        raise KeyError(
            "Cannot build the Appendix B.1.4 winsorization summary; missing columns: "
            f"{missing_columns}."
        )

    rows: list[dict[str, object]] = []
    for label, column in [
        ("Original BHAR[2,20]", raw_column),
        ("Winsorized BHAR[2,20]", winsorized_column),
    ]:
        values = pd.to_numeric(dataset[column], errors="coerce").dropna()
        rows.append(
            {
                "BHAR_Treatment": label,
                "N": int(values.size),
                "Mean_Percentage_Points": values.mean(),
                "SD_Percentage_Points": values.std(ddof=1),
                "Min_Percentage_Points": values.min(),
                "P5_Percentage_Points": values.quantile(0.05),
                "Median_Percentage_Points": values.median(),
                "P95_Percentage_Points": values.quantile(0.95),
                "Max_Percentage_Points": values.max(),
            }
        )
    return pd.DataFrame(rows)


def filter_specs_for_non_original_suite(specs: list[RegressionSpec]) -> list[RegressionSpec]:
    return [
        spec
        for spec in specs
        if spec.family not in NON_ORIGINAL_EXCLUDED_FAMILIES
    ]


def build_heteroskedasticity_robust_specs(
    specs: list[RegressionSpec],
) -> list[RegressionSpec]:
    """Return otherwise identical specifications estimated with HC1 standard errors."""
    return [
        RegressionSpec(
            key=spec.key,
            family=spec.family,
            label=spec.label,
            formula=spec.formula,
            cluster_spec="heteroskedasticity_robust",
            row_filter_query=spec.row_filter_query,
            enabled=spec.enabled,
            notes=spec.notes,
            fixed_effect_terms_to_exclude=spec.fixed_effect_terms_to_exclude,
            ordered_time_periods=spec.ordered_time_periods,
            ordered_time_regressor=spec.ordered_time_regressor,
            ordered_time_period_column=spec.ordered_time_period_column,
        )
        for spec in specs
    ]


def build_fixed_effect_variant_specs(
    specs: list[RegressionSpec],
    *,
    include_firm_and_quarter_fixed_effects: bool,
) -> list[RegressionSpec]:
    """Return each specification with a uniform firm/quarter FE treatment.

    The baseline suite uses firm and announcement-quarter fixed effects, but
    its technical FE checks apply only to the main regression.  This helper
    creates full-suite counterparts by applying the requested treatment to
    every reported specification.  When firm effects are removed, the two H3
    specifications explicitly restore the firm-size main effect that is
    otherwise absorbed by firm fixed effects.
    """
    transformed_specs: list[RegressionSpec] = []
    for spec in specs:
        dependent_variable, separator, rhs = spec.formula.partition("~")
        if not separator:
            raise ValueError(f"Invalid formula {spec.formula!r}.")

        rhs_terms = [
            term.strip()
            for term in rhs.split("+")
            if term.strip() not in FIRM_AND_QUARTER_FIXED_EFFECT_TERMS
        ]
        if include_firm_and_quarter_fixed_effects:
            rhs_terms.extend(FIRM_AND_QUARTER_FIXED_EFFECT_TERMS)
        elif spec.key in {
            H3_SIZE_TIME_SPEC_KEY,
            H3_SIZE_LINEAR_TIME_TREND_SPEC_KEY,
        } and CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN not in rhs_terms:
            rhs_terms.append(CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN)

        notes = spec.notes
        if include_firm_and_quarter_fixed_effects:
            variant_note = (
                "This full-suite variant includes firm and announcement-quarter fixed effects "
                "in every specification."
            )
        else:
            variant_note = (
                "This full-suite variant excludes firm and announcement-quarter fixed effects "
                "in every specification."
            )
            notes = notes.replace(
                "and the time-invariant firm-size main effect is absorbed by firm fixed effects.",
                "and includes the centered firm-size main effect because it is not absorbed by firm fixed effects.",
            )
        notes = " ".join(part for part in (notes, variant_note) if part)

        transformed_specs.append(
            replace(
                spec,
                formula=f"{dependent_variable.strip()} ~ {' + '.join(rhs_terms)}",
                notes=notes,
                fixed_effect_terms_to_exclude=(
                    FIRM_AND_QUARTER_FIXED_EFFECT_TERMS
                    if include_firm_and_quarter_fixed_effects
                    else ()
                ),
            )
        )
    return transformed_specs


def build_main_dependent_variant_specs(
    specs: list[RegressionSpec],
    *,
    dependent_variable_map: dict[str, str],
    dependent_label_map: dict[str, str],
    excluded_spec_keys: set[str] | None = None,
) -> list[RegressionSpec]:
    excluded_keys = excluded_spec_keys or set()
    transformed_specs: list[RegressionSpec] = []
    for spec in specs:
        if spec.key in excluded_keys:
            continue

        formula = spec.formula
        label = spec.label
        for source_variable, target_variable in dependent_variable_map.items():
            prefix = f"{source_variable} ~"
            if formula.startswith(prefix):
                formula = formula.replace(prefix, f"{target_variable} ~", 1)
                label = label.replace(
                    dependent_label_map.get(source_variable, source_variable),
                    dependent_label_map.get(target_variable, target_variable),
                )
                break

        transformed_specs.append(
            RegressionSpec(
                key=spec.key,
                family=spec.family,
                label=label,
                formula=formula,
                cluster_spec=spec.cluster_spec,
                row_filter_query=spec.row_filter_query,
                enabled=spec.enabled,
            notes=spec.notes,
            fixed_effect_terms_to_exclude=spec.fixed_effect_terms_to_exclude,
            ordered_time_periods=spec.ordered_time_periods,
            ordered_time_regressor=spec.ordered_time_regressor,
            ordered_time_period_column=spec.ordered_time_period_column,
            )
        )
    return transformed_specs


def build_family_specs_map(
    specs: list[RegressionSpec],
    completed_spec_keys: set[str],
) -> list[tuple[str, list[RegressionSpec]]]:
    main_spec = next(
        (
            spec
            for spec in specs
            if spec.key == "main_regression" and spec.key in completed_spec_keys
        ),
        None,
    )

    combined_family_specs: list[tuple[str, list[RegressionSpec]]] = []
    for family in dict.fromkeys(spec.family for spec in specs):
        family_specs = [
            spec
            for spec in specs
            if spec.family == family and spec.key in completed_spec_keys
        ]
        if not family_specs:
            continue
        if family != "main_regression" and main_spec is not None:
            family_specs = [main_spec] + [
                spec for spec in family_specs if spec.key != main_spec.key
            ]
        combined_family_specs.append((family, family_specs))
    return combined_family_specs


def build_suite_metadata(job: SuiteJob, *, base_dataset_fingerprint: str) -> dict[str, object]:
    payload = {
        "regression_suite_output_version": REGRESSION_SUITE_OUTPUT_VERSION,
        "suite_key": job.key,
        "base_dataset_fingerprint": base_dataset_fingerprint,
        "source_dataset_key": job.source_dataset_key,
        "table_mode": job.table_mode,
        "title": job.title,
        "time_period_buckets": [list(period) for period in TIME_PERIOD_BUCKETS],
        "dataset_recipe": job.dataset_recipe,
        "suite_parameters": job.suite_parameters or {},
    }
    payload_text = json.dumps(payload, sort_keys=True)
    payload["suite_fingerprint"] = sha256(payload_text.encode("utf-8")).hexdigest()
    return payload


def validate_time_period_buckets(dataset: pd.DataFrame) -> None:
    observed_labels = set(
        dataset[TIME_PERIOD_COLUMN].dropna().astype("string").tolist()
    )
    if observed_labels != EXPECTED_TIME_PERIOD_LABELS:
        raise ValueError(
            "Unexpected time-period labels in prepared regression dataset: "
            f"{sorted(observed_labels)}."
        )


def add_signed_log_bhar_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    transformed = dataset.copy()
    for source_column, target_column in LOG_BHAR_COLUMN_MAP.items():
        if source_column not in transformed.columns:
            continue
        values = pd.to_numeric(transformed[source_column], errors="coerce")
        transformed[target_column] = np.sign(values) * np.log1p(np.abs(values))
    return transformed


def add_regression_suite_sue_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    grouped = build_prior_year_group_columns(
        dataset,
        group_columns=[
            (SUE_COMPUTATION_GROUP_COUNT, RAW_MAIN_REGRESSION_GROUP_COLUMN),
            *[
                (group_count, column_name)
                for group_count, column_name in ALTERNATIVE_REGRESSION_GROUP_COLUMNS.items()
                if group_count != SUE_COMPUTATION_GROUP_COUNT
            ],
        ],
    )
    # SUEQ is the baseline SUE quintile centered at Q3: Q1-Q5 map to -2 through 2.
    grouped[MAIN_REGRESSION_GROUP_COLUMN] = (
        pd.to_numeric(grouped[RAW_MAIN_REGRESSION_GROUP_COLUMN], errors="coerce") - 3
    ).astype("Int64")
    return grouped


def add_winsorized_bhar_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    transformed = dataset.copy()
    for source_column, target_column in WINSORIZED_BHAR_COLUMN_MAP.items():
        if source_column not in transformed.columns:
            continue
        values = pd.to_numeric(transformed[source_column], errors="coerce")
        valid_values = values.dropna()
        if valid_values.empty:
            transformed[target_column] = values
            continue

        lower_bound = float(valid_values.quantile(0.05))
        upper_bound = float(valid_values.quantile(0.95))
        transformed[target_column] = values.clip(
            lower=lower_bound,
            upper=upper_bound,
        )
    return transformed


def add_variant_grid_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    transformed = build_prior_year_group_columns(
        dataset,
        group_columns=[
            (config["group_count"], config["column"])
            for config in VARIANT_GRID_SUE_SPLITS
        ],
    )

    for bhar_config in VARIANT_GRID_BHAR_CONFIGS:
        source_column = bhar_config["source_column"]
        if source_column not in transformed.columns:
            continue

        source_values = pd.to_numeric(transformed[source_column], errors="coerce")
        winsorized_values = source_values
        valid_values = source_values.dropna()
        if not valid_values.empty:
            lower_bound = float(valid_values.quantile(0.05))
            upper_bound = float(valid_values.quantile(0.95))
            winsorized_values = source_values.clip(lower=lower_bound, upper=upper_bound)

        raw_column = f"{source_column}_Variant_Raw"
        win_column = f"{source_column}_Variant_Win"
        sym_raw_column = f"{source_column}_Variant_Sym"
        sym_win_column = f"{source_column}_Variant_WinSym"

        transformed[raw_column] = source_values
        transformed[win_column] = winsorized_values
        transformed[sym_raw_column] = np.sign(source_values) * np.log1p(np.abs(source_values))
        transformed[sym_win_column] = np.sign(winsorized_values) * np.log1p(np.abs(winsorized_values))

    return transformed


def validate_variant_transforms(base_dataset: pd.DataFrame) -> None:
    signed_log_dataset = add_signed_log_bhar_columns(base_dataset)
    if len(signed_log_dataset) != len(base_dataset):
        raise ValueError("Signed-log BHAR transform changed the dataset row count.")

    for source_column, target_column in LOG_BHAR_COLUMN_MAP.items():
        if source_column not in base_dataset.columns:
            continue
        source_values = pd.to_numeric(base_dataset[source_column], errors="coerce")
        transformed_values = pd.to_numeric(signed_log_dataset[target_column], errors="coerce")
        finite_mask = source_values.notna() & np.isfinite(source_values)
        if transformed_values.loc[finite_mask].isna().any():
            raise ValueError(f"Signed-log transform introduced missing values for {source_column}.")

    winsorized_dataset = add_winsorized_bhar_columns(base_dataset)
    if len(winsorized_dataset) != len(base_dataset):
        raise ValueError("Winsorized BHAR transform changed the dataset row count.")

    required_group_columns = {
        MAIN_REGRESSION_GROUP_COLUMN,
        RAW_MAIN_REGRESSION_GROUP_COLUMN,
        *ALTERNATIVE_REGRESSION_GROUP_COLUMNS.values(),
    }
    missing_group_columns = sorted(
        column for column in required_group_columns if column not in base_dataset.columns
    )
    if missing_group_columns:
        raise ValueError(
            "Prepared regression dataset is missing required regression-suite SUE columns: "
            f"{missing_group_columns}."
        )


def build_prior_year_group_columns(
    dataset: pd.DataFrame,
    *,
    group_columns: list[tuple[int, str]],
) -> pd.DataFrame:
    grouped = dataset.copy()
    formation_years = pd.to_numeric(grouped[FORMATION_YEAR_COLUMN], errors="coerce")
    unique_years = sorted(formation_years.dropna().astype(int).unique().tolist())

    for _, column_name in group_columns:
        grouped[column_name] = pd.Series(pd.NA, index=grouped.index, dtype="Int64")

    for year in unique_years:
        current_mask = formation_years.eq(year)
        prior_mask = formation_years.eq(year - 1)
        current_events = grouped.loc[current_mask].copy()
        prior_year_events = grouped.loc[prior_mask].copy()

        for group_count, column_name in group_columns:
            assigned = assign_prior_year_sue_groups(
                current_events,
                prior_year_events,
                group_count=group_count,
            )
            grouped.loc[current_mask, column_name] = assigned[SUE_GROUP_COLUMN].astype("Int64")

    return grouped


def add_market_cap_breakpoint_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    transformed = dataset.copy()
    for breakpoint_config in GRID_MARKET_CAP_BREAKPOINTS:
        working = transformed.drop(
            columns=[
                column
                for column in (
                    ANALYSIS_MARKET_CAP_SPLIT_PERCENTILE_COLUMN,
                    ANALYSIS_MARKET_CAP_SPLIT_BREAKPOINT_COLUMN,
                    ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN,
                )
                if column in transformed.columns
            ],
            errors="ignore",
        )
        split_df = apply_market_cap_analysis_split(
            working,
            data_dir=DATA_DIR,
            split_percentile=float(breakpoint_config["percentile"]),
            market_cap_column=PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN,
        )
        transformed[breakpoint_config["column"]] = pd.Categorical(
            split_df[ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN].to_numpy(),
            categories=split_df[ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN].cat.categories,
            ordered=split_df[ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN].cat.ordered,
        )
    return transformed


def build_market_cap_grid_specs(sample_config: RegressionSampleConfig) -> list[RegressionSpec]:
    specs: list[RegressionSpec] = []
    for base_sample in GRID_BASE_SAMPLES:
        row_filter_query = ""
        if base_sample["apply_market_cap_price_filter"]:
            row_filter_query = (
                f"`{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN}` >= {MARKET_CAP_FILTER_THRESHOLD} and "
                f"`Price` >= {PRICE_FILTER_THRESHOLD}"
            )
        base_sample_label = sample_config.label
        if base_sample["apply_market_cap_price_filter"]:
            base_sample_label = (
                f"{sample_config.label} plus market cap >= $5m and price >= $1"
            )

        for breakpoint_config in GRID_MARKET_CAP_BREAKPOINTS:
            split_column = breakpoint_config["column"]
            specs.append(
                RegressionSpec(
                    key=f"grid_{base_sample['key']}_{breakpoint_config['key']}",
                    family="grid_search",
                    label=f"Grid: {base_sample_label} / {breakpoint_config['label']}",
                    formula=(
                        f"{MAIN_DEPENDENT_VARIABLE} ~ {MAIN_REGRESSION_GROUP_COLUMN} "
                        f"+ C({split_column}) "
                        f"+ {MAIN_REGRESSION_GROUP_COLUMN}:C({split_column}) "
                        f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
                    ),
                    cluster_spec="firm_quarter",
                    row_filter_query=row_filter_query,
                )
            )

    return specs


def build_variant_grid_specs(sample_config: RegressionSampleConfig) -> list[RegressionSpec]:
    specs: list[RegressionSpec] = []
    for base_sample in GRID_BASE_SAMPLES:
        row_filter_query = ""
        if base_sample["apply_market_cap_price_filter"]:
            row_filter_query = (
                f"`{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN}` >= {MARKET_CAP_FILTER_THRESHOLD} and "
                f"`Price` >= {PRICE_FILTER_THRESHOLD}"
            )
        base_sample_label = sample_config.label
        if base_sample["apply_market_cap_price_filter"]:
            base_sample_label = (
                f"{sample_config.label} plus market cap >= $5m and price >= $1"
            )

        for bhar_config in VARIANT_GRID_BHAR_CONFIGS:
            source_column = bhar_config["source_column"]
            dependent_columns = {
                ("raw", "lin"): f"{source_column}_Variant_Raw",
                ("win", "lin"): f"{source_column}_Variant_Win",
                ("raw", "sym"): f"{source_column}_Variant_Sym",
                ("win", "sym"): f"{source_column}_Variant_WinSym",
            }
            for winsor_option in VARIANT_GRID_WINSOR_OPTIONS:
                for log_option in VARIANT_GRID_LOG_OPTIONS:
                    dependent_column = dependent_columns[(winsor_option["key"], log_option["key"])]
                    for sue_split in VARIANT_GRID_SUE_SPLITS:
                        specs.append(
                            RegressionSpec(
                                key=(
                                    f"variant_grid_{base_sample['key']}_{bhar_config['key']}_"
                                    f"{winsor_option['key']}_{log_option['key']}_{sue_split['key']}"
                                ),
                                family="grid_search",
                                label=(
                                    f"Variant grid: {base_sample_label} / {bhar_config['label']} / "
                                    f"{winsor_option['label']} / {log_option['label']} / {sue_split['label']}"
                                ),
                                formula=(
                                    f"{dependent_column} ~ {sue_split['column']} "
                                    f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
                                ),
                                cluster_spec="firm_quarter",
                                row_filter_query=row_filter_query,
                                notes=(
                                    "Winsorization is applied before the SymLog transform when both are enabled."
                                ),
                            )
                        )

    return specs


def build_time_variation_specs(
    *,
    sue_group_count: int = SUE_COMPUTATION_GROUP_COUNT,
) -> list[RegressionSpec]:
    """Build the ordered time-variation specification for one SUE grouping."""
    period_labels = tuple(
        f"{start_year}-{end_year}"
        for start_year, end_year in TIME_PERIOD_BUCKETS
    )
    normalized_group_count = int(sue_group_count)
    regressor_column = (
        MAIN_REGRESSION_GROUP_COLUMN
        if normalized_group_count == int(SUE_COMPUTATION_GROUP_COUNT)
        else ALTERNATIVE_REGRESSION_GROUP_COLUMNS[normalized_group_count]
    )
    regressor_label = describe_sue_group_count(normalized_group_count)
    return [
        RegressionSpec(
            key=TIME_VARIATION_SPEC_KEY,
            family="time_variation",
            label=f"Time variation: early, middle, and late {regressor_label.lower()} slopes",
            formula=(
                f"{MAIN_DEPENDENT_VARIABLE} ~ {regressor_column} "
                f"+ {regressor_column}:C({TIME_PERIOD_COLUMN}) "
                f"+ C({ANNOUNCEMENT_QUARTER_COLUMN}) + C({FIRM_IDENTIFIER_COLUMN})"
            ),
            cluster_spec="firm_quarter",
            notes=(
                "Early, middle, and late formation periods are "
                f"{period_labels[0]}, {period_labels[1]}, and {period_labels[2]}, respectively."
            ),
            ordered_time_periods=period_labels,
            ordered_time_regressor=regressor_column,
            ordered_time_period_column=TIME_PERIOD_COLUMN,
        )
    ]


def build_full_sue_grouping_specs(sue_group_count: int) -> list[RegressionSpec]:
    """Re-estimate every reported Suite 11 specification under one SUE grouping.

    The standalone alternative-SUE baseline rows are omitted because the chosen
    grouping is the regressor throughout this suite, rather than one comparison
    within a quintile-based suite. Methodology-duration specifications are kept
    so the suite remains a complete counterpart to the baseline suite.
    """
    normalized_group_count = int(sue_group_count)
    if normalized_group_count not in ALTERNATIVE_REGRESSION_GROUP_COLUMNS:
        raise ValueError(f"No alternative regression SUE column configured for {normalized_group_count} groups.")

    alternative_group_column = ALTERNATIVE_REGRESSION_GROUP_COLUMNS[normalized_group_count]
    alternative_group_label = describe_sue_group_count(normalized_group_count)
    alternative_split_spec_keys = {
        f"variable_spec_{format_sue_split_key(group_count)}"
        for group_count in ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS
    }
    base_specs = [
        spec
        for spec in REGRESSION_SPECS
        if spec.family not in DISABLED_MAIN_OUTPUT_FAMILIES
        and spec.key not in alternative_split_spec_keys
    ] + build_methodology_continuation_specs() + build_time_variation_specs(
        sue_group_count=normalized_group_count
    )

    return [
        replace(
            spec,
            formula=spec.formula.replace(MAIN_REGRESSION_GROUP_COLUMN, alternative_group_column),
            label=(
                spec.label
                .replace("SUE-quintile", alternative_group_label)
                .replace("SUEQ", alternative_group_label)
                .replace(MAIN_REGRESSION_GROUP_LABEL, alternative_group_label)
            ),
        )
        for spec in base_specs
    ]


def base_dependent_label_map() -> dict[str, str]:
    return {
        bhar_column: format_bhar_label(int(parts[1]), int(parts[2]))
        for bhar_column in REGRESSION_BHAR_COLUMNS
        for parts in [bhar_column.split("_")]
    }


def add_baseline_sample_filter(
    dataset: pd.DataFrame,
    *,
    min_analyst_count: int,
) -> pd.DataFrame:
    return dataset.loc[
        pd.to_numeric(dataset["Forecast_Analyst_Count"], errors="coerce").ge(int(min_analyst_count))
    ].copy()


def build_stratified_firm_split(
    dataset: pd.DataFrame,
    *,
    window_selection_share: float = SPLIT_SAMPLE_WINDOW_SELECTION_SHARE,
    random_seed: int = SPLIT_SAMPLE_RANDOM_SEED,
) -> pd.DataFrame:
    """Assign whole firms to reproducible formation-year-stratified subsamples.

    Firms can have multiple formation-year observations. To avoid assigning one
    firm to both subsamples, its earliest available formation year defines its
    stratum. Half the firms are reserved for drift-window estimation and the
    remaining half are reserved for all BHAR[2,20] analyses.
    """
    if not 0 < window_selection_share < 1:
        raise ValueError("window_selection_share must lie strictly between zero and one.")

    working = dataset.copy()
    formation_year = pd.to_numeric(working[FORMATION_YEAR_COLUMN], errors="coerce")
    if working[FIRM_IDENTIFIER_COLUMN].isna().any() or formation_year.isna().any():
        raise ValueError(
            "The stratified firm split requires non-missing firm identifiers and formation years."
        )

    firm_strata = (
        pd.DataFrame(
            {
                FIRM_IDENTIFIER_COLUMN: working[FIRM_IDENTIFIER_COLUMN],
                SPLIT_SAMPLE_STRATUM_COLUMN: formation_year.astype(int),
            }
        )
        .groupby(FIRM_IDENTIFIER_COLUMN, as_index=False, sort=True)[SPLIT_SAMPLE_STRATUM_COLUMN]
        .min()
    )

    rng = np.random.default_rng(random_seed)
    assignment_rows: list[dict[str, object]] = []
    for stratum, stratum_firms in firm_strata.groupby(SPLIT_SAMPLE_STRATUM_COLUMN, sort=True):
        firm_ids = stratum_firms[FIRM_IDENTIFIER_COLUMN].to_numpy(copy=True)
        rng.shuffle(firm_ids)
        firm_count = len(firm_ids)
        if firm_count == 1:
            window_firm_count = int(rng.random() < window_selection_share)
        else:
            window_firm_count = int(np.floor(firm_count * window_selection_share + 0.5))
            window_firm_count = min(max(window_firm_count, 1), firm_count - 1)

        assignment_rows.extend(
            {
                FIRM_IDENTIFIER_COLUMN: firm_id,
                SPLIT_SAMPLE_STRATUM_COLUMN: int(stratum),
                SPLIT_SAMPLE_ASSIGNMENT_COLUMN: (
                    "window_selection_50pct"
                    if position < window_firm_count
                    else "analysis_50pct"
                ),
            }
            for position, firm_id in enumerate(firm_ids)
        )

    assignments = pd.DataFrame(assignment_rows).set_index(FIRM_IDENTIFIER_COLUMN)
    working[SPLIT_SAMPLE_ASSIGNMENT_COLUMN] = working[FIRM_IDENTIFIER_COLUMN].map(
        assignments[SPLIT_SAMPLE_ASSIGNMENT_COLUMN]
    )
    working[SPLIT_SAMPLE_STRATUM_COLUMN] = working[FIRM_IDENTIFIER_COLUMN].map(
        assignments[SPLIT_SAMPLE_STRATUM_COLUMN]
    )
    if working[SPLIT_SAMPLE_ASSIGNMENT_COLUMN].isna().any():
        raise AssertionError("At least one firm did not receive a split-sample assignment.")
    return working


def select_split_sample_group(dataset: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return one firm-disjoint subsample from the deterministic stratified split."""
    assigned = build_stratified_firm_split(dataset)
    return assigned.loc[assigned[SPLIT_SAMPLE_ASSIGNMENT_COLUMN].eq(group)].copy()


def add_centered_firm_average_log_market_cap(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add a year ordinal and a fixed firm-level average log market capitalization."""
    working = dataset.copy()
    formation_year = pd.to_numeric(working[FORMATION_YEAR_COLUMN], errors="coerce")
    working[YEAR_ORDINAL_COLUMN] = formation_year - TIME_TREND_REFERENCE_YEAR
    market_cap = pd.to_numeric(working[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce")
    valid = market_cap.gt(0)
    if not bool(valid.any()):
        working[FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN] = np.nan
        working[CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN] = np.nan
        working[FIRM_SIZE_MIDDLE_PERIOD_INTERACTION_COLUMN] = np.nan
        working[FIRM_SIZE_LATE_PERIOD_INTERACTION_COLUMN] = np.nan
        return working

    log_market_cap = pd.Series(np.where(valid, np.log(market_cap), np.nan), index=working.index)
    firm_average_log_market_cap = log_market_cap.groupby(
        working[FIRM_IDENTIFIER_COLUMN]
    ).transform("mean")
    working[FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN] = firm_average_log_market_cap
    firm_level_log_market_cap = firm_average_log_market_cap.groupby(
        working[FIRM_IDENTIFIER_COLUMN]
    ).first().dropna()
    working[CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN] = (
        working[FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN]
        - firm_level_log_market_cap.mean()
    )
    centered_firm_size = working[CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN]
    working[FIRM_SIZE_MIDDLE_PERIOD_INTERACTION_COLUMN] = (
        centered_firm_size * working[TIME_PERIOD_COLUMN].eq(MIDDLE_TIME_PERIOD_LABEL)
    )
    working[FIRM_SIZE_LATE_PERIOD_INTERACTION_COLUMN] = (
        centered_firm_size * working[TIME_PERIOD_COLUMN].eq(LATE_TIME_PERIOD_LABEL)
    )
    return working


def add_post_2000_baseline_sample_filter(
    dataset: pd.DataFrame,
    *,
    min_analyst_count: int,
) -> pd.DataFrame:
    analyst_filtered = add_baseline_sample_filter(
        dataset,
        min_analyst_count=min_analyst_count,
    )
    return analyst_filtered.loc[
        pd.to_numeric(analyst_filtered[FORMATION_YEAR_COLUMN], errors="coerce").ge(MIN_SAMPLE_YEAR)
    ].copy()


def add_post_2000_market_cap_price_filter(
    dataset: pd.DataFrame,
    *,
    min_analyst_count: int,
) -> pd.DataFrame:
    filtered = add_post_2000_baseline_sample_filter(
        dataset,
        min_analyst_count=min_analyst_count,
    )
    return filtered.loc[
        pd.to_numeric(filtered[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce").ge(MARKET_CAP_FILTER_THRESHOLD)
        & pd.to_numeric(filtered["Price"], errors="coerce").ge(PRICE_FILTER_THRESHOLD)
    ].copy()


def build_market_cap_price_filter_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize LSE-adjusted and U.S.-reference pre-announcement screens.

    The regression dataset is earnings-announcement-event level.  The audit
    therefore reports both event counts and unique stock-year counts, so that
    pre-announcement screen effects are not confused with regression sample size.
    """
    market_cap = pd.to_numeric(dataset[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce")
    price = pd.to_numeric(dataset["Price"], errors="coerce")
    stock_year_columns = ["Instrument", FORMATION_YEAR_COLUMN]

    screens = [
        (
            "Candidate regression dataset",
            "No pre-announcement market-capitalization or price screen",
            pd.Series(True, index=dataset.index),
            np.nan,
            np.nan,
        ),
        (
            "LSE-adjusted market-capitalization screen",
            f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {LSE_MARKET_CAP_FILTER_THRESHOLD} USD million",
            market_cap.ge(LSE_MARKET_CAP_FILTER_THRESHOLD),
            LSE_MARKET_CAP_FILTER_THRESHOLD,
            np.nan,
        ),
        (
            "LSE-adjusted combined screen",
            f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {LSE_MARKET_CAP_FILTER_THRESHOLD} USD million and "
            f"Price >= {LSE_PRICE_FILTER_THRESHOLD} USD",
            market_cap.ge(LSE_MARKET_CAP_FILTER_THRESHOLD)
            & price.ge(LSE_PRICE_FILTER_THRESHOLD),
            LSE_MARKET_CAP_FILTER_THRESHOLD,
            LSE_PRICE_FILTER_THRESHOLD,
        ),
        (
            "U.S.-reference market-capitalization screen",
            f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD} USD million",
            market_cap.ge(MARKET_CAP_FILTER_THRESHOLD),
            MARKET_CAP_FILTER_THRESHOLD,
            np.nan,
        ),
        (
            "U.S.-reference combined screen",
            f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD} USD million and "
            f"Price >= {PRICE_FILTER_THRESHOLD} USD",
            market_cap.ge(MARKET_CAP_FILTER_THRESHOLD)
            & price.ge(PRICE_FILTER_THRESHOLD),
            MARKET_CAP_FILTER_THRESHOLD,
            PRICE_FILTER_THRESHOLD,
        ),
    ]

    initial_event_count = int(len(dataset))
    initial_stock_year_count = int(dataset.loc[:, stock_year_columns].drop_duplicates().shape[0])
    rows: list[dict[str, object]] = []
    for stage_order, (screen_name, criterion, mask, market_cap_threshold, price_threshold) in enumerate(
        screens,
        start=1,
    ):
        retained = dataset.loc[mask].copy()
        event_count = int(len(retained))
        stock_year_count = int(retained.loc[:, stock_year_columns].drop_duplicates().shape[0])
        rows.append(
            {
                "Observation_Unit": "Earnings-announcement events and unique stock-years",
                "Stage_Order": stage_order,
                "Screen_Name": screen_name,
                "Filter_Criterion": criterion,
                "Market_Cap_Threshold_USD_Millions": market_cap_threshold,
                "Price_Threshold_USD": price_threshold,
                "Events_Retained": event_count,
                "Events_Excluded_Cumulative": initial_event_count - event_count,
                "Event_Retention_Percent": 100 * event_count / initial_event_count,
                "Stock_Years_Retained": stock_year_count,
                "Stock_Years_Excluded_Cumulative": initial_stock_year_count - stock_year_count,
                "Stock_Year_Retention_Percent": 100 * stock_year_count / initial_stock_year_count,
                "Unique_Firms_Retained": int(retained[FIRM_IDENTIFIER_COLUMN].nunique(dropna=True)),
                "Missing_Market_Cap_Events": int(market_cap.loc[mask].isna().sum()),
                "Missing_Price_Events": int(price.loc[mask].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _format_time_variation_number(value: object, *, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def build_time_variation_tests_latex(diagnostics: pd.Series) -> str:
    joint_statistic = _format_time_variation_number(diagnostics.get("Time_Variation_Joint_F_Statistic"))
    joint_p_value = _format_time_variation_number(diagnostics.get("Time_Variation_Joint_F_p_value"))
    joint_numerator_df = _format_time_variation_number(
        diagnostics.get("Time_Variation_Joint_F_DF_Numerator"), decimals=0
    )
    joint_denominator_df = _format_time_variation_number(
        diagnostics.get("Time_Variation_Joint_F_DF_Denominator"), decimals=0
    )
    middle_estimate = _format_time_variation_number(diagnostics.get("Middle_vs_Early_Estimate"))
    middle_std_error = _format_time_variation_number(diagnostics.get("Middle_vs_Early_Std_Error"))
    middle_t_statistic = _format_time_variation_number(diagnostics.get("Middle_vs_Early_t_Statistic"))
    middle_p_value = _format_time_variation_number(
        diagnostics.get("Middle_vs_Early_One_Sided_p_value")
    )
    late_estimate = _format_time_variation_number(diagnostics.get("Late_vs_Middle_Estimate"))
    late_std_error = _format_time_variation_number(diagnostics.get("Late_vs_Middle_Std_Error"))
    late_t_statistic = _format_time_variation_number(diagnostics.get("Late_vs_Middle_t_Statistic"))
    late_p_value = _format_time_variation_number(
        diagnostics.get("Late_vs_Middle_One_Sided_p_value")
    )

    lines = [
        r"\begingroup",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Test & Estimate & Clustered SE & Statistic & $p$-value \\",
        r"\hline",
        (
            "F-test "
            f"($F_{{{joint_numerator_df},{joint_denominator_df}}}$) &  &  & "
            f"{joint_statistic} & {joint_p_value} \\\\"
        ),
        fr"Middle versus early ($\theta_M < 0$) & {middle_estimate} & {middle_std_error} & {middle_t_statistic} & {middle_p_value} \\",
        fr"Late versus middle ($\theta_L-\theta_M < 0$) & {late_estimate} & {late_std_error} & {late_t_statistic} & {late_p_value} \\",
        r"\hline",
        r"\end{tabular}",
        r"\endgroup",
    ]
    return "\n".join(lines)


def build_time_variation_results_latex(diagnostics: pd.Series) -> str:
    """Build the Results-chapter fragment from the specified H2 and H2a tests."""
    lines = [
        r"\section{Time Variation and Attenuation of \ac{PEAD}}",
        r"\label{sec:time-variation-results}",
        (
            "The following table reports the joint test of whether the middle- and late-period "
            "ranked-\ac{SUE} slopes differ from the early-period slope (H2), followed by the "
            "two one-sided adjacent-period contrasts for monotonic attenuation (H2a)."
        ),
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption[Time variation and attenuation tests]{\textbf{Time variation and attenuation tests.} "
        r"The F-test evaluates $H_0^{\mathrm{time}}:\theta_M=\theta_L=0$. "
        r"The subsequent one-sided tests evaluate $H_A^{\mathrm{middle\text{-}early}}:\theta_M<0$ "
        r"and $H_A^{\mathrm{late\text{-}middle}}:\theta_L-\theta_M<0$. Standard errors are "
        r"two-way clustered by firm and announcement year-quarter. Reported F-tests are based on the "
        r"corresponding two-way cluster-robust covariance matrix.}",
        r"\label{tab:time-variation-attenuation-tests}",
        build_time_variation_tests_latex(diagnostics),
        r"\end{table}",
    ]
    return "\n".join(lines)


def _parameter_covariance(result, parameter_names: list[str]) -> np.ndarray:
    """Return the fitted covariance matrix aligned to ``parameter_names``."""
    covariance = getattr(result, "cov", None)
    if covariance is None and hasattr(result, "cov_params"):
        covariance = result.cov_params()
    if covariance is None:
        raise ValueError("The fitted model does not expose a parameter covariance matrix.")
    if isinstance(covariance, pd.DataFrame):
        return covariance.loc[parameter_names, parameter_names].to_numpy(dtype=float)

    covariance_array = np.asarray(covariance, dtype=float)
    expected_shape = (len(parameter_names), len(parameter_names))
    if covariance_array.shape != expected_shape:
        raise ValueError(
            "The parameter covariance matrix has shape "
            f"{covariance_array.shape}, expected {expected_shape}."
        )
    return covariance_array


def _linear_combination_summary(
    *,
    parameter_values: np.ndarray,
    covariance: np.ndarray,
    contrast: np.ndarray,
    degrees_of_freedom: float,
) -> dict[str, float]:
    """Summarise a coefficient linear combination using clustered covariance."""
    estimate = float(contrast @ parameter_values)
    variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The linear-combination variance must be finite and positive.")

    standard_error = float(np.sqrt(variance))
    t_statistic = estimate / standard_error
    if np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0:
        two_sided_p_value = float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
        one_sided_p_value_greater = float(stats.t.sf(t_statistic, degrees_of_freedom))
        one_sided_p_value_less = float(stats.t.cdf(t_statistic, degrees_of_freedom))
    else:
        two_sided_p_value = float(2.0 * stats.norm.sf(abs(t_statistic)))
        one_sided_p_value_greater = float(stats.norm.sf(t_statistic))
        one_sided_p_value_less = float(stats.norm.cdf(t_statistic))
    return {
        "Estimate": estimate,
        "Clustered_SE": standard_error,
        "t_Statistic": t_statistic,
        "Two_Sided_p_value": two_sided_p_value,
        "One_Sided_p_value_Greater": one_sided_p_value_greater,
        "One_Sided_p_value_Less": one_sided_p_value_less,
    }


def _find_time_interaction_term(
    parameter_names: list[str],
    *,
    period_label: str,
    regressor_column: str = MAIN_REGRESSION_GROUP_COLUMN,
) -> str:
    period_term = f"C({TIME_PERIOD_COLUMN})[T.{period_label}]"
    candidates = (
        f"{regressor_column}:{period_term}",
        f"{period_term}:{regressor_column}",
    )
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    raise ValueError(
        "Could not locate the SUE-quintile interaction term for "
        f"time period {period_label!r}."
    )


def _time_variation_reporting_model(
    dataset: pd.DataFrame,
    *,
    spec: RegressionSpec | None = None,
) -> dict[str, object]:
    """Refit the designated time-variation model for derived reporting outputs.

    The reusable regression-suite cache stores coefficient summaries but not their
    covariance matrices. Re-fitting this single model allows period-specific
    slopes to use the correct covariance of the base slope and interaction terms.
    """
    if spec is None:
        spec = build_time_variation_specs()[0]
    return fit_formula_model(
        dataset,
        formula=spec.formula,
        model_label=spec.label,
        cluster_spec=spec.cluster_spec,
    )


def _h3_size_time_reporting_model(dataset: pd.DataFrame) -> dict[str, object]:
    """Refit the H3 size--time model with its covariance matrix available."""
    spec = next(spec for spec in REGRESSION_SPECS if spec.key == H3_SIZE_TIME_SPEC_KEY)
    working_dataset = (
        dataset.query(spec.row_filter_query).copy()
        if spec.row_filter_query is not None
        else dataset.copy()
    )
    return fit_formula_model(
        working_dataset,
        formula=spec.formula,
        model_label=spec.label,
        cluster_spec=spec.cluster_spec,
    )


def _find_h3_triple_interaction_term(
    parameter_names: list[str],
    *,
    period_label: str,
) -> str:
    """Locate the H3 SUEQ-by-size-by-period interaction for one period."""
    period_term = f"C({TIME_PERIOD_COLUMN})[T.{period_label}]"
    matching_terms = [
        term
        for term in parameter_names
        if MAIN_REGRESSION_GROUP_COLUMN in term
        and CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN in term
        and period_term in term
    ]
    if len(matching_terms) != 1:
        raise ValueError(
            "Expected one H3 triple-interaction term for period "
            f"{period_label!r}, found {matching_terms!r}."
        )
    return matching_terms[0]


def _find_h3_base_size_interaction_term(parameter_names: list[str]) -> str:
    """Locate the early-period SUEQ-by-size interaction in the H3 model."""
    matching_terms = [
        term
        for term in parameter_names
        if MAIN_REGRESSION_GROUP_COLUMN in term
        and CENTERED_FIRM_AVERAGE_LOG_MARKET_CAP_COLUMN in term
        and f"C({TIME_PERIOD_COLUMN})" not in term
    ]
    if len(matching_terms) != 1:
        raise ValueError(
            "Expected one H3 base SUEQ-by-size interaction term, "
            f"found {matching_terms!r}."
        )
    return matching_terms[0]


def _build_h3_joint_test_result(
    *,
    result: object,
    analysis_df: pd.DataFrame,
    parameter_names: list[str],
    tested_terms: list[str],
    test: str,
    alternative: str,
) -> dict[str, object]:
    """Build one F-test for a set of H3 interaction terms."""
    tested_indices = [parameter_names.index(term) for term in tested_terms]
    parameter_values = result.params.loc[parameter_names].to_numpy(dtype=float)
    covariance = _parameter_covariance(result, parameter_names)
    restricted_values = parameter_values[tested_indices]
    restricted_covariance = covariance[np.ix_(tested_indices, tested_indices)]
    numerator_df = int(np.linalg.matrix_rank(restricted_covariance))
    denominator_df = _as_float(getattr(result, "df_resid", np.nan))
    if numerator_df <= 0 or not np.isfinite(denominator_df) or denominator_df <= 0:
        raise ValueError("The H3 joint test has invalid degrees of freedom.")
    restriction_quadratic = float(
        restricted_values.T @ np.linalg.pinv(restricted_covariance) @ restricted_values
    )
    f_statistic = restriction_quadratic / numerator_df
    p_value = float(stats.f.sf(f_statistic, numerator_df, denominator_df))
    return {
        "Test": test,
        "Alternative": alternative,
        "Statistic": f_statistic,
        "p_value": p_value,
        "DF_Numerator": numerator_df,
        "DF_Denominator": denominator_df,
        "Tested_Terms": " | ".join(tested_terms),
        "N": int(len(analysis_df)),
        "Firm_Cluster_Count": int(analysis_df[FIRM_IDENTIFIER_COLUMN].nunique()),
        "Quarter_Cluster_Count": int(analysis_df[ANNOUNCEMENT_QUARTER_COLUMN].nunique()),
        "Status": "completed",
    }


def build_h3_joint_tests(
    dataset: pd.DataFrame,
    *,
    model_output: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Test any and time-varying firm-size moderation from the full H3 model."""
    test_definitions = (
        (
            "H3a: any firm-size moderation",
            r"$\gamma_B \neq 0$ or $\gamma_M \neq 0$ or $\gamma_L \neq 0$",
        ),
        (
            "H3b: joint time variation in firm-size moderation",
            r"$\gamma_M \neq 0$ or $\gamma_L \neq 0$",
        ),
    )
    try:
        if model_output is None:
            model_output = _h3_size_time_reporting_model(dataset)
        result = model_output["result"]
        analysis_df = model_output["data"]
        parameter_names = [str(name) for name in result.params.index]
        triple_interaction_terms = [
            _find_h3_triple_interaction_term(parameter_names, period_label=period_label)
            for period_label in tuple(f"{start}-{end}" for start, end in TIME_PERIOD_BUCKETS)[1:]
        ]
        base_size_interaction_term = _find_h3_base_size_interaction_term(parameter_names)
        return [
            _build_h3_joint_test_result(
                result=result,
                analysis_df=analysis_df,
                parameter_names=parameter_names,
                tested_terms=[base_size_interaction_term, *triple_interaction_terms],
                test=test_definitions[0][0],
                alternative=test_definitions[0][1],
            ),
            _build_h3_joint_test_result(
                result=result,
                analysis_df=analysis_df,
                parameter_names=parameter_names,
                tested_terms=triple_interaction_terms,
                test=test_definitions[1][0],
                alternative=test_definitions[1][1],
            ),
        ]
    except Exception as exc:
        return [
            {
                "Test": test,
                "Alternative": alternative,
                "Statistic": np.nan,
                "p_value": np.nan,
                "DF_Numerator": np.nan,
                "DF_Denominator": np.nan,
                "Tested_Terms": "",
                "N": np.nan,
                "Firm_Cluster_Count": np.nan,
                "Quarter_Cluster_Count": np.nan,
                "Status": f"failed: {type(exc).__name__}: {exc}",
            }
            for test, alternative in test_definitions
        ]


def build_time_variation_period_slopes(
    dataset: pd.DataFrame,
    *,
    spec: RegressionSpec | None = None,
) -> pd.DataFrame:
    """Build clustered early-, middle-, and late-period SUEQ slope estimates."""
    if spec is None:
        spec = build_time_variation_specs()[0]
    model_output = _time_variation_reporting_model(dataset, spec=spec)
    result = model_output["result"]
    analysis_df = model_output["data"]
    parameter_names = [str(name) for name in result.params.index]
    parameter_values = result.params.loc[parameter_names].to_numpy(dtype=float)
    covariance = _parameter_covariance(result, parameter_names)
    degrees_of_freedom = _as_float(getattr(result, "df_resid", np.nan))

    regressor_column = spec.ordered_time_regressor or MAIN_REGRESSION_GROUP_COLUMN
    period_labels = spec.ordered_time_periods or tuple(
        f"{start}-{end}" for start, end in TIME_PERIOD_BUCKETS
    )
    period_column = spec.ordered_time_period_column or TIME_PERIOD_COLUMN
    base_index = parameter_names.index(regressor_column)
    contrasts: dict[str, np.ndarray] = {}

    early_contrast = np.zeros(len(parameter_names))
    early_contrast[base_index] = 1.0
    contrasts[period_labels[0]] = early_contrast

    for period_label in period_labels[1:]:
        contrast = early_contrast.copy()
        interaction_index = parameter_names.index(
            _find_time_interaction_term(
                parameter_names,
                period_label=period_label,
                regressor_column=regressor_column,
            )
        )
        contrast[interaction_index] = 1.0
        contrasts[period_label] = contrast

    rows: list[dict[str, object]] = []
    for period_order, period_label in enumerate(period_labels, start=1):
        period_data = analysis_df.loc[analysis_df[period_column].eq(period_label)]
        summary = _linear_combination_summary(
            parameter_values=parameter_values,
            covariance=covariance,
            contrast=contrasts[period_label],
            degrees_of_freedom=degrees_of_freedom,
        )
        rows.append(
            {
                "Period": period_label,
                "Period_Order": period_order,
                "Event_Count": int(len(period_data)),
                "Firm_Count": int(period_data[FIRM_IDENTIFIER_COLUMN].nunique()),
                "Reported_p_value_type": "one-sided greater than zero",
                "Reported_p_value": summary["One_Sided_p_value_Greater"],
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_h3_period_specific_effects(
    dataset: pd.DataFrame,
    *,
    model_output: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build covariance-correct H3 period slopes and firm-size gradients.

    Panel A gives the ranked-SUE slope at centered mean firm size. Panel B gives
    the change in that slope for a one-unit increase in firm-average log market
    capitalization. Both are linear combinations of the fully interacted H3
    specification.
    """
    if model_output is None:
        model_output = _h3_size_time_reporting_model(dataset)
    result = model_output["result"]
    analysis_df = model_output["data"]
    parameter_names = [str(name) for name in result.params.index]
    parameter_values = result.params.loc[parameter_names].to_numpy(dtype=float)
    covariance = _parameter_covariance(result, parameter_names)
    degrees_of_freedom = _as_float(getattr(result, "df_resid", np.nan))
    period_labels = tuple(f"{start}-{end}" for start, end in TIME_PERIOD_BUCKETS)

    base_sue_index = parameter_names.index(MAIN_REGRESSION_GROUP_COLUMN)
    base_size_index = parameter_names.index(
        _find_h3_base_size_interaction_term(parameter_names)
    )
    base_sue_contrast = np.zeros(len(parameter_names))
    base_sue_contrast[base_sue_index] = 1.0
    base_size_contrast = np.zeros(len(parameter_names))
    base_size_contrast[base_size_index] = 1.0

    slope_contrasts = {period_labels[0]: base_sue_contrast}
    gradient_contrasts = {period_labels[0]: base_size_contrast}
    for period_label in period_labels[1:]:
        slope_contrast = base_sue_contrast.copy()
        slope_contrast[parameter_names.index(
            _find_time_interaction_term(parameter_names, period_label=period_label)
        )] = 1.0
        slope_contrasts[period_label] = slope_contrast

        gradient_contrast = base_size_contrast.copy()
        gradient_contrast[parameter_names.index(
            _find_h3_triple_interaction_term(parameter_names, period_label=period_label)
        )] = 1.0
        gradient_contrasts[period_label] = gradient_contrast

    rows: list[dict[str, object]] = []
    effect_definitions = (
        (
            "A: Ranked-SUE slope at centered mean firm size",
            "Ranked-SUE slope",
            slope_contrasts,
            "one-sided greater than zero",
            (r"beta", r"beta + theta_M", r"beta + theta_L"),
        ),
        (
            "B: Firm-size gradient in the ranked-SUE slope",
            "Firm-size gradient",
            gradient_contrasts,
            "two-sided different from zero",
            (r"gamma", r"gamma + delta_M", r"gamma + delta_L"),
        ),
    )
    for (
        panel,
        effect,
        contrasts,
        reported_p_value_type,
        coefficient_combinations,
    ) in effect_definitions:
        for period_order, period_label in enumerate(period_labels, start=1):
            period_data = analysis_df.loc[analysis_df[TIME_PERIOD_COLUMN].eq(period_label)]
            summary = _linear_combination_summary(
                parameter_values=parameter_values,
                covariance=covariance,
                contrast=contrasts[period_label],
                degrees_of_freedom=degrees_of_freedom,
            )
            reported_p_value = (
                summary["One_Sided_p_value_Greater"]
                if panel.startswith("A:")
                else summary["Two_Sided_p_value"]
            )
            rows.append(
                {
                    "Panel": panel,
                    "Effect": effect,
                    "Period": period_label,
                    "Period_Order": period_order,
                    "Coefficient_combination": coefficient_combinations[period_order - 1],
                    "Event_Count": int(len(period_data)),
                    "Firm_Count": int(period_data[FIRM_IDENTIFIER_COLUMN].nunique()),
                    "Reported_p_value_type": reported_p_value_type,
                    "Reported_p_value": reported_p_value,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def save_time_variation_outputs(
    output_dir: Path,
    dataset: pd.DataFrame,
    suite_results: dict[str, object],
    *,
    time_variation_spec: RegressionSpec | None = None,
) -> None:
    diagnostics = suite_results["diagnostics"]
    if diagnostics.empty or "Spec_Key" not in diagnostics.columns:
        return
    matching = diagnostics.loc[diagnostics["Spec_Key"].eq(TIME_VARIATION_SPEC_KEY)].copy()
    if matching.empty:
        return
    result_row = matching.iloc[0]
    _save_table_to_dir(output_dir, matching, "time_variation_tests")
    _save_latex_to_dir(
        output_dir,
        "time_variation_tests",
        build_time_variation_tests_latex(result_row),
    )
    period_slopes = build_time_variation_period_slopes(
        dataset,
        spec=time_variation_spec,
    )
    _save_table_to_dir(output_dir, period_slopes, "time_variation_period_slopes")
    if (
        output_dir == OUTPUT_DIR / BASELINE_SUITE_LABEL
        and result_row.get("Time_Variation_Tests_Status") == "completed"
    ):
        TIME_VARIATION_RESULTS_TEX_PATH.write_text(
            build_time_variation_results_latex(result_row),
            encoding="utf-8",
        )


def build_time_variation_output_callback(
    time_variation_spec: RegressionSpec,
) -> Callable[[Path, pd.DataFrame, dict[str, object]], None]:
    """Return a callback that preserves a suite's transformed time-variation spec."""

    def callback(
        output_dir: Path,
        dataset: pd.DataFrame,
        suite_results: dict[str, object],
    ) -> None:
        save_time_variation_outputs(
            output_dir,
            dataset,
            suite_results,
            time_variation_spec=time_variation_spec,
        )

    return callback


def _as_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def _one_sided_p_value(
    estimate: object,
    two_sided_p_value: object,
    *,
    alternative: str,
) -> float:
    """Convert a symmetric two-sided regression p-value to its directional form."""
    estimate_value = _as_float(estimate)
    p_value = _as_float(two_sided_p_value)
    if not np.isfinite(estimate_value) or not np.isfinite(p_value):
        return np.nan
    half_p_value = min(max(p_value, 0.0), 1.0) / 2.0
    if alternative == "greater":
        return half_p_value if estimate_value > 0 else 1.0 - half_p_value
    if alternative == "less":
        return half_p_value if estimate_value < 0 else 1.0 - half_p_value
    raise ValueError(f"Unsupported one-sided alternative: {alternative!r}")


def _coefficient_row(
    coefficients: pd.DataFrame,
    *,
    spec_key: str,
    term: str,
    contains: bool = False,
) -> pd.Series | None:
    if coefficients.empty:
        return None
    matching = coefficients.loc[coefficients["Spec_Key"].eq(spec_key)].copy()
    if contains:
        matching = matching.loc[matching["Term"].astype(str).str.contains(term, regex=False)]
    else:
        matching = matching.loc[matching["Term"].eq(term)]
    return matching.iloc[0] if not matching.empty else None


def _diagnostics_row(diagnostics: pd.DataFrame, spec_key: str) -> pd.Series | None:
    if diagnostics.empty:
        return None
    matching = diagnostics.loc[diagnostics["Spec_Key"].eq(spec_key)]
    return matching.iloc[0] if not matching.empty else None


def build_unbiasedness_regression_results(dataset: pd.DataFrame) -> pd.DataFrame:
    """Export the unrestricted unbiasedness regression and its slope restriction."""
    model_output = fit_simple_ols(
        dataset,
        x_col=ANNOUNCEMENT_WINDOW_COLUMN,
        y_col=FULL_WINDOW_COLUMN,
    )
    coefficients = model_output["coefficients"].copy()
    tests = build_unbiasedness_tests(model_output)
    model_summary = model_output["model"]

    rows: list[dict[str, object]] = []
    for _, coefficient in coefficients.iterrows():
        term = str(coefficient["Term"])
        if term == "Intercept":
            restriction_null = _as_float(tests["Intercept_Null"])
            restriction_statistic = _as_float(tests["Intercept_t_stat"])
            restriction_two_sided_p_value = _as_float(tests["Intercept_p_value"])
            restriction_one_sided_p_value = np.nan
            restriction_alternative = "Intercept != 0"
        else:
            restriction_null = _as_float(tests["Slope_Null"])
            restriction_statistic = _as_float(tests["Slope_t_stat"])
            restriction_two_sided_p_value = _as_float(tests["Slope_p_value"])
            restriction_one_sided_p_value = _as_float(
                tests["Slope_greater_than_null_p_value"]
            )
            restriction_alternative = "Slope > 1"

        rows.append(
            {
                "Term": term,
                "Estimate": _as_float(coefficient["Coefficient"]),
                "Clustered_SE": _as_float(coefficient["Std_Error"]),
                "Coefficient_t_Statistic": _as_float(coefficient["t_stat"]),
                "Coefficient_Two_Sided_p_value": _as_float(coefficient["p_value"]),
                "Restriction_Null": restriction_null,
                "Restriction_Alternative": restriction_alternative,
                "Restriction_t_Statistic": restriction_statistic,
                "Restriction_Two_Sided_p_value": restriction_two_sided_p_value,
                "Restriction_One_Sided_p_value": restriction_one_sided_p_value,
                "N": int(model_summary["N"]),
                "Firm_Cluster_Count": int(model_summary["Firm_Cluster_Count"]),
                "Quarter_Cluster_Count": int(model_summary["Quarter_Cluster_Count"]),
            }
        )
    return pd.DataFrame(rows)


def _regression_test_row(
    *,
    test: str,
    alternative: str,
    coefficient: pd.Series | None,
    diagnostics: pd.Series | None,
    direction: str,
    null_value: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, object]:
    if coefficient is None:
        return {
            "Test_Group": "Regression",
            "Test": test,
            "Alternative": alternative,
            "Estimate": np.nan,
            "Clustered_SE": np.nan,
            "Statistic": np.nan,
            "p_value": np.nan,
            "N": np.nan,
            "Inference": "Two-way clustered by firm and announcement year-quarter",
            "Alpha": alpha,
            "Reject_Null": np.nan,
            "Status": "unavailable: coefficient not estimated",
        }

    estimate = _as_float(coefficient.get("Coefficient"))
    standard_error = _as_float(coefficient.get("Std_Error"))
    statistic = (estimate - null_value) / standard_error if standard_error > 0 else np.nan
    p_value = _one_sided_p_value(
        estimate - null_value,
        coefficient.get("p_value"),
        alternative=direction,
    )
    sample_size = _as_float(diagnostics.get("Sample_Size")) if diagnostics is not None else np.nan
    return {
        "Test_Group": "Regression",
        "Test": test,
        "Alternative": alternative,
        "Estimate": estimate,
        "Clustered_SE": standard_error,
        "Statistic": statistic,
        "p_value": p_value,
        "N": sample_size,
        "Inference": "Two-way clustered by firm and announcement year-quarter",
        "Alpha": alpha,
        "Reject_Null": bool(p_value < alpha) if np.isfinite(p_value) else np.nan,
        "Status": "completed",
    }


def _sign_test_row(
    *,
    test: str,
    alternative: str,
    sample: pd.Series,
    direction: str,
) -> dict[str, object]:
    nonzero_sample = pd.to_numeric(sample, errors="coerce").dropna()
    nonzero_sample = nonzero_sample.loc[nonzero_sample.ne(0)]
    if nonzero_sample.empty:
        return {
            "Test_Group": "Sign test",
            "Test": test,
            "Alternative": alternative,
            "Estimate": np.nan,
            "Clustered_SE": np.nan,
            "Statistic": np.nan,
            "p_value": np.nan,
            "N": 0,
            "Inference": "One-sided exact binomial sign test",
            "Alpha": 0.05,
            "Reject_Null": np.nan,
            "Status": "unavailable: no non-zero observations",
        }

    positive_count = int(nonzero_sample.gt(0).sum())
    result = stats.binomtest(
        k=positive_count,
        n=len(nonzero_sample),
        p=0.5,
        alternative=direction,
    )
    return {
        "Test_Group": "Sign test",
        "Test": test,
        "Alternative": alternative,
        "Estimate": float(nonzero_sample.gt(0).mean()),
        "Clustered_SE": np.nan,
        "Statistic": positive_count,
        "p_value": float(result.pvalue),
        "N": int(len(nonzero_sample)),
        "Inference": "One-sided exact binomial sign test",
        "Alpha": 0.05,
        "Reject_Null": bool(result.pvalue < 0.05),
        "Status": "completed",
    }


def build_methodology_hypothesis_tests(
    dataset: pd.DataFrame,
    suite_results: dict[str, object],
    *,
    unbiasedness_dataset: pd.DataFrame | None = None,
    h3_joint_tests: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Run and collect every formal test specified in the methodology chapter."""
    coefficients = suite_results["coefficients"]
    diagnostics = suite_results["diagnostics"]
    rows: list[dict[str, object]] = []

    main_coefficient = _coefficient_row(
        coefficients,
        spec_key="main_regression",
        term=MAIN_REGRESSION_GROUP_COLUMN,
    )
    rows.append(
        _regression_test_row(
            test="H1: initial drift interval",
            alternative=r"$\beta_{[2,20]} > 0$",
            coefficient=main_coefficient,
            diagnostics=_diagnostics_row(diagnostics, "main_regression"),
            direction="greater",
            alpha=H1_TEST_ALPHA,
        )
    )

    for interval in METHODOLOGY_CONTINUATION_INTERVALS:
        day_start, day_end = interval
        spec_key = METHODOLOGY_CONTINUATION_SPEC_KEYS[interval]
        rows.append(
            _regression_test_row(
                test=f"Duration continuation: [{day_start},{day_end}]",
                alternative=fr"$\beta_{{[{day_start},{day_end}]}} > 0$",
                coefficient=_coefficient_row(
                    coefficients,
                    spec_key=spec_key,
                    term=MAIN_REGRESSION_GROUP_COLUMN,
                ),
                diagnostics=_diagnostics_row(diagnostics, spec_key),
                direction="greater",
                alpha=CONTINUATION_TEST_ALPHA,
            )
        )

    bottom_group = 1
    top_group = int(SUE_COMPUTATION_GROUP_COUNT)
    rows.append(
        _sign_test_row(
            test="Q1 BHAR sign",
            alternative=r"$P(\mathrm{BHAR}>0) < 0.5$",
            sample=dataset.loc[
                dataset[RAW_MAIN_REGRESSION_GROUP_COLUMN].eq(bottom_group), MAIN_DEPENDENT_VARIABLE
            ],
            direction="less",
        )
    )
    rows.append(
        _sign_test_row(
            test="Q5 BHAR sign",
            alternative=r"$P(\mathrm{BHAR}>0) > 0.5$",
            sample=dataset.loc[
                dataset[RAW_MAIN_REGRESSION_GROUP_COLUMN].eq(top_group), MAIN_DEPENDENT_VARIABLE
            ],
            direction="greater",
        )
    )
    yearly_extremes = dataset.loc[
        dataset[RAW_MAIN_REGRESSION_GROUP_COLUMN].isin([bottom_group, top_group]),
        [FORMATION_YEAR_COLUMN, RAW_MAIN_REGRESSION_GROUP_COLUMN, MAIN_DEPENDENT_VARIABLE],
    ].dropna()
    yearly_means = yearly_extremes.pivot_table(
        index=FORMATION_YEAR_COLUMN,
        columns=RAW_MAIN_REGRESSION_GROUP_COLUMN,
        values=MAIN_DEPENDENT_VARIABLE,
        aggfunc="mean",
    )
    if bottom_group in yearly_means.columns and top_group in yearly_means.columns:
        annual_spread = yearly_means[top_group] - yearly_means[bottom_group]
    else:
        annual_spread = pd.Series(dtype=float)
    rows.append(
        _sign_test_row(
            test="Annual Q5-minus-Q1 spread sign",
            alternative=r"$P(\overline{\mathrm{BHAR}}^{Q5-Q1}>0) > 0.5$",
            sample=annual_spread,
            direction="greater",
        )
    )

    if unbiasedness_dataset is None:
        unbiasedness_dataset = dataset

    try:
        unbiasedness_model = fit_simple_ols(
            unbiasedness_dataset,
            x_col=ANNOUNCEMENT_WINDOW_COLUMN,
            y_col=FULL_WINDOW_COLUMN,
        )
        unbiasedness_coefficient = unbiasedness_model["coefficients"].set_index("Term").loc[
            ANNOUNCEMENT_WINDOW_COLUMN
        ]
        unbiasedness_summary = unbiasedness_model["model"]
        unbiasedness_tests = build_unbiasedness_tests(unbiasedness_model)
        slope_test_statistic = _as_float(unbiasedness_tests["Slope_t_stat"])
        slope_test_p_value = _as_float(
            unbiasedness_tests["Slope_greater_than_null_p_value"]
        )
        rows.append(
            {
                "Test_Group": "Regression",
                "Test": "Unbiasedness slope",
                "Alternative": r"$\lambda > 1$",
                "Estimate": _as_float(unbiasedness_coefficient["Coefficient"]),
                "Clustered_SE": _as_float(unbiasedness_coefficient["Std_Error"]),
                "Statistic": slope_test_statistic,
                "p_value": slope_test_p_value,
                "N": _as_float(unbiasedness_summary["N"]),
                "Inference": "Two-way clustered by firm and announcement year-quarter",
                "Alpha": 0.05,
                "Reject_Null": bool(slope_test_p_value < 0.05),
                "Status": "completed",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test_Group": "Regression",
                "Test": "Unbiasedness slope",
                "Alternative": r"$\lambda > 1$",
                "Estimate": np.nan,
                "Clustered_SE": np.nan,
                "Statistic": np.nan,
                "p_value": np.nan,
                "N": np.nan,
                "Inference": "Two-way clustered by firm and announcement year-quarter",
                "Alpha": 0.05,
                "Reject_Null": np.nan,
                "Status": f"failed: {type(exc).__name__}: {exc}",
            }
        )

    time_diagnostics = _diagnostics_row(diagnostics, TIME_VARIATION_SPEC_KEY)
    if time_diagnostics is not None:
        rows.extend(
            [
                {
                    "Test_Group": "Time variation",
                    "Test": "H2: joint time variation",
                    "Alternative": r"$\theta_M \neq 0$ or $\theta_L \neq 0$",
                    "Estimate": np.nan,
                    "Clustered_SE": np.nan,
                    "Statistic": _as_float(time_diagnostics.get("Time_Variation_Joint_F_Statistic")),
                    "p_value": _as_float(time_diagnostics.get("Time_Variation_Joint_F_p_value")),
                    "N": _as_float(time_diagnostics.get("Sample_Size")),
                    "Inference": "F-test",
                    "Alpha": 0.05,
                    "Reject_Null": (
                        bool(_as_float(time_diagnostics.get("Time_Variation_Joint_F_p_value")) < 0.05)
                        if np.isfinite(_as_float(time_diagnostics.get("Time_Variation_Joint_F_p_value")))
                        else np.nan
                    ),
                    "Status": str(time_diagnostics.get("Time_Variation_Tests_Status", "unavailable")),
                },
                {
                    "Test_Group": "Time variation",
                    "Test": "H2a: middle versus early",
                    "Alternative": r"$\theta_M < 0$",
                    "Estimate": _as_float(time_diagnostics.get("Middle_vs_Early_Estimate")),
                    "Clustered_SE": _as_float(time_diagnostics.get("Middle_vs_Early_Std_Error")),
                    "Statistic": _as_float(time_diagnostics.get("Middle_vs_Early_t_Statistic")),
                    "p_value": _as_float(time_diagnostics.get("Middle_vs_Early_One_Sided_p_value")),
                    "N": _as_float(time_diagnostics.get("Sample_Size")),
                    "Inference": "One-sided, two-way clustered contrast",
                    "Alpha": 0.05,
                    "Reject_Null": (
                        bool(_as_float(time_diagnostics.get("Middle_vs_Early_One_Sided_p_value")) < 0.05)
                        if np.isfinite(_as_float(time_diagnostics.get("Middle_vs_Early_One_Sided_p_value")))
                        else np.nan
                    ),
                    "Status": str(time_diagnostics.get("Time_Variation_Tests_Status", "unavailable")),
                },
                {
                    "Test_Group": "Time variation",
                    "Test": "H2a: late versus middle",
                    "Alternative": r"$\theta_L-\theta_M < 0$",
                    "Estimate": _as_float(time_diagnostics.get("Late_vs_Middle_Estimate")),
                    "Clustered_SE": _as_float(time_diagnostics.get("Late_vs_Middle_Std_Error")),
                    "Statistic": _as_float(time_diagnostics.get("Late_vs_Middle_t_Statistic")),
                    "p_value": _as_float(time_diagnostics.get("Late_vs_Middle_One_Sided_p_value")),
                    "N": _as_float(time_diagnostics.get("Sample_Size")),
                    "Inference": "One-sided, two-way clustered contrast",
                    "Alpha": 0.05,
                    "Reject_Null": (
                        bool(_as_float(time_diagnostics.get("Late_vs_Middle_One_Sided_p_value")) < 0.05)
                        if np.isfinite(_as_float(time_diagnostics.get("Late_vs_Middle_One_Sided_p_value")))
                        else np.nan
                    ),
                    "Status": str(time_diagnostics.get("Time_Variation_Tests_Status", "unavailable")),
                },
            ]
        )

    for h3_joint_test in h3_joint_tests or []:
        h3_p_value = _as_float(h3_joint_test.get("p_value"))
        rows.append(
            {
                "Test_Group": "Firm-size heterogeneity",
                "Test": str(h3_joint_test["Test"]),
                "Alternative": str(h3_joint_test["Alternative"]),
                "Estimate": np.nan,
                "Clustered_SE": np.nan,
                "Statistic": _as_float(h3_joint_test.get("Statistic")),
                "p_value": h3_p_value,
                "N": _as_float(h3_joint_test.get("N")),
                "Inference": "F-test",
                "Alpha": 0.10,
                "Reject_Null": bool(h3_p_value < 0.10) if np.isfinite(h3_p_value) else np.nan,
                "Status": str(h3_joint_test["Status"]),
            }
        )

    return pd.DataFrame(rows)


def _format_methodology_test_number(value: object, *, decimals: int = 3) -> str:
    numeric_value = _as_float(value)
    return "" if not np.isfinite(numeric_value) else f"{numeric_value:.{decimals}f}"


def build_methodology_hypothesis_tests_latex(test_results: pd.DataFrame) -> str:
    rows = []
    for _, result in test_results.iterrows():
        rows.append(
            " & ".join(
                [
                    str(result["Test"]),
                    str(result["Alternative"]),
                    _format_methodology_test_number(result["Estimate"]),
                    _format_methodology_test_number(result["Clustered_SE"]),
                    _format_methodology_test_number(result["Statistic"]),
                    _format_methodology_test_number(result["p_value"], decimals=4),
                    _format_methodology_test_number(result["N"], decimals=0),
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Methodology-specified hypothesis tests}",
        r"\label{tab:methodology-hypothesis-tests}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Test & Alternative & Estimate & SE & Statistic & $p$-value & $N$ " + r"\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.96\linewidth}",
        r"\footnotesize \textit{Notes:} Directional regression tests use the two-way-clustered covariance "
        r"matrix (firm and announcement year-quarter). The H1 test uses a 5\% threshold; the two prespecified "
        r"continuation tests use a 10\% threshold. The Q1, Q5, and annual Q5-minus-Q1 tests are one-sided exact "
        r"binomial sign tests. The H2 and H3 joint time-variation tests are two-sided; the two adjacent-period attenuation "
        r"contrasts and all other directional tests report one-sided $p$-values.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_methodology_hypothesis_test_outputs(
    output_dir: Path,
    dataset: pd.DataFrame,
    suite_results: dict[str, object],
    *,
    unbiasedness_dataset: pd.DataFrame | None = None,
) -> None:
    """Save methodology tests, retaining the full eligible sample for unbiasedness."""
    save_time_variation_outputs(output_dir, dataset, suite_results)
    if unbiasedness_dataset is None:
        unbiasedness_dataset = dataset
    _save_table_to_dir(
        output_dir,
        build_unbiasedness_regression_results(unbiasedness_dataset),
        "unbiasedness_regression_results",
    )
    h3_model_output = _h3_size_time_reporting_model(dataset)
    h3_joint_tests = build_h3_joint_tests(dataset, model_output=h3_model_output)
    _save_table_to_dir(
        output_dir,
        pd.DataFrame(h3_joint_tests),
        "h3_joint_tests",
    )
    _save_table_to_dir(
        output_dir,
        build_h3_period_specific_effects(dataset, model_output=h3_model_output),
        "h3_period_specific_effects",
    )
    test_results = build_methodology_hypothesis_tests(
        dataset,
        suite_results,
        unbiasedness_dataset=unbiasedness_dataset,
        h3_joint_tests=h3_joint_tests,
    )
    _save_table_to_dir(output_dir, test_results, "methodology_hypothesis_tests")
    _save_latex_to_dir(
        output_dir,
        "methodology_hypothesis_tests",
        build_methodology_hypothesis_tests_latex(test_results),
    )


def execute_suite_job(
    job: SuiteJob,
    *,
    base_dataset: pd.DataFrame,
    base_dataset_fingerprint: str,
    master_progress_bar,
) -> dict[str, object]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = add_centered_firm_average_log_market_cap(job.dataset_builder(base_dataset))
    if job.pre_filter_audit_builder is not None:
        _save_table_to_dir(
            job.output_dir,
            build_market_cap_price_filter_audit(job.pre_filter_audit_builder(base_dataset)),
            "pre_announcement_market_cap_price_filter_audit",
        )
    dataset_column_overview = build_dataset_column_overview(dataset)
    suite_metadata = build_suite_metadata(job, base_dataset_fingerprint=base_dataset_fingerprint)

    def _progress_callback(_: dict[str, object]) -> None:
        if master_progress_bar is not None:
            master_progress_bar.update(1)

    suite_results = run_regression_suite(
        dataset,
        job.specs,
        output_dir=job.output_dir,
        suite_metadata=suite_metadata,
        progress_callback=_progress_callback if master_progress_bar is not None else None,
    )

    if job.display_outputs:
        display(dataset.head())
        display(dataset_column_overview)
        display(suite_results["registry"])
        display(suite_results["status"])
        display(suite_results["diagnostics"])
        display(suite_results["coefficients"])

    _save_table_to_dir(job.output_dir, dataset.head(200), "analysis_dataset_preview_first_200_rows")
    _save_table_to_dir(job.output_dir, dataset_column_overview, "analysis_dataset_column_overview")
    if job.key == BASELINE_SUITE_LABEL:
        _save_table_to_dir(
            job.output_dir,
            build_appendix_b_1_3_missing_return_summary(dataset),
            "appendix_b_1_3_missing_return_summary",
        )
    if job.key == WINSORIZED_BHAR20_MAIN_LABEL:
        _save_table_to_dir(
            job.output_dir,
            build_appendix_b_1_4_winsorization_summary(dataset),
            "appendix_b_1_4_winsorization_summary",
        )
    _save_table_to_dir(job.output_dir, suite_results["registry"], "regression_registry")
    _save_table_to_dir(job.output_dir, suite_results["status"], "regression_status")
    if not suite_results["status"].empty:
        error_rows = suite_results["status"].loc[
            suite_results["status"]["Status"].eq("failed")
        ].copy()
        _save_table_to_dir(job.output_dir, error_rows, "regression_error_diagnostics")
    _save_table_to_dir(job.output_dir, suite_results["diagnostics"], "model_diagnostics")
    _save_table_to_dir(
        job.output_dir,
        suite_results["diagnostics"],
        "regression_table_statistics",
    )
    if "SUE_Interaction_Joint_Test_Applied" in suite_results["diagnostics"].columns:
        joint_interaction_tests = suite_results["diagnostics"].loc[
            suite_results["diagnostics"]["SUE_Interaction_Joint_Test_Applied"].fillna(False)
        ].copy()
        _save_table_to_dir(
            job.output_dir,
            joint_interaction_tests,
            "joint_sue_interaction_tests",
        )
    _save_table_to_dir(job.output_dir, suite_results["coefficients"], "non_fixed_effect_coefficients")
    _save_table_to_dir(
        job.output_dir,
        suite_results["coefficients"],
        "regression_table_coefficients",
    )

    completed_spec_keys = set(
        suite_results["status"].loc[
            suite_results["status"]["Status"].eq("completed"),
            "Spec_Key",
        ].astype(str).tolist()
    ) if not suite_results["status"].empty else set()

    suite_execution_errors: list[dict[str, object]] = []

    try:
        if job.table_mode == "family":
            combined_family_specs = build_family_specs_map(job.specs, completed_spec_keys)
            if combined_family_specs:
                combined_latex_document = build_combined_regression_latex_document_from_summaries(
                    combined_family_specs,
                    suite_results["coefficients"],
                    suite_results["diagnostics"],
                )
                _save_latex_to_dir(
                    job.output_dir,
                    f"{job.key}_regression_tables",
                    combined_latex_document,
                )
        elif job.table_mode == "chunked":
            completed_specs = [
                spec for spec in job.specs if spec.key in completed_spec_keys
            ]
            if completed_specs:
                chunked_latex_document = build_chunked_regression_latex_document(
                    completed_specs,
                    suite_results["coefficients"],
                    suite_results["diagnostics"],
                    title=job.title or GRID_SEARCH_TITLE,
                    regressions_per_table=GRID_REGRESSIONS_PER_TABLE,
                    tables_per_page=GRID_TABLES_PER_PAGE,
                )
                _save_latex_to_dir(
                    job.output_dir,
                    f"{job.key}_regression_tables",
                    chunked_latex_document,
                )
        else:
            raise ValueError(f"Unsupported table mode {job.table_mode!r}.")
    except Exception as exc:
        suite_execution_errors.append(
            {
                "Suite_Key": job.key,
                "Error_Stage": "table_generation",
                "Error_Type": type(exc).__name__,
                "Error_Module": type(exc).__module__,
                "Reason": str(exc),
                "Traceback": traceback.format_exc(),
            }
        )

    if job.post_run_callback is not None:
        try:
            job.post_run_callback(job.output_dir, dataset, suite_results)
        except Exception as exc:
            suite_execution_errors.append(
                {
                    "Suite_Key": job.key,
                    "Error_Stage": "post_run_callback",
                    "Error_Type": type(exc).__name__,
                    "Error_Module": type(exc).__module__,
                    "Reason": str(exc),
                    "Traceback": traceback.format_exc(),
                }
            )

    if suite_execution_errors:
        _save_table_to_dir(
            job.output_dir,
            pd.DataFrame(suite_execution_errors),
            "suite_execution_errors",
        )

    return suite_results


def build_suite_jobs(
    base_dataset: pd.DataFrame,
    *,
    sample_config: RegressionSampleConfig,
) -> list[SuiteJob]:
    base_labels = base_dependent_label_map()
    log_labels = {
        **base_labels,
        **{
            LOG_BHAR_COLUMN_MAP[bhar_column]: f"SymLog {bhar_label}"
            for bhar_column, bhar_label in base_labels.items()
        },
    }
    winsorized_labels = {
        **base_labels,
        **{
            WINSORIZED_BHAR_COLUMN_MAP[bhar_column]: f"Win {bhar_label}"
            for bhar_column, bhar_label in base_labels.items()
        },
    }
    bhar_2_60_labels = {
        **base_labels,
        BHAR_2_60_MAIN_COLUMN: BHAR_2_60_MAIN_LABEL_TEXT,
    }

    time_variation_specs = build_time_variation_specs()
    reporting_specs = [
        spec
        for spec in REGRESSION_SPECS
        if spec.family not in DISABLED_MAIN_OUTPUT_FAMILIES
    ] + time_variation_specs
    methodology_duration_specs = build_methodology_continuation_specs()
    baseline_specs = reporting_specs + methodology_duration_specs
    no_fixed_effect_specs = build_fixed_effect_variant_specs(
        baseline_specs,
        include_firm_and_quarter_fixed_effects=False,
    )
    firm_quarter_fixed_effect_specs = build_fixed_effect_variant_specs(
        baseline_specs,
        include_firm_and_quarter_fixed_effects=True,
    )
    split_window_selection_specs = [
        next(spec for spec in reporting_specs if spec.key == "main_regression"),
        *methodology_duration_specs,
    ]
    split_analysis_specs = [
        spec
        for spec in reporting_specs
        if spec.formula.startswith(f"{MAIN_DEPENDENT_VARIABLE} ~")
    ]
    decile_full_suite_specs = build_full_sue_grouping_specs(10)
    two_bin_full_suite_specs = build_full_sue_grouping_specs(2)
    heteroskedasticity_robust_specs = build_heteroskedasticity_robust_specs(
        reporting_specs + methodology_duration_specs
    )
    non_original_specs = filter_specs_for_non_original_suite(
        reporting_specs + methodology_duration_specs
    )
    grid_specs = build_market_cap_grid_specs(sample_config)
    variant_grid_specs = build_variant_grid_specs(sample_config)
    analyst_filter_label = f"Forecast_Analyst_Count >= {sample_config.min_analyst_count}"

    def suite_output_dir(name: str) -> Path:
        return sample_config.output_dir / name

    split_sample_output_dir = suite_output_dir(SPLIT_SAMPLE_PARENT_LABEL)

    jobs = [
        SuiteJob(
            key=BASELINE_SUITE_LABEL,
            output_dir=suite_output_dir(BASELINE_SUITE_LABEL),
            specs=baseline_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
            },
        ),
        SuiteJob(
            key=HETEROSKEDASTICITY_ROBUST_SUITE_LABEL,
            output_dir=suite_output_dir(HETEROSKEDASTICITY_ROBUST_SUITE_LABEL),
            specs=heteroskedasticity_robust_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [],
                "standard_error_treatment": "heteroskedasticity_robust_hc1",
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "standard_error_treatment": "HC1 heteroskedasticity-robust",
            },
        ),
        SuiteJob(
            key=NO_FIXED_EFFECTS_FULL_SUITE_LABEL,
            output_dir=suite_output_dir(NO_FIXED_EFFECTS_FULL_SUITE_LABEL),
            specs=no_fixed_effect_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [],
                "fixed_effect_treatment": "none",
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "fixed_effect_treatment": "none",
            },
        ),
        SuiteJob(
            key=FIRM_QUARTER_FIXED_EFFECTS_FULL_SUITE_LABEL,
            output_dir=suite_output_dir(FIRM_QUARTER_FIXED_EFFECTS_FULL_SUITE_LABEL),
            specs=firm_quarter_fixed_effect_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [],
                "fixed_effect_treatment": "firm_and_announcement_quarter",
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "fixed_effect_treatment": "firm and announcement-quarter",
            },
        ),
        SuiteJob(
            key=SPLIT_SAMPLE_WINDOW_SELECTION_LABEL,
            output_dir=split_sample_output_dir / "window_selection_50pct",
            specs=split_window_selection_specs,
            dataset_builder=lambda dataset: select_split_sample_group(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                ),
                "window_selection_50pct",
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [
                    "firm-disjoint stratified random split by earliest formation year",
                    "50 percent window-selection subsample",
                ],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "random_seed": SPLIT_SAMPLE_RANDOM_SEED,
                "stratification": "firm earliest formation year",
                "firm_allocation": "50 percent window selection / 50 percent analysis",
                "dependent_variables": ["BHAR_2_20", "BHAR_21_40", "BHAR_41_60"],
            },
        ),
        SuiteJob(
            key=SPLIT_SAMPLE_ANALYSIS_LABEL,
            output_dir=split_sample_output_dir / "analysis_50pct",
            specs=split_analysis_specs,
            dataset_builder=lambda dataset: select_split_sample_group(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                ),
                "analysis_50pct",
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [
                    "firm-disjoint stratified random split by earliest formation year",
                    "50 percent main-analysis subsample",
                    "retain only specifications with BHAR_2_20 as the dependent variable",
                ],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "random_seed": SPLIT_SAMPLE_RANDOM_SEED,
                "stratification": "firm earliest formation year",
                "firm_allocation": "50 percent window selection / 50 percent analysis",
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "window_selection_result_used_for_analysis": False,
            },
        ),
        SuiteJob(
            key=SUE_DECILE_FULL_SUITE_LABEL,
            output_dir=suite_output_dir(SUE_DECILE_FULL_SUITE_LABEL),
            specs=decile_full_suite_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": ["replace SUE quintile with SUE decile in every reported specification"],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "sue_group_count": 10,
                "sue_grouping": "prior-year SUE deciles",
            },
        ),
        SuiteJob(
            key=SUE_TWO_BIN_FULL_SUITE_LABEL,
            output_dir=suite_output_dir(SUE_TWO_BIN_FULL_SUITE_LABEL),
            specs=two_bin_full_suite_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": ["replace SUE quintile with a 2-bin SUE group in every reported specification"],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "sue_group_count": 2,
                "sue_grouping": "prior-year 2-bin SUE groups",
            },
        ),
        SuiteJob(
            key=MARKET_CAP_PRICE_FILTER_LABEL,
            output_dir=suite_output_dir(MARKET_CAP_PRICE_FILTER_LABEL),
            specs=non_original_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ).loc[
                pd.to_numeric(dataset[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce").ge(MARKET_CAP_FILTER_THRESHOLD)
                & pd.to_numeric(dataset["Price"], errors="coerce").ge(PRICE_FILTER_THRESHOLD)
            ].copy(),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [
                    analyst_filter_label,
                    f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD}",
                    f"Price >= {PRICE_FILTER_THRESHOLD}",
                ],
                "transforms": [],
            },
            table_mode="family",
            suite_parameters={
                "sample": (
                    f"{analyst_filter_label} and "
                    f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD} and "
                    f"Price >= {PRICE_FILTER_THRESHOLD}"
                ),
            },
            pre_filter_audit_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
        ),
        SuiteJob(
            key=POST_2000_BASELINE_LABEL,
            output_dir=suite_output_dir(POST_2000_BASELINE_LABEL),
            specs=non_original_specs,
            dataset_builder=lambda dataset: add_post_2000_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [
                    analyst_filter_label,
                    f"{FORMATION_YEAR_COLUMN} >= {MIN_SAMPLE_YEAR}",
                ],
                "transforms": [],
            },
            table_mode="family",
            suite_parameters={
                "sample": (
                    f"{analyst_filter_label} and "
                    f"{FORMATION_YEAR_COLUMN} >= {MIN_SAMPLE_YEAR}"
                ),
            },
        ),
        SuiteJob(
            key=POST_2000_MARKET_CAP_PRICE_FILTER_LABEL,
            output_dir=suite_output_dir(POST_2000_MARKET_CAP_PRICE_FILTER_LABEL),
            specs=non_original_specs,
            dataset_builder=lambda dataset: add_post_2000_market_cap_price_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [
                    analyst_filter_label,
                    f"{FORMATION_YEAR_COLUMN} >= {MIN_SAMPLE_YEAR}",
                    f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD}",
                    f"Price >= {PRICE_FILTER_THRESHOLD}",
                ],
                "transforms": [],
            },
            table_mode="family",
            suite_parameters={
                "sample": (
                    f"{analyst_filter_label} and "
                    f"{FORMATION_YEAR_COLUMN} >= {MIN_SAMPLE_YEAR} and "
                    f"{PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN} >= {MARKET_CAP_FILTER_THRESHOLD} and "
                    f"Price >= {PRICE_FILTER_THRESHOLD}"
                ),
            },
        ),
        SuiteJob(
            key=COMPLETE_CASE_RETURN_LABEL,
            output_dir=suite_output_dir(COMPLETE_CASE_RETURN_LABEL),
            specs=non_original_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": sample_config.complete_case_return_filename,
                "filters": [analyst_filter_label],
                "transforms": [],
                "missing_return_treatment": "complete_case_by_bhar_window",
            },
            source_dataset_key=COMPLETE_CASE_RETURN_LABEL,
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "missing_return_treatment": (
                    "BHAR is missing, and the regression observation is excluded, when "
                    "the corresponding post-announcement return window is incomplete"
                ),
            },
        ),
        SuiteJob(
            key=TERMINAL_LOSS_RETURN_LABEL,
            output_dir=suite_output_dir(TERMINAL_LOSS_RETURN_LABEL),
            specs=non_original_specs,
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": sample_config.terminal_loss_return_filename,
                "filters": [analyst_filter_label],
                "transforms": [],
                "missing_return_treatment": "interior_zero_terminal_minus_100_then_zero",
            },
            source_dataset_key=TERMINAL_LOSS_RETURN_LABEL,
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
                "missing_return_treatment": (
                    "interior missing returns are zero; the first terminal missing return "
                    "is -100 percent and subsequent returns are zero"
                ),
            },
        ),
        SuiteJob(
            key=LOG_BHAR20_MAIN_LABEL,
            output_dir=suite_output_dir(LOG_BHAR20_MAIN_LABEL),
            specs=filter_specs_for_non_original_suite(build_main_dependent_variant_specs(
                REGRESSION_SPECS + time_variation_specs,
                dependent_variable_map=LOG_BHAR_COLUMN_MAP,
                dependent_label_map=log_labels,
            )),
            dataset_builder=lambda dataset: add_signed_log_bhar_columns(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                )
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": ["signed_log1p_bhar_columns"],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": LOG_BHAR20_MAIN_COLUMN,
                "transform": "sign(x) * log1p(abs(x))",
            },
        ),
        SuiteJob(
            key=WINSORIZED_BHAR20_MAIN_LABEL,
            output_dir=suite_output_dir(WINSORIZED_BHAR20_MAIN_LABEL),
            specs=filter_specs_for_non_original_suite(build_main_dependent_variant_specs(
                REGRESSION_SPECS + methodology_duration_specs + time_variation_specs,
                dependent_variable_map=WINSORIZED_BHAR_COLUMN_MAP,
                dependent_label_map=winsorized_labels,
            )),
            dataset_builder=lambda dataset: add_winsorized_bhar_columns(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                )
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [
                    "winsorize BHAR[2,20], BHAR[21,40], and BHAR[41,60] separately at their 5th and 95th percentiles"
                ],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": WINSORIZED_BHAR20_MAIN_COLUMN,
                "transform": "each BHAR interval winsorized separately at its 5th and 95th percentiles within the suite sample",
            },
        ),
        SuiteJob(
            key=BHAR_2_60_MAIN_LABEL,
            output_dir=suite_output_dir(BHAR_2_60_MAIN_LABEL),
            specs=filter_specs_for_non_original_suite(build_main_dependent_variant_specs(
                REGRESSION_SPECS + time_variation_specs,
                dependent_variable_map={
                    MAIN_DEPENDENT_VARIABLE: BHAR_2_60_MAIN_COLUMN,
                },
                dependent_label_map=bhar_2_60_labels,
                excluded_spec_keys={"variable_spec_bhar_2_60"},
            )),
            dataset_builder=lambda dataset: add_baseline_sample_filter(
                dataset,
                min_analyst_count=sample_config.min_analyst_count,
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": [],
            },
            table_mode="family",
            suite_parameters={
                "sample": analyst_filter_label,
                "main_dependent_variable": BHAR_2_60_MAIN_COLUMN,
            },
        ),
        SuiteJob(
            key=GRID_SEARCH_LABEL,
            output_dir=suite_output_dir(GRID_SEARCH_LABEL),
            specs=grid_specs,
            dataset_builder=lambda dataset: add_market_cap_breakpoint_columns(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                )
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": ["market_cap_breakpoint_split_columns"],
            },
            table_mode="chunked",
            title=GRID_SEARCH_TITLE,
            suite_parameters={
                "base_samples": [
                    sample_config.label,
                ],
                "market_cap_breakpoints": [
                    breakpoint_config["percentile"] for breakpoint_config in GRID_MARKET_CAP_BREAKPOINTS
                ],
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
            },
        ),
        SuiteJob(
            key=VARIANT_GRID_SEARCH_LABEL,
            output_dir=suite_output_dir(VARIANT_GRID_SEARCH_LABEL),
            specs=variant_grid_specs,
            dataset_builder=lambda dataset: add_variant_grid_columns(
                add_baseline_sample_filter(
                    dataset,
                    min_analyst_count=sample_config.min_analyst_count,
                )
            ),
            dataset_recipe={
                "source": "base_dataset",
                "filters": [analyst_filter_label],
                "transforms": ["variant_grid_bhar_columns", "variant_grid_sue_group_columns"],
            },
            table_mode="chunked",
            title=VARIANT_GRID_SEARCH_TITLE,
            suite_parameters={
                "base_samples": [
                    sample_config.label,
                    f"{sample_config.label} plus market cap >= $5m and price >= $1",
                ],
                "bhars": [config["source_column"] for config in VARIANT_GRID_BHAR_CONFIGS],
                "winsorized": [option["apply_winsorized"] for option in VARIANT_GRID_WINSOR_OPTIONS],
                "symlog": [option["apply_symlog"] for option in VARIANT_GRID_LOG_OPTIONS],
                "sue_group_counts": [config["group_count"] for config in VARIANT_GRID_SUE_SPLITS],
                "main_dependent_variable": MAIN_DEPENDENT_VARIABLE,
            },
        ),
    ]

    jobs = [
        replace(
            job,
            specs=[
                replace(spec, exclude_intercept_from_reporting=True)
                for spec in job.specs
            ],
        )
        for job in jobs
    ]

    jobs = [
        replace(job, post_run_callback=save_methodology_hypothesis_test_outputs)
        if sample_config.key == MAIN_REGRESSION_SAMPLE.key and job.key == BASELINE_SUITE_LABEL
        else replace(
            job,
            post_run_callback=build_time_variation_output_callback(
                next(
                    spec
                    for spec in job.specs
                    if spec.key == TIME_VARIATION_SPEC_KEY
                )
            ),
        )
        if any(spec.key == TIME_VARIATION_SPEC_KEY for spec in job.specs)
        else job
        for job in jobs
    ]

    # Run every configured robustness alternative, except suites explicitly
    # retained as disabled exploratory analyses. The analyst >= 1 sample is
    # narrowed below to its designated baseline suite.
    filtered_jobs = [job for job in jobs if job.key not in DISABLED_SUITE_KEYS]
    if sample_config.key == MIN1_REGRESSION_SAMPLE.key:
        filtered_jobs = [
            job for job in filtered_jobs if job.key in MIN1_ENABLED_SUITE_KEYS
        ]
    return filtered_jobs


def main() -> None:
    suite_jobs: list[tuple[dict[str, pd.DataFrame], dict[str, str], SuiteJob]] = []
    total_planned_regressions = 0

    for sample_config in REGRESSION_SAMPLE_CONFIGS:
        dataset_sources = {
            BASELINE_SUITE_LABEL: sample_config.baseline_return_filename,
            COMPLETE_CASE_RETURN_LABEL: sample_config.complete_case_return_filename,
            TERMINAL_LOSS_RETURN_LABEL: sample_config.terminal_loss_return_filename,
        }
        full_base_datasets = {
            dataset_key: load_or_build_regression_suite_dataset(
                DATA_DIR,
                sample_config.output_dir,
                abnormal_returns_filename=filename,
                cache_key=(
                    dataset_key
                    if sample_config.key == MAIN_REGRESSION_SAMPLE.key
                    else f"{sample_config.key}_{dataset_key}"
                ),
            )
            for dataset_key, filename in dataset_sources.items()
        }
        # Prior-year SUE grouping makes the first formation year ineligible for
        # SUE-based tests. Keep the full data separately because the
        # unbiasedness regression does not use a SUE group and must retain it.
        base_datasets = {
            dataset_key: drop_first_formation_year(dataset)
            for dataset_key, dataset in full_base_datasets.items()
        }
        base_dataset = base_datasets[BASELINE_SUITE_LABEL]
        validate_time_period_buckets(base_dataset)
        validate_variant_transforms(base_dataset)
        base_dataset_fingerprints = {
            dataset_key: compute_base_dataset_fingerprint(dataset)
            for dataset_key, dataset in base_datasets.items()
        }

        sample_suite_jobs = build_suite_jobs(
            base_dataset,
            sample_config=sample_config,
        )
        if sample_config.key == MAIN_REGRESSION_SAMPLE.key:
            full_unbiasedness_dataset = add_baseline_sample_filter(
                full_base_datasets[BASELINE_SUITE_LABEL],
                min_analyst_count=sample_config.min_analyst_count,
            )

            def save_main_methodology_outputs(
                output_dir: Path,
                sue_dataset: pd.DataFrame,
                suite_results: dict[str, object],
                *,
                _unbiasedness_dataset: pd.DataFrame = full_unbiasedness_dataset,
            ) -> None:
                save_methodology_hypothesis_test_outputs(
                    output_dir,
                    sue_dataset,
                    suite_results,
                    unbiasedness_dataset=_unbiasedness_dataset,
                )

            sample_suite_jobs = [
                replace(
                    job,
                    post_run_callback=save_main_methodology_outputs,
                )
                if job.key == BASELINE_SUITE_LABEL
                else job
                for job in sample_suite_jobs
            ]
        for suite_job in sample_suite_jobs:
            plan = plan_regression_suite_run(
                suite_job.specs,
                output_dir=suite_job.output_dir,
                suite_metadata=build_suite_metadata(
                    suite_job,
                    base_dataset_fingerprint=base_dataset_fingerprints[
                        suite_job.source_dataset_key
                    ],
                ),
            )
            total_planned_regressions += len(plan["runnable_specs"])
            suite_jobs.append((base_datasets, base_dataset_fingerprints, suite_job))

    master_progress_bar = None
    if tqdm is not None:
        master_progress_bar = tqdm(
            total=total_planned_regressions,
            desc="11_regression_suite total regressions",
        )

    try:
        for base_datasets, base_dataset_fingerprints, suite_job in suite_jobs:
            execute_suite_job(
                suite_job,
                base_dataset=base_datasets[suite_job.source_dataset_key],
                base_dataset_fingerprint=base_dataset_fingerprints[
                    suite_job.source_dataset_key
                ],
                master_progress_bar=master_progress_bar,
            )
    finally:
        if master_progress_bar is not None:
            master_progress_bar.close()


if __name__ == "__main__":
    main()
