from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from hashlib import sha256
from pathlib import Path
import re
import traceback
from typing import TypeVar
import warnings

import numpy as np
import pandas as pd
from scipy import stats
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..core.pipeline_config import (
    ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS,
    MAIN_REGRESSION_BHAR_WINDOW,
    SUE_COMPUTATION_GROUP_COUNT,
    SUE_PLOT_GROUP_COUNT,
)
from ..pead.sue_groups import SUE_GROUP_COLUMN, SUE_PLOT_GROUP_COLUMN, add_plot_group_column
from .time_varying_analysis import (
    ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN,
    ANNOUNCEMENT_QUARTER_COLUMN,
    ANNOUNCEMENT_YEAR_COLUMN,
    FIRM_IDENTIFIER_COLUMN,
    FORMATION_YEAR_COLUMN,
    TIME_PERIOD_COLUMN,
    apply_market_cap_analysis_split,
    assign_time_periods,
    attach_universe_snapshot,
    build_coefficient_table,
    build_time_periods,
    collapse_to_event_level,
    fit_formula_model,
    load_abnormal_returns_with_groups,
    load_stock_universe_snapshots,
)


DEFAULT_FIXED_EFFECT_TERMS = (
    f"C({ANNOUNCEMENT_QUARTER_COLUMN})",
    f"C({FIRM_IDENTIFIER_COLUMN})",
)
SUITE_METADATA_FILENAME = "suite_run_metadata.json"
REGRESSION_OUTPUT_SCHEMA_VERSION = "3"
MAX_REGRESSIONS_PER_LATEX_TABLE = 6
PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN = "Market_Cap_Pre_Announcement_1D"
PRE_ANNOUNCEMENT_MARKET_CAP_DATE_COLUMN = "Market_Cap_Pre_Announcement_1D_Date"
PRE_ANNOUNCEMENT_MARKET_CAP_METHOD_COLUMN = "Market_Cap_Pre_Announcement_1D_Method"
PRE_ANNOUNCEMENT_MARKET_CAP_LAG_TRADING_DAYS = 1
SUPPLEMENTAL_PRE_ANNOUNCEMENT_MARKET_CAP_FILENAME = (
    "supplemental_pre_announcement_market_caps.csv"
)
FORMATION_DATE_MARKET_CAP_FALLBACK_METHOD = "formation_date_market_cap_fallback"
RegressionTableItem = TypeVar("RegressionTableItem")


@dataclass(frozen=True)
class RegressionSpec:
    key: str
    family: str
    label: str
    formula: str
    cluster_spec: str = "firm_quarter"
    row_filter_query: str = ""
    enabled: bool = True
    notes: str = ""
    fixed_effect_terms_to_exclude: tuple[str, ...] = DEFAULT_FIXED_EFFECT_TERMS
    exclude_intercept_from_reporting: bool = False
    ordered_time_periods: tuple[str, str, str] | None = None
    ordered_time_regressor: str = ""
    ordered_time_period_column: str = ""


@dataclass(frozen=True)
class CompletedRegressionResult:
    spec: RegressionSpec
    result: object
    cluster_label: str
    diagnostics: dict[str, object]
    regression_input_path: str


def _spec_signature(spec: RegressionSpec) -> str:
    payload = {
        "key": spec.key,
        "family": spec.family,
        "formula": spec.formula,
        "cluster_spec": spec.cluster_spec,
        "row_filter_query": spec.row_filter_query,
        "enabled": spec.enabled,
        "notes": spec.notes,
        "fixed_effect_terms_to_exclude": list(spec.fixed_effect_terms_to_exclude),
        "exclude_intercept_from_reporting": spec.exclude_intercept_from_reporting,
        "ordered_time_periods": list(spec.ordered_time_periods) if spec.ordered_time_periods else None,
        "ordered_time_regressor": spec.ordered_time_regressor,
        "ordered_time_period_column": spec.ordered_time_period_column,
        "regression_output_schema_version": REGRESSION_OUTPUT_SCHEMA_VERSION,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def _escape_latex(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_significance_stars(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value <= 0.01:
        return "***"
    if p_value <= 0.05:
        return "**"
    if p_value <= 0.10:
        return "*"
    return ""


def _format_coefficient_cell(row: pd.Series) -> str:
    coefficient = row.get("Coefficient")
    std_error = row.get("Std_Error")
    if pd.isna(coefficient):
        return ""

    stars = _format_significance_stars(row.get("p_value"))
    coefficient_text = f"{float(coefficient):.3f}{stars}"
    if pd.isna(std_error):
        return coefficient_text
    return r"\shortstack[c]{" + coefficient_text + r" \\ " + f"({float(std_error):.3f})" + "}"


def _term_sort_key(term: str) -> tuple[int, str]:
    configured_group_terms = {
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}": 21,
        **{
            f"Regression_SUE_Group_{int(group_count)}": 22 + index
            for index, group_count in enumerate(ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS, start=1)
        },
    }
    explicit_order = {
        "Intercept": 0,
        "Low_Analyst_Following_LT_10": 10,
        "Low_Analyst_Following_LT_5": 11,
        "SUE": 20,
        "SUE_Plot_Group": 21,
        "SUE_Group": 22,
        "SUE_Plot_Group:Low_Analyst_Following_LT_10": 30,
        "SUE_Plot_Group:Low_Analyst_Following_LT_5": 31,
        "SUE_Group:Low_Analyst_Following_LT_10": 33,
        "SUE_Group:Low_Analyst_Following_LT_5": 34,
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}:Low_Analyst_Following_LT_10": 35,
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}:Low_Analyst_Following_LT_5": 36,
        "Centered_Log_Pre_Announcement_Market_Cap": 50,
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}:Centered_Log_Pre_Announcement_Market_Cap": 51,
        "Below_Pre_Announcement_Market_Cap_P20": 52,
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}:Below_Pre_Announcement_Market_Cap_P20": 53,
        **configured_group_terms,
    }
    return (explicit_order.get(term, 1000), term)


def _derive_short_label(spec: RegressionSpec) -> str:
    max_label_length = 8
    configured_bhar_windows = (
        MAIN_REGRESSION_BHAR_WINDOW,
        *ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    )
    bhar_key_codes = {
        f"b{int(day_end)}": str(index)
        for index, (_, day_end) in enumerate(configured_bhar_windows, start=1)
    }
    split_key_codes = {
        f"q{int(group_count)}": f"Q{int(group_count)}"
        for group_count in ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS
    }

    def finalize(label: str) -> str:
        if len(label) <= max_label_length:
            return label
        compact = re.sub(r"[^A-Za-z0-9+]", "", label)
        if len(compact) <= max_label_length:
            return compact
        return compact[:max_label_length] if compact else label[:max_label_length]

    if spec.key == "main_regression":
        return "Main"
    firm_size_labels = {
        "added_variables_firm_average_log_market_cap_interaction": "SizeLog",
        "added_variables_firm_average_market_cap_percentile_interaction": "SizePct",
        "added_variables_bottom_firm_average_market_cap_interaction": "Small20",
        "added_variables_firm_average_log_market_cap_time_interactions": "LogTime",
        "added_variables_firm_average_market_cap_percentile_time_interactions": "PctTime",
        "added_variables_bottom_firm_average_market_cap_time_interactions": "SmTime",
    }
    if spec.key in firm_size_labels:
        return firm_size_labels[spec.key]
    if spec.key.startswith("grid_"):
        match = re.fullmatch(
            r"grid_(base|mcap5m_price1)_(p\d+)",
            spec.key,
        )
        if match:
            sample_code = "Base" if match.group(1) == "base" else "MCap"
            breakpoint_code = match.group(2).upper()
            return finalize(f"{sample_code}{breakpoint_code}")
    if spec.key.startswith("variant_grid_"):
        match = re.fullmatch(
            r"variant_grid_(base|mcap5m_price1)_(b\d+)_(raw|win)_(lin|sym)_(q\d+)",
            spec.key,
        )
        if match:
            sample_code = "B" if match.group(1) == "base" else "M"
            bhar_code = bhar_key_codes.get(match.group(2), "0")
            winsor_code = "W" if match.group(3) == "win" else "R"
            log_code = "S" if match.group(4) == "sym" else "L"
            split_code = split_key_codes.get(match.group(5), "Q")
            return finalize(f"{sample_code}{bhar_code}{winsor_code}{log_code}{split_code}")

    label = spec.label
    prefix_delimiters = (": ", " - ")
    for delimiter in prefix_delimiters:
        if delimiter in label:
            label = label.split(delimiter, 1)[1]
            break
    label = label.strip()
    replacements = {
        "low analyst following below 10 with SUE-quintile interaction": "LT10",
        "low analyst following below 5 with SUE-quintile interaction": "LT5",
        "low analyst following below 10 and centered log market cap with SUE-quintile interactions": "LT10ME",
        "low analyst following below 5 and centered log market cap with SUE-quintile interactions": "LT5ME",
        "time-period interactions with time-varying SUE-quintile slopes": "TimeInt",
        "annual interactions: yearly SUE-quintile slopes": "YrInt",
        "SUE decile instead of quintile": "D10",
        "SUE median split instead of quintile": "MedSplt",
        **{
            f"BHAR({int(day_start)},{int(day_end)})": f"B{int(day_end)}"
            for day_start, day_end in configured_bhar_windows
        },
        **{
            f"SymLog BHAR({int(day_start)},{int(day_end)})": f"SymB{int(day_end)}"
            for day_start, day_end in configured_bhar_windows
        },
        **{
            f"Win BHAR({int(day_start)},{int(day_end)})": f"WinB{int(day_end)}"
            for day_start, day_end in configured_bhar_windows
        },
        "no fixed effects": "NoFE",
        "quarter fixed effects only": "QtrFE",
        "no clustering": "NoClus",
        "clustered by firm": "FirmClu",
        f"BHAR({int(MAIN_REGRESSION_BHAR_WINDOW[0])},{int(MAIN_REGRESSION_BHAR_WINDOW[1])}) on SUE quintile with firm and quarter fixed effects": "Main",
    }
    short_label = replacements.get(label, label)
    return finalize(short_label)


def _humanize_term(term: str) -> str:
    grouped_match = re.fullmatch(r"(?:(?:Grid|Variant_Grid|Regression)_)?SUE_Group_(\d+)", term)
    if grouped_match:
        group_count = int(grouped_match.group(1))
        if group_count == 10:
            return "SUE decile"
        if group_count == 5:
            return "SUE quintile"
        if group_count == 2:
            return "SUE 2-bin group"
        return f"SUE {group_count}-bin group"

    replacements = {
        "Intercept": "Intercept",
        "SUE_Plot_Group": "SUE quintile",
        "SUE_Group": "SUE decile",
        f"Regression_SUE_Group_{int(SUE_COMPUTATION_GROUP_COUNT)}": (
            "Ranked SUE"
            if int(SUE_COMPUTATION_GROUP_COUNT) == 5
            else "Ranked SUE decile"
            if int(SUE_COMPUTATION_GROUP_COUNT) == 10
            else f"Ranked SUE {int(SUE_COMPUTATION_GROUP_COUNT)}-bin group"
        ),
        "SUE": "SUE",
        "np.log(SUE)": "Log SUE",
        "Low_Analyst_Following_LT_10": "Low coverage < 10",
        "Low_Analyst_Following_LT_5": "Low coverage < 5",
        "Centered_Log_Pre_Announcement_Market_Cap": (
            "Centered log market cap (trading day before announcement)"
        ),
        "Centered_Pre_Announcement_Market_Cap_Percentile": (
            "Centered pre-announcement market-cap percentile"
        ),
        "Firm_Size_x_Middle_Period": "Firm size x Middle period",
        "Firm_Size_x_Late_Period": "Firm size x Late period",
        "Below_Pre_Announcement_Market_Cap_P20": "Bottom pre-announcement market-cap quintile",
    }
    if term in replacements:
        return replacements[term]

    humanized = term
    humanized = humanized.replace("C(SUE_Plot_Group)[T.", "SUE quintile ")
    humanized = humanized.replace("C(SUE_Group)[T.", "SUE decile ")
    humanized = re.sub(
        r"C\(Regression_SUE_Group_(\d+)\)\[T\.",
        lambda match: (
            "SUE decile "
            if int(match.group(1)) == 10
            else "SUE quintile "
            if int(match.group(1)) == 5
            else f"SUE {int(match.group(1))}-bin group "
        ),
        humanized,
    )
    humanized = humanized.replace("SUE_Plot_Group:", "SUE quintile x ")
    humanized = humanized.replace("SUE_Group:", "SUE decile x ")
    humanized = re.sub(
        r"Regression_SUE_Group_(\d+):",
        lambda match: (
            "Ranked SUE decile x "
            if int(match.group(1)) == 10
            else "Ranked SUE x "
            if int(match.group(1)) == 5
            else f"Ranked SUE {int(match.group(1))}-bin group x "
        ),
        humanized,
    )
    humanized = humanized.replace("SUE_Plot_Group", "SUE quintile")
    humanized = humanized.replace("SUE_Group", "SUE decile")
    humanized = re.sub(
        r"Regression_SUE_Group_(\d+)",
        lambda match: (
            "Ranked SUE decile"
            if int(match.group(1)) == 10
            else "Ranked SUE"
            if int(match.group(1)) == 5
            else f"Ranked SUE {int(match.group(1))}-bin group"
        ),
        humanized,
    )
    humanized = humanized.replace("C(SUE_Plot_Group)", "SUE quintile")
    humanized = humanized.replace("C(SUE_Group)", "SUE decile")
    humanized = re.sub(
        r"C\(Grid_MCap_Split_(\d+)\)\[T\.(.+)\]",
        lambda match: f"{match.group(2)} (p{match.group(1)})",
        humanized,
    )
    humanized = humanized.replace(".0]", "]")
    humanized = humanized.replace("Low_Analyst_Following_LT_10", "low coverage < 10")
    humanized = humanized.replace("Low_Analyst_Following_LT_5", "low coverage < 5")
    humanized = humanized.replace(
        "Below_Pre_Announcement_Market_Cap_P20", "bottom pre-announcement market-cap quintile"
    )
    humanized = humanized.replace(
        "Centered_Log_Pre_Announcement_Market_Cap",
        "centered log market cap (trading day before announcement)",
    )
    humanized = humanized.replace(
        "Centered_Pre_Announcement_Market_Cap_Percentile",
        "centered pre-announcement market-cap percentile",
    )
    humanized = humanized.replace(
        "C(Time_Period)[T.2003-2013]", "Middle period"
    )
    humanized = humanized.replace(
        "C(Time_Period)[T.2014-2024]", "Late period"
    )
    humanized = humanized.replace(
        "C(Martineau_Time_Period)[T.2006-2015]", "Martineau middle period"
    )
    humanized = humanized.replace(
        "C(Martineau_Time_Period)[T.2016+]", "Martineau late period"
    )
    humanized = humanized.replace("Middle_Period", "Middle period")
    humanized = humanized.replace("Late_Period", "Late period")
    humanized = humanized.replace("np.log(Market_Cap_Current)", "log market cap")
    humanized = humanized.replace(
        "I(log market cap - log market cap.mean())",
        "centered log market cap",
    )
    humanized = humanized.replace(
        "C(Market_Cap_Analysis_Split_Group)[T.Top 60% by market cap]",
        "top market-cap bucket",
    )
    humanized = humanized.replace(":", " x ")
    return humanized


def _has_formula_term(formula: str, term: str) -> bool:
    return term in formula


def _build_footer_value(spec: RegressionSpec, diagnostics: dict[str, object], row_label: str) -> str:
    if row_label == "Firm FE":
        return "Yes" if _has_formula_term(spec.formula, f"C({FIRM_IDENTIFIER_COLUMN})") else "No"
    if row_label == "Quarter FE":
        return "Yes" if _has_formula_term(spec.formula, f"C({ANNOUNCEMENT_QUARTER_COLUMN})") else "No"
    if row_label == "Clustering":
        cluster_label = str(diagnostics.get("Std_Error_Treatment", ""))
        cluster_replacements = {
            "Heteroskedasticity-robust (HC1)": "HC1 robust",
            "Clustered by firm and quarter": "Firm + quarter",
            "Clustered by quarter": "Quarter",
            "Clustered by firm": "Firm",
            "No clustering": "None",
        }
        return cluster_replacements.get(cluster_label, cluster_label)
    if row_label == "N":
        value = diagnostics.get("Sample_Size")
        return "" if pd.isna(value) else f"{int(float(value))}"
    if row_label == "Firm clusters":
        value = diagnostics.get("Firm_Cluster_Count")
        return "" if pd.isna(value) else f"{int(float(value))}"
    if row_label == "Quarter clusters":
        value = diagnostics.get("Quarter_Cluster_Count")
        return "" if pd.isna(value) else f"{int(float(value))}"
    if row_label == "Joint SUE F-test p-value":
        value = diagnostics.get("SUE_Interaction_Joint_Test_p_value")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "Joint time-variation F-test p-value":
        value = diagnostics.get("Time_Variation_Joint_F_p_value")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "Middle vs early one-sided p-value":
        value = diagnostics.get("Middle_vs_Early_One_Sided_p_value")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "Late vs middle one-sided p-value":
        value = diagnostics.get("Late_vs_Middle_One_Sided_p_value")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "R-squared":
        value = diagnostics.get("R_Squared")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "Within R-squared":
        value = diagnostics.get("Within_R_Squared")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "Adj. R-squared":
        value = diagnostics.get("Adjusted_R_Squared")
        return "" if pd.isna(value) else f"{float(value):.3f}"
    if row_label == "DF model":
        value = diagnostics.get("DF_Model")
        return "" if pd.isna(value) else f"{int(round(float(value)))}"
    if row_label == "DF resid":
        value = diagnostics.get("DF_Residual")
        return "" if pd.isna(value) else f"{int(round(float(value)))}"
    raise KeyError(f"Unsupported footer row {row_label!r}.")


def _family_title_from_key(family: str) -> str:
    replacements = {
        "main_regression": "Main Regression",
        "technical_fixed_effect_alternatives": "Technical Fixed-Effect Alternatives",
        "technical_clustering_alternatives": "Technical Clustering Alternatives",
        "variable_specification_alternatives": "Variable Specification Alternatives",
        "added_variable_alternatives": "Added-Variable Alternatives",
        "sample_split_interactions": "Sample-Split Interactions",
        "grid_search": "Grid Search",
        "time_variation": "Time Variation and Attenuation",
    }
    if family in replacements:
        return replacements[family]
    return family.replace("_", " ").title()


def _resolve_table_family(completed_results: list[CompletedRegressionResult]) -> str:
    families = {result.spec.family for result in completed_results}
    if not families:
        raise ValueError("completed_results must not be empty.")

    if len(families) == 1:
        return next(iter(families))

    non_main_families = families - {"main_regression"}
    if families == {"main_regression"} | non_main_families and len(non_main_families) == 1:
        return next(iter(non_main_families))

    family_list = ", ".join(sorted(families))
    raise ValueError(
        "All completed results must belong to the same family, except that "
        f"'main_regression' may accompany one comparison family. Found: {family_list}."
    )


def _resolve_table_family_from_specs(specs: list[RegressionSpec]) -> str:
    families = {spec.family for spec in specs}
    if not families:
        raise ValueError("specs must not be empty.")

    if len(families) == 1:
        return next(iter(families))

    non_main_families = families - {"main_regression"}
    if families == {"main_regression"} | non_main_families and len(non_main_families) == 1:
        return next(iter(non_main_families))

    family_list = ", ".join(sorted(families))
    raise ValueError(
        "All specs must belong to the same family, except that "
        f"'main_regression' may accompany one comparison family. Found: {family_list}."
    )


def _build_table_from_summary_frames(
    specs: list[RegressionSpec],
    coefficient_frames: list[pd.DataFrame],
    diagnostics_by_spec_key: dict[str, dict[str, object]],
) -> str:
    if not specs:
        raise ValueError("specs must not be empty.")

    model_numbers = [f"({index})" for index in range(1, len(specs) + 1)]
    short_labels = [_derive_short_label(spec) for spec in specs]

    term_order: list[str] = []
    seen_terms: set[str] = set()
    prepared_frames: list[pd.DataFrame] = []
    for frame in coefficient_frames:
        filtered = frame.loc[frame["Term"].notna()].copy()
        for term in filtered["Term"].tolist():
            if term not in seen_terms:
                seen_terms.add(term)
                term_order.append(term)
        if "Formatted_Cell" not in filtered.columns:
            filtered["Formatted_Cell"] = filtered.apply(_format_coefficient_cell, axis=1)
        prepared_frames.append(filtered[["Term", "Formatted_Cell"]].copy())

    ordered_terms = sorted(term_order, key=_term_sort_key)

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.35}")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.34\textwidth}"
        + (r">{\centering\arraybackslash}X" * len(specs))
        + "}"
    )
    lines.append(r"\hline")
    lines.append(
        "Regressor & " + " & ".join(_escape_latex(number) for number in model_numbers) + r" \\"
    )
    lines.append(
        " & " + " & ".join(r"\shortstack[c]{" + _escape_latex(label) + "}" for label in short_labels) + r" \\"
    )
    lines.append(r"\hline")

    for term in ordered_terms:
        row_values: list[str] = []
        for frame in prepared_frames:
            matching = frame.loc[frame["Term"] == term, "Formatted_Cell"]
            row_values.append(matching.iloc[0] if not matching.empty else "")
        lines.append(
            _escape_latex(_humanize_term(term)) + " & " + " & ".join(row_values) + r" \\"
        )

    lines.append(r"\hline")
    footer_rows = (
        "Firm FE",
        "Quarter FE",
        "Clustering",
        "Firm clusters",
        "Quarter clusters",
        "Joint time-variation F-test p-value",
        "Middle vs early one-sided p-value",
        "Late vs middle one-sided p-value",
        "N",
        "DF model",
        "DF resid",
        "R-squared",
        "Within R-squared",
        "Adj. R-squared",
    )
    for row_label in footer_rows:
        values = []
        for spec in specs:
            diagnostics = diagnostics_by_spec_key.get(spec.key, {})
            values.append(_escape_latex(_build_footer_value(spec, diagnostics, row_label)))
        lines.append(_escape_latex(row_label) + " & " + " & ".join(values) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabularx}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def build_family_regression_latex_table(completed_results: list[CompletedRegressionResult]) -> str:
    if not completed_results:
        raise ValueError("completed_results must not be empty.")

    _resolve_table_family(completed_results)

    coefficient_frames: list[pd.DataFrame] = []
    diagnostics_by_spec_key: dict[str, dict[str, object]] = {}
    specs = [completed.spec for completed in completed_results]
    for completed in completed_results:
        coefficient_table = build_coefficient_table(completed.result)
        filtered = filter_non_fixed_effect_coefficients(
            coefficient_table,
            completed.spec.fixed_effect_terms_to_exclude,
            exclude_intercept=False,
        )
        coefficient_frames.append(filtered)
        diagnostics_by_spec_key[completed.spec.key] = completed.diagnostics

    return _build_table_from_summary_frames(specs, coefficient_frames, diagnostics_by_spec_key)


def build_family_regression_latex_document(completed_results: list[CompletedRegressionResult]) -> str:
    if not completed_results:
        raise ValueError("completed_results must not be empty.")

    family_title = _family_title_from_key(_resolve_table_family(completed_results))
    table_fragment = build_family_regression_latex_table(completed_results)

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        r"\section*{" + _escape_latex(family_title) + "}",
        table_fragment,
        r"\end{document}",
    ]
    return "\n".join(lines)


def _split_regressions_for_latex_table(
    items: list[RegressionTableItem],
) -> list[list[RegressionTableItem]]:
    """Split regression columns so generated tables remain readable on a page."""
    return [
        items[start:start + MAX_REGRESSIONS_PER_LATEX_TABLE]
        for start in range(0, len(items), MAX_REGRESSIONS_PER_LATEX_TABLE)
    ]


def build_combined_regression_latex_document(
    family_results_map: list[tuple[str, list[CompletedRegressionResult]]],
) -> str:
    if not family_results_map:
        raise ValueError("family_results_map must not be empty.")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
    ]

    table_index = 0
    for family, completed_results in family_results_map:
        if not completed_results:
            continue
        family_title = _family_title_from_key(family)
        table_chunks = _split_regressions_for_latex_table(completed_results)
        for chunk_index, table_chunk in enumerate(table_chunks):
            if table_index > 0:
                lines.append(r"\newpage")
            title_suffix = "" if chunk_index == 0 else " (continued)"
            lines.append(r"\thispagestyle{empty}")
            lines.append(r"\section*{" + _escape_latex(family_title + title_suffix) + "}")
            lines.append(build_family_regression_latex_table(table_chunk))
            table_index += 1

    lines.append(r"\end{document}")
    return "\n".join(lines)


def build_family_regression_latex_table_from_summaries(
    specs: list[RegressionSpec],
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    if not specs:
        raise ValueError("specs must not be empty.")

    _resolve_table_family_from_specs(specs)
    diagnostics_by_spec_key = {
        str(row["Spec_Key"]): row.dropna().to_dict()
        for _, row in diagnostics.loc[diagnostics["Spec_Key"].isin([spec.key for spec in specs])].iterrows()
    } if not diagnostics.empty and "Spec_Key" in diagnostics.columns else {}

    coefficient_frames: list[pd.DataFrame] = []
    for spec in specs:
        if coefficients.empty or "Spec_Key" not in coefficients.columns:
            coefficient_frames.append(pd.DataFrame(columns=["Term", "Coefficient", "Std_Error", "p_value"]))
            continue
        coefficient_frames.append(coefficients.loc[coefficients["Spec_Key"] == spec.key].copy())

    return _build_table_from_summary_frames(specs, coefficient_frames, diagnostics_by_spec_key)


def build_family_regression_latex_document_from_summaries(
    specs: list[RegressionSpec],
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    if not specs:
        raise ValueError("specs must not be empty.")

    family_title = _family_title_from_key(_resolve_table_family_from_specs(specs))
    table_fragment = build_family_regression_latex_table_from_summaries(
        specs,
        coefficients,
        diagnostics,
    )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        r"\section*{" + _escape_latex(family_title) + "}",
        table_fragment,
        r"\end{document}",
    ]
    return "\n".join(lines)


def build_combined_regression_latex_document_from_summaries(
    family_specs_map: list[tuple[str, list[RegressionSpec]]],
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    if not family_specs_map:
        raise ValueError("family_specs_map must not be empty.")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
    ]

    table_index = 0
    for family, family_specs in family_specs_map:
        if not family_specs:
            continue
        family_title = _family_title_from_key(family)
        table_chunks = _split_regressions_for_latex_table(family_specs)
        for chunk_index, table_chunk in enumerate(table_chunks):
            if table_index > 0:
                lines.append(r"\newpage")
            title_suffix = "" if chunk_index == 0 else " (continued)"
            lines.append(r"\thispagestyle{empty}")
            lines.append(r"\section*{" + _escape_latex(family_title + title_suffix) + "}")
            lines.append(
                build_family_regression_latex_table_from_summaries(
                    table_chunk,
                    coefficients,
                    diagnostics,
                )
            )
            table_index += 1

    lines.append(r"\end{document}")
    return "\n".join(lines)


def build_chunked_regression_latex_document(
    specs: list[RegressionSpec],
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    title: str,
    regressions_per_table: int = 6,
    tables_per_page: int = 2,
) -> str:
    if not specs:
        raise ValueError("specs must not be empty.")
    if regressions_per_table <= 0:
        raise ValueError("regressions_per_table must be positive.")
    if tables_per_page <= 0:
        raise ValueError("tables_per_page must be positive.")

    chunks = [
        specs[start:start + regressions_per_table]
        for start in range(0, len(specs), regressions_per_table)
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        r"\section*{" + _escape_latex(title) + "}",
    ]

    for index, chunk in enumerate(chunks):
        if index > 0:
            if index % tables_per_page == 0:
                lines.append(r"\newpage")
            else:
                lines.append(r"\vspace{1.5em}")
        start = index * regressions_per_table + 1
        end = start + len(chunk) - 1
        lines.append(r"\subsection*{" + _escape_latex(f"Models {start}-{end}") + "}")
        lines.append(
            build_family_regression_latex_table_from_summaries(
                chunk,
                coefficients,
                diagnostics,
            )
        )

    lines.append(r"\end{document}")
    return "\n".join(lines)


def _normalize_saved_bhar_columns(
    event_level: pd.DataFrame,
    *,
    columns: tuple[str, ...] = tuple(
        dict.fromkeys(
            [
                *(f"BHAR_{int(day_start)}_{int(day_end)}" for day_start, day_end in ALTERNATIVE_REGRESSION_BHAR_WINDOWS),
                f"BHAR_{int(MAIN_REGRESSION_BHAR_WINDOW[0])}_{int(MAIN_REGRESSION_BHAR_WINDOW[1])}",
            ]
        )
    ),
) -> pd.DataFrame:
    out = event_level.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def attach_pre_announcement_market_cap(
    event_level: pd.DataFrame,
    data_dir: Path,
    *,
    lag_trading_days: int = PRE_ANNOUNCEMENT_MARKET_CAP_LAG_TRADING_DAYS,
) -> pd.DataFrame:
    """Attach each event's market cap from the preceding observed trading day.

    The daily market-cap cache is keyed by formation year and instrument.  This
    deliberately keeps the event-study control separate from the formation-date
    market cap used to construct the characteristic benchmark portfolios. If no
    preceding daily observation is available after checking the supplemental
    Datastream cache, the positive formation-date market cap is used as a
    documented final fallback.
    """
    required_columns = {FORMATION_YEAR_COLUMN, FIRM_IDENTIFIER_COLUMN, "Ann_Date"}
    missing_columns = sorted(required_columns.difference(event_level.columns))
    if missing_columns:
        raise KeyError(
            "Cannot attach pre-announcement market capitalization. Missing columns: "
            f"{missing_columns}."
        )
    if lag_trading_days < 1:
        raise ValueError("lag_trading_days must be at least one.")

    out = event_level.copy()
    out[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN] = np.nan
    out[PRE_ANNOUNCEMENT_MARKET_CAP_DATE_COLUMN] = pd.NaT
    out[PRE_ANNOUNCEMENT_MARKET_CAP_METHOD_COLUMN] = pd.NA
    out["Ann_Date"] = pd.to_datetime(out["Ann_Date"], errors="coerce").dt.normalize()
    out[FIRM_IDENTIFIER_COLUMN] = out[FIRM_IDENTIFIER_COLUMN].astype("string")

    formation_years = pd.to_numeric(out[FORMATION_YEAR_COLUMN], errors="coerce")
    for formation_year in sorted(formation_years.dropna().astype(int).unique()):
        event_mask = formation_years.eq(formation_year) & out["Ann_Date"].notna()
        if not bool(event_mask.any()):
            continue

        cache_path = (
            data_dir
            / "yearly"
            / str(formation_year)
            / "_cache"
            / "daily_market_caps_completed.csv"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                "Missing daily market-cap cache required for the regression suite: "
                f"{cache_path}."
            )

        event_instruments = set(out.loc[event_mask, FIRM_IDENTIFIER_COLUMN].dropna().astype(str))
        cache_chunks: list[pd.DataFrame] = []
        cache_paths = [
            cache_path,
            cache_path.parent / SUPPLEMENTAL_PRE_ANNOUNCEMENT_MARKET_CAP_FILENAME,
        ]
        for cache_priority, candidate_path in enumerate(cache_paths):
            if not candidate_path.exists():
                continue
            for chunk in pd.read_csv(
                candidate_path,
                usecols=["Date", "Instrument", "MarketCap", "MarketCapMethod"],
                chunksize=250_000,
            ):
                chunk = chunk.loc[
                    chunk["Instrument"].astype(str).isin(event_instruments)
                ].copy()
                if not chunk.empty:
                    chunk["_Cache_Priority"] = cache_priority
                    cache_chunks.append(chunk)
        if not cache_chunks:
            continue

        market_caps = pd.concat(cache_chunks, ignore_index=True)
        market_caps["Date"] = pd.to_datetime(market_caps["Date"], errors="coerce").dt.normalize()
        market_caps["MarketCap"] = pd.to_numeric(market_caps["MarketCap"], errors="coerce")
        market_caps = market_caps.dropna(subset=["Date", "Instrument"]).sort_values(
            ["Instrument", "Date", "_Cache_Priority"], kind="stable"
        )
        # A supplemental request must take precedence over the original cache when
        # both provide the same instrument-date observation.
        market_caps = market_caps.drop_duplicates(["Instrument", "Date"], keep="last")

        for instrument, event_indices in out.loc[event_mask].groupby(FIRM_IDENTIFIER_COLUMN).groups.items():
            instrument_caps = market_caps.loc[
                market_caps["Instrument"].astype(str).eq(str(instrument))
            ].reset_index(drop=True)
            if instrument_caps.empty:
                continue
            available_dates = instrument_caps["Date"].to_numpy(dtype="datetime64[ns]")
            event_dates = out.loc[event_indices, "Ann_Date"].to_numpy(dtype="datetime64[ns]")
            # ``side='left'`` excludes the announcement date itself, then selects
            # the immediately preceding observed trading date for each security.
            positions = np.searchsorted(available_dates, event_dates, side="left") - lag_trading_days
            valid_positions = positions >= 0
            if not bool(valid_positions.any()):
                continue
            target_indices = np.asarray(event_indices)[valid_positions]
            selected = instrument_caps.iloc[positions[valid_positions]]
            out.loc[target_indices, PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN] = selected[
                "MarketCap"
            ].to_numpy()
            out.loc[target_indices, PRE_ANNOUNCEMENT_MARKET_CAP_DATE_COLUMN] = selected[
                "Date"
            ].to_numpy()
            out.loc[target_indices, PRE_ANNOUNCEMENT_MARKET_CAP_METHOD_COLUMN] = selected[
                "MarketCapMethod"
            ].to_numpy()

    formation_market_cap = (
        pd.to_numeric(out["Market_Cap_Current"], errors="coerce")
        if "Market_Cap_Current" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    pre_announcement_market_cap = pd.to_numeric(
        out[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce"
    )
    formation_fallback_mask = (
        ~pre_announcement_market_cap.gt(0) & formation_market_cap.gt(0)
    )
    if bool(formation_fallback_mask.any()):
        out.loc[formation_fallback_mask, PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN] = (
            formation_market_cap.loc[formation_fallback_mask]
        )
        if "Formation_Date" in out.columns:
            out.loc[formation_fallback_mask, PRE_ANNOUNCEMENT_MARKET_CAP_DATE_COLUMN] = pd.to_datetime(
                out.loc[formation_fallback_mask, "Formation_Date"], errors="coerce"
            ).to_numpy()
        out.loc[formation_fallback_mask, PRE_ANNOUNCEMENT_MARKET_CAP_METHOD_COLUMN] = (
            FORMATION_DATE_MARKET_CAP_FALLBACK_METHOD
        )

    return out


def _add_low_analyst_following_dummies(
    frame: pd.DataFrame,
    cutoffs: tuple[int, ...] = (10, 5, 3),
) -> pd.DataFrame:
    out = frame.copy()
    analyst_count = pd.to_numeric(out["Forecast_Analyst_Count"], errors="coerce")
    out["Forecast_Analyst_Count"] = analyst_count
    for cutoff in cutoffs:
        column = f"Low_Analyst_Following_LT_{int(cutoff)}"
        out[column] = pd.Series(
            np.where(analyst_count.notna(), (analyst_count < int(cutoff)).astype(int), np.nan),
            index=out.index,
            dtype=float,
        )
    return out


def build_regression_suite_dataset(
    data_dir: Path,
    *,
    time_period_length_years: int = 5,
    explicit_time_periods: tuple[tuple[int, int], ...] | None = None,
    analyst_following_cutoffs: tuple[int, ...] = (10, 5, 3),
    abnormal_returns_filename: str = "earnings_abnormal_returns.csv",
    additional_bhar_windows: tuple[tuple[int, int], ...] = (),
) -> pd.DataFrame:
    abnormal_returns = load_abnormal_returns_with_groups(
        data_dir,
        abnormal_returns_filename=abnormal_returns_filename,
    )
    additional_bhar_columns = tuple(
        f"BHAR_{int(day_start)}_{int(day_end)}"
        for day_start, day_end in additional_bhar_windows
    )
    event_level = collapse_to_event_level(
        abnormal_returns,
        additional_bhar_columns=additional_bhar_columns,
    )

    if SUE_PLOT_GROUP_COLUMN not in event_level.columns:
        event_level = add_plot_group_column(
            event_level,
            group_column=SUE_GROUP_COLUMN,
            plot_group_column=SUE_PLOT_GROUP_COLUMN,
            computation_group_count=SUE_COMPUTATION_GROUP_COUNT,
            plot_group_count=SUE_PLOT_GROUP_COUNT,
        )

    event_level = _normalize_saved_bhar_columns(
        event_level,
        columns=tuple(
            dict.fromkeys(
                [
                    *(
                        f"BHAR_{int(day_start)}_{int(day_end)}"
                        for day_start, day_end in ALTERNATIVE_REGRESSION_BHAR_WINDOWS
                    ),
                    f"BHAR_{int(MAIN_REGRESSION_BHAR_WINDOW[0])}_{int(MAIN_REGRESSION_BHAR_WINDOW[1])}",
                    *additional_bhar_columns,
                ]
            )
        ),
    )
    event_level = _add_low_analyst_following_dummies(
        event_level,
        cutoffs=analyst_following_cutoffs,
    )

    stock_universe = load_stock_universe_snapshots(data_dir)
    event_level = attach_universe_snapshot(event_level, stock_universe)
    event_level = attach_pre_announcement_market_cap(event_level, data_dir)
    formation_market_cap_fallback_count = int(
        event_level[PRE_ANNOUNCEMENT_MARKET_CAP_METHOD_COLUMN]
        .eq(FORMATION_DATE_MARKET_CAP_FALLBACK_METHOD)
        .sum()
    )
    print(
        f"Formation-date market-cap fallback used for {formation_market_cap_fallback_count:,} "
        f"of {len(event_level):,} earnings events after checking both daily caches."
    )
    pre_announcement_market_cap = pd.to_numeric(
        event_level[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce"
    )
    missing_or_nonpositive_market_cap_count = int(
        (~pre_announcement_market_cap.gt(0)).sum()
    )
    if missing_or_nonpositive_market_cap_count:
        warnings.warn(
            f"{missing_or_nonpositive_market_cap_count:,} of {len(event_level):,} earnings events "
            f"have no positive {PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN}. Size-based regression "
            "specifications and market-cap filters will exclude these events.",
            stacklevel=2,
        )
    event_level = apply_market_cap_analysis_split(
        event_level,
        data_dir,
        market_cap_column=PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN,
    )

    available_years = sorted(
        event_level[FORMATION_YEAR_COLUMN].dropna().astype(int).unique().tolist()
    )
    time_periods = build_time_periods(
        available_years,
        period_length=time_period_length_years,
        explicit_periods=list(explicit_time_periods) if explicit_time_periods is not None else None,
    )
    event_level = assign_time_periods(event_level, time_periods)

    if ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN in event_level.columns:
        event_level[ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN] = event_level[
            ANALYSIS_MARKET_CAP_SPLIT_GROUP_COLUMN
        ].astype("string")

    return event_level


def build_regression_registry(specs: list[RegressionSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = asdict(spec)
        row["fixed_effect_terms_to_exclude"] = " | ".join(spec.fixed_effect_terms_to_exclude)
        row["spec_signature"] = _spec_signature(spec)
        rows.append(row)
    return pd.DataFrame(rows)


def filter_non_fixed_effect_coefficients(
    coefficient_table: pd.DataFrame,
    fixed_effect_terms_to_exclude: tuple[str, ...],
    *,
    exclude_intercept: bool = False,
) -> pd.DataFrame:
    if coefficient_table.empty:
        return coefficient_table.copy()

    keep_mask = pd.Series(True, index=coefficient_table.index)
    if exclude_intercept:
        keep_mask &= coefficient_table["Term"].ne("Intercept")
    for term_prefix in fixed_effect_terms_to_exclude:
        keep_mask &= ~coefficient_table["Term"].str.contains(term_prefix, regex=False)
    return coefficient_table.loc[keep_mask].reset_index(drop=True)


def add_coefficient_significance_columns(coefficient_table: pd.DataFrame) -> pd.DataFrame:
    """Add explicit two- and one-sided p-values plus conventional significance markers."""
    result = coefficient_table.copy()
    if result.empty:
        result["Two_Sided_p_Value"] = pd.Series(dtype=float)
        result["One_Sided_p_Value_Greater"] = pd.Series(dtype=float)
        result["One_Sided_p_Value_Less"] = pd.Series(dtype=float)
        result["Significance_Stars"] = pd.Series(dtype=str)
        return result

    p_values = pd.to_numeric(result["p_value"], errors="coerce")
    estimates = pd.to_numeric(result["Coefficient"], errors="coerce")
    half_p_values = p_values.clip(lower=0.0, upper=1.0) / 2.0
    result["Two_Sided_p_Value"] = p_values
    result["One_Sided_p_Value_Greater"] = np.where(
        estimates > 0,
        half_p_values,
        1.0 - half_p_values,
    )
    result["One_Sided_p_Value_Less"] = np.where(
        estimates < 0,
        half_p_values,
        1.0 - half_p_values,
    )
    result["Significance_Stars"] = np.select(
        [p_values <= 0.01, p_values <= 0.05, p_values <= 0.10],
        ["***", "**", "*"],
        default="",
    )
    return result


def _spec_uses_absorbed_fixed_effects(spec: RegressionSpec) -> bool:
    absorbed_terms = (
        f"C({ANNOUNCEMENT_QUARTER_COLUMN})",
        f"C({FIRM_IDENTIFIER_COLUMN})",
    )
    return any(term in spec.formula for term in absorbed_terms)


def _build_status_row(
    spec: RegressionSpec,
    *,
    status: str,
    reason: str = "",
    error_stage: str = "",
    error_type: str = "",
    error_module: str = "",
    traceback_text: str = "",
    dependent_variable: str = "",
    non_missing_dependent_values: int | None = None,
    sample_size: int | None = None,
    saved_coefficient_count: int | None = None,
    regression_input_path: str = "",
    warning_count: int | None = None,
    warning_summary: str = "",
) -> dict[str, object]:
    return {
        "Spec_Key": spec.key,
        "Family": spec.family,
        "Label": spec.label,
        "Formula": spec.formula,
        "Cluster_Spec": spec.cluster_spec,
        "Row_Filter_Query": spec.row_filter_query,
        "Spec_Signature": _spec_signature(spec),
        "Enabled": spec.enabled,
        "Status": status,
        "Reason": reason,
        "Error_Stage": error_stage,
        "Error_Type": error_type,
        "Error_Module": error_module,
        "Traceback": traceback_text,
        "Dependent_Variable": dependent_variable,
        "Non_Missing_Dependent_Values": non_missing_dependent_values,
        "Sample_Size": sample_size,
        "Saved_Coefficient_Count": saved_coefficient_count,
        "Regression_Input_Path": regression_input_path,
        "Warning_Count": warning_count,
        "Warning_Summary": warning_summary,
        "Notes": spec.notes,
    }


def _exception_details(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error_module": type(exc).__module__,
        "traceback_text": traceback.format_exc(),
    }


def _flatten_diagnostics_table(diagnostics_table: pd.DataFrame) -> dict[str, object]:
    if "Value" not in diagnostics_table.columns:
        return diagnostics_table.to_dict()
    return diagnostics_table["Value"].to_dict()


def safe_scalar(value: object) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(scalar):
        return None
    return scalar


def _first_test_scalar(value: object) -> float | None:
    values = np.asarray(value).reshape(-1)
    return safe_scalar(values[0]) if values.size else None


def _build_joint_sue_interaction_test(
    result,
    *,
    interaction_terms: list[str] | None = None,
) -> dict[str, object]:
    """Test whether the selected SUE interaction coefficients are jointly zero."""
    parameter_names = [str(name) for name in result.params.index]
    if interaction_terms is None:
        interaction_terms = [
            term for term in parameter_names if "SUE" in term and ":" in term
        ]
    missing_terms = [term for term in interaction_terms if term not in parameter_names]
    if missing_terms:
        raise ValueError(
            "The requested SUE interaction term(s) were not estimated: "
            + ", ".join(missing_terms)
        )
    if not interaction_terms:
        return {
            "SUE_Interaction_Joint_Test_Applied": False,
            "SUE_Interaction_Joint_Test_p_value": np.nan,
            "SUE_Interaction_Joint_Test_Statistic": np.nan,
            "SUE_Interaction_Joint_Test_DF": np.nan,
            "SUE_Interaction_Joint_Test_DF_Denominator": np.nan,
            "SUE_Interaction_Joint_Test_Distribution": "",
            "SUE_Interaction_Joint_Test_Terms": "",
            "SUE_Interaction_Joint_Test_Status": "not_applicable_no_sue_interaction",
        }

    tested_indices = [parameter_names.index(term) for term in interaction_terms]
    restriction_matrix = np.zeros((len(tested_indices), len(parameter_names)))
    restriction_matrix[np.arange(len(tested_indices)), tested_indices] = 1.0

    numerator_df = len(tested_indices)
    denominator_df = safe_scalar(getattr(result, "df_resid", np.nan))

    try:
        if hasattr(result, "f_test"):
            test_result = result.f_test(restriction_matrix)
            statistic = _first_test_scalar(getattr(test_result, "fvalue", np.nan))
            p_value = _first_test_scalar(getattr(test_result, "pvalue", np.nan))
            test_denominator_df = safe_scalar(getattr(test_result, "df_denom", np.nan))
            if test_denominator_df is not None:
                denominator_df = test_denominator_df
        else:
            # AbsorbingLS provides an asymptotic Wald statistic. Dividing that
            # statistic by the number of restrictions and using the residual
            # degrees of freedom yields the requested finite-sample F form.
            wald_result = result.wald_test(restriction_matrix)
            wald_statistic = _first_test_scalar(
                getattr(wald_result, "statistic", getattr(wald_result, "stat", np.nan))
            )
            if wald_statistic is None or denominator_df is None or denominator_df <= 0:
                raise ValueError("The Wald-to-F conversion requires a finite statistic and residual degrees of freedom.")
            statistic = wald_statistic / numerator_df
            p_value = float(stats.f.sf(statistic, numerator_df, denominator_df))
    except Exception as exc:
        return {
            "SUE_Interaction_Joint_Test_Applied": True,
            "SUE_Interaction_Joint_Test_p_value": np.nan,
            "SUE_Interaction_Joint_Test_Statistic": np.nan,
            "SUE_Interaction_Joint_Test_DF": numerator_df,
            "SUE_Interaction_Joint_Test_DF_Denominator": denominator_df,
            "SUE_Interaction_Joint_Test_Distribution": "F",
            "SUE_Interaction_Joint_Test_Terms": " | ".join(
                parameter_names[index] for index in tested_indices
            ),
            "SUE_Interaction_Joint_Test_Status": f"failed: {type(exc).__name__}: {exc}",
        }

    return {
        "SUE_Interaction_Joint_Test_Applied": True,
        "SUE_Interaction_Joint_Test_p_value": p_value,
        "SUE_Interaction_Joint_Test_Statistic": statistic,
        "SUE_Interaction_Joint_Test_DF": numerator_df,
        "SUE_Interaction_Joint_Test_DF_Denominator": denominator_df,
        "SUE_Interaction_Joint_Test_Distribution": "F",
        "SUE_Interaction_Joint_Test_Terms": " | ".join(
            parameter_names[index] for index in tested_indices
        ),
        "SUE_Interaction_Joint_Test_Status": "completed",
    }


def _build_joint_reported_coefficient_test(
    result,
    spec: RegressionSpec,
) -> dict[str, object]:
    """Test whether all non-intercept, non-fixed-effect coefficients are jointly zero."""
    coefficient_table = filter_non_fixed_effect_coefficients(
        build_coefficient_table(result),
        spec.fixed_effect_terms_to_exclude,
        exclude_intercept=True,
    )
    parameter_names = [str(name) for name in result.params.index]
    reported_terms = coefficient_table["Term"].astype(str).tolist()
    if "Intercept" in reported_terms:
        raise AssertionError("Joint F-statistics must not include the intercept.")
    missing_terms = [term for term in reported_terms if term not in parameter_names]
    if missing_terms:
        raise ValueError(
            "The reported coefficient term(s) were not estimated: " + ", ".join(missing_terms)
        )
    if not reported_terms:
        return {
            "Reported_Coefficient_Joint_Test_Applied": False,
            "Reported_Coefficient_Joint_F_Statistic": np.nan,
            "Reported_Coefficient_Joint_F_p_value": np.nan,
            "Reported_Coefficient_Joint_F_DF_Numerator": np.nan,
            "Reported_Coefficient_Joint_F_DF_Denominator": np.nan,
            "Reported_Coefficient_Joint_F_Terms": "",
            "Reported_Coefficient_Joint_F_Status": "not_applicable_no_reported_coefficients",
        }

    tested_indices = [parameter_names.index(term) for term in reported_terms]
    denominator_df = safe_scalar(getattr(result, "df_resid", np.nan))

    try:
        parameter_values = result.params.loc[parameter_names].to_numpy(dtype=float)
        covariance = _parameter_covariance(result, parameter_names)
        restricted_values = parameter_values[tested_indices]
        restricted_covariance = covariance[np.ix_(tested_indices, tested_indices)]
        numerator_df = int(np.linalg.matrix_rank(restricted_covariance))
        if numerator_df <= 0 or denominator_df is None or denominator_df <= 0:
            raise ValueError("The Wald-to-F conversion requires a full-rank covariance matrix and residual degrees of freedom.")
        wald_statistic = float(
            restricted_values.T @ np.linalg.pinv(restricted_covariance) @ restricted_values
        )
        if not np.isfinite(wald_statistic) or wald_statistic < 0:
            raise ValueError("The joint Wald statistic must be finite and non-negative.")
        statistic = wald_statistic / numerator_df
        p_value = float(stats.f.sf(statistic, numerator_df, denominator_df))
    except Exception as exc:
        return {
            "Reported_Coefficient_Joint_Test_Applied": True,
            "Reported_Coefficient_Joint_F_Statistic": np.nan,
            "Reported_Coefficient_Joint_F_p_value": np.nan,
            "Reported_Coefficient_Joint_F_DF_Numerator": len(tested_indices),
            "Reported_Coefficient_Joint_F_DF_Denominator": denominator_df,
            "Reported_Coefficient_Joint_F_Terms": " | ".join(reported_terms),
            "Reported_Coefficient_Joint_F_Status": f"failed: {type(exc).__name__}: {exc}",
        }

    return {
        "Reported_Coefficient_Joint_Test_Applied": True,
        "Reported_Coefficient_Joint_F_Statistic": statistic,
        "Reported_Coefficient_Joint_F_p_value": p_value,
        "Reported_Coefficient_Joint_F_DF_Numerator": numerator_df,
        "Reported_Coefficient_Joint_F_DF_Denominator": denominator_df,
        "Reported_Coefficient_Joint_F_Terms": " | ".join(reported_terms),
        "Reported_Coefficient_Joint_F_Status": "completed",
    }


def _empty_ordered_time_variation_tests() -> dict[str, object]:
    return {
        "Time_Variation_Tests_Applied": False,
        "Time_Variation_Joint_F_Statistic": np.nan,
        "Time_Variation_Joint_F_p_value": np.nan,
        "Time_Variation_Joint_F_DF_Numerator": np.nan,
        "Time_Variation_Joint_F_DF_Denominator": np.nan,
        "Middle_vs_Early_Estimate": np.nan,
        "Middle_vs_Early_Std_Error": np.nan,
        "Middle_vs_Early_t_Statistic": np.nan,
        "Middle_vs_Early_One_Sided_p_value": np.nan,
        "Late_vs_Middle_Estimate": np.nan,
        "Late_vs_Middle_Std_Error": np.nan,
        "Late_vs_Middle_t_Statistic": np.nan,
        "Late_vs_Middle_One_Sided_p_value": np.nan,
        "Time_Variation_Tests_Status": "not_applicable",
    }


def _parameter_covariance(result, parameter_names: list[str]) -> np.ndarray:
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


def _find_time_interaction_term(
    parameter_names: list[str],
    *,
    regressor: str,
    period_column: str,
    period_label: str,
) -> str:
    period_term = f"C({period_column})[T.{period_label}]"
    candidates = (
        f"{regressor}:{period_term}",
        f"{period_term}:{regressor}",
    )
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    raise ValueError(
        "Could not locate the time-period interaction term for "
        f"{period_label!r}; expected one of {candidates}."
    )


def _build_one_sided_negative_contrast(
    *,
    parameter_values: np.ndarray,
    covariance: np.ndarray,
    contrast: np.ndarray,
    degrees_of_freedom: float | None,
) -> dict[str, float]:
    estimate = float(contrast @ parameter_values)
    variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The contrast variance must be finite and positive.")

    std_error = float(np.sqrt(variance))
    t_statistic = estimate / std_error
    if degrees_of_freedom is not None and np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0:
        p_value = float(stats.t.cdf(t_statistic, degrees_of_freedom))
    else:
        p_value = float(stats.norm.cdf(t_statistic))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "t_statistic": t_statistic,
        "one_sided_p_value": p_value,
    }


def _build_ordered_time_variation_tests(
    result,
    spec: RegressionSpec,
) -> dict[str, object]:
    diagnostics = _empty_ordered_time_variation_tests()
    if not spec.ordered_time_periods:
        return diagnostics
    if not spec.ordered_time_regressor or not spec.ordered_time_period_column:
        diagnostics["Time_Variation_Tests_Status"] = "failed: incomplete time-variation test configuration"
        return diagnostics

    try:
        _, middle_period, late_period = spec.ordered_time_periods
        parameter_names = [str(name) for name in result.params.index]
        middle_term = _find_time_interaction_term(
            parameter_names,
            regressor=spec.ordered_time_regressor,
            period_column=spec.ordered_time_period_column,
            period_label=middle_period,
        )
        late_term = _find_time_interaction_term(
            parameter_names,
            regressor=spec.ordered_time_regressor,
            period_column=spec.ordered_time_period_column,
            period_label=late_period,
        )

        parameter_values = result.params.loc[parameter_names].to_numpy(dtype=float)
        covariance = _parameter_covariance(result, parameter_names)
        middle_index = parameter_names.index(middle_term)
        late_index = parameter_names.index(late_term)
        middle_contrast = np.zeros(len(parameter_names))
        middle_contrast[middle_index] = 1.0
        late_minus_middle_contrast = np.zeros(len(parameter_names))
        late_minus_middle_contrast[late_index] = 1.0
        late_minus_middle_contrast[middle_index] = -1.0
        degrees_of_freedom = safe_scalar(getattr(result, "df_resid", np.nan))

        middle_test = _build_one_sided_negative_contrast(
            parameter_values=parameter_values,
            covariance=covariance,
            contrast=middle_contrast,
            degrees_of_freedom=degrees_of_freedom,
        )
        late_middle_test = _build_one_sided_negative_contrast(
            parameter_values=parameter_values,
            covariance=covariance,
            contrast=late_minus_middle_contrast,
            degrees_of_freedom=degrees_of_freedom,
        )
        joint_test = _build_joint_sue_interaction_test(
            result,
            interaction_terms=[middle_term, late_term],
        )

        diagnostics.update(
            {
                "Time_Variation_Tests_Applied": True,
                "Time_Variation_Joint_F_Statistic": joint_test["SUE_Interaction_Joint_Test_Statistic"],
                "Time_Variation_Joint_F_p_value": joint_test["SUE_Interaction_Joint_Test_p_value"],
                "Time_Variation_Joint_F_DF_Numerator": joint_test["SUE_Interaction_Joint_Test_DF"],
                "Time_Variation_Joint_F_DF_Denominator": joint_test[
                    "SUE_Interaction_Joint_Test_DF_Denominator"
                ],
                "Middle_vs_Early_Estimate": middle_test["estimate"],
                "Middle_vs_Early_Std_Error": middle_test["std_error"],
                "Middle_vs_Early_t_Statistic": middle_test["t_statistic"],
                "Middle_vs_Early_One_Sided_p_value": middle_test["one_sided_p_value"],
                "Late_vs_Middle_Estimate": late_middle_test["estimate"],
                "Late_vs_Middle_Std_Error": late_middle_test["std_error"],
                "Late_vs_Middle_t_Statistic": late_middle_test["t_statistic"],
                "Late_vs_Middle_One_Sided_p_value": late_middle_test["one_sided_p_value"],
                "Time_Variation_Tests_Status": "completed",
            }
        )
    except Exception as exc:
        diagnostics["Time_Variation_Tests_Status"] = f"failed: {type(exc).__name__}: {exc}"
    return diagnostics


def _warning_origin(record: warnings.WarningMessage) -> str:
    filename = Path(str(record.filename)).name if getattr(record, "filename", None) else "<unknown>"
    lineno = getattr(record, "lineno", None)
    return filename if lineno is None else f"{filename}:{lineno}"


def _serialize_warning_records(
    warning_records: list[warnings.WarningMessage],
    *,
    stage: str,
) -> list[dict[str, str]]:
    return [
        {
            "stage": stage,
            "category": record.category.__name__,
            "message": str(record.message),
            "origin": _warning_origin(record),
        }
        for record in warning_records
    ]


def _format_warning_summary(warning_records: list[dict[str, str]]) -> str:
    if not warning_records:
        return ""

    parts: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in warning_records:
        key = (
            record.get("stage", ""),
            record.get("category", ""),
            record.get("message", ""),
            record.get("origin", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{key[0]}: {key[1]} at {key[3]} ({key[2]})")
    return " | ".join(parts)


def _find_problematic_coefficient_terms(
    coefficient_table: pd.DataFrame,
    *,
    exclude_intercept: bool = False,
) -> list[str]:
    if coefficient_table.empty:
        return []

    numeric_columns = [
        column
        for column in ("Std_Error", "t_stat", "p_value", "CI_95_Low", "CI_95_High")
        if column in coefficient_table.columns
    ]
    if not numeric_columns:
        return []

    invalid_mask = coefficient_table[numeric_columns].isna().any(axis=1)
    if exclude_intercept:
        invalid_mask &= coefficient_table["Term"].ne("Intercept")
    return coefficient_table.loc[invalid_mask, "Term"].astype(str).tolist()


def _emit_regression_warning(
    spec: RegressionSpec,
    warning_records: list[dict[str, str]],
    *,
    sample_size: int | None = None,
    problematic_terms: list[str] | None = None,
) -> None:
    if not warning_records:
        return

    details: list[str] = [f"spec={spec.key}"]
    if sample_size is not None:
        details.append(f"n={sample_size}")
    if problematic_terms:
        displayed_terms = ", ".join(problematic_terms[:8])
        details.append(f"problem_terms={displayed_terms}")
        if len(problematic_terms) > 8:
            details.append(f"+{len(problematic_terms) - 8} more")

    message = f"[regression warning] {'; '.join(details)}; {_format_warning_summary(warning_records)}"
    if hasattr(tqdm, "write"):
        tqdm.write(message)
    else:
        print(message)


def _build_suite_diagnostics_row(
    model_output: dict[str, object],
    spec: RegressionSpec,
) -> dict[str, object]:
    result = model_output["result"]
    diagnostics = {
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
        "Warning_Count": int(len(model_output.get("warning_records", []))),
        "Warning_Summary": _format_warning_summary(model_output.get("warning_records", [])),
    }
    diagnostics.update(_build_joint_reported_coefficient_test(result, spec))
    diagnostics.update(_build_joint_sue_interaction_test(result))
    diagnostics.update(_build_ordered_time_variation_tests(result, spec))
    diagnostics.update(_build_within_r_squared_diagnostics(model_output, spec))
    return diagnostics


def _dependent_variable(formula: str) -> str:
    if "~" not in formula:
        raise ValueError(f"Invalid formula {formula!r}.")
    return formula.split("~", 1)[0].strip()


def _build_fixed_effects_only_formula(spec: RegressionSpec) -> str | None:
    """Return the nested model containing only the fixed effects in ``spec``.

    The resulting comparison holds the estimation sample and included fixed
    effects constant, so the reduction in SSR measures fit attributable to the
    non-fixed-effect regressors.  Formulae in this suite use additive top-level
    terms; interactions that merely contain a fixed-effect variable are not
    retained in the restricted model.
    """
    dependent_variable, separator, rhs = spec.formula.partition("~")
    if not separator:
        raise ValueError(f"Invalid formula {spec.formula!r}.")

    fixed_effect_terms = {
        term.strip() for term in spec.fixed_effect_terms_to_exclude
    }
    retained_terms = [
        term.strip()
        for term in rhs.split("+")
        if term.strip() in fixed_effect_terms
    ]
    if not retained_terms:
        return None
    return f"{dependent_variable.strip()} ~ {' + '.join(retained_terms)}"


def _build_within_r_squared_diagnostics(
    model_output: dict[str, object],
    spec: RegressionSpec,
) -> dict[str, object]:
    """Calculate FE-conditional R-squared from nested-model residual sums.

    For a model with fixed effects, within R-squared is
    ``1 - SSR(full model) / SSR(fixed-effects-only model)``.  It is not
    defined for specifications without fixed effects, for which full-model
    R-squared remains the appropriate goodness-of-fit measure.
    """
    restricted_formula = _build_fixed_effects_only_formula(spec)
    if restricted_formula is None:
        return {
            "Within_R_Squared": np.nan,
            "Within_R_Squared_Status": "not_applicable_no_fixed_effects",
            "Within_R_Squared_Restricted_Formula": "",
        }

    try:
        restricted_output = fit_formula_model(
            model_output["data"],
            formula=restricted_formula,
            model_label=f"{spec.label}: fixed-effects-only restricted model",
            cluster_spec=spec.cluster_spec,
        )
        full_result = model_output["result"]
        restricted_result = restricted_output["result"]
        if int(full_result.nobs) != int(restricted_result.nobs):
            raise ValueError(
                "full and fixed-effects-only models use different estimation samples"
            )

        full_ssr = float(np.square(np.asarray(full_result.resid, dtype=float)).sum())
        restricted_ssr = float(
            np.square(np.asarray(restricted_result.resid, dtype=float)).sum()
        )
        if not np.isfinite(restricted_ssr) or restricted_ssr <= 0:
            raise ValueError("fixed-effects-only residual sum of squares is not positive")
        within_r_squared = 1.0 - (full_ssr / restricted_ssr)
        return {
            "Within_R_Squared": within_r_squared,
            "Within_R_Squared_Status": "completed",
            "Within_R_Squared_Restricted_Formula": restricted_formula,
        }
    except Exception as exc:
        return {
            "Within_R_Squared": np.nan,
            "Within_R_Squared_Status": f"failed: {type(exc).__name__}: {exc}",
            "Within_R_Squared_Restricted_Formula": restricted_formula,
        }


def _apply_row_filter(dataset: pd.DataFrame, row_filter_query: str) -> pd.DataFrame:
    if not row_filter_query.strip():
        return dataset
    return dataset.query(row_filter_query, engine="python").copy()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_suite_metadata(output_dir: Path) -> dict[str, object] | None:
    metadata_path = output_dir / SUITE_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_reusable_suite_state(
    output_dir: Path,
    specs: list[RegressionSpec],
    suite_metadata: dict[str, object] | None,
) -> dict[str, object]:
    if suite_metadata is None:
        return {"completed_spec_keys": set()}

    saved_metadata = _load_suite_metadata(output_dir)
    if saved_metadata != suite_metadata:
        return {"completed_spec_keys": set()}

    status = _read_csv_if_exists(output_dir / "regression_status.csv")
    coefficients = _read_csv_if_exists(output_dir / "non_fixed_effect_coefficients.csv")
    diagnostics = _read_csv_if_exists(output_dir / "model_diagnostics.csv")

    if status.empty or "Spec_Key" not in status.columns:
        return {"completed_spec_keys": set()}

    spec_signature_by_key = {spec.key: _spec_signature(spec) for spec in specs}
    valid_completed_spec_keys: set[str] = set()
    valid_rows: list[pd.Series] = []

    for _, row in status.iterrows():
        spec_key = str(row.get("Spec_Key", ""))
        if spec_key not in spec_signature_by_key:
            continue
        if str(row.get("Status", "")) != "completed":
            continue
        if str(row.get("Spec_Signature", "")) != spec_signature_by_key[spec_key]:
            continue
        regression_input_path = str(row.get("Regression_Input_Path", "")).strip()
        if not regression_input_path or not Path(regression_input_path).exists():
            continue
        valid_completed_spec_keys.add(spec_key)
        valid_rows.append(row)

    if not valid_completed_spec_keys:
        return {"completed_spec_keys": set()}

    valid_status = pd.DataFrame(valid_rows).reset_index(drop=True)
    if not coefficients.empty and "Spec_Key" in coefficients.columns:
        coefficients = coefficients.loc[coefficients["Spec_Key"].isin(valid_completed_spec_keys)].copy()
    else:
        coefficients = pd.DataFrame()
    if not diagnostics.empty and "Spec_Key" in diagnostics.columns:
        diagnostics = diagnostics.loc[diagnostics["Spec_Key"].isin(valid_completed_spec_keys)].copy()
    else:
        diagnostics = pd.DataFrame()

    return {
        "completed_spec_keys": valid_completed_spec_keys,
        "status": valid_status,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
    }


def plan_regression_suite_run(
    specs: list[RegressionSpec],
    *,
    output_dir: Path,
    suite_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    reusable_state = _load_reusable_suite_state(output_dir, specs, suite_metadata)
    completed_spec_keys = reusable_state.get("completed_spec_keys", set())
    runnable_specs = [spec for spec in specs if spec.key not in completed_spec_keys]
    return {
        "runnable_specs": runnable_specs,
        "reused_spec_keys": sorted(completed_spec_keys),
    }


def run_regression_suite(
    dataset: pd.DataFrame,
    specs: list[RegressionSpec],
    *,
    output_dir: Path,
    suite_metadata: dict[str, object] | None = None,
    progress_callback=None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    regression_artifact_dir = output_dir / "regression_inputs"
    regression_artifact_dir.mkdir(parents=True, exist_ok=True)

    reusable_state = _load_reusable_suite_state(output_dir, specs, suite_metadata)
    reused_spec_keys: set[str] = set(reusable_state.get("completed_spec_keys", set()))

    coefficient_frames: list[pd.DataFrame] = []
    if not reusable_state.get("coefficients", pd.DataFrame()).empty:
        coefficient_frames.append(reusable_state["coefficients"].copy())
    diagnostics_rows: list[dict[str, object]] = []
    if not reusable_state.get("diagnostics", pd.DataFrame()).empty:
        diagnostics_rows.extend(reusable_state["diagnostics"].to_dict(orient="records"))
    status_rows: list[dict[str, object]] = []
    if not reusable_state.get("status", pd.DataFrame()).empty:
        status_rows.extend(reusable_state["status"].to_dict(orient="records"))
    completed_results: list[CompletedRegressionResult] = []

    runnable_specs = [spec for spec in specs if spec.key not in reused_spec_keys]
    iterator = runnable_specs
    if progress_callback is None:
        iterator = tqdm(runnable_specs, desc="Running regressions", total=len(runnable_specs))

    for spec in iterator:
        warning_records: list[dict[str, str]] = []
        working_dataset = dataset
        if spec.row_filter_query:
            try:
                working_dataset = _apply_row_filter(dataset, spec.row_filter_query)
            except Exception as exc:
                error_details = _exception_details(exc)
                status_rows.append(
                    _build_status_row(
                        spec,
                        status="failed",
                        reason=f"row filter failed: {exc}",
                        error_stage="row_filter",
                        **error_details,
                    )
                )
                if progress_callback is not None:
                    progress_callback({"spec_key": spec.key, "status": "failed"})
                continue

        if not spec.enabled:
            status_rows.append(
                _build_status_row(
                    spec,
                    status="skipped",
                    reason="spec disabled",
                    error_stage="precheck",
                )
            )
            if progress_callback is not None:
                progress_callback({"spec_key": spec.key, "status": "skipped"})
            continue

        dependent_variable = _dependent_variable(spec.formula)
        if dependent_variable not in working_dataset.columns:
            status_rows.append(
                _build_status_row(
                    spec,
                    status="skipped",
                    reason=f"missing dependent-variable column {dependent_variable!r}",
                    error_stage="precheck",
                    dependent_variable=dependent_variable,
                )
            )
            if progress_callback is not None:
                progress_callback({"spec_key": spec.key, "status": "skipped"})
            continue

        non_missing_dependent_values = int(working_dataset[dependent_variable].notna().sum())
        if non_missing_dependent_values < 3:
            status_rows.append(
                _build_status_row(
                    spec,
                    status="skipped",
                    reason=(
                        f"dependent-variable column {dependent_variable!r} has fewer than "
                        "3 non-missing observations in the prepared dataset"
                    ),
                    error_stage="precheck",
                    dependent_variable=dependent_variable,
                    non_missing_dependent_values=non_missing_dependent_values,
                )
            )
            if progress_callback is not None:
                progress_callback({"spec_key": spec.key, "status": "skipped"})
            continue

        try:
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always")
                model_output = fit_formula_model(
                    working_dataset,
                    formula=spec.formula,
                    model_label=spec.label,
                    cluster_spec=spec.cluster_spec,
                )
            warning_records.extend(_serialize_warning_records(fit_warnings, stage="fit"))
        except Exception as exc:
            error_details = _exception_details(exc)
            status_rows.append(
                _build_status_row(
                    spec,
                    status="failed",
                    reason=str(exc),
                    error_stage="fit_model",
                    dependent_variable=dependent_variable,
                    non_missing_dependent_values=non_missing_dependent_values,
                    warning_count=len(warning_records),
                    warning_summary=_format_warning_summary(warning_records),
                    **error_details,
                )
            )
            if progress_callback is not None:
                progress_callback({"spec_key": spec.key, "status": "failed"})
            continue

        try:
            regression_input_path = regression_artifact_dir / f"{_slugify(spec.key)}.json"
            regression_input_path.write_text(
                json.dumps(
                    {
                        "spec_key": spec.key,
                        "spec_signature": _spec_signature(spec),
                        "formula": spec.formula,
                        "cluster_spec": spec.cluster_spec,
                        "row_filter_query": spec.row_filter_query,
                        "sample_size": int(model_output["result"].nobs),
                        "year_values": (
                            sorted(
                                pd.to_numeric(
                                    model_output["data"][ANNOUNCEMENT_YEAR_COLUMN],
                                    errors="coerce",
                                ).dropna().astype(int).unique().tolist()
                            )
                            if ANNOUNCEMENT_YEAR_COLUMN in model_output["data"].columns
                            else []
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as coefficient_warnings:
                warnings.simplefilter("always")
                coefficient_table = build_coefficient_table(model_output["result"])
            warning_records.extend(
                _serialize_warning_records(coefficient_warnings, stage="build_coefficient_table")
            )
            coefficient_table = filter_non_fixed_effect_coefficients(
                coefficient_table,
                spec.fixed_effect_terms_to_exclude,
                exclude_intercept=spec.exclude_intercept_from_reporting,
            )
            coefficient_table = add_coefficient_significance_columns(coefficient_table)
            problematic_terms = _find_problematic_coefficient_terms(
                coefficient_table,
                exclude_intercept=spec.exclude_intercept_from_reporting,
            )
            coefficient_table.insert(0, "Spec_Key", spec.key)
            coefficient_table.insert(1, "Family", spec.family)
            coefficient_table.insert(2, "Label", spec.label)
            coefficient_table.insert(3, "Formula", spec.formula)
            coefficient_table.insert(4, "Cluster_Spec", spec.cluster_spec)
            coefficient_table.insert(5, "Cluster_Label", model_output["cluster_label"])
            coefficient_frames.append(coefficient_table)

            model_output["warning_records"] = warning_records.copy()
            with warnings.catch_warnings(record=True) as diagnostics_warnings:
                warnings.simplefilter("always")
                diagnostics_row = _build_suite_diagnostics_row(model_output, spec)
            warning_records.extend(
                _serialize_warning_records(diagnostics_warnings, stage="build_diagnostics")
            )
            model_output["warning_records"] = warning_records.copy()
            diagnostics_row["Warning_Count"] = len(warning_records)
            diagnostics_row["Warning_Summary"] = _format_warning_summary(warning_records)
            diagnostics_row["Problematic_Terms"] = " | ".join(problematic_terms)
            diagnostics_row.update(
                {
                    "Spec_Key": spec.key,
                    "Family": spec.family,
                    "Label": spec.label,
                    "Cluster_Spec": spec.cluster_spec,
                    "Regression_Input_Path": str(regression_input_path),
                }
            )
            diagnostics_rows.append(diagnostics_row)
            completed_results.append(
                CompletedRegressionResult(
                    spec=spec,
                    result=model_output["result"],
                    cluster_label=model_output["cluster_label"],
                    diagnostics=diagnostics_row.copy(),
                    regression_input_path=str(regression_input_path),
                )
            )

            status_rows.append(
                _build_status_row(
                    spec,
                    status="completed",
                    error_stage="completed",
                    dependent_variable=dependent_variable,
                    non_missing_dependent_values=non_missing_dependent_values,
                    sample_size=int(model_output["result"].nobs),
                    saved_coefficient_count=int(len(coefficient_table)),
                    regression_input_path=str(regression_input_path),
                    warning_count=len(warning_records),
                    warning_summary=_format_warning_summary(warning_records),
                )
            )
            _emit_regression_warning(
                spec,
                warning_records,
                sample_size=int(model_output["result"].nobs),
                problematic_terms=problematic_terms,
            )
        except Exception as exc:
            error_details = _exception_details(exc)
            status_rows.append(
                _build_status_row(
                    spec,
                    status="failed",
                    reason=str(exc),
                    error_stage="post_fit_processing",
                    dependent_variable=dependent_variable,
                    non_missing_dependent_values=non_missing_dependent_values,
                    warning_count=len(warning_records),
                    warning_summary=_format_warning_summary(warning_records),
                    **error_details,
                )
            )
            if progress_callback is not None:
                progress_callback({"spec_key": spec.key, "status": "failed"})
            continue
        if progress_callback is not None:
            progress_callback({"spec_key": spec.key, "status": "completed"})

    coefficient_columns = [
        "Spec_Key",
        "Family",
        "Label",
        "Formula",
        "Cluster_Spec",
        "Cluster_Label",
        "Term",
        "Coefficient",
        "Std_Error",
        "t_stat",
        "p_value",
        "Two_Sided_p_Value",
        "One_Sided_p_Value_Greater",
        "One_Sided_p_Value_Less",
        "Significance_Stars",
        "CI_95_Low",
        "CI_95_High",
    ]
    diagnostics_columns = [
        "Model",
        "Formula",
        "Std_Error_Treatment",
        "Firm_Cluster_Count",
        "Quarter_Cluster_Count",
        "Reported_Coefficient_Joint_Test_Applied",
        "Reported_Coefficient_Joint_F_Statistic",
        "Reported_Coefficient_Joint_F_p_value",
        "Reported_Coefficient_Joint_F_DF_Numerator",
        "Reported_Coefficient_Joint_F_DF_Denominator",
        "Reported_Coefficient_Joint_F_Terms",
        "Reported_Coefficient_Joint_F_Status",
        "SUE_Interaction_Joint_Test_Applied",
        "SUE_Interaction_Joint_Test_p_value",
        "SUE_Interaction_Joint_Test_Statistic",
        "SUE_Interaction_Joint_Test_DF",
        "SUE_Interaction_Joint_Test_DF_Denominator",
        "SUE_Interaction_Joint_Test_Distribution",
        "SUE_Interaction_Joint_Test_Terms",
        "SUE_Interaction_Joint_Test_Status",
        "R_Squared",
        "Within_R_Squared",
        "Within_R_Squared_Status",
        "Within_R_Squared_Restricted_Formula",
        "Adjusted_R_Squared",
        "Sample_Size",
        "DF_Model",
        "DF_Residual",
        "Warning_Count",
        "Warning_Summary",
        "Problematic_Terms",
        "Spec_Key",
        "Family",
        "Label",
        "Cluster_Spec",
        "Regression_Input_Path",
    ]
    status_columns = [
        "Spec_Key",
        "Family",
        "Label",
        "Formula",
        "Cluster_Spec",
        "Row_Filter_Query",
        "Spec_Signature",
        "Enabled",
        "Status",
        "Reason",
        "Error_Stage",
        "Error_Type",
        "Error_Module",
        "Traceback",
        "Dependent_Variable",
        "Non_Missing_Dependent_Values",
        "Sample_Size",
        "Saved_Coefficient_Count",
        "Regression_Input_Path",
        "Warning_Count",
        "Warning_Summary",
        "Notes",
    ]

    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(columns=coefficient_columns)
    )
    diagnostics = (
        pd.DataFrame(diagnostics_rows)
        if diagnostics_rows
        else pd.DataFrame(columns=diagnostics_columns)
    )
    status = (
        pd.DataFrame(status_rows)
        if status_rows
        else pd.DataFrame(columns=status_columns)
    )
    registry = build_regression_registry(specs)
    if suite_metadata is not None:
        (output_dir / SUITE_METADATA_FILENAME).write_text(
            json.dumps(suite_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return {
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "status": status,
        "registry": registry,
        "completed_results": completed_results,
        "reused_spec_keys": sorted(reused_spec_keys),
        "planned_spec_count": len(runnable_specs),
    }
