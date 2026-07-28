"""Summarise every available BHAR horizon by SUE quintile at event level.

The output is deliberately in long format: each row is one BHAR window and
one SUE quintile.  This preserves the horizon-specific non-missing event count
instead of implicitly treating all windows as if they used the same sample.
"""

from __future__ import annotations

import re

import pandas as pd
from scipy import stats

from _analysis_shared import AnalysisOutputManager, DATA_DIR
from src.analysis.time_varying_analysis import (
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
)
from src.core.pipeline_config import (
    ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
    MAIN_REGRESSION_BHAR_WINDOW,
    SUE_COMPUTATION_GROUP_COUNT,
)
from src.pead.sue_groups import SUE_GROUP_COLUMN


OUTPUTS = AnalysisOutputManager(__file__)
BHAR_COLUMN_PATTERN = re.compile(r"BHAR_(\d+)_(\d+)$")
GROUP_COUNT = SUE_COMPUTATION_GROUP_COUNT


def parse_bhar_window(column: str) -> tuple[int, int] | None:
    """Return the inclusive event-time bounds encoded in a BHAR column name."""
    match = BHAR_COLUMN_PATTERN.fullmatch(column)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def sue_quintile_label(quintile: int) -> str:
    if quintile == 1:
        return "Q1 (lowest)"
    if quintile == GROUP_COUNT:
        return f"Q{GROUP_COUNT} (highest)"
    return f"Q{quintile}"


def summarise_values(values: pd.Series) -> dict[str, float | int]:
    """Calculate descriptive statistics using non-missing event-level BHAR."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return {
            "N": 0,
            "Mean": float("nan"),
            "Median": float("nan"),
            "SD": float("nan"),
            "Min": float("nan"),
            "Q1": float("nan"),
            "Q3": float("nan"),
            "Max": float("nan"),
        }

    return {
        "N": int(clean_values.size),
        "Mean": clean_values.mean(),
        "Median": clean_values.median(),
        "SD": clean_values.std(ddof=1),
        "Min": clean_values.min(),
        "Q1": clean_values.quantile(0.25),
        "Q3": clean_values.quantile(0.75),
        "Max": clean_values.max(),
    }


def correlation_statistics(
    sue_quintiles: pd.Series,
    bhar_values: pd.Series,
) -> dict[str, float | int]:
    """Return linear and rank correlations using complete SUE--BHAR pairs only."""
    paired_values = pd.DataFrame(
        {
            "SUE_Quintile": pd.to_numeric(sue_quintiles, errors="coerce"),
            "BHAR": pd.to_numeric(bhar_values, errors="coerce"),
        }
    ).dropna()

    if len(paired_values) < 2 or paired_values["SUE_Quintile"].nunique() < 2:
        return {
            "N": int(len(paired_values)),
            "Pearson_Correlation": float("nan"),
            "Pearson_P_Value": float("nan"),
            "Spearman_Correlation": float("nan"),
            "Spearman_P_Value": float("nan"),
        }

    pearson = stats.pearsonr(
        paired_values["SUE_Quintile"],
        paired_values["BHAR"],
    )
    spearman = stats.spearmanr(
        paired_values["SUE_Quintile"],
        paired_values["BHAR"],
    )
    return {
        "N": int(len(paired_values)),
        "Pearson_Correlation": pearson.statistic,
        "Pearson_P_Value": pearson.pvalue,
        "Spearman_Correlation": spearman.statistic,
        "Spearman_P_Value": spearman.pvalue,
    }


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)

# Retain every stored BHAR window, while also requesting the thesis's configured
# cumulative regression horizons if a raw input needs them reconstructed.
stored_windows = {
    parsed
    for column in abnormal_returns.columns
    if (parsed := parse_bhar_window(str(column))) is not None
}
configured_windows = {
    MAIN_REGRESSION_BHAR_WINDOW,
    *ALTERNATIVE_REGRESSION_BHAR_WINDOWS,
}
windows = tuple(sorted(stored_windows | configured_windows))
if not windows:
    raise KeyError("No BHAR_<start>_<end> columns were found in abnormal-return data.")

bhar_columns = tuple(f"BHAR_{start_day}_{end_day}" for start_day, end_day in windows)
event_level = collapse_to_event_level(
    abnormal_returns,
    additional_bhar_columns=bhar_columns,
)

if SUE_GROUP_COLUMN not in event_level.columns:
    raise KeyError(f"Event-level data are missing {SUE_GROUP_COLUMN!r}.")

event_level[SUE_GROUP_COLUMN] = pd.to_numeric(
    event_level[SUE_GROUP_COLUMN], errors="coerce"
)
event_level = event_level.loc[
    event_level[SUE_GROUP_COLUMN].between(1, GROUP_COUNT)
].copy()
event_level[SUE_GROUP_COLUMN] = event_level[SUE_GROUP_COLUMN].astype(int)

summary_rows: list[dict[str, float | int | str]] = []
correlation_rows: list[dict[str, float | int | str]] = []
for (start_day, end_day), bhar_column in zip(windows, bhar_columns, strict=True):
    if bhar_column not in event_level.columns:
        raise KeyError(f"Event-level data are missing {bhar_column!r}.")

    values = pd.to_numeric(event_level[bhar_column], errors="coerce")
    correlation_rows.append(
        {
            "BHAR_Window": f"BHAR[{start_day},{end_day}]",
            "Window_Start_Day": start_day,
            "Window_End_Day": end_day,
            "Trading_Days": end_day - start_day + 1,
            **correlation_statistics(event_level[SUE_GROUP_COLUMN], values),
        }
    )
    for quintile in range(1, GROUP_COUNT + 1):
        quintile_values = values.loc[event_level[SUE_GROUP_COLUMN] == quintile]
        summary_rows.append(
            {
                "BHAR_Window": f"BHAR[{start_day},{end_day}]",
                "Window_Start_Day": start_day,
                "Window_End_Day": end_day,
                "Trading_Days": end_day - start_day + 1,
                "SUE_Quintile": quintile,
                "SUE_Quintile_Label": sue_quintile_label(quintile),
                "Eligible_Events": int(quintile_values.size),
                "Missing_BHAR_Events": int(quintile_values.isna().sum()),
                **summarise_values(quintile_values),
            }
        )

    summary_rows.append(
        {
            "BHAR_Window": f"BHAR[{start_day},{end_day}]",
            "Window_Start_Day": start_day,
            "Window_End_Day": end_day,
            "Trading_Days": end_day - start_day + 1,
            "SUE_Quintile": "Total",
            "SUE_Quintile_Label": "All quintiles",
            "Eligible_Events": int(values.size),
            "Missing_BHAR_Events": int(values.isna().sum()),
            **summarise_values(values),
        }
    )

summary_table = pd.DataFrame(summary_rows)
OUTPUTS.save_table(summary_table, "bhar_horizon_summary_by_sue_quintile")

correlation_table = pd.DataFrame(correlation_rows)
OUTPUTS.save_table(correlation_table, "sue_quintile_bhar_horizon_correlations")
