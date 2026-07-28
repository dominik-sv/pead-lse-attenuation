from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.pipeline_config import SUE_COMPUTATION_GROUP_COUNT, SUE_PLOT_GROUP_COUNT

SUE_GROUP_COLUMN = "SUE_Group"
SUE_PLOT_GROUP_COLUMN = "SUE_Plot_Group"
LEGACY_SUE_GROUP_COLUMN = "SUE_Decile"
LEGACY_SUE_PLOT_GROUP_COLUMN = "SUE_Quintile"


def validate_group_count(group_count: int, label: str) -> int:
    normalized_count = int(group_count)
    if normalized_count < 2:
        raise ValueError(f"{label} must be at least 2, got {group_count}.")
    return normalized_count


def normalize_sue_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if SUE_GROUP_COLUMN not in frame.columns and LEGACY_SUE_GROUP_COLUMN in frame.columns:
        rename_map[LEGACY_SUE_GROUP_COLUMN] = SUE_GROUP_COLUMN
    if (
        SUE_PLOT_GROUP_COLUMN not in frame.columns
        and LEGACY_SUE_PLOT_GROUP_COLUMN in frame.columns
    ):
        rename_map[LEGACY_SUE_PLOT_GROUP_COLUMN] = SUE_PLOT_GROUP_COLUMN
    if not rename_map:
        return frame
    return frame.rename(columns=rename_map)


def drop_sue_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_sue_group_columns(frame.copy())
    columns_to_drop = [
        column
        for column in (
            SUE_GROUP_COLUMN,
            SUE_PLOT_GROUP_COLUMN,
            LEGACY_SUE_GROUP_COLUMN,
            LEGACY_SUE_PLOT_GROUP_COLUMN,
        )
        if column in out.columns
    ]
    if not columns_to_drop:
        return out
    return out.drop(columns=columns_to_drop)


def assign_prior_year_sue_groups(
    current_events: pd.DataFrame,
    prior_year_events: pd.DataFrame | None,
    group_count: int = SUE_COMPUTATION_GROUP_COUNT,
) -> pd.DataFrame:
    normalized_group_count = validate_group_count(group_count, "SUE computation group count")
    out = normalize_sue_group_columns(current_events.copy())

    if prior_year_events is None or prior_year_events.empty:
        out[SUE_GROUP_COLUMN] = pd.NA
        return out

    prior_year_events = normalize_sue_group_columns(prior_year_events)
    required_columns = {"SUE"}
    missing_current = required_columns.difference(out.columns)
    missing_prior = required_columns.difference(prior_year_events.columns)

    if missing_current:
        raise KeyError(f"Current earnings events missing columns: {sorted(missing_current)}.")
    if missing_prior:
        raise KeyError(f"Prior-year earnings events missing columns: {sorted(missing_prior)}.")

    prior_sue = pd.to_numeric(prior_year_events["SUE"], errors="coerce").dropna()
    current_sue = pd.to_numeric(out["SUE"], errors="coerce")

    if prior_sue.nunique() < normalized_group_count:
        out[SUE_GROUP_COLUMN] = pd.NA
        return out

    group_edges = prior_sue.quantile(
        np.linspace(
            1.0 / normalized_group_count,
            (normalized_group_count - 1) / normalized_group_count,
            normalized_group_count - 1,
        )
    ).to_numpy()

    groups = pd.Series(
        np.searchsorted(group_edges, current_sue, side="right") + 1,
        index=out.index,
        dtype="Int64",
    )

    out[SUE_GROUP_COLUMN] = groups.where(current_sue.notna())

    return out


def map_groups_to_plot_groups(
    group_values: pd.Series,
    computation_group_count: int = SUE_COMPUTATION_GROUP_COUNT,
    plot_group_count: int = SUE_PLOT_GROUP_COUNT,
) -> pd.Series:
    normalized_computation_count = validate_group_count(
        computation_group_count, "SUE computation group count"
    )
    normalized_plot_count = validate_group_count(plot_group_count, "SUE plot group count")
    if normalized_plot_count > normalized_computation_count:
        raise ValueError(
            "SUE plot group count cannot exceed the SUE computation group count."
        )

    numeric_groups = pd.to_numeric(group_values, errors="coerce")
    zero_based_plot_groups = np.floor(
        ((numeric_groups - 1) * normalized_plot_count) / normalized_computation_count
    )
    mapped_groups = pd.Series(zero_based_plot_groups + 1, index=group_values.index, dtype="Int64")
    return mapped_groups.where(numeric_groups.notna())


def add_plot_group_column(
    frame: pd.DataFrame,
    group_column: str = SUE_GROUP_COLUMN,
    plot_group_column: str = SUE_PLOT_GROUP_COLUMN,
    computation_group_count: int = SUE_COMPUTATION_GROUP_COUNT,
    plot_group_count: int = SUE_PLOT_GROUP_COUNT,
) -> pd.DataFrame:
    out = normalize_sue_group_columns(frame.copy())
    if group_column not in out.columns:
        raise KeyError(f"Frame is missing required SUE group column: {group_column}.")

    out[plot_group_column] = map_groups_to_plot_groups(
        out[group_column],
        computation_group_count=computation_group_count,
        plot_group_count=plot_group_count,
    )
    return out

def build_plot_group_labels(
    computation_group_count: int = SUE_COMPUTATION_GROUP_COUNT,
    plot_group_count: int = SUE_PLOT_GROUP_COUNT,
) -> dict[int, str]:
    normalized_computation_count = validate_group_count(
        computation_group_count, "SUE computation group count"
    )
    normalized_plot_count = validate_group_count(plot_group_count, "SUE plot group count")

    labels: dict[int, str] = {}
    for plot_group in range(1, normalized_plot_count + 1):
        label = f"SUE quintile {plot_group}"
        if plot_group == 1:
            label += " (lowest)"
        elif plot_group == normalized_plot_count:
            label += " (highest)"

        labels[plot_group] = label

    return labels
    
def get_extreme_group_values(
    group_count: int = SUE_COMPUTATION_GROUP_COUNT,
) -> tuple[int, int]:
    normalized_group_count = validate_group_count(group_count, "SUE computation group count")
    return (1, normalized_group_count)
