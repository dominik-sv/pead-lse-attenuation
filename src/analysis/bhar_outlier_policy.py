from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.io_utils import load_json, save_json


DEFAULT_OUTLIER_POLICY_FILENAME = "bhar_outlier_policy.json"
EVENT_COMPANY_COLUMN = "Instrument"
EVENT_DATE_COLUMN = "Ann_Date"
BHAR_COLUMNS = ("BHAR_0_1", "BHAR_2_60", "BHAR_0_60")
SelectorTail = Literal["upper", "lower"]


def resolve_outlier_policy_path(data_dir: Path) -> Path:
    return Path(data_dir) / DEFAULT_OUTLIER_POLICY_FILENAME


def build_event_keys(
    frame: pd.DataFrame,
    *,
    company_column: str = EVENT_COMPANY_COLUMN,
    date_column: str = EVENT_DATE_COLUMN,
) -> pd.Series:
    missing_columns = [
        column for column in (company_column, date_column) if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(
            "Cannot build outlier-policy event keys. Missing columns: "
            f"{sorted(missing_columns)}."
        )

    normalized_company = frame[company_column].astype("string").str.strip()
    normalized_dates = pd.to_datetime(frame[date_column], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return normalized_company.fillna("<missing>") + "|" + normalized_dates.fillna("<missing>")


def _empty_policy() -> dict[str, object]:
    return {
        "version": 1,
        "generated_at_utc": None,
        "observation_actions": [],
        "global_rules": [],
    }


def load_outlier_policy(policy_path: Path) -> dict[str, object]:
    loaded = load_json(policy_path, default=None)
    if not loaded:
        return _empty_policy()

    policy = _empty_policy()
    policy.update(loaded)

    if not isinstance(policy.get("observation_actions"), list):
        policy["observation_actions"] = []
    if not isinstance(policy.get("global_rules"), list):
        policy["global_rules"] = []
    return policy


def save_outlier_policy(policy: dict[str, object], policy_path: Path) -> None:
    payload = _empty_policy()
    payload.update(policy)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(payload, policy_path)


def event_level_view(frame: pd.DataFrame) -> pd.DataFrame:
    event_keys = build_event_keys(frame)
    event_level = frame.copy()
    event_level["_Event_Key"] = event_keys

    sort_columns = [column for column in ("Formation_Year", "_Event_Key", "Relative_Day") if column in event_level.columns]
    if sort_columns:
        event_level = event_level.sort_values(sort_columns)

    event_level = event_level.drop_duplicates(subset=["_Event_Key"], keep="first").copy()
    return event_level


def select_outlier_candidates(
    event_level: pd.DataFrame,
    *,
    column: str,
    top_n: int | None = None,
    percentile: float | None = None,
    tail: SelectorTail = "upper",
) -> pd.DataFrame:
    if "_Event_Key" not in event_level.columns:
        event_level = event_level_view(event_level)

    if column not in event_level.columns:
        raise KeyError(f"Column {column!r} is not available in the event-level sample.")
    if (top_n is None) == (percentile is None):
        raise ValueError("Provide exactly one of top_n or percentile.")
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")

    working = event_level.copy()
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.loc[working[column].notna()].copy()
    if working.empty:
        raise ValueError(f"No non-missing observations are available for {column}.")

    if top_n is not None:
        requested_top_n = int(top_n)
        if requested_top_n <= 0:
            raise ValueError("top_n must be positive.")
        if requested_top_n >= len(working):
            raise ValueError(
                f"top_n={requested_top_n} is too large for a sample with {len(working)} eligible observations."
            )

        ascending = tail == "lower"
        working = working.sort_values([column, "_Event_Key"], ascending=[ascending, True]).reset_index(drop=True)
        selected = working.head(requested_top_n).copy()
        replacement_value = float(working.iloc[requested_top_n][column])
        threshold_value = float(selected.iloc[-1][column])
        selected["Selection_Method"] = "top_n"
        selected["Selection_Value"] = requested_top_n
        selected["Selection_Threshold"] = threshold_value
        selected["Replacement_Value"] = replacement_value
    else:
        requested_percentile = float(percentile)
        if not 0.0 < requested_percentile < 1.0:
            raise ValueError("percentile must lie strictly between 0 and 1.")

        threshold_value = float(working[column].quantile(requested_percentile))
        if tail == "upper":
            selected = working.loc[working[column] > threshold_value].copy()
        else:
            selected = working.loc[working[column] < threshold_value].copy()
        if selected.empty:
            raise ValueError(
                f"No observations fall in the requested {tail} tail for {column} at percentile={requested_percentile:.4f}."
            )

        selected = selected.sort_values([column, "_Event_Key"], ascending=[tail == "lower", True]).reset_index(drop=True)
        selected["Selection_Method"] = "percentile"
        selected["Selection_Value"] = requested_percentile
        selected["Selection_Threshold"] = threshold_value
        selected["Replacement_Value"] = threshold_value

    selected["Selection_Tail"] = tail
    selected["Selected_Rank"] = range(1, len(selected) + 1)
    return selected


def build_winsorized_values(
    event_level: pd.DataFrame,
    *,
    selected_keys: list[str],
    columns: list[str],
    top_n: int | None = None,
    percentile: float | None = None,
    tail: SelectorTail = "upper",
) -> dict[str, dict[str, float]]:
    if not selected_keys:
        return {}

    selected_key_set = set(selected_keys)
    winsorized_values: dict[str, dict[str, float]] = {}
    event_level_with_keys = event_level.copy()
    if "_Event_Key" not in event_level_with_keys.columns:
        event_level_with_keys = event_level_view(event_level_with_keys)

    for column in columns:
        selected = select_outlier_candidates(
            event_level_with_keys,
            column=column,
            top_n=top_n,
            percentile=percentile,
            tail=tail,
        )
        replacement_value = float(selected["Replacement_Value"].iloc[0])
        column_values = pd.to_numeric(
            event_level_with_keys.set_index("_Event_Key")[column],
            errors="coerce",
        )

        for event_key in selected_key_set:
            if event_key not in column_values.index:
                continue
            original_value = column_values.loc[event_key]
            if pd.isna(original_value):
                continue

            if tail == "upper":
                clipped_value = min(float(original_value), replacement_value)
            else:
                clipped_value = max(float(original_value), replacement_value)

            if clipped_value == float(original_value):
                continue
            winsorized_values.setdefault(event_key, {})[column] = clipped_value

    return winsorized_values


def update_policy_actions(
    existing_policy: dict[str, object],
    *,
    new_actions: list[dict[str, object]],
    replace_existing: bool = True,
) -> dict[str, object]:
    policy = _empty_policy()
    policy.update(existing_policy)

    current_actions = []
    seen_keys: set[tuple[str, str]] = set()
    for action in policy["observation_actions"]:
        if not isinstance(action, dict):
            continue
        company = str(action.get("instrument", "")).strip()
        ann_date = str(action.get("ann_date", "")).strip()
        action_key = (company, ann_date)
        if replace_existing and action_key in {
            (str(new_action.get("instrument", "")).strip(), str(new_action.get("ann_date", "")).strip())
            for new_action in new_actions
        }:
            continue
        if action_key in seen_keys:
            continue
        seen_keys.add(action_key)
        current_actions.append(action)

    for action in new_actions:
        company = str(action.get("instrument", "")).strip()
        ann_date = str(action.get("ann_date", "")).strip()
        if not company or not ann_date:
            continue
        current_actions.append(action)

    policy["observation_actions"] = current_actions
    return policy


def apply_outlier_policy(frame: pd.DataFrame, *, policy: dict[str, object]) -> pd.DataFrame:
    observation_actions = [
        action for action in policy.get("observation_actions", [])
        if isinstance(action, dict)
    ]
    global_rules = [
        rule for rule in policy.get("global_rules", [])
        if isinstance(rule, dict)
    ]
    if not observation_actions and not global_rules:
        return frame.copy()

    out = frame.copy()
    out["_Event_Key"] = build_event_keys(out)

    excluded_keys = {
        f"{str(action.get('instrument', '')).strip()}|{str(action.get('ann_date', '')).strip()}"
        for action in observation_actions
        if str(action.get("action", "")).strip().lower() == "exclude"
    }
    if excluded_keys:
        out = out.loc[~out["_Event_Key"].isin(excluded_keys)].copy()

    event_level = event_level_view(out)

    for rule in global_rules:
        if str(rule.get("action", "")).strip().lower() != "winsorize":
            continue

        column = str(rule.get("column", "")).strip()
        lower_quantile = rule.get("lower_quantile")
        upper_quantile = rule.get("upper_quantile")
        if not column or column not in event_level.columns:
            continue

        event_values = pd.to_numeric(event_level[column], errors="coerce")
        valid_values = event_values.dropna()
        if valid_values.empty:
            continue

        lower_bound = None if lower_quantile is None else float(valid_values.quantile(float(lower_quantile)))
        upper_bound = None if upper_quantile is None else float(valid_values.quantile(float(upper_quantile)))
        clipped_values = event_values.copy()
        if lower_bound is not None:
            clipped_values = clipped_values.clip(lower=lower_bound)
        if upper_bound is not None:
            clipped_values = clipped_values.clip(upper=upper_bound)

        replacement_by_key = pd.Series(
            clipped_values.to_numpy(),
            index=event_level["_Event_Key"],
        ).to_dict()
        out[column] = out["_Event_Key"].map(replacement_by_key).where(
            out["_Event_Key"].map(replacement_by_key).notna(),
            out[column],
        )

    winsorization_actions = [
        action for action in observation_actions
        if str(action.get("action", "")).strip().lower() == "winsorize"
    ]
    for action in winsorization_actions:
        event_key = (
            f"{str(action.get('instrument', '')).strip()}|"
            f"{str(action.get('ann_date', '')).strip()}"
        )
        columns = action.get("columns", {})
        if not isinstance(columns, dict):
            continue
        mask = out["_Event_Key"].eq(event_key)
        if not mask.any():
            continue
        for column, replacement_value in columns.items():
            if column not in out.columns:
                continue
            out.loc[mask, column] = float(replacement_value)

    return out.drop(columns="_Event_Key")
