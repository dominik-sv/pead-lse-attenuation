import ast
from typing import Sequence

import pandas as pd


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def normalize_history_columns(
    hist: pd.DataFrame,
    instrument_batch: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = hist.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = _normalize_history_multiindex(out.columns)
    elif (
        instrument_batch is not None
        and len(instrument_batch) == 1
        and all(_is_history_field_label(column) for column in out.columns)
    ):
        instrument = str(instrument_batch[0])
        out.columns = pd.MultiIndex.from_tuples(
            [
                (_normalize_history_field_label(column), instrument)
                for column in out.columns
            ],
            names=["Field", "Instrument"],
        )
    else:
        out.columns = pd.Index([str(column) for column in out.columns])
        out.columns.name = None

    out.index.name = "Date"
    return out


def restore_history_multiindex(columns: pd.Index) -> pd.MultiIndex | None:
    restored_columns = []
    restored_any_history = False

    for column in columns:
        restored = _restore_history_column(column)

        if restored is None:
            restored_columns.append((str(column), ""))
            continue

        restored_columns.append(restored)
        restored_any_history = True

    if not restored_any_history:
        return None

    return pd.MultiIndex.from_tuples(
        restored_columns,
        names=["Field", "Instrument"],
    )


def _normalize_history_multiindex(columns: pd.MultiIndex) -> pd.MultiIndex:
    field_level = _find_history_field_level(columns)
    instrument_level = next(
        level for level in range(columns.nlevels) if level != field_level
    )

    normalized_columns = [
        (
            _normalize_history_field_label(column[field_level]),
            str(column[instrument_level]),
        )
        for column in columns
    ]

    return pd.MultiIndex.from_tuples(
        normalized_columns,
        names=["Field", "Instrument"],
    )


def _find_history_field_level(columns: pd.MultiIndex) -> int:
    for level in range(columns.nlevels):
        labels = columns.get_level_values(level)
        if any(_is_history_field_label(label) for label in labels):
            return level

    return columns.nlevels - 1


def _is_history_field_label(label) -> bool:
    normalized = _compact_label(label)
    return "TOTALRETURN" in normalized or "PRICECLOSE" in normalized


def _normalize_history_field_label(label) -> str:
    normalized = _compact_label(label)

    if "TOTALRETURN" in normalized:
        return "TotalReturn"

    if "PRICECLOSE" in normalized:
        return "PriceClose"

    return str(label)


def _compact_label(label) -> str:
    return (
        str(label)
        .replace(" ", "")
        .replace("_", "")
        .replace(".", "")
        .upper()
    )


def _restore_history_column(column) -> tuple[str, str] | None:
    try:
        parsed = ast.literal_eval(str(column))
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, tuple) or len(parsed) != 2:
        return None

    left, right = parsed

    if _is_history_field_label(left):
        return (_normalize_history_field_label(left), str(right))

    if _is_history_field_label(right):
        return (_normalize_history_field_label(right), str(left))

    return None
