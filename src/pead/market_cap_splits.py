from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN = "Market_Cap_Size_Split_Percentile"
MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN = "Market_Cap_Size_Split_Breakpoint"
MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN = "Market_Cap_Size_Split_Group"
MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN = "Market_Cap_Size_Split_Is_Bottom"

MARKET_CAP_BOTTOM_GROUP_LABEL = "Microcap"
MARKET_CAP_TOP_GROUP_LABEL = "All-but-microcap"
MARKET_CAP_SIZE_GROUP_ORDER = [
    MARKET_CAP_TOP_GROUP_LABEL,
    MARKET_CAP_BOTTOM_GROUP_LABEL,
]
MARKET_CAP_DECILE_BREAKPOINT_PERCENTILES = tuple(
    percentile / 100 for percentile in range(10, 100, 10)
)


def validate_market_cap_size_split_percentile(percentile: float) -> float:
    normalized_percentile = float(percentile)
    if not 0.0 < normalized_percentile < 1.0:
        raise ValueError(
            "Market-cap size-split percentile must be strictly between 0 and 1."
        )
    return normalized_percentile


def categorize_market_cap_size_groups(values) -> pd.Categorical:
    return pd.Categorical(
        values,
        categories=MARKET_CAP_SIZE_GROUP_ORDER,
        ordered=True,
    )


def build_market_cap_decile_breakpoints(
    market_caps: pd.Series,
    percentiles: tuple[float, ...] = MARKET_CAP_DECILE_BREAKPOINT_PERCENTILES,
) -> dict[str, float]:
    valid_market_caps = pd.to_numeric(market_caps, errors="coerce").dropna()
    return {
        f"{int(percentile * 100)}th percentile": float(
            valid_market_caps.quantile(percentile)
        )
        for percentile in percentiles
    }


def build_market_cap_size_split_metadata(
    market_caps: pd.Series,
    split_percentile: float,
    breakpoint_value: float,
) -> dict:
    normalized_percentile = validate_market_cap_size_split_percentile(split_percentile)
    valid_market_caps = pd.to_numeric(market_caps, errors="coerce").dropna()
    is_bottom_group = valid_market_caps <= breakpoint_value

    return {
        "Market cap size split percentile": normalized_percentile,
        "Market cap size split breakpoint": float(breakpoint_value),
        "Market cap size split breakpoint unit": "USD mn",
        "Market cap decile breakpoints": build_market_cap_decile_breakpoints(
            valid_market_caps
        ),
        "Market cap size split reference universe count": int(valid_market_caps.shape[0]),
        "Microcap count": int(is_bottom_group.sum()),
        "All-but-microcap count": int((~is_bottom_group).sum()),
    }


def assign_market_cap_size_split_from_breakpoint(
    frame: pd.DataFrame,
    split_percentile: float,
    breakpoint_value: float,
    market_cap_column: str = "Market_Cap_Current",
) -> pd.DataFrame:
    normalized_percentile = validate_market_cap_size_split_percentile(split_percentile)

    if market_cap_column not in frame.columns:
        raise KeyError(
            f"Cannot assign market-cap size split. Missing column: {market_cap_column!r}."
        )

    out = frame.copy()
    market_caps = pd.to_numeric(out[market_cap_column], errors="coerce")
    is_bottom_group = market_caps <= float(breakpoint_value)

    size_group = pd.Series(pd.NA, index=out.index, dtype="object")
    size_group.loc[market_caps.notna() & is_bottom_group] = MARKET_CAP_BOTTOM_GROUP_LABEL
    size_group.loc[market_caps.notna() & ~is_bottom_group] = MARKET_CAP_TOP_GROUP_LABEL

    out[MARKET_CAP_SIZE_SPLIT_PERCENTILE_COLUMN] = normalized_percentile
    out[MARKET_CAP_SIZE_SPLIT_BREAKPOINT_COLUMN] = float(breakpoint_value)
    out[MARKET_CAP_SIZE_SPLIT_FLAG_COLUMN] = pd.Series(
        np.where(market_caps.notna(), is_bottom_group.astype(int), pd.NA),
        index=out.index,
        dtype="Int64",
    )
    out[MARKET_CAP_SIZE_SPLIT_GROUP_COLUMN] = categorize_market_cap_size_groups(size_group)

    return out


def assign_market_cap_size_split(
    frame: pd.DataFrame,
    split_percentile: float,
    market_cap_column: str = "Market_Cap_Current",
) -> tuple[pd.DataFrame, dict]:
    normalized_percentile = validate_market_cap_size_split_percentile(split_percentile)

    if market_cap_column not in frame.columns:
        raise KeyError(
            f"Cannot assign market-cap size split. Missing column: {market_cap_column!r}."
        )

    market_caps = pd.to_numeric(frame[market_cap_column], errors="coerce")
    valid_market_caps = market_caps.dropna()
    if valid_market_caps.empty:
        raise ValueError("Cannot assign market-cap size split because all market caps are missing.")

    breakpoint_value = float(valid_market_caps.quantile(normalized_percentile))
    out = assign_market_cap_size_split_from_breakpoint(
        frame=frame,
        split_percentile=normalized_percentile,
        breakpoint_value=breakpoint_value,
        market_cap_column=market_cap_column,
    )
    metadata = build_market_cap_size_split_metadata(
        market_caps=market_caps,
        split_percentile=normalized_percentile,
        breakpoint_value=breakpoint_value,
    )

    return out, metadata
