from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH_BIG_STOCK_MARKET_CAP_SHARE = 0.90
FRENCH_BM_PERCENTILES = (0.30, 0.70)

# Fama-French international-style 2x3 size/BM portfolios:
# S/B = Small/Big
# G/N/V = Growth/Neutral/Value
STANDARD_PORTFOLIO_LABELS = ("SG", "SN", "SV", "BG", "BN", "BV")

SIZE_GROUP_TO_Q = {"S": 1, "B": 2}
BM_GROUP_TO_Q = {"G": 1, "N": 2, "V": 3}


def parse_official_french_benchmark_returns(path: str | Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    title_index = next(
        index
        for index, line in enumerate(lines)
        if "Average Value Weighted Returns -- Daily" in line
    )
    header_index = next(
        index for index in range(title_index + 1, len(lines)) if lines[index].strip()
    )

    raw = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    raw.columns = [str(column).strip() for column in raw.columns]

    date_column = raw.columns[0]
    raw[date_column] = raw[date_column].astype("string").str.strip()
    raw = raw.loc[raw[date_column].str.fullmatch(r"\d{8}", na=False)].copy()
    raw = raw.rename(columns={date_column: "Date"})

    portfolio_columns = [column for column in raw.columns if column != "Date"]
    if len(portfolio_columns) != len(STANDARD_PORTFOLIO_LABELS):
        raise ValueError(
            "Unexpected number of French portfolio columns in official benchmark file. "
            f"Expected {len(STANDARD_PORTFOLIO_LABELS)}, got {len(portfolio_columns)}."
        )

    rename_map = {
        column: STANDARD_PORTFOLIO_LABELS[index]
        for index, column in enumerate(portfolio_columns)
    }
    raw = raw.rename(columns=rename_map)

    out = raw.copy()
    out["Date"] = pd.to_datetime(out["Date"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["Date"]).set_index("Date").sort_index()

    for column in STANDARD_PORTFOLIO_LABELS:
        series = pd.to_numeric(out[column], errors="coerce")
        series = series.mask(series == -99.99, pd.NA)
        out[column] = series

    out.index.name = "Date"
    return out.loc[:, list(STANDARD_PORTFOLIO_LABELS)]


def filter_official_benchmark_returns_for_window(
    official_returns: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return official_returns.loc[
        (official_returns.index >= start_ts) & (official_returns.index <= end_ts)
    ].copy()

def compute_french_benchmark_breakpoints(
    reference_universe: pd.DataFrame,
    formation_date: str,
    exchange_definitions: tuple[tuple[str, str], ...],
) -> dict:
    required_columns = {
        "Instrument",
        "Market_Cap_Current",
        "BM_French",
    }
    missing_columns = required_columns.difference(reference_universe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "French benchmark reference universe is missing required columns: "
            f"{missing}."
        )

    out = _prepare_french_benchmark_universe(reference_universe)

    if out.empty:
        raise RuntimeError(
            "French benchmark reference universe is empty after applying the "
            "required market-cap and book-to-market filters."
        )

    big_stock_universe = _select_big_stock_universe(out)

    # In the FF international methodology, Big stocks are the stocks that
    # cumulatively account for the top 90% of aggregate market cap.
    # The size breakpoint is therefore the smallest market cap still included
    # in that Big-stock subset.
    size_breakpoint = float(big_stock_universe["Market_Cap_Current"].min())

    # B/M breakpoints are the 30th and 70th percentiles of B/M among Big stocks.
    bm_breakpoints = [
        float(big_stock_universe["BM_French"].quantile(percentile))
        for percentile in FRENCH_BM_PERCENTILES
    ]

    return {
        "formation_date": formation_date,
        "reference_universe_count": int(out.shape[0]),
        "french_benchmark_exchanges": [
            {"country": country, "exchange_code": exchange_code}
            for country, exchange_code in exchange_definitions
        ],
        "benchmark_sort": "2x3_size_bm",
        "portfolio_labels": list(STANDARD_PORTFOLIO_LABELS),
        "big_stock_market_cap_share": float(FRENCH_BIG_STOCK_MARKET_CAP_SHARE),
        "big_stock_count": int(big_stock_universe.shape[0]),
        "big_stock_market_cap_floor": size_breakpoint,
        "size_breakpoint": size_breakpoint,
        "size_breakpoints": [size_breakpoint],  # kept for backward compatibility
        "bm_percentiles": list(FRENCH_BM_PERCENTILES),
        "bm_breakpoints": bm_breakpoints,
    }

def build_french_benchmark_sample_metadata(breakpoints: dict) -> dict:
    metadata = {
        "French benchmark formation date": breakpoints["formation_date"],
        "French benchmark sort": breakpoints.get("benchmark_sort", "2x3_size_bm"),
        "French benchmark reference universe count": breakpoints[
            "reference_universe_count"
        ],
        "French benchmark exchanges": [
            exchange["exchange_code"]
            for exchange in breakpoints["french_benchmark_exchanges"]
        ],
        "French benchmark portfolio labels": breakpoints.get(
            "portfolio_labels", list(STANDARD_PORTFOLIO_LABELS)
        ),
        "French benchmark big stock market cap share": breakpoints[
            "big_stock_market_cap_share"
        ],
        "French benchmark big stock count": breakpoints["big_stock_count"],
        "French benchmark size breakpoint": breakpoints["size_breakpoint"],
        "French benchmark big stock market cap floor": breakpoints[
            "big_stock_market_cap_floor"
        ],
        "French benchmark B/M percentiles": breakpoints["bm_percentiles"],
        "French benchmark B/M breakpoints": breakpoints["bm_breakpoints"],
    }

    optional_key_map = {
        "benchmark_universe_source": "Benchmark universe source",
        "benchmark_return_window_start": "Benchmark return window start",
        "benchmark_return_window_end": "Benchmark return window end",
        "benchmark_weighting_method": "Benchmark weighting method",
        "benchmark_return_request_batches": "Benchmark return request batches",
        "benchmark_constituent_count": "Benchmark constituent count",
    }
    for raw_key, metadata_key in optional_key_map.items():
        if raw_key in breakpoints:
            metadata[metadata_key] = breakpoints[raw_key]

    return metadata

def assign_french_benchmark_portfolios(
    stock_universe: pd.DataFrame,
    breakpoints: dict,
) -> pd.DataFrame:
    out = stock_universe.copy()

    if out.empty:
        out["BM"] = pd.Series(dtype="float64")
        out["BM_French"] = pd.Series(dtype="float64")
        out["Benchmark_Size_Group"] = pd.Series(dtype="string")
        out["Benchmark_BM_Group"] = pd.Series(dtype="string")
        out["Benchmark_Size_Q"] = pd.Series(dtype="Int64")
        out["Benchmark_BM_Q"] = pd.Series(dtype="Int64")
        out["Size_Q"] = pd.Series(dtype="Int64")
        out["BM_Q"] = pd.Series(dtype="Int64")
        out["Benchmark_Portfolio"] = pd.Series(dtype="string")
        return out

    out["Market_Cap_Current"] = pd.to_numeric(
        out["Market_Cap_Current"],
        errors="coerce",
    )
    out["BM_French"] = pd.to_numeric(
        out["BM_French"],
        errors="coerce",
    )
    out["BM"] = out["BM_French"]

    out["Benchmark_Size_Group"] = _assign_size_group(
        out["Market_Cap_Current"],
        breakpoints["size_breakpoint"],
    )
    out["Benchmark_BM_Group"] = _assign_bm_group(
        out["BM_French"],
        breakpoints["bm_breakpoints"],
    )

    out["Benchmark_Size_Q"] = (
        out["Benchmark_Size_Group"]
        .map(SIZE_GROUP_TO_Q)
        .astype("Int64")
    )
    out["Benchmark_BM_Q"] = (
        out["Benchmark_BM_Group"]
        .map(BM_GROUP_TO_Q)
        .astype("Int64")
    )

    # Keep these legacy column names so downstream code that expects Size_Q/BM_Q
    # does not immediately break. Their meaning is now 2x3, not 5x5.
    out["Size_Q"] = out["Benchmark_Size_Q"]
    out["BM_Q"] = out["Benchmark_BM_Q"]

    portfolio = pd.Series(pd.NA, index=out.index, dtype="string")
    valid = (
        out["Benchmark_Size_Group"].notna()
        & out["Benchmark_BM_Group"].notna()
    )
    portfolio.loc[valid] = (
        out.loc[valid, "Benchmark_Size_Group"].astype("string")
        + out.loc[valid, "Benchmark_BM_Group"].astype("string")
    )

    out["Benchmark_Portfolio"] = portfolio

    return out

def build_benchmark_portfolio_constituents(
    reference_universe: pd.DataFrame,
    breakpoints: dict,
) -> pd.DataFrame:
    assigned = assign_french_benchmark_portfolios(reference_universe, breakpoints)
    assigned["Market_Cap_Current"] = pd.to_numeric(
        assigned["Market_Cap_Current"],
        errors="coerce",
    )
    assigned = assigned.dropna(
        subset=["Instrument", "Benchmark_Portfolio", "Market_Cap_Current"]
    ).copy()
    assigned = assigned.loc[assigned["Market_Cap_Current"] > 0].copy()

    if assigned.empty:
        assigned["Benchmark_Weight"] = pd.Series(dtype="float64")
        return assigned

    assigned["Benchmark_Weight"] = assigned.groupby("Benchmark_Portfolio")[
        "Market_Cap_Current"
    ].transform(lambda values: values / values.sum())
    assigned = assigned.sort_values(
        ["Benchmark_Portfolio", "Instrument"]
    ).reset_index(drop=True)
    return assigned


def build_self_constructed_benchmark_returns(
    benchmark_constituents: pd.DataFrame,
    return_history: pd.DataFrame,
    *,
    columns: tuple[str, ...] = STANDARD_PORTFOLIO_LABELS,
) -> pd.DataFrame:
    if return_history.empty:
        raise ValueError("Benchmark return history is empty.")

    returns_frame = return_history.copy()
    returns_frame.index = pd.to_datetime(returns_frame.index, errors="coerce")
    returns_frame = returns_frame.loc[returns_frame.index.notna()].sort_index()
    returns_frame = returns_frame.loc[~returns_frame.index.duplicated(keep="first")]

    benchmark_returns = pd.DataFrame(
        index=returns_frame.index,
        columns=list(columns),
        dtype="float64",
    )
    benchmark_returns.index.name = "Date"

    if benchmark_constituents.empty:
        return benchmark_returns

    for portfolio, portfolio_constituents in benchmark_constituents.groupby(
        "Benchmark_Portfolio",
        observed=True,
    ):
        if portfolio not in benchmark_returns.columns:
            continue

        weights = (
            portfolio_constituents.loc[:, ["Instrument", "Benchmark_Weight"]]
            .dropna(subset=["Instrument", "Benchmark_Weight"])
            .drop_duplicates(subset=["Instrument"], keep="first")
            .set_index("Instrument")["Benchmark_Weight"]
            .astype("float64")
        )
        available_instruments = [
            instrument
            for instrument in weights.index.tolist()
            if instrument in returns_frame.columns
        ]
        if not available_instruments:
            continue

        aligned_weights = weights.reindex(available_instruments)
        aligned_returns = returns_frame.loc[:, available_instruments].apply(
            pd.to_numeric,
            errors="coerce",
        )
        observed_weight = aligned_returns.notna().mul(aligned_weights, axis=1).sum(axis=1)
        weighted_sum = aligned_returns.mul(aligned_weights, axis=1).sum(axis=1, min_count=1)
        benchmark_returns[portfolio] = weighted_sum.div(observed_weight.where(observed_weight > 0))

    return benchmark_returns.reindex(columns=list(columns))


def _prepare_french_benchmark_universe(reference_universe: pd.DataFrame) -> pd.DataFrame:
    out = reference_universe.copy()

    out["Market_Cap_Current"] = pd.to_numeric(
        out["Market_Cap_Current"],
        errors="coerce",
    )
    out["BM_French"] = pd.to_numeric(
        out["BM_French"],
        errors="coerce",
    )
    out["BM"] = out["BM_French"]

    valid = (
        out["Instrument"].notna()
        & out["Market_Cap_Current"].gt(0)
        & out["BM_French"].gt(0)
        & pd.Series(np.isfinite(out["BM_French"]), index=out.index)
    )

    return out.loc[valid].copy()


def _select_big_stock_universe(reference_universe: pd.DataFrame) -> pd.DataFrame:
    sorted_desc = reference_universe.sort_values(
        ["Market_Cap_Current", "Instrument"],
        ascending=[False, True],
    ).reset_index(drop=True)

    cumulative_share = (
        sorted_desc["Market_Cap_Current"].cumsum()
        / sorted_desc["Market_Cap_Current"].sum()
    )

    include_mask = cumulative_share.le(FRENCH_BIG_STOCK_MARKET_CAP_SHARE)

    # Include the marginal stock that crosses the 90% market-cap-share cutoff.
    crossing_positions = np.flatnonzero(
        cumulative_share.to_numpy() > FRENCH_BIG_STOCK_MARKET_CAP_SHARE
    )
    if crossing_positions.size:
        include_mask.iloc[int(crossing_positions[0])] = True

    big_stock_universe = sorted_desc.loc[include_mask].copy()
    if big_stock_universe.empty:
        raise RuntimeError(
            "Cannot compute French B/M breakpoints because the big-stock subset is empty."
        )

    return big_stock_universe


def _assign_size_group(
    market_caps: pd.Series,
    size_breakpoint: float,
) -> pd.Series:
    numeric_market_caps = pd.to_numeric(market_caps, errors="coerce")
    out = pd.Series(pd.NA, index=numeric_market_caps.index, dtype="string")

    valid = numeric_market_caps.notna() & numeric_market_caps.gt(0)

    # Marginal stock at the breakpoint is assigned to Big, matching the
    # construction where the crossing stock is included in the Big subset.
    out.loc[valid & numeric_market_caps.ge(float(size_breakpoint))] = "B"
    out.loc[valid & numeric_market_caps.lt(float(size_breakpoint))] = "S"

    return out


def _assign_bm_group(
    bm_values: pd.Series,
    bm_breakpoints: list[float],
) -> pd.Series:
    if len(bm_breakpoints) != 2:
        raise ValueError(
            "2x3 Fama-French benchmark assignment requires exactly two B/M "
            "breakpoints: 30th and 70th percentiles."
        )

    low_breakpoint, high_breakpoint = sorted(float(value) for value in bm_breakpoints)

    numeric_bm = pd.to_numeric(bm_values, errors="coerce")
    out = pd.Series(pd.NA, index=numeric_bm.index, dtype="string")

    valid = numeric_bm.notna() & pd.Series(np.isfinite(numeric_bm), index=numeric_bm.index)

    out.loc[valid & numeric_bm.le(low_breakpoint)] = "G"
    out.loc[
        valid
        & numeric_bm.gt(low_breakpoint)
        & numeric_bm.le(high_breakpoint)
    ] = "N"
    out.loc[valid & numeric_bm.gt(high_breakpoint)] = "V"

    return out

def _select_big_stock_universe(reference_universe: pd.DataFrame) -> pd.DataFrame:
    sorted_desc = reference_universe.sort_values(
        ["Market_Cap_Current", "Instrument"],
        ascending=[False, True],
    ).reset_index(drop=True)
    cumulative_share = sorted_desc["Market_Cap_Current"].cumsum() / sorted_desc[
        "Market_Cap_Current"
    ].sum()
    include_mask = cumulative_share.le(FRENCH_BIG_STOCK_MARKET_CAP_SHARE)

    crossing_positions = np.flatnonzero(
        cumulative_share.to_numpy() > FRENCH_BIG_STOCK_MARKET_CAP_SHARE
    )
    if crossing_positions.size:
        include_mask.iloc[int(crossing_positions[0])] = True

    big_stock_universe = sorted_desc.loc[include_mask].copy()
    if big_stock_universe.empty:
        raise RuntimeError(
            "Cannot compute French B/M breakpoints because the big-stock subset is empty."
        )

    return big_stock_universe
