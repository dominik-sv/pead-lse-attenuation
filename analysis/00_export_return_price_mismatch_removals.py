from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import (  # noqa: E402
    FORMATION_YEARS,
    NEGATIVE_OUTLIER_RETURN_THRESHOLD_PCT,
    POSITIVE_OUTLIER_RETURN_THRESHOLD_PCT,
    PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS,
)
from src.core.project_paths import DATA_DIR  # noqa: E402
from src.core.year_context import build_year_context  # noqa: E402
from src.pead.market_data_fetch import read_market_data_file  # noqa: E402
from src.pead.market_data_repairs import (  # noqa: E402
    PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY,
    _apply_stock_day_filters,
    _build_long_histories,
    _is_outlier_return,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "return_price_mismatch_removed_stock_days.csv"
)
ANALYSIS_MISMATCH_SAMPLE_SIZE_KEY = (
    f"Analysis branch: {PRICE_IMPLIED_MISMATCH_SAMPLE_SIZE_KEY}"
)


def load_base_builder_module():
    module_path = PROJECT_ROOT / "scripts" / "02_build_universe_and_market_data.py"
    spec = importlib.util.spec_from_file_location("base_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_neighbor_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["Instrument", "Date"], kind="stable").copy()
    grouped = out.groupby("Instrument", sort=False)
    for offset, prefix in [(-1, "Previous"), (1, "Following")]:
        out[f"{prefix}_Trading_Date"] = grouped["Date"].shift(-offset)
        out[f"{prefix}_TotalReturn"] = grouped["TotalReturn"].shift(-offset)
        out[f"{prefix}_PriceClose"] = grouped["Current_PriceClose"].shift(-offset)
        out[f"{prefix}_Price_Return_From_Close"] = grouped["Price_Return_From_Close"].shift(
            -offset
        )
    return out


def build_year_mismatch_rows(year: int, base_builder) -> pd.DataFrame:
    year_context = build_year_context(year, DATA_DIR)
    if not year_context.shared_post_cleaning_universe_path.exists():
        raise FileNotFoundError(year_context.shared_post_cleaning_universe_path)
    if not year_context.shared_market_data_path.exists():
        raise FileNotFoundError(year_context.shared_market_data_path)

    shared_universe = pd.read_csv(year_context.shared_post_cleaning_universe_path)
    shared_market_data = read_market_data_file(year_context.shared_market_data_path)

    _, filtered_market_data, _ = base_builder.apply_stock_level_return_filters(
        universe=shared_universe,
        market_data=shared_market_data,
        year_context=year_context,
        sample_size_prefix="Analysis branch: ",
    )

    pre_mismatch_market_data, _ = _apply_stock_day_filters(filtered_market_data)
    return_long, price_long = _build_long_histories(pre_mismatch_market_data)
    if return_long.empty or price_long.empty:
        return pd.DataFrame()

    merged = return_long.merge(
        price_long.loc[
            :,
            [
                "Instrument",
                "Date",
                "Instrument_First_Price_Date",
                "Previous_PriceClose",
                "Current_PriceClose",
                "Price_Return_From_Close",
            ],
        ],
        on=["Instrument", "Date"],
        how="left",
    )
    merged["Return_Error"] = merged["Price_Return_From_Close"] - merged["TotalReturn"]
    merged["Absolute_Return_Error"] = merged["Return_Error"].abs()
    merged["Price_Return_Available"] = merged["Price_Return_From_Close"].notna()
    merged["Ignore_For_Validation"] = (
        merged["Current_PriceClose"].notna()
        & merged["Previous_PriceClose"].isna()
        & (merged["Date"] == merged["Instrument_First_Price_Date"])
    )
    merged["Outlier_Return_Flag"] = merged["TotalReturn"].map(_is_outlier_return)

    with_neighbors = add_neighbor_context(merged)
    mismatch_mask = (
        with_neighbors["Outlier_Return_Flag"]
        & with_neighbors["Price_Return_Available"]
        & ~with_neighbors["Ignore_For_Validation"]
        & (
            with_neighbors["Absolute_Return_Error"]
            > float(PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS)
        )
    )
    mismatches = with_neighbors.loc[mismatch_mask].copy()
    if mismatches.empty:
        return pd.DataFrame()

    mismatches.insert(0, "Formation_Year", int(year))
    mismatches["Removed_Trading_Date"] = mismatches["Date"]
    mismatches["Removed_TotalReturn"] = mismatches["TotalReturn"]
    mismatches["Removed_PriceClose"] = mismatches["Current_PriceClose"]
    mismatches["Removed_Price_Return_From_Close"] = mismatches[
        "Price_Return_From_Close"
    ]
    mismatches["Validation_Previous_PriceClose"] = mismatches["Previous_PriceClose"]
    mismatches["Validation_Current_PriceClose"] = mismatches["Current_PriceClose"]
    mismatches["Mismatch_Tolerance_Pct_Points"] = float(
        PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS
    )
    mismatches["Positive_Outlier_Threshold_Pct"] = float(
        POSITIVE_OUTLIER_RETURN_THRESHOLD_PCT
    )
    mismatches["Negative_Outlier_Threshold_Pct"] = float(
        NEGATIVE_OUTLIER_RETURN_THRESHOLD_PCT
    )

    columns = [
        "Formation_Year",
        "Instrument",
        "Previous_Trading_Date",
        "Previous_TotalReturn",
        "Previous_PriceClose",
        "Previous_Price_Return_From_Close",
        "Removed_Trading_Date",
        "Removed_TotalReturn",
        "Removed_PriceClose",
        "Removed_Price_Return_From_Close",
        "Following_Trading_Date",
        "Following_TotalReturn",
        "Following_PriceClose",
        "Following_Price_Return_From_Close",
        "Validation_Previous_PriceClose",
        "Validation_Current_PriceClose",
        "Return_Error",
        "Absolute_Return_Error",
        "Mismatch_Tolerance_Pct_Points",
        "Positive_Outlier_Threshold_Pct",
        "Negative_Outlier_Threshold_Pct",
    ]
    return mismatches.loc[:, columns].sort_values(
        ["Formation_Year", "Instrument", "Removed_Trading_Date"],
        kind="stable",
    )


def recorded_analysis_mismatch_count(year: int) -> int | None:
    sample_size_path = build_year_context(year, DATA_DIR).sample_size_path
    if not sample_size_path.exists():
        return None
    with sample_size_path.open("r", encoding="utf-8") as handle:
        sample_size = json.load(handle)
    value = sample_size.get(ANALYSIS_MISMATCH_SAMPLE_SIZE_KEY)
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export stock-day returns removed by the return/price mismatch repair "
            "with previous, removed, and following trading-day return/price context."
        )
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=FORMATION_YEARS,
        help="Formation years to inspect. Defaults to all configured years.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--include-zero-count-years",
        action="store_true",
        help=(
            "Replay years whose sample_size.json records zero analysis-branch "
            "mismatch removals. By default those years are skipped for speed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_builder = load_base_builder_module()

    yearly_rows: list[pd.DataFrame] = []
    for year in args.years:
        recorded_count = recorded_analysis_mismatch_count(int(year))
        if (
            recorded_count == 0
            and not args.include_zero_count_years
        ):
            print(f"Skipping formation year {year}: recorded mismatch removals = 0")
            continue
        print(f"Inspecting formation year {year}...")
        rows = build_year_mismatch_rows(int(year), base_builder)
        if recorded_count is None:
            print(f"  removed mismatch stock-days: {len(rows)}")
        else:
            print(
                "  removed mismatch stock-days: "
                f"{len(rows)} (sample_size recorded {recorded_count})"
            )
        if not rows.empty:
            yearly_rows.append(rows)

    output = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
