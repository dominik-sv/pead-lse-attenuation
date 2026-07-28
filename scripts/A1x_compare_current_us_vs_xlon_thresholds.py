from __future__ import annotations

from pathlib import Path
import sys

import lseg.data as ld
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import BASE_DATA_DIR, CURRENCY, ORDINARY_SHARE_TYPES
from src.pead.universe_filters import (
    build_historical_security_definition_mask,
    lseg_get_data_with_retry,
    rename_lseg_columns,
)
from src.utils.io_utils import save_json

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_ROOT / BASE_DATA_DIR
OUTPUT_DIR = DATA_DIR / "current_exchange_threshold_comparison"

US_EXCHANGES = ("XNYS", "XNAS", "XASE")
XLON_EXCHANGES = ("XLON",)

MARKET_CAP_THRESHOLD = 5.0
PRICE_THRESHOLD = 1.0
MARKET_CAP_STUDY_THRESHOLD = 500_000.0
PRICE_STUDY_THRESHOLD = 0.0025

US_COLOR = "#1f77b4"
LSE_COLOR = "#ff7f0e"
US_STUDY_THRESHOLD_LABEL = "Thresholds used in U.S.-based studies"
THIS_STUDY_THRESHOLD_LABEL = "Thresholds used in this study"

SNAPSHOT_FIELDS = {
    "TR.RIC": "RIC",
    "TR.ExchangeTicker": "Ticker",
    "TR.CommonName": "Name",
    "TR.InstrumentType": "Instrument_Type",
    "TR.SecurityType": "Security_Type",
    "TR.ExchangeName": "Exchange_Name",
    "TR.ExchangeMarketIdCode": "Exchange_Code",
    "TR.PriceClose": "Price",
    "TR.CompanyMarketCap(Scale=6)": "Market_Cap_Current",
}


def fetch_exchange_snapshot(exchange_code: str, currency: str) -> pd.DataFrame:
    universe = [
        f"""SCREEN(
            U(IN(Equity(active or inactive,public,primary))/*UNV:Public*/),
            IN(TR.ExchangeMarketIdCode,"{exchange_code}"),
            CURN="{currency}"
        )"""
    ]

    raw = lseg_get_data_with_retry(
        universe=universe,
        fields=list(SNAPSHOT_FIELDS.keys()),
        parameters={"Curn": currency},
        header_type=ld.HeaderType.NAME,
    )

    if raw is None or raw.empty:
        return build_empty_snapshot_frame()

    out = rename_lseg_columns(raw, SNAPSHOT_FIELDS).copy()
    out["Snapshot_Exchange_Request"] = exchange_code
    return normalize_snapshot_frame(out)


def build_empty_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Instrument",
            "RIC",
            "Ticker",
            "Name",
            "Instrument_Type",
            "Security_Type",
            "Exchange_Name",
            "Exchange_Code",
            "Price",
            "Market_Cap_Current",
            "Snapshot_Exchange_Request",
        ]
    )


def normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = build_empty_snapshot_frame()
    for column in df.columns:
        out[column] = df[column]

    for column in (
        "RIC",
        "Ticker",
        "Name",
        "Instrument_Type",
        "Security_Type",
        "Exchange_Name",
        "Exchange_Code",
        "Snapshot_Exchange_Request",
    ):
        out[column] = out[column].astype("string").str.strip()

    for column in ("Price", "Market_Cap_Current"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["Exchange_Code"] = out["Exchange_Code"].str.upper()
    out["Snapshot_Exchange_Request"] = out["Snapshot_Exchange_Request"].str.upper()
    out["Instrument"] = coalesce_identifier(out)
    return out


def coalesce_identifier(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="string")
    for column in ("RIC", "Ticker", "Name"):
        values = df[column].astype("string").str.strip()
        valid = values.notna() & (values != "")
        result = result.where(result.notna() & (result != ""), values.where(valid))
    return result


def filter_current_ordinary_share_sample(
    df: pd.DataFrame,
    sample_name: str,
    allowed_exchanges: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    out = normalize_snapshot_frame(df)
    diagnostics = {"raw_rows": int(len(out))}

    allowed_set = {exchange.strip().upper() for exchange in allowed_exchanges}
    out = out.loc[out["Exchange_Code"].isin(allowed_set)].copy()
    diagnostics["target_exchange_rows"] = int(len(out))

    out = out.loc[out["Instrument"].notna() & (out["Instrument"] != "")].copy()
    diagnostics["usable_identifier_rows"] = int(len(out))

    out = out.loc[build_historical_security_definition_mask(out)].copy()
    diagnostics["common_stock_candidate_rows"] = int(len(out))

    out = out.loc[out["Instrument_Type"].isin(ORDINARY_SHARE_TYPES)].copy()
    diagnostics["ordinary_share_rows"] = int(len(out))

    completeness_score = (
        out["Market_Cap_Current"].notna().astype(int)
        + out["Price"].notna().astype(int)
    )
    out = (
        out.assign(_completeness_score=completeness_score)
        .sort_values(
            [
                "_completeness_score",
                "Market_Cap_Current",
                "Price",
                "RIC",
            ],
            ascending=[False, False, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=["Instrument"], keep="first")
        .drop(columns="_completeness_score")
        .reset_index(drop=True)
    )
    diagnostics["deduplicated_rows"] = int(len(out))
    diagnostics["sample_name"] = sample_name
    diagnostics["allowed_exchanges"] = list(allowed_exchanges)
    return out, diagnostics


def build_sample_metrics(
    sample: pd.DataFrame,
    sample_name: str,
    market_cap_threshold: float,
    price_threshold: float,
) -> dict:
    market_cap = pd.to_numeric(sample["Market_Cap_Current"], errors="coerce")
    price = pd.to_numeric(sample["Price"], errors="coerce")
    joint_mask = market_cap.notna() & price.notna()

    market_cap_share = compute_share_above(market_cap, market_cap_threshold)
    price_share = compute_share_above(price, price_threshold)
    joint_share = compute_joint_share_above(
        market_cap,
        price,
        market_cap_threshold,
        price_threshold,
    )

    return {
        "sample_name": sample_name,
        "row_count": int(len(sample)),
        "market_cap_non_missing_count": int(market_cap.notna().sum()),
        "price_non_missing_count": int(price.notna().sum()),
        "joint_non_missing_count": int(joint_mask.sum()),
        "percent_above_market_cap_threshold": market_cap_share * 100.0,
        "percent_above_price_threshold": price_share * 100.0,
        "percent_above_both_thresholds": joint_share * 100.0,
    }


def compute_share_above(values: pd.Series, threshold: float) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float((valid > threshold).mean())


def compute_joint_share_above(
    market_cap: pd.Series,
    price: pd.Series,
    market_cap_threshold: float,
    price_threshold: float,
) -> float:
    market_cap_valid = pd.to_numeric(market_cap, errors="coerce")
    price_valid = pd.to_numeric(price, errors="coerce")
    valid_mask = market_cap_valid.notna() & price_valid.notna()
    if not valid_mask.any():
        return float("nan")
    joint = (
        market_cap_valid.loc[valid_mask].gt(market_cap_threshold)
        & price_valid.loc[valid_mask].gt(price_threshold)
    )
    return float(joint.mean())


def build_equivalent_threshold_result(
    values: pd.Series,
    target_share_above: float,
    metric_name: str,
) -> dict:
    finite = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
        .to_numpy(dtype=float)
    )

    if finite.size == 0 or pd.isna(target_share_above):
        return {
            "metric": metric_name,
            "threshold": None,
            "achieved_share_above": None,
            "target_share_above": None if pd.isna(target_share_above) else float(target_share_above),
            "exact_match": False,
        }

    candidate_thresholds = build_candidate_thresholds(finite)
    candidate_shares = np.array(
        [(finite > threshold).mean() for threshold in candidate_thresholds],
        dtype=float,
    )
    distance = np.abs(candidate_shares - float(target_share_above))
    best_index = int(distance.argmin())

    threshold = float(candidate_thresholds[best_index])
    achieved_share = float(candidate_shares[best_index])
    return {
        "metric": metric_name,
        "threshold": threshold,
        "achieved_share_above": achieved_share,
        "target_share_above": float(target_share_above),
        "exact_match": bool(np.isclose(achieved_share, target_share_above, atol=1e-12)),
    }


def build_candidate_thresholds(sorted_values: np.ndarray) -> np.ndarray:
    unique_values = np.unique(sorted_values)
    if unique_values.size == 1:
        only_value = float(unique_values[0])
        epsilon = max(abs(only_value) * 1e-6, 1e-6)
        return np.array([only_value - epsilon, only_value], dtype=float)

    candidate_thresholds = []
    min_value = float(unique_values[0])
    epsilon = max(abs(min_value) * 1e-6, 1e-6)
    candidate_thresholds.append(min_value - epsilon)

    for left, right in zip(unique_values[:-1], unique_values[1:]):
        candidate_thresholds.append(float(left + (right - left) / 2.0))

    candidate_thresholds.append(float(unique_values[-1]))
    return np.array(candidate_thresholds, dtype=float)


def build_distribution_plot(
    us_sample: pd.DataFrame,
    xlon_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    plot_cumulative_distribution(
        axis=axes[0],
        us_values=us_sample["Market_Cap_Current"] * 1_000_000.0,
        xlon_values=xlon_sample["Market_Cap_Current"] * 1_000_000.0,
        threshold=MARKET_CAP_THRESHOLD * 1_000_000.0,
        study_threshold=MARKET_CAP_STUDY_THRESHOLD,
        title="Cumulative Market Cap Distribution of Active Securities",
        xlabel="Market cap (USD)",
    )
    plot_cumulative_distribution(
        axis=axes[1],
        us_values=us_sample["Price"],
        xlon_values=xlon_sample["Price"],
        threshold=PRICE_THRESHOLD,
        study_threshold=PRICE_STUDY_THRESHOLD,
        title="Cumulative Price Distribution of Active Securities",
        xlabel="Price (USD)",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_distribution(
    axis,
    us_values: pd.Series,
    xlon_values: pd.Series,
    threshold: float,
    study_threshold: float,
    title: str,
    xlabel: str,
) -> None:
    us_positive = positive_numeric_values(us_values)
    xlon_positive = positive_numeric_values(xlon_values)

    if us_positive.empty and xlon_positive.empty:
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cumulative share (%)")
        axis.text(0.5, 0.5, "No positive observations", ha="center", va="center")
        return

    combined = pd.concat([us_positive, xlon_positive], ignore_index=True)
    min_value = float(combined.min())
    max_value = float(combined.max())

    if np.isclose(min_value, max_value):
        min_value = min_value / 2.0
        max_value = max_value * 2.0

    if not us_positive.empty:
        plot_empirical_cdf(
            axis=axis,
            values=us_positive,
            label="NYSE, NASDAQ and AMEX combined",
            color=US_COLOR,
        )
    if not xlon_positive.empty:
        plot_empirical_cdf(
            axis=axis,
            values=xlon_positive,
            label="LSE",
            color=LSE_COLOR,
        )

    axis.axvline(
        threshold,
        color=US_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=US_STUDY_THRESHOLD_LABEL,
    )
    axis.axvline(
        study_threshold,
        color=LSE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=THIS_STUDY_THRESHOLD_LABEL,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Cumulative share (%)")
    axis.set_xlim(min_value, max_value)
    axis.set_ylim(0, 100)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_major_formatter(FuncFormatter(format_log_tick_as_plain_number))
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)


def positive_numeric_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.loc[numeric > 0].dropna()


def plot_empirical_cdf(
    axis,
    values: pd.Series,
    label: str,
    color: str,
) -> None:
    sorted_values = np.sort(values.to_numpy(dtype=float))
    if sorted_values.size == 0:
        return

    cumulative_share = (np.arange(1, sorted_values.size + 1) / sorted_values.size) * 100.0
    axis.step(
        sorted_values,
        cumulative_share,
        where="post",
        label=label,
        color=color,
        linewidth=1.8,
    )


def format_log_tick_as_plain_number(value: float, _position: int) -> str:
    if value <= 0 or not np.isfinite(value):
        return ""
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def save_samples(
    us_sample: pd.DataFrame,
    xlon_sample: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    us_sample.to_csv(output_dir / "us_sample.csv", index=False)
    xlon_sample.to_csv(output_dir / "xlon_sample.csv", index=False)


def print_summary(summary: dict) -> None:
    us_metrics = summary["samples"]["us_combined"]["metrics"]
    xlon_metrics = summary["samples"]["xlon"]["metrics"]
    xlon_market_cap_equivalent = summary["xlon_equivalent_thresholds"]["market_cap"]
    xlon_price_equivalent = summary["xlon_equivalent_thresholds"]["price"]

    print("\n=== Current US vs XLON threshold comparison ===")
    print(f"Output directory: {summary['output_directory']}")
    print(
        "US combined rows after ordinary-share filtering: "
        f"{summary['samples']['us_combined']['metrics']['row_count']}"
    )
    print(
        "XLON rows after ordinary-share filtering: "
        f"{summary['samples']['xlon']['metrics']['row_count']}"
    )
    print(
        f"US % above ${MARKET_CAP_THRESHOLD:g}m market cap: "
        f"{us_metrics['percent_above_market_cap_threshold']:.2f}%"
    )
    print(
        f"XLON % above ${MARKET_CAP_THRESHOLD:g}m market cap: "
        f"{xlon_metrics['percent_above_market_cap_threshold']:.2f}%"
    )
    print(
        f"US % above ${PRICE_THRESHOLD:g} price: "
        f"{us_metrics['percent_above_price_threshold']:.2f}%"
    )
    print(
        f"XLON % above ${PRICE_THRESHOLD:g} price: "
        f"{xlon_metrics['percent_above_price_threshold']:.2f}%"
    )
    print(
        f"US % above both thresholds: "
        f"{us_metrics['percent_above_both_thresholds']:.2f}%"
    )
    print(
        f"XLON % above both thresholds: "
        f"{xlon_metrics['percent_above_both_thresholds']:.2f}%"
    )
    print(
        "XLON-equivalent market-cap threshold for the US > $5m survival rate: "
        f"{format_threshold(xlon_market_cap_equivalent['threshold'])} "
        f"(achieved share {format_percent(xlon_market_cap_equivalent['achieved_share_above'])})"
    )
    print(
        "XLON-equivalent price threshold for the US > $1 survival rate: "
        f"{format_threshold(xlon_price_equivalent['threshold'])} "
        f"(achieved share {format_percent(xlon_price_equivalent['achieved_share_above'])})"
    )


def format_threshold(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.6g}"


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.2f}%"


def build_summary(
    us_sample: pd.DataFrame,
    xlon_sample: pd.DataFrame,
    us_diagnostics: dict,
    xlon_diagnostics: dict,
    output_dir: Path,
) -> dict:
    us_metrics = build_sample_metrics(
        us_sample,
        sample_name="us_combined",
        market_cap_threshold=MARKET_CAP_THRESHOLD,
        price_threshold=PRICE_THRESHOLD,
    )
    xlon_metrics = build_sample_metrics(
        xlon_sample,
        sample_name="xlon",
        market_cap_threshold=MARKET_CAP_THRESHOLD,
        price_threshold=PRICE_THRESHOLD,
    )

    us_market_cap_share = us_metrics["percent_above_market_cap_threshold"] / 100.0
    us_price_share = us_metrics["percent_above_price_threshold"] / 100.0

    xlon_market_cap_equivalent = build_equivalent_threshold_result(
        xlon_sample["Market_Cap_Current"],
        target_share_above=us_market_cap_share,
        metric_name="market_cap",
    )
    xlon_price_equivalent = build_equivalent_threshold_result(
        xlon_sample["Price"],
        target_share_above=us_price_share,
        metric_name="price",
    )

    return {
        "currency": CURRENCY,
        "market_cap_unit": "USD millions",
        "price_unit": "USD",
        "threshold_definition": "strictly greater than",
        "market_cap_threshold": MARKET_CAP_THRESHOLD,
        "price_threshold": PRICE_THRESHOLD,
        "output_directory": str(output_dir),
        "samples": {
            "us_combined": {
                "exchanges": list(US_EXCHANGES),
                "diagnostics": us_diagnostics,
                "metrics": us_metrics,
            },
            "xlon": {
                "exchanges": list(XLON_EXCHANGES),
                "diagnostics": xlon_diagnostics,
                "metrics": xlon_metrics,
            },
        },
        "xlon_equivalent_thresholds": {
            "market_cap": xlon_market_cap_equivalent,
            "price": xlon_price_equivalent,
        },
        "output_files": {
            "us_sample_csv": str(output_dir / "us_sample.csv"),
            "xlon_sample_csv": str(output_dir / "xlon_sample.csv"),
            "summary_json": str(output_dir / "summary.json"),
            "distribution_plot_png": str(output_dir / "distribution_plot.png"),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ld.open_session()
    try:
        us_frames = [
            fetch_exchange_snapshot(exchange_code=exchange_code, currency=CURRENCY)
            for exchange_code in US_EXCHANGES
        ]
        xlon_frames = [
            fetch_exchange_snapshot(exchange_code=exchange_code, currency=CURRENCY)
            for exchange_code in XLON_EXCHANGES
        ]

        us_raw = (
            pd.concat(us_frames, ignore_index=True)
            if us_frames
            else build_empty_snapshot_frame()
        )
        xlon_raw = (
            pd.concat(xlon_frames, ignore_index=True)
            if xlon_frames
            else build_empty_snapshot_frame()
        )

        us_sample, us_diagnostics = filter_current_ordinary_share_sample(
            df=us_raw,
            sample_name="us_combined",
            allowed_exchanges=US_EXCHANGES,
        )
        xlon_sample, xlon_diagnostics = filter_current_ordinary_share_sample(
            df=xlon_raw,
            sample_name="xlon",
            allowed_exchanges=XLON_EXCHANGES,
        )

        save_samples(us_sample, xlon_sample, OUTPUT_DIR)
        build_distribution_plot(
            us_sample=us_sample,
            xlon_sample=xlon_sample,
            output_path=OUTPUT_DIR / "distribution_plot.png",
        )

        summary = build_summary(
            us_sample=us_sample,
            xlon_sample=xlon_sample,
            us_diagnostics=us_diagnostics,
            xlon_diagnostics=xlon_diagnostics,
            output_dir=OUTPUT_DIR,
        )
        save_json(summary, OUTPUT_DIR / "summary.json")
        print_summary(summary)
    finally:
        try:
            ld.close_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()


