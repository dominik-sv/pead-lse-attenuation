from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("ggplot")
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "current_exchange_threshold_comparison"
OUTPUT_DIR = INPUT_DIR / "A2_from_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_PATH = INPUT_DIR / "cumulative_distribution_plot.png"

SUMMARY_PATH = INPUT_DIR / "summary.json"
US_SAMPLE_PATH = INPUT_DIR / "us_sample.csv"
XLON_SAMPLE_PATH = INPUT_DIR / "xlon_sample.csv"

METRIC_SPECS = {
    "Market_Cap_Current": {
        "title": "Cumulative Market Cap Distribution of Active Securities",
        "x_label": "Market cap ($ millions)",
        "threshold_key": "market_cap_threshold",
        "equivalent_key": "market_cap",
        "summary_prefix": "market_cap",
        "display_multiplier": 1.0,
        "study_threshold": 0.5,
    },
    "Price": {
        "title": "Cumulative Price Distribution of Active Securities",
        "x_label": "Price ($)",
        "threshold_key": "price_threshold",
        "equivalent_key": "price",
        "summary_prefix": "price",
        "display_multiplier": 1.0,
        "study_threshold": 0.0025,
    },
}

US_COMPARATOR_COLOR = "#1f77b4"
LSE_COMPARATOR_COLOR = "#ff7f0e"
US_STUDY_THRESHOLD_LABEL = "Thresholds used in U.S.-based studies"
THIS_STUDY_THRESHOLD_LABEL = "Thresholds used in this study"


def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    us_sample = pd.read_csv(US_SAMPLE_PATH, low_memory=False)
    xlon_sample = pd.read_csv(XLON_SAMPLE_PATH, low_memory=False)

    for frame in [us_sample, xlon_sample]:
        for column in ["Price", "Market_Cap_Current"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return summary, us_sample, xlon_sample


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = values.dropna()
    clean = clean.loc[clean > 0]
    clean = clean.sort_values(kind="mergesort").to_numpy(dtype=float)
    if clean.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    y = np.arange(1, clean.size + 1, dtype=float) / clean.size * 100.0
    return clean, y


def cumulative_share_at_threshold(values: pd.Series, threshold: float) -> float:
    clean = values.dropna()
    clean = clean.loc[clean > 0]
    if clean.empty:
        return float("nan")
    return float((clean <= threshold).mean() * 100.0)


def value_at_cumulative_share(values: pd.Series, cumulative_share_pct: float) -> float:
    clean = values.dropna()
    clean = clean.loc[clean > 0]
    if clean.empty:
        return float("nan")
    quantile = cumulative_share_pct / 100.0
    return float(clean.quantile(quantile, interpolation="linear"))


def build_percentile_grid() -> list[float]:
    base = list(np.arange(0.0, 1.0, 0.1))
    base.extend(np.arange(1.0, 10.0, 0.5))
    base.extend(np.arange(10.0, 100.0 + 0.0001, 1.0))
    rounded = sorted({round(value, 10) for value in base})
    return rounded


def format_percentile(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value))}"
    if math.isclose(value * 10, round(value * 10), abs_tol=1e-9):
        return f"{value:.1f}"
    return f"{value:.2f}"


def apply_log_x_grid(axis) -> None:
    axis.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    log_formatter = LogFormatterMathtext(base=10)
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position: (
                log_formatter(value, position)
                if round(math.log10(value)) % 2 == 0
                else ""
            )
        )
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", which="minor", labelbottom=False)
    axis.tick_params(axis="x", labelrotation=0)
    axis.grid(False)
    axis.grid(True, which="major", axis="both", alpha=0.3)
    axis.grid(True, which="minor", axis="x", alpha=0.2)


def build_percentile_summary(us_sample: pd.DataFrame, xlon_sample: pd.DataFrame) -> pd.DataFrame:
    percentiles = build_percentile_grid()
    rows: list[dict[str, float | str]] = []

    for percentile in percentiles:
        q = percentile / 100.0
        row: dict[str, float | str] = {
            "percentile": percentile,
            "percentile_label": format_percentile(percentile),
        }
        for column, spec in METRIC_SPECS.items():
            prefix = spec["summary_prefix"]
            row[f"us_{prefix}"] = float(
                us_sample[column]
                .dropna()
                .loc[us_sample[column].dropna() > 0]
                .quantile(q, interpolation="linear")
            )
            row[f"xlon_{prefix}"] = float(
                xlon_sample[column]
                .dropna()
                .loc[xlon_sample[column].dropna() > 0]
                .quantile(q, interpolation="linear")
            )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_trimmed_distribution(
    summary: dict,
    us_sample: pd.DataFrame,
    xlon_sample: pd.DataFrame,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.6))

    annotation_rows: list[dict[str, float | str]] = []

    for ax, (column, spec) in zip(axes, METRIC_SPECS.items()):
        display_multiplier = float(spec["display_multiplier"])
        threshold = float(summary[spec["threshold_key"]]) * display_multiplier
        study_threshold = float(spec["study_threshold"])

        us_x, us_y = ecdf(us_sample[column])
        xlon_x, xlon_y = ecdf(xlon_sample[column])
        us_x = us_x * display_multiplier
        xlon_x = xlon_x * display_multiplier

        ax.plot(us_x, us_y, label="NYSE, NASDAQ and AMEX combined", linewidth=2.0, color=US_COMPARATOR_COLOR)
        ax.plot(xlon_x, xlon_y, label="LSE", linewidth=2.0, color=LSE_COMPARATOR_COLOR)
        ax.axvline(
            threshold,
            color=US_COMPARATOR_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=US_STUDY_THRESHOLD_LABEL,
        )
        ax.axvline(
            study_threshold,
            color=LSE_COMPARATOR_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=THIS_STUDY_THRESHOLD_LABEL,
        )

        source_threshold = threshold / display_multiplier
        us_y_at_threshold = cumulative_share_at_threshold(us_sample[column], source_threshold)
        xlon_x_at_us_y = value_at_cumulative_share(xlon_sample[column], us_y_at_threshold) * display_multiplier

        positive_values = pd.concat(
            [
                us_sample[column].dropna().loc[us_sample[column].dropna() > 0],
                xlon_sample[column].dropna().loc[xlon_sample[column].dropna() > 0],
            ],
            ignore_index=True,
        )
        left_min = min(
            float(positive_values.min()) * display_multiplier,
            study_threshold,
            threshold,
        )
        ax.set_xscale("log")
        ax.set_xlim(left_min, max(float(positive_values.max()) * display_multiplier, threshold))
        ax.set_ylim(0, 100)
        ax.set_xlabel(spec["x_label"])
        apply_log_x_grid(ax)

        annotation_rows.append(
            {
                "metric": spec["summary_prefix"],
                "threshold": threshold,
                "us_cumulative_share_at_threshold_pct": us_y_at_threshold,
                "xlon_value_at_us_threshold_share": xlon_x_at_us_y,
                "xlon_equivalent_threshold_from_summary": float(
                    summary["xlon_equivalent_thresholds"][spec["equivalent_key"]]["threshold"]
                )
                * display_multiplier,
            }
        )

    axes[0].set_ylabel("Cumulative share (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(PLOT_OUTPUT_PATH, dpi=180, bbox_inches="tight")
    # The thesis retains only cumulative_distribution_plot.png.
    # fig.savefig(OUTPUT_DIR / "cumulative_distribution_plot_left_of_threshold.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(annotation_rows)


def main() -> None:
    summary, us_sample, xlon_sample = load_inputs()
    threshold_reference = plot_trimmed_distribution(summary, us_sample, xlon_sample)
    percentile_summary = build_percentile_summary(us_sample, xlon_sample)

    threshold_reference.to_csv(OUTPUT_DIR / "threshold_reference_summary.csv", index=False)
    percentile_summary.to_csv(OUTPUT_DIR / "percentile_summary_price_and_market_cap.csv", index=False)

    print(f"Wrote distribution plot to {PLOT_OUTPUT_PATH}")
    # The duplicate trimmed plot export is intentionally disabled.
    print(f"Wrote threshold reference summary to {OUTPUT_DIR / 'threshold_reference_summary.csv'}")
    print(f"Wrote percentile summary to {OUTPUT_DIR / 'percentile_summary_price_and_market_cap.csv'}")


if __name__ == "__main__":
    main()
