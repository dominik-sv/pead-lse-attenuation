from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_SUBDIR = "D_compustat_datastream_sample_size_comparison"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / OUTPUT_SUBDIR
DEFAULT_COMPARISON_INPUT = (
    DEFAULT_OUTPUT_DIR / "compustat_vs_datastream_sample_sizes_by_year.csv"
)
DEFAULT_COMPUSTAT_SOURCE = (
    PROJECT_ROOT / "data" / "XLON_membership" / "compustat_filtered_june.csv"
)
DEFAULT_ACTIVE_STOCKS_INPUT = (
    DEFAULT_OUTPUT_DIR / "datastream_active_screener_stocks.csv"
)
COMPUSTAT_PRIMARY_COLUMN = "compustat_primary_ordinary_stocks"
COMPUSTAT_COLOR = "#1f77b4"
DATASTREAM_COLOR = "#ff7f0e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare yearly Compustat primary ordinary-share sample sizes with "
            "Datastream/LSEG June historical-return availability."
        )
    )
    parser.add_argument(
        "--comparison-input",
        type=Path,
        default=DEFAULT_COMPARISON_INPUT,
        help=(
            "Precomputed yearly comparison CSV. D2 never runs Datastream queries. "
            f"Default: {DEFAULT_COMPARISON_INPUT}"
        ),
    )
    parser.add_argument(
        "--compustat-source",
        type=Path,
        default=DEFAULT_COMPUSTAT_SOURCE,
        help=(
            "Existing Compustat June-window CSV used to recompute primary issues. "
            f"Default: {DEFAULT_COMPUSTAT_SOURCE}"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk when reading the Compustat CSV. Default: 250000",
    )
    parser.add_argument(
        "--active-stocks-input",
        type=Path,
        default=DEFAULT_ACTIVE_STOCKS_INPUT,
        help=(
            "Optional precomputed active screener CSV used only for statistics. "
            f"Default: {DEFAULT_ACTIVE_STOCKS_INPUT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for statistics and plot outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed comparison CSV: {path}. "
            "D2 does not run D1x or query Datastream; provide --comparison-input."
        )

    comparison = pd.read_csv(path)
    legacy_columns = {
        "datastream_historical_ric_resolved_stocks",
        "datastream_historical_ric_resolution_rate",
        "datastream_june_ric_available_stocks",
        "datastream_june_ric_availability_rate",
        "datastream_june_price_available_stocks",
        "datastream_june_price_availability_rate",
    }.intersection(comparison.columns)
    if legacy_columns:
        raise ValueError(
            "Comparison CSV was produced by an older D1x query and cannot be "
            "safely interpreted as the current June historical-return dataset. "
            f"Legacy columns found: {sorted(legacy_columns)}. Rebuild the CSV "
            "with D1x before running D2."
        )
    required_columns = {
        "year",
        "datastream_active_screener_stocks",
        "datastream_june_return_available_stocks",
        "datastream_june_return_availability_rate",
    }
    missing = required_columns.difference(comparison.columns)
    if missing:
        raise KeyError(
            f"Comparison CSV is missing required columns: {sorted(missing)}"
        )

    comparison["year"] = pd.to_numeric(comparison["year"], errors="coerce")
    comparison = comparison.loc[comparison["year"].notna()].copy()
    comparison["year"] = comparison["year"].astype(int)
    return comparison.sort_values("year").reset_index(drop=True)


def load_compustat_primary_sample_sizes(
    path: Path,
    *,
    chunksize: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Compustat June-window CSV: {path}")

    required_columns = {"gvkey", "iid", "prirow", "datadate", "exchg", "tpci"}
    header = pd.read_csv(path, nrows=0)
    available_columns = {str(column).lower().strip() for column in header.columns}
    missing = required_columns.difference(available_columns)
    if missing:
        raise KeyError(f"Compustat CSV is missing required columns: {sorted(missing)}")

    primary_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=list(required_columns),
        dtype=str,
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk.columns = chunk.columns.str.lower().str.strip()
        for column in ["gvkey", "iid", "prirow", "exchg", "tpci"]:
            chunk[column] = chunk[column].astype("string").str.strip().str.upper()

        dates = pd.to_datetime(chunk["datadate"], errors="coerce")
        primary_mask = (
            chunk["exchg"].eq("194")
            & chunk["tpci"].eq("0")
            & chunk["iid"].ne("")
            & chunk["prirow"].ne("")
            & chunk["iid"].eq(chunk["prirow"])
            & chunk["gvkey"].ne("")
            & dates.dt.month.eq(6)
            & dates.dt.day.between(20, 30)
        )
        if not primary_mask.any():
            continue

        primary_frames.append(
            pd.DataFrame(
                {
                    "year": dates.loc[primary_mask].dt.year.astype(int),
                    "gvkey": chunk.loc[primary_mask, "gvkey"],
                    "iid": chunk.loc[primary_mask, "iid"],
                }
            ).drop_duplicates()
        )

    if not primary_frames:
        raise ValueError(f"No primary Compustat ordinary shares were found in {path}")

    primary_securities = pd.concat(primary_frames, ignore_index=True).drop_duplicates(
        ["year", "gvkey", "iid"]
    )
    return (
        primary_securities.groupby("year")
        .size()
        .rename(COMPUSTAT_PRIMARY_COLUMN)
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )


def attach_compustat_primary_sample_sizes(
    comparison: pd.DataFrame,
    compustat_summary: pd.DataFrame,
) -> pd.DataFrame:
    required_years = set(comparison["year"].astype(int))
    available_years = set(compustat_summary["year"].astype(int))
    missing_years = sorted(required_years.difference(available_years))
    if missing_years:
        raise ValueError(
            f"Compustat primary-issue counts are missing years: {missing_years}"
        )

    old_compustat_columns = [
        column
        for column in comparison.columns
        if str(column).startswith("compustat_")
    ]
    out = comparison.drop(columns=old_compustat_columns, errors="ignore")
    return out.merge(compustat_summary, on="year", how="left", validate="one_to_one")


def load_active_stocks_count(path: Path, comparison: pd.DataFrame) -> int:
    if path.exists():
        return int(len(pd.read_csv(path)))

    return int(
        pd.to_numeric(
            comparison["datastream_active_screener_stocks"],
            errors="coerce",
        )
        .fillna(0)
        .max()
    )


def build_statistics(comparison: pd.DataFrame, active_stocks_count: int) -> dict[str, object]:
    compustat = comparison[COMPUSTAT_PRIMARY_COLUMN].astype(int)
    datastream = comparison["datastream_june_return_available_stocks"].fillna(0).astype(int)
    gap = compustat - datastream

    max_compustat_idx = compustat.idxmax()
    max_datastream_idx = datastream.idxmax()
    max_gap_idx = gap.idxmax()
    latest = comparison.iloc[-1]

    return {
        "years": {
            "min": int(comparison["year"].min()),
            "max": int(comparison["year"].max()),
            "count": int(comparison["year"].nunique()),
        },
        "active_datastream_screener_stocks": int(active_stocks_count),
        "total_compustat_primary_ordinary_stock_years": int(compustat.sum()),
        "total_datastream_june_return_available_stock_years": int(datastream.sum()),
        "total_compustat_primary_minus_datastream_gap": int(gap.sum()),
        "mean_datastream_june_return_availability_rate": float(
            comparison["datastream_june_return_availability_rate"].fillna(0).mean()
        ),
        "peak_compustat_primary_ordinary_year": {
            "year": int(comparison.loc[max_compustat_idx, "year"]),
            COMPUSTAT_PRIMARY_COLUMN: int(compustat.loc[max_compustat_idx]),
        },
        "peak_datastream_year": {
            "year": int(comparison.loc[max_datastream_idx, "year"]),
            "datastream_june_return_available_stocks": int(datastream.loc[max_datastream_idx]),
        },
        "largest_compustat_primary_minus_datastream_gap_year": {
            "year": int(comparison.loc[max_gap_idx, "year"]),
            "gap": int(gap.loc[max_gap_idx]),
        },
        "latest_year": {
            "year": int(latest["year"]),
            COMPUSTAT_PRIMARY_COLUMN: int(latest[COMPUSTAT_PRIMARY_COLUMN]),
            "datastream_june_return_available_stocks": int(
                latest["datastream_june_return_available_stocks"]
            ),
            "datastream_june_return_availability_rate": float(
                latest["datastream_june_return_availability_rate"]
            ),
        },
    }


def plot_summary_with_matplotlib(comparison: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ModuleNotFoundError:
        return False

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        comparison["year"],
        comparison[COMPUSTAT_PRIMARY_COLUMN],
        linewidth=2,
        label="Compustat Global",
        color=COMPUSTAT_COLOR,
    )
    ax.plot(
        comparison["year"],
        comparison["datastream_june_return_available_stocks"],
        linewidth=2,
        label="LSEG Datastream",
        color=DATASTREAM_COLOR,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique stocks")
    ax.set_xlim(1990, 2024)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1990, 2025, 5))
    ax.tick_params(axis="x", labelrotation=0)
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.margins(x=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def draw_text(
    draw: object,
    xy: tuple[int, int],
    text: str,
    fill: str,
    font: object,
    anchor: str | None = None,
) -> None:
    kwargs = {"fill": fill, "font": font}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(xy, text, **kwargs)


def plot_summary_with_pillow(comparison: pd.DataFrame, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 2160, 1080
    margin_left, margin_right = 150, 90
    margin_top, margin_bottom = 130, 160
    plot_left, plot_top = margin_left, margin_top
    plot_right, plot_bottom = width - margin_right, height - margin_bottom
    plot_width, plot_height = plot_right - plot_left, plot_bottom - plot_top

    image = Image.new("RGB", (width, height), "#f2f2f2")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    title_font = ImageFont.truetype(str(font_path), 38) if font_path.exists() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default()
    tick_font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    legend_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()

    years = comparison["year"].astype(int).tolist()
    compustat = comparison[COMPUSTAT_PRIMARY_COLUMN].astype(float).tolist()
    datastream = (
        comparison["datastream_june_return_available_stocks"]
        .fillna(0)
        .astype(float)
        .tolist()
    )
    max_value = max(compustat + datastream) if years else 1
    y_tick_interval = 500
    y_max = max(
        y_tick_interval,
        int(((max_value + y_tick_interval - 1) // y_tick_interval) * y_tick_interval),
    )

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill="#e5e5e5")
    for tick in range(0, y_max + 1, y_tick_interval):
        y = plot_bottom - int((tick / y_max) * plot_height)
        draw.line((plot_left, y, plot_right, y), fill="#ffffff", width=2)
        draw_text(draw, (plot_left - 18, y), str(tick), "#666666", tick_font, anchor="rm")

    def point(idx: int, value: float) -> tuple[int, int]:
        year = years[idx]
        x = plot_left + int((year - 1990) / (2024 - 1990) * plot_width)
        y = plot_bottom - int((value / y_max) * plot_height)
        return x, y

    for values, color in [(compustat, COMPUSTAT_COLOR), (datastream, DATASTREAM_COLOR)]:
        points = [point(idx, value) for idx, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=5)

    for year in range(1990, 2025, 5):
        x = plot_left + int((year - 1990) / (2024 - 1990) * plot_width)
        draw_text(
            draw,
            (x, plot_bottom + 18),
            str(year),
            "#666666",
            tick_font,
            anchor="ma",
        )

    draw_text(
        draw,
        (width // 2, 45),
        "Compustat vs Datastream Historical Sample sizes",
        "#222222",
        title_font,
        anchor="mm",
    )
    draw_text(draw, (width // 2, height - 55), "Year", "#555555", label_font, anchor="mm")

    rotated_label = Image.new("RGBA", (360, 60), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(rotated_label)
    label_draw.text((180, 30), "Unique stocks", fill="#555555", font=label_font, anchor="mm")
    rotated = rotated_label.rotate(90, expand=True)
    image.paste(rotated, (25, plot_top + plot_height // 2 - 180), rotated)

    legend_x, legend_y = plot_right - 710, plot_top + 28
    draw.rounded_rectangle(
        (legend_x - 20, legend_y - 20, legend_x + 680, legend_y + 80),
        radius=4,
        fill="#eeeeee",
        outline="#d4d4d4",
    )
    draw.line((legend_x, legend_y + 12, legend_x + 50, legend_y + 12), fill=COMPUSTAT_COLOR, width=5)
    draw_text(draw, (legend_x + 65, legend_y + 12), "Compustat Global", "#222222", legend_font, anchor="lm")
    draw.line((legend_x, legend_y + 54, legend_x + 50, legend_y + 54), fill=DATASTREAM_COLOR, width=5)
    draw_text(
        draw,
        (legend_x + 65, legend_y + 54),
        "LSEG Datastream",
        "#222222",
        legend_font,
        anchor="lm",
    )

    image.save(output_path)


def plot_summary(comparison: pd.DataFrame, output_path: Path) -> str:
    if plot_summary_with_matplotlib(comparison, output_path):
        return "matplotlib"
    plot_summary_with_pillow(comparison, output_path)
    return "pillow"


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_input = resolve_path(args.comparison_input)
    compustat_source = resolve_path(args.compustat_source)
    active_stocks_input = resolve_path(args.active_stocks_input)
    comparison = load_comparison(comparison_input)
    compustat_summary = load_compustat_primary_sample_sizes(
        compustat_source,
        chunksize=args.chunksize,
    )
    comparison = attach_compustat_primary_sample_sizes(comparison, compustat_summary)
    active_stocks_count = load_active_stocks_count(active_stocks_input, comparison)
    statistics = build_statistics(comparison, active_stocks_count)

    statistics_path = output_dir / "compustat_vs_datastream_statistics.json"
    plot_path = output_dir / "compustat_vs_datastream_historical_sample_sizes.png"

    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    plot_backend = plot_summary(comparison, plot_path)

    print(f"Loaded precomputed yearly comparison from: {comparison_input}")
    print(f"Recomputed Compustat primary ordinary-share counts from: {compustat_source}")
    if active_stocks_input.exists():
        print(f"Loaded active screener stock count from: {active_stocks_input}")
    else:
        print("Active screener stock count inferred from comparison CSV.")
    print(f"Wrote statistics to: {statistics_path}")
    print(f"Wrote plot to: {plot_path} ({plot_backend})")


if __name__ == "__main__":
    main()
