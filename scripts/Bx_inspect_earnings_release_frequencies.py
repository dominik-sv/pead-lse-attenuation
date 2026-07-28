from html import escape
from pathlib import Path
import math
import sys

import lseg.data as ld
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import EARNINGS_REQUEST_INSTRUMENT_LIMIT
from src.core.year_context import build_year_context
from src.pead.earnings_events import clean_earnings_events, fetch_earnings_events
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR, available_year_gbp_constituent_paths
from src.utils.io_utils import save_json

pd.set_option("future.no_silent_downcasting", True)

DATA_DIR = PROJECT_DATA_DIR
OUTPUT_DIR_NAME = "earnings_release_diagnostics"
PLOTS_DIR_NAME = "plots"

# Keep the inspection scope local to this script rather than depending on the
# pipeline config. This makes the diagnostics stage self-contained.
INSPECT_EARNINGS_RELEASE_FREQUENCIES = {
    "FY": "Annual earnings report",
    "FS": "Semi-annual earnings report",
    "FQ": "Quarterly earnings report",
    "FI": "Interim / other accepted earnings report",
}

ANNOUNCEMENT_TYPE_LABELS = {
    "annual": "Annual earnings report",
    "semi_annual": "Semi-annual earnings report",
    "quarterly": "Quarterly earnings report",
    "interim_other": "Interim / other accepted earnings report",
    "other": "Other / unknown earnings report",
}

ANNOUNCEMENT_TYPE_COLUMNS = {
    "annual": "Annual_Announcements",
    "semi_annual": "Semi_Annual_Announcements",
    "quarterly": "Quarterly_Announcements",
    "interim_other": "Interim_Other_Announcements",
    "other": "Other_Announcements",
}


def find_available_constituent_files(base_dir: Path) -> list[Path]:
    return available_year_gbp_constituent_paths(base_dir)


def load_sedol_constituent_universe(path: Path) -> pd.DataFrame:
    stock_universe = pd.read_csv(path)

    if "lseg_identifier" not in stock_universe.columns:
        raise KeyError(f"Missing required column 'lseg_identifier' in {path}.")

    out = stock_universe.copy()
    out["Instrument"] = out["lseg_identifier"].astype("string").str.strip()
    out = out[out["Instrument"].notna() & (out["Instrument"] != "")].copy()
    out = out.rename(columns={"conm": "Name", "sedol": "Ticker"})

    return out


def load_base_firm_years(base_dir: Path) -> pd.DataFrame:
    universe_files = find_available_constituent_files(base_dir)

    if not universe_files:
        raise FileNotFoundError(
            "No yearly constituent files found under "
            f"{base_dir.resolve()}."
        )

    firm_years = []
    keep_columns = [
        "Instrument",
        "Ticker",
        "Name",
        "Formation_Year",
        "lseg_identifier",
        "sedol",
        "gvkey",
        "iid",
    ]

    for path in universe_files:
        formation_year = int(path.parent.name)
        stock_universe = load_sedol_constituent_universe(path)
        stock_universe["Formation_Year"] = formation_year

        available_columns = [
            column for column in keep_columns if column in stock_universe.columns
        ]
        firm_years.append(stock_universe[available_columns].copy())

    return pd.concat(firm_years, ignore_index=True).drop_duplicates(
        subset=["Formation_Year", "Instrument"]
    )


def classify_announcement_type(row: pd.Series) -> str:
    requested_frequency = str(row.get("Report_Frequency", "")).strip().upper()

    if requested_frequency == "FY":
        return "annual"
    if requested_frequency == "FS":
        return "semi_annual"
    if requested_frequency == "FQ":
        return "quarterly"
    if requested_frequency == "FI":
        return "interim_other"
    return "other"


def standardize_announcement_labels(earnings_events: pd.DataFrame) -> pd.DataFrame:
    out = earnings_events.copy()

    if "Report_Frequency" not in out.columns:
        out["Report_Frequency"] = pd.NA

    out["Requested_Report_Frequency"] = out["Report_Frequency"].astype("string").str.strip()
    out["Announcement_Type"] = out.apply(classify_announcement_type, axis=1)
    out["Announcement_Type_Label"] = out["Announcement_Type"].map(
        ANNOUNCEMENT_TYPE_LABELS
    )
    out["Report_Type"] = out["Announcement_Type_Label"]

    return out


def fetch_release_events_for_available_years(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_files = find_available_constituent_files(base_dir)

    if not universe_files:
        raise FileNotFoundError(
            "No yearly constituent files found under "
            f"{base_dir.resolve()}."
        )

    all_events = []
    sample_size_rows = []
    failure_records = []
    failure_stats = {}

    for path in universe_files:
        formation_year = int(path.parent.name)
        year_context = build_year_context(formation_year, DATA_DIR)
        stock_universe = load_sedol_constituent_universe(path)

        print(
            "\n=== Fetching earnings release events for "
            f"formation year {formation_year} ==="
        )

        sample_size = {
            "Base sample firms": len(stock_universe),
        }
        raw_events = fetch_earnings_events(
            stock_universe=stock_universe,
            year=formation_year,
            frequencies=INSPECT_EARNINGS_RELEASE_FREQUENCIES,
            currency="USD",
            failure_stats=failure_stats,
            failure_records=failure_records,
        )
        sample_size["Raw earnings rows returned"] = len(raw_events)

        events = clean_earnings_events(raw_events, formation_year, sample_size)
        events["Formation_Year"] = formation_year
        events["Source_File"] = str(path)

        year_context.cache_dir.mkdir(parents=True, exist_ok=True)
        events.to_csv(year_context.earnings_release_events_path, index=False)

        sample_size_rows.extend(
            {
                "Formation_Year": formation_year,
                "Step": step,
                "Sample_Size": count,
            }
            for step, count in sample_size.items()
        )
        all_events.append(events)

    if not all_events:
        return pd.DataFrame(), pd.DataFrame(sample_size_rows), pd.DataFrame(failure_records), pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in failure_stats.items()]
        )

    earnings_events = pd.concat(all_events, ignore_index=True)
    earnings_events = standardize_announcement_labels(earnings_events)

    return (
        earnings_events,
        pd.DataFrame(sample_size_rows),
        pd.DataFrame(failure_records),
        pd.DataFrame([{"Metric": key, "Value": value} for key, value in failure_stats.items()]),
    )


def build_firm_year_release_counts(
    earnings_events: pd.DataFrame,
    base_firm_years: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        earnings_events.groupby(["Formation_Year", "Instrument", "Announcement_Type"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=list(ANNOUNCEMENT_TYPE_COLUMNS), fill_value=0)
    counts = counts.rename(columns=ANNOUNCEMENT_TYPE_COLUMNS).reset_index()

    base_columns = [
        column
        for column in ["Formation_Year", "Instrument", "Ticker", "Name"]
        if column in base_firm_years.columns
    ]
    counts = base_firm_years[base_columns].merge(
        counts,
        on=["Formation_Year", "Instrument"],
        how="left",
    )

    for column in ANNOUNCEMENT_TYPE_COLUMNS.values():
        counts[column] = counts[column].fillna(0).astype(int)

    counts["Total_Announcements"] = counts[list(ANNOUNCEMENT_TYPE_COLUMNS.values())].sum(
        axis=1
    )
    counts["Release_Pattern"] = counts.apply(label_release_pattern, axis=1)

    return counts.sort_values(["Formation_Year", "Instrument"]).reset_index(drop=True)


def label_release_pattern(row: pd.Series) -> str:
    present_labels = []
    for short_label, count_column in ANNOUNCEMENT_TYPE_COLUMNS.items():
        if int(row[count_column]) > 0:
            label = ANNOUNCEMENT_TYPE_LABELS[short_label]
            present_labels.append(label.replace(" earnings report", ""))

    total_announcements = int(row["Total_Announcements"])
    distinct_types = len(present_labels)

    if total_announcements == 0:
        return "No announcements"
    if total_announcements == 1 and distinct_types == 1:
        return f"{present_labels[0]} only"
    if total_announcements == distinct_types:
        return " + ".join(present_labels)
    return f"{' + '.join(present_labels)} (multiple)"


def build_firm_release_summary(firm_year_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        firm_year_counts.groupby("Instrument")
        .agg(
            Years_Observed=("Formation_Year", "nunique"),
            Total_Announcements=("Total_Announcements", "sum"),
            Mean_Announcements_Per_Year=("Total_Announcements", "mean"),
            Median_Announcements_Per_Year=("Total_Announcements", "median"),
            Mean_Annual_Announcements=("Annual_Announcements", "mean"),
            Mean_Semi_Annual_Announcements=("Semi_Annual_Announcements", "mean"),
            Mean_Quarterly_Announcements=("Quarterly_Announcements", "mean"),
            Mean_Interim_Other_Announcements=("Interim_Other_Announcements", "mean"),
            Mean_Other_Announcements=("Other_Announcements", "mean"),
            Most_Common_Release_Pattern=(
                "Release_Pattern",
                lambda values: values.mode().iloc[0] if not values.mode().empty else pd.NA,
            ),
        )
        .reset_index()
        .sort_values(
            ["Mean_Announcements_Per_Year", "Total_Announcements"],
            ascending=False,
        )
    )

    return summary


def build_yearly_pattern_distribution(
    firm_year_counts: pd.DataFrame,
) -> pd.DataFrame:
    yearly_distribution = (
        firm_year_counts.groupby(["Formation_Year", "Release_Pattern"])
        .size()
        .reset_index(name="Firm_Years")
    )

    yearly_totals = yearly_distribution.groupby("Formation_Year")["Firm_Years"].transform(
        "sum"
    )
    yearly_distribution["Share"] = yearly_distribution["Firm_Years"] / yearly_totals

    return yearly_distribution.sort_values(["Formation_Year", "Release_Pattern"])


def build_report_type_distribution(earnings_events: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        earnings_events["Announcement_Type_Label"]
        .value_counts(dropna=False)
        .rename_axis("Announcement_Type")
        .reset_index(name="Announcements")
    )
    distribution["Share"] = (
        distribution["Announcements"] / distribution["Announcements"].sum()
    )

    return distribution


def build_yearly_report_type_distribution(
    earnings_events: pd.DataFrame,
) -> pd.DataFrame:
    yearly_distribution = (
        earnings_events.groupby(["Formation_Year", "Announcement_Type_Label"], dropna=False)
        .size()
        .reset_index(name="Announcements")
    )

    yearly_totals = yearly_distribution.groupby("Formation_Year")[
        "Announcements"
    ].transform("sum")
    yearly_distribution["Share"] = yearly_distribution["Announcements"] / yearly_totals

    return yearly_distribution.sort_values(["Formation_Year", "Announcement_Type_Label"])


def build_distribution_table(
    firm_year_counts: pd.DataFrame,
    column: str,
    label: str,
) -> pd.DataFrame:
    distribution = (
        firm_year_counts[column]
        .value_counts()
        .rename_axis(label)
        .reset_index(name="Firm_Years")
        .sort_values(label)
    )
    distribution["Share"] = distribution["Firm_Years"] / distribution["Firm_Years"].sum()

    return distribution


def build_pattern_distribution(firm_year_counts: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        firm_year_counts["Release_Pattern"]
        .value_counts()
        .rename_axis("Release_Pattern")
        .reset_index(name="Firm_Years")
    )
    distribution["Share"] = distribution["Firm_Years"] / distribution["Firm_Years"].sum()

    return distribution


def build_summary_stats(
    earnings_events: pd.DataFrame,
    firm_year_counts: pd.DataFrame,
) -> dict:
    formation_years = sorted(firm_year_counts["Formation_Year"].unique().tolist())
    summary = {
        "formation_years": formation_years,
        "formation_year_count": len(formation_years),
        "announcement_rows": int(len(earnings_events)),
        "firm_year_rows": int(len(firm_year_counts)),
        "unique_firms": int(firm_year_counts["Instrument"].nunique()),
        "mean_total_announcements_per_firm_year": round(
            float(firm_year_counts["Total_Announcements"].mean()), 4
        ),
        "median_total_announcements_per_firm_year": round(
            float(firm_year_counts["Total_Announcements"].median()), 4
        ),
        "share_of_firm_years_with_no_announcements": round(
            float((firm_year_counts["Total_Announcements"] == 0).mean()), 4
        ),
    }

    for short_label, count_column in ANNOUNCEMENT_TYPE_COLUMNS.items():
        label = ANNOUNCEMENT_TYPE_LABELS[short_label]
        key_prefix = short_label
        summary[f"{key_prefix}_announcement_rows"] = int(
            earnings_events["Announcement_Type"].eq(short_label).sum()
        )
        summary[f"{key_prefix}_mean_announcements_per_firm_year"] = round(
            float(firm_year_counts[count_column].mean()), 4
        )
        summary[f"{key_prefix}_median_announcements_per_firm_year"] = round(
            float(firm_year_counts[count_column].median()), 4
        )
        summary[f"{key_prefix}_share_of_firm_years_with_at_least_one"] = round(
            float((firm_year_counts[count_column] > 0).mean()), 4
        )
        summary[f"{key_prefix}_label"] = label

    return summary


def build_request_summary(base_firm_years: pd.DataFrame) -> dict:
    per_year_rows = []
    total_frequency_requests = 0

    instrument_counts = (
        base_firm_years.groupby("Formation_Year")["Instrument"].nunique().sort_index()
    )
    for formation_year, instrument_count in instrument_counts.items():
        requests_per_frequency = math.ceil(
            int(instrument_count) / max(EARNINGS_REQUEST_INSTRUMENT_LIMIT, 1)
        )
        total_year_requests = requests_per_frequency * len(
            INSPECT_EARNINGS_RELEASE_FREQUENCIES
        )
        total_frequency_requests += requests_per_frequency
        per_year_rows.append(
            {
                "Formation_Year": int(formation_year),
                "Instrument_Count": int(instrument_count),
                "Requests_Per_Frequency": int(requests_per_frequency),
                "Frequencies_Requested": sorted(
                    INSPECT_EARNINGS_RELEASE_FREQUENCIES.keys()
                ),
                "Total_Requests": int(total_year_requests),
            }
        )

    return {
        "instrument_batch_limit": int(EARNINGS_REQUEST_INSTRUMENT_LIMIT),
        "configured_frequencies": INSPECT_EARNINGS_RELEASE_FREQUENCIES,
        "years_covered": int(len(per_year_rows)),
        "requests_per_frequency_total": int(total_frequency_requests),
        "total_requests_before_retries": int(
            total_frequency_requests * len(INSPECT_EARNINGS_RELEASE_FREQUENCIES)
        ),
        "per_year": per_year_rows,
    }


def save_bar_chart_svg(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    if table.empty:
        return

    width = 1200
    height = 800
    left_margin = 90
    right_margin = 40
    top_margin = 80
    bottom_margin = 220
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    values = pd.to_numeric(table[y_column], errors="coerce").fillna(0).tolist()
    labels = [str(value) for value in table[x_column].tolist()]
    max_value = max(values) if values else 0
    max_value = max(max_value, 1)

    bar_count = len(table)
    gap = 12
    available_width = plot_width - gap * max(bar_count - 1, 0)
    bar_width = max(18, available_width / max(bar_count, 1))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{left_margin + plot_width}" y2="{top_margin + plot_height}" stroke="#222" stroke-width="2" />',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#222" stroke-width="2" />',
    ]

    for tick_index in range(6):
        tick_value = max_value * tick_index / 5
        y = top_margin + plot_height - (tick_value / max_value) * plot_height
        svg_lines.append(
            f'<line x1="{left_margin - 6}" y1="{y}" x2="{left_margin + plot_width}" y2="{y}" stroke="#dddddd" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{left_margin - 12}" y="{y + 5}" text-anchor="end" font-family="Arial" font-size="14">{escape(f"{tick_value:.0f}")}</text>'
        )

    current_x = left_margin
    for label, value in zip(labels, values):
        bar_height = (value / max_value) * plot_height
        y = top_margin + plot_height - bar_height
        center_x = current_x + bar_width / 2
        svg_lines.append(
            f'<rect x="{current_x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#1f77b4" />'
        )
        svg_lines.append(
            f'<text x="{center_x}" y="{y - 10}" text-anchor="middle" font-family="Arial" font-size="13">{escape(f"{value:.0f}")}</text>'
        )
        svg_lines.append(
            f'<text x="{center_x}" y="{top_margin + plot_height + 18}" text-anchor="end" transform="rotate(-35 {center_x} {top_margin + plot_height + 18})" font-family="Arial" font-size="13">{escape(label)}</text>'
        )
        current_x += bar_width + gap

    svg_lines.append(
        f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="16">{escape(x_column.replace("_", " "))}</text>'
    )
    svg_lines.append(
        f'<text x="28" y="{top_margin + plot_height / 2}" text-anchor="middle" transform="rotate(-90 28 {top_margin + plot_height / 2})" font-family="Arial" font-size="16">{escape(y_column.replace("_", " "))}</text>'
    )
    svg_lines.append("</svg>")

    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def save_outputs(
    output_dir: Path,
    plots_dir: Path,
    earnings_events: pd.DataFrame,
    firm_year_counts: pd.DataFrame,
    firm_summary: pd.DataFrame,
    sample_sizes: pd.DataFrame,
    failure_records: pd.DataFrame,
    failure_stats: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    summary_stats: dict,
    request_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    earnings_events.to_csv(output_dir / "all_release_events.csv", index=False)
    firm_year_counts.to_csv(output_dir / "firm_year_release_counts.csv", index=False)
    firm_summary.to_csv(output_dir / "firm_release_summary.csv", index=False)
    sample_sizes.to_csv(output_dir / "release_event_sample_sizes.csv", index=False)
    failure_records.to_csv(output_dir / "release_event_failure_records.csv", index=False)
    failure_stats.to_csv(output_dir / "release_event_failure_stats.csv", index=False)

    for name, table in distributions.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    save_json(summary_stats, output_dir / "summary_stats.json")
    save_json(summary_stats, plots_dir / "summary_stats.json")
    save_json(request_summary, output_dir / "request_summary.json")
    save_json(request_summary, plots_dir / "request_summary.json")
    save_json(
        {
            "failure_records_rows": len(failure_records),
            "failure_stats_rows": len(failure_stats),
        },
        output_dir / "failure_summary.json",
    )


def save_plots(
    plots_dir: Path,
    distributions: dict[str, pd.DataFrame],
) -> None:
    plot_specs = [
        (
            "report_type_distribution",
            "Announcement_Type",
            "Announcements",
            "Distribution of Earnings Announcements by Announcement Type",
            "report_type_distribution.svg",
        ),
        (
            "total_announcements_distribution",
            "Total_Announcements",
            "Firm_Years",
            "Distribution of Total Earnings Announcements per Firm-Year",
            "total_announcements_distribution.svg",
        ),
        (
            "annual_announcements_distribution",
            "Annual_Announcements",
            "Firm_Years",
            "Distribution of Annual Earnings Announcements per Firm-Year",
            "annual_announcements_distribution.svg",
        ),
        (
            "semi_annual_announcements_distribution",
            "Semi_Annual_Announcements",
            "Firm_Years",
            "Distribution of Semi-Annual Earnings Announcements per Firm-Year",
            "semi_annual_announcements_distribution.svg",
        ),
        (
            "quarterly_announcements_distribution",
            "Quarterly_Announcements",
            "Firm_Years",
            "Distribution of Quarterly Earnings Announcements per Firm-Year",
            "quarterly_announcements_distribution.svg",
        ),
        (
            "interim_other_announcements_distribution",
            "Interim_Other_Announcements",
            "Firm_Years",
            "Distribution of Interim / Other Accepted Earnings Announcements per Firm-Year",
            "interim_other_announcements_distribution.svg",
        ),
        (
            "other_announcements_distribution",
            "Other_Announcements",
            "Firm_Years",
            "Distribution of Other / Unknown Earnings Announcements per Firm-Year",
            "other_announcements_distribution.svg",
        ),
        (
            "release_pattern_distribution",
            "Release_Pattern",
            "Firm_Years",
            "Distribution of Release Patterns per Firm-Year",
            "release_pattern_distribution.svg",
        ),
    ]

    for distribution_name, x_column, y_column, title, file_name in plot_specs:
        table = distributions.get(distribution_name)
        if table is None or table.empty:
            continue
        save_bar_chart_svg(
            table=table,
            x_column=x_column,
            y_column=y_column,
            title=title,
            output_path=plots_dir / file_name,
        )


def print_distribution(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(table.to_string(index=False))


def main() -> None:
    output_dir = DATA_DIR / OUTPUT_DIR_NAME
    plots_dir = output_dir / PLOTS_DIR_NAME

    base_firm_years = load_base_firm_years(DATA_DIR)
    request_summary = build_request_summary(base_firm_years)

    ld.open_session()
    earnings_events, sample_sizes, failure_records, failure_stats = fetch_release_events_for_available_years(
        DATA_DIR
    )

    firm_year_counts = build_firm_year_release_counts(
        earnings_events=earnings_events,
        base_firm_years=base_firm_years,
    )
    firm_summary = build_firm_release_summary(firm_year_counts)

    distributions = {
        "report_type_distribution": build_report_type_distribution(earnings_events),
        "yearly_report_type_distribution": build_yearly_report_type_distribution(
            earnings_events
        ),
        "total_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Total_Announcements",
            "Total_Announcements",
        ),
        "annual_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Annual_Announcements",
            "Annual_Announcements",
        ),
        "semi_annual_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Semi_Annual_Announcements",
            "Semi_Annual_Announcements",
        ),
        "quarterly_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Quarterly_Announcements",
            "Quarterly_Announcements",
        ),
        "interim_other_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Interim_Other_Announcements",
            "Interim_Other_Announcements",
        ),
        "other_announcements_distribution": build_distribution_table(
            firm_year_counts,
            "Other_Announcements",
            "Other_Announcements",
        ),
        "release_pattern_distribution": build_pattern_distribution(firm_year_counts),
        "yearly_release_pattern_distribution": build_yearly_pattern_distribution(
            firm_year_counts
        ),
    }

    summary_stats = build_summary_stats(
        earnings_events=earnings_events,
        firm_year_counts=firm_year_counts,
    )

    save_outputs(
        output_dir=output_dir,
        plots_dir=plots_dir,
        earnings_events=earnings_events,
        firm_year_counts=firm_year_counts,
        firm_summary=firm_summary,
        sample_sizes=sample_sizes,
        failure_records=failure_records,
        failure_stats=failure_stats,
        distributions=distributions,
        summary_stats=summary_stats,
        request_summary=request_summary,
    )
    save_plots(plots_dir, distributions)

    save_json(
        {
            "source": str(DATA_DIR),
            "configured_earnings_frequencies": INSPECT_EARNINGS_RELEASE_FREQUENCIES,
            "announcement_type_source": (
                "Requested frequency bucket used for each pull; no native returned "
                "announcement-type field is currently used."
            ),
            "formation_years": sorted(earnings_events["Formation_Year"].unique().tolist()),
            "announcement_rows": len(earnings_events),
            "firm_year_rows": len(firm_year_counts),
            "plots_directory": str(plots_dir),
            "total_requests_before_retries": request_summary[
                "total_requests_before_retries"
            ],
        },
        output_dir / "diagnostics_summary.json",
    )

    print(f"Fetched {len(earnings_events):,} earnings release rows.")
    print(
        "Covered "
        f"{firm_year_counts['Formation_Year'].nunique():,} formation years and "
        f"{firm_year_counts['Instrument'].nunique():,} unique firms."
    )
    print(f"Firm-year observations: {len(firm_year_counts):,}")
    print(f"Estimated LSEG requests before retries: {request_summary['total_requests_before_retries']:,}")
    print(f"Saved diagnostics to: {output_dir.resolve()}")
    print(f"Saved plots and copied summary stats to: {plots_dir.resolve()}")

    print_distribution(
        "Distribution of earnings announcements by announcement type",
        distributions["report_type_distribution"],
    )
    print_distribution(
        "Distribution of total earnings announcements per firm-year",
        distributions["total_announcements_distribution"],
    )
    print_distribution(
        "Distribution of annual earnings announcements per firm-year",
        distributions["annual_announcements_distribution"],
    )
    print_distribution(
        "Distribution of semi-annual earnings announcements per firm-year",
        distributions["semi_annual_announcements_distribution"],
    )
    print_distribution(
        "Distribution of quarterly earnings announcements per firm-year",
        distributions["quarterly_announcements_distribution"],
    )
    print_distribution(
        "Distribution of interim / other accepted earnings announcements per firm-year",
        distributions["interim_other_announcements_distribution"],
    )
    print_distribution(
        "Distribution of other / unknown earnings announcements per firm-year",
        distributions["other_announcements_distribution"],
    )
    print_distribution(
        "Distribution of release patterns per firm-year",
        distributions["release_pattern_distribution"],
    )

    print("\nTop firms by average announcement frequency")
    print("-------------------------------------------")
    print(firm_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()


