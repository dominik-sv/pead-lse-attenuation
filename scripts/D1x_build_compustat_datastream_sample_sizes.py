from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - local environments can still read yearly sample sizes.
    ld = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import ORDINARY_SHARE_TYPES


OUTPUT_SUBDIR = "D_compustat_datastream_sample_size_comparison"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / OUTPUT_SUBDIR
DEFAULT_SAMPLE_SIZE_DIR = PROJECT_ROOT / "data sample size" / "yearly"
NAME_SCREEN_METRIC = "Historical ordinary/common share candidates"
NAME_SCREEN_COLUMN = "compustat_name_screened_ordinary_stocks"
XLON_MARKET_ID_CODE = "XLON"
JUNE_WINDOW_START_DAY = 20
JUNE_WINDOW_END_DAY = 30
DEFAULT_CURRENCY = "GBP"
DEFAULT_BATCH_SIZE = 250
_LSEG_SESSION_OPENED = False

warnings.filterwarnings(
    "ignore",
    message=r"Downcasting behavior in `replace` is deprecated.*",
    category=FutureWarning,
    module=r"lseg\.data\._tools\._dataframe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine yearly Compustat name-screened ordinary-share sample sizes "
            "with Datastream/LSEG June historical-return availability for D2."
        )
    )
    parser.add_argument(
        "--sample-size-dir",
        type=Path,
        default=DEFAULT_SAMPLE_SIZE_DIR,
        help=(
            f"Directory containing yearly/<year>/sample_size.json data with "
            f"'{NAME_SCREEN_METRIC}'. Default: {DEFAULT_SAMPLE_SIZE_DIR}"
        ),
    )
    parser.add_argument(
        "--currency",
        default=DEFAULT_CURRENCY,
        help=f"LSEG query currency. Default: {DEFAULT_CURRENCY}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"LSEG instruments per historical return query batch. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for precomputed D2 inputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def get_series(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def resolve_sample_size_dir(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing yearly sample-size directory: {resolved}")
    return resolved


def load_name_screened_yearly_sample_sizes(sample_size_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for sample_path in sample_size_dir.glob("*/sample_size.json"):
        if not sample_path.parent.name.isdigit():
            continue
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        screened_count = payload.get(NAME_SCREEN_METRIC) if isinstance(payload, dict) else None
        if not isinstance(screened_count, (int, float)) or isinstance(screened_count, bool):
            raise KeyError(
                f"Missing numeric '{NAME_SCREEN_METRIC}' in {sample_path}"
            )
        rows.append(
            {
                "year": int(sample_path.parent.name),
                NAME_SCREEN_COLUMN: int(screened_count),
            }
        )

    if not rows:
        raise ValueError(
            f"No yearly sample_size.json files containing '{NAME_SCREEN_METRIC}' "
            f"were found below {sample_size_dir}"
        )
    summary = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if summary["year"].duplicated().any():
        duplicates = sorted(summary.loc[summary["year"].duplicated(), "year"].tolist())
        raise ValueError(f"Duplicate yearly sample-size records found: {duplicates}")
    return summary


def require_lseg() -> None:
    if ld is None:
        raise ModuleNotFoundError(
            "lseg.data is required to run the Datastream/LSEG screener and "
            "historical June return-availability check."
        )


def ensure_lseg_session() -> None:
    require_lseg()
    global _LSEG_SESSION_OPENED
    if _LSEG_SESSION_OPENED:
        return
    ld.open_session()
    _LSEG_SESSION_OPENED = True


def build_active_xlon_screener_expr(exchange: str = XLON_MARKET_ID_CODE) -> str:
    return f"""SCREEN(
        U(IN(Equity(active,public,primary))/*UNV:Public*/),
        IN(TR.ExchangeMarketIdCode,"{exchange}")
    )"""


def fetch_active_datastream_stocks(
    *,
    exchange: str = XLON_MARKET_ID_CODE,
    currency: str = DEFAULT_CURRENCY,
) -> pd.DataFrame:
    ensure_lseg_session()
    frame = ld.get_data(
        universe=[build_active_xlon_screener_expr(exchange)],
        fields=[
            "TR.RIC",
            "TR.CommonName",
            "TR.ExchangeTicker",
            "TR.ExchangeMarketIdCode",
            "TR.ISIN",
            "TR.SEDOL",
            "TR.InstrumentType",
        ],
        parameters={"Curn": currency},
        header_type=ld.HeaderType.NAME,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    return prepare_datastream_screener_frame(frame)


def prepare_datastream_screener_frame(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "INSTRUMENT": "screen_instrument",
        "TR.RIC": "current_ric",
        "TR.COMMONNAME": "name",
        "TR.EXCHANGETICKER": "ticker",
        "TR.EXCHANGEMARKETIDCODE": "exchange_code",
        "TR.ISIN": "isin",
        "TR.SEDOL": "sedol",
        "TR.INSTRUMENTTYPE": "instrument_type",
    }
    out = frame.rename(
        columns=lambda column: column_map.get(str(column).upper(), column)
    ).copy()

    for column in [
        "screen_instrument",
        "current_ric",
        "name",
        "ticker",
        "exchange_code",
        "isin",
        "sedol",
        "instrument_type",
    ]:
        if column in out.columns:
            out[column] = normalize_text(out[column])

    if "instrument_type" not in out.columns:
        raise KeyError("LSEG screener response is missing TR.InstrumentType.")

    out = out.loc[normalize_text(get_series(out, "screen_instrument", "")).ne("")].copy()
    out = out.loc[out["instrument_type"].isin(ORDINARY_SHARE_TYPES)].copy()
    dedupe_columns = [
        column
        for column in ["screen_instrument", "current_ric", "isin", "sedol"]
        if column in out.columns
    ]
    if dedupe_columns:
        out = out.drop_duplicates(subset=dedupe_columns, keep="first")
    return out.reset_index(drop=True)


def query_datastream_june_price_availability_for_year(
    instruments: list[str],
    *,
    year: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    currency: str = DEFAULT_CURRENCY,
) -> pd.DataFrame:
    ensure_lseg_session()
    june_dates = [
        f"{int(year)}-06-{day:02d}"
        for day in range(JUNE_WINDOW_START_DAY, JUNE_WINDOW_END_DAY + 1)
    ]
    frames: list[pd.DataFrame] = []

    for start in range(0, len(instruments), batch_size):
        batch = instruments[start:start + batch_size]
        frames.extend(
            fetch_datastream_price_batches_with_fallback(
                batch,
                june_dates=june_dates,
                currency=currency,
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "year",
                "screen_instrument",
                "price_date",
                "historical_price",
                "historical_price_available",
            ]
        )

    out = pd.concat(frames, ignore_index=True).rename(columns={"Instrument": "screen_instrument"})
    out["year"] = int(year)

    if "screen_instrument" in out.columns:
        out["screen_instrument"] = normalize_text(out["screen_instrument"])

    price_columns_by_date = {
        date_value: historical_price_column(date_value)
        for date_value in june_dates
        if historical_price_column(date_value) in out.columns
    }
    price_values = out.loc[:, list(price_columns_by_date.values())].apply(
        pd.to_numeric,
        errors="coerce",
    )
    out["historical_price_available"] = price_values.notna().any(axis=1)
    out["price_date"] = pd.NA
    out["historical_price"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    for date_value, column in price_columns_by_date.items():
        unresolved = out["historical_price"].isna()
        available = pd.to_numeric(out[column], errors="coerce")
        take = unresolved & available.notna()
        out.loc[take, "price_date"] = date_value
        out.loc[take, "historical_price"] = available.loc[take]

    out = (
        out.sort_values(
            ["year", "screen_instrument", "historical_price_available"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["year", "screen_instrument"], keep="first")
        .reset_index(drop=True)
    )
    keep_columns = [
        "year",
        "screen_instrument",
        "price_date",
        "historical_price",
        "historical_price_available",
    ]
    return out.loc[:, keep_columns]


def fetch_datastream_price_batches_with_fallback(
    instruments: list[str],
    *,
    june_dates: list[str],
    currency: str,
) -> list[pd.DataFrame]:
    if not instruments:
        return []

    try:
        frame = ld.get_data(
            universe=instruments,
            fields=[f"TR.PriceClose(SDate={date_value})" for date_value in june_dates],
            parameters={"Curn": currency},
            header_type=ld.HeaderType.NAME,
        )
        if frame is None or frame.empty:
            return []
    except Exception:
        if len(instruments) == 1:
            return []

        midpoint = len(instruments) // 2
        return (
            fetch_datastream_price_batches_with_fallback(
                instruments[:midpoint],
                june_dates=june_dates,
                currency=currency,
            )
            + fetch_datastream_price_batches_with_fallback(
                instruments[midpoint:],
                june_dates=june_dates,
                currency=currency,
            )
        )

    return [canonicalize_historical_price_response(frame, june_dates)]


def historical_price_column(date_value: str) -> str:
    return f"price_{date_value.replace('-', '_')}"


def find_historical_price_columns(frame: pd.DataFrame, june_dates: list[str]) -> dict[str, str]:
    columns_by_normalized_name = {
        str(column).lower().replace(" ", ""): str(column)
        for column in frame.columns
    }
    found: dict[str, str] = {}
    for date_value in june_dates:
        expected = f"TR.PriceClose(SDate={date_value})"
        normalized_expected = expected.lower().replace(" ", "")
        if normalized_expected in columns_by_normalized_name:
            found[date_value] = columns_by_normalized_name[normalized_expected]
            continue

        for column in frame.columns:
            normalized_column = str(column).lower().replace(" ", "")
            if normalized_column.startswith("tr.priceclose(") and f"sdate={date_value.lower()}" in normalized_column:
                found[date_value] = str(column)
                break
    return found


def canonicalize_historical_price_response(
    frame: pd.DataFrame,
    june_dates: list[str],
) -> pd.DataFrame:
    instrument_positions = [
        index
        for index, column in enumerate(frame.columns)
        if str(column).lower().replace(" ", "") == "instrument"
    ]
    if not instrument_positions:
        raise KeyError("LSEG price response is missing its Instrument column.")

    found = find_historical_price_columns(frame, june_dates)
    result = pd.DataFrame({"Instrument": frame.iloc[:, instrument_positions[0]]})
    if len(found) == len(june_dates):
        for date_value in june_dates:
            result[historical_price_column(date_value)] = frame[found[date_value]]
        return result

    value_positions = [
        index for index in range(len(frame.columns)) if index not in instrument_positions
    ]
    if len(value_positions) != len(june_dates):
        raise KeyError(
            "Could not map LSEG price response columns to the 11 requested dates. "
            f"Returned columns: {list(frame.columns)}"
        )
    for date_value, position in zip(june_dates, value_positions):
        result[historical_price_column(date_value)] = frame.iloc[:, position]
    return result


def build_datastream_june_price_availability_summary(
    years: list[int],
    *,
    active_stocks: pd.DataFrame | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    currency: str = DEFAULT_CURRENCY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_lseg_session()
    if active_stocks is None:
        active_stocks = fetch_active_datastream_stocks(currency=currency)

    instruments = (
        normalize_text(get_series(active_stocks, "screen_instrument", ""))
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int | float]] = []

    for year in tqdm(
        sorted(int(year) for year in years),
        desc="Querying Datastream June historical prices by year",
        unit="year",
    ):
        year_detail = query_datastream_june_price_availability_for_year(
            instruments,
            year=year,
            batch_size=batch_size,
            currency=currency,
        )
        detail_frames.append(year_detail)

        queried_count = len(instruments)
        available_count = (
            int(year_detail["historical_price_available"].sum())
            if not year_detail.empty
            else 0
        )
        returned_rows = int(len(year_detail))
        summary_rows.append(
            {
                "year": year,
                "datastream_active_screener_stocks": queried_count,
                "datastream_returned_rows": returned_rows,
                "datastream_june_price_available_stocks": available_count,
                "datastream_june_price_availability_rate": (
                    available_count / queried_count if queried_count else 0.0
                ),
            }
        )

    details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    return summary, details, active_stocks


def find_return_response_column(
    columns: pd.Index,
    *,
    kind: str,
    excluded: set[object] | None = None,
) -> object:
    excluded = excluded or set()
    for column in columns:
        if column in excluded:
            continue
        normalized = str(column).upper().replace(" ", "").replace("_", "")
        if kind == "instrument" and normalized == "INSTRUMENT":
            return column
        if kind == "date" and (
            normalized == "DATE"
            or ("TOTALRETURN1D" in normalized and "DATE" in normalized)
        ):
            return column
        if kind == "return" and "TOTALRETURN" in normalized and "DATE" not in normalized:
            return column
    raise KeyError(
        f"LSEG return response is missing its {kind} column. "
        f"Returned columns: {list(columns)}"
    )


def canonicalize_historical_return_response(frame: pd.DataFrame) -> pd.DataFrame:
    instrument_column = find_return_response_column(frame.columns, kind="instrument")
    date_column = find_return_response_column(
        frame.columns,
        kind="date",
        excluded={instrument_column},
    )
    return_column = find_return_response_column(
        frame.columns,
        kind="return",
        excluded={instrument_column, date_column},
    )
    out = frame.loc[:, [instrument_column, date_column, return_column]].copy()
    out.columns = ["Instrument", "return_date", "historical_return"]
    out["Instrument"] = normalize_text(out["Instrument"])
    out["return_date"] = pd.to_datetime(
        out["return_date"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    out["historical_return"] = pd.to_numeric(
        out["historical_return"], errors="coerce"
    )
    return out.loc[out["Instrument"].ne("") & out["return_date"].notna()].reset_index(
        drop=True
    )


def fetch_datastream_return_batches_with_fallback(
    instruments: list[str],
    *,
    start_date: str,
    end_date: str,
) -> list[pd.DataFrame]:
    if not instruments:
        return []
    try:
        frame = ld.get_data(
            universe=instruments,
            fields=["TR.TotalReturn1D.date", "TR.TotalReturn1D"],
            parameters={"SDate": start_date, "EDate": end_date, "Frq": "D"},
            header_type=ld.HeaderType.NAME,
        )
        if frame is None or frame.empty:
            return []
    except Exception:
        if len(instruments) == 1:
            return []
        midpoint = len(instruments) // 2
        return (
            fetch_datastream_return_batches_with_fallback(
                instruments[:midpoint],
                start_date=start_date,
                end_date=end_date,
            )
            + fetch_datastream_return_batches_with_fallback(
                instruments[midpoint:],
                start_date=start_date,
                end_date=end_date,
            )
        )
    return [canonicalize_historical_return_response(frame)]


def query_datastream_june_return_availability_for_year(
    instruments: list[str],
    *,
    year: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    ensure_lseg_session()
    start_date = pd.Timestamp(year=int(year), month=6, day=JUNE_WINDOW_START_DAY)
    end_date = pd.Timestamp(year=int(year), month=6, day=JUNE_WINDOW_END_DAY)
    frames: list[pd.DataFrame] = []
    for start in range(0, len(instruments), batch_size):
        frames.extend(
            fetch_datastream_return_batches_with_fallback(
                instruments[start:start + batch_size],
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
        )

    base = pd.DataFrame({"screen_instrument": instruments})
    if frames:
        observations = pd.concat(frames, ignore_index=True).rename(
            columns={"Instrument": "screen_instrument"}
        )
        observations["screen_instrument"] = normalize_text(
            observations["screen_instrument"]
        )
        valid = observations.loc[
            observations["return_date"].between(start_date, end_date)
            & observations["historical_return"].notna()
        ].copy()
        if not valid.empty:
            first_observations = (
                valid.sort_values(["screen_instrument", "return_date"])
                .drop_duplicates("screen_instrument", keep="first")
                .loc[:, ["screen_instrument", "return_date", "historical_return"]]
            )
            observation_counts = (
                valid.groupby("screen_instrument")
                .size()
                .rename("return_observation_count")
                .reset_index()
            )
            base = base.merge(first_observations, on="screen_instrument", how="left")
            base = base.merge(observation_counts, on="screen_instrument", how="left")

    for column in ["return_date", "historical_return", "return_observation_count"]:
        if column not in base.columns:
            base[column] = pd.NA
    base["historical_return"] = pd.to_numeric(
        base["historical_return"], errors="coerce"
    ).astype("Float64")
    base["return_observation_count"] = (
        pd.to_numeric(base["return_observation_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    base["historical_return_available"] = base["return_observation_count"].gt(0)
    base["year"] = int(year)
    return base.loc[
        :,
        [
            "year",
            "screen_instrument",
            "return_date",
            "historical_return",
            "return_observation_count",
            "historical_return_available",
        ],
    ]


def build_datastream_june_return_availability_summary(
    years: list[int],
    *,
    active_stocks: pd.DataFrame | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    currency: str = DEFAULT_CURRENCY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_lseg_session()
    if active_stocks is None:
        active_stocks = fetch_active_datastream_stocks(currency=currency)
    instruments = (
        normalize_text(get_series(active_stocks, "screen_instrument", ""))
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int | float]] = []
    for year in tqdm(
        sorted(int(year) for year in years),
        desc="Querying Datastream June historical returns by year",
        unit="year",
    ):
        year_detail = query_datastream_june_return_availability_for_year(
            instruments,
            year=year,
            batch_size=batch_size,
        )
        detail_frames.append(year_detail)
        queried_count = len(instruments)
        available_count = int(year_detail["historical_return_available"].sum())
        summary_rows.append(
            {
                "year": year,
                "datastream_active_screener_stocks": queried_count,
                "datastream_june_return_observations": int(
                    year_detail["return_observation_count"].sum()
                ),
                "datastream_june_return_available_stocks": available_count,
                "datastream_june_return_availability_rate": (
                    available_count / queried_count if queried_count else 0.0
                ),
            }
        )

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    return summary, details, active_stocks


def build_origin_sample_size_comparison(
    *,
    sample_size_dir: Path = DEFAULT_SAMPLE_SIZE_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    currency: str = DEFAULT_CURRENCY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compustat_summary = load_name_screened_yearly_sample_sizes(sample_size_dir)
    datastream_summary, datastream_detail, active_stocks = build_datastream_june_return_availability_summary(
        compustat_summary["year"].astype(int).tolist(),
        batch_size=batch_size,
        currency=currency,
    )
    comparison = compustat_summary.merge(datastream_summary, on="year", how="left")
    return comparison, datastream_detail, active_stocks


def resolve_output_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_precomputed_outputs(
    *,
    output_dir: Path,
    comparison: pd.DataFrame,
    datastream_detail: pd.DataFrame,
    active_stocks: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / "compustat_vs_datastream_sample_sizes_by_year.csv",
        index=False,
    )
    datastream_detail.to_csv(
        output_dir / "datastream_june_return_availability_detail.csv",
        index=False,
    )
    active_stocks.to_csv(
        output_dir / "datastream_active_screener_stocks.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    sample_size_dir = resolve_sample_size_dir(args.sample_size_dir)
    output_dir = resolve_output_dir(args.output_dir)

    print(
        f"Using yearly '{NAME_SCREEN_METRIC}' sample sizes from: {sample_size_dir}"
    )
    comparison, datastream_detail, active_stocks = build_origin_sample_size_comparison(
        sample_size_dir=sample_size_dir,
        batch_size=args.batch_size,
        currency=args.currency,
    )
    write_precomputed_outputs(
        output_dir=output_dir,
        comparison=comparison,
        datastream_detail=datastream_detail,
        active_stocks=active_stocks,
    )
    print(f"Active Datastream/LSEG screener stocks: {len(active_stocks):,}")
    print(f"Wrote precomputed D2 inputs to: {output_dir}")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
