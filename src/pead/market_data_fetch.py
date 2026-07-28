from pathlib import Path
import time
from typing import cast

import pandas as pd
try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - local cached-data workflows do not need LSEG.
    ld = None
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..core.pipeline_config import (
    BASE_PIPELINE_VERSION,
    SLEEP_BTWN_PULLS,
    TARGETED_PRICE_BATCH_SIZE,
    TARGETED_PRICE_INTERVAL_DAYS,
    TARGETED_RETURN_BATCH_SIZE,
)
from ..utils.io_utils import load_json, save_json
from ..utils.pandas_utils import (
    chunk_list,
    restore_history_multiindex,
)

EXPECTED_HISTORY_FIELDS = {"TotalReturn", "PriceClose"}
EXPECTED_PRICE_WINDOW_COLUMNS = ["Instrument", "Date", "PriceClose"]
EXPECTED_RETURN_WINDOW_COLUMNS = ["Instrument", "Date", "TotalReturn"]
COMBINED_MARKET_DATA_FIELDS = [
    "TR.PriceClose.Date",
    "TR.PriceClose",
    "TR.TotalReturn1D",
]


def require_lseg() -> None:
    if ld is None:
        raise ModuleNotFoundError(
            "lseg.data is required for remote market-data downloads, but it is not "
            "installed in the active Python runtime. Use the cached local-only workflow "
            "or run this script in an environment with the LSEG SDK available."
        )


def robust_get_history(
    universe,
    fields,
    interval,
    start,
    end,
    max_retries=2,
    base_sleep=1.0,
) -> pd.DataFrame:
    require_lseg()
    last_error = None

    for attempt in range(max_retries):
        try:
            return ld.get_history(
                universe=universe,
                fields=fields,
                interval=interval,
                start=start,
                end=end,
            )
        except Exception as error:
            last_error = error
            sleep_s = base_sleep * (2**attempt)
            print(f"[retry {attempt + 1}/{max_retries}] get_history failed: {error}")
            print(f"Sleeping {sleep_s:.1f}s before retry...")
            time.sleep(sleep_s)

    raise last_error  # type: ignore[misc]


def robust_get_data(
    universe,
    fields,
    parameters,
    header_type,
    max_retries=2,
    base_sleep=1.0,
) -> pd.DataFrame:
    require_lseg()
    last_error = None

    for attempt in range(max_retries):
        try:
            return ld.get_data(
                universe=universe,
                fields=fields,
                parameters=parameters,
                header_type=header_type,
            )
        except Exception as error:
            last_error = error
            sleep_s = base_sleep * (2**attempt)
            print(f"[retry {attempt + 1}/{max_retries}] get_data failed: {error}")
            print(f"Sleeping {sleep_s:.1f}s before retry...")
            time.sleep(sleep_s)

    raise last_error  # type: ignore[misc]


def download_market_data_for_year(
    stock_universe,
    year_context,
    market_data_window_start,
    market_data_window_end,
    batch_size,
    currency,
):
    chunks_dir, checkpoint_file, error_log_file = get_market_data_file_paths(
        year_context
    )
    completed_batches, errors = load_download_state(
        checkpoint_file,
        error_log_file,
        expected_metadata=build_market_data_metadata(currency),
    )

    download_market_data_chunks(
        stock_universe=stock_universe,
        market_data_window_start=market_data_window_start,
        market_data_window_end=market_data_window_end,
        batch_size=batch_size,
        chunks_dir=chunks_dir,
        checkpoint_file=checkpoint_file,
        error_log_file=error_log_file,
        completed_batches=completed_batches,
        errors=errors,
        currency=currency,
    )

    validate_market_data_download(
        stock_universe=stock_universe,
        batch_size=batch_size,
        chunks_dir=chunks_dir,
        checkpoint_file=checkpoint_file,
        error_log_file=error_log_file,
        expected_metadata=build_market_data_metadata(currency),
    )

    return combine_market_data_chunks(chunks_dir)


def get_market_data_file_paths(year_context):
    chunks_dir = year_context.market_data_chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)

    return (
        chunks_dir,
        year_context.market_data_checkpoint_path,
        year_context.market_data_errors_path,
    )


def build_market_data_metadata(currency: str) -> dict[str, str]:
    return {
        "pipeline_version": BASE_PIPELINE_VERSION,
        "currency": currency,
    }


def checkpoint_metadata_matches(
    checkpoint: dict,
    expected_metadata: dict[str, str],
) -> bool:
    checkpoint_metadata = {
        "pipeline_version": checkpoint.get("pipeline_version"),
        "currency": checkpoint.get("currency"),
    }
    return checkpoint_metadata == expected_metadata


def load_download_state(checkpoint_file, error_log_file, expected_metadata):
    checkpoint = load_json(checkpoint_file, default={"completed_batches": []})
    if not checkpoint_metadata_matches(checkpoint, expected_metadata):
        save_json(
            {"completed_batches": [], **expected_metadata},
            checkpoint_file,
        )
        save_json([], error_log_file)
        return set(), []

    completed_batches = set(checkpoint["completed_batches"])
    errors = list(load_json(error_log_file, default=[]))
    return completed_batches, errors


def market_data_file_has_expected_columns(path: str | Path) -> bool:
    return _file_has_expected_history_header(Path(path))


def market_data_download_is_complete(
    stock_universe: pd.DataFrame,
    batch_size: int,
    chunks_dir: str | Path,
    checkpoint_file: str | Path,
    error_log_file: str | Path,
    expected_metadata: dict[str, str] | None = None,
) -> bool:
    chunks_path = Path(chunks_dir)
    checkpoint = load_json(checkpoint_file, default={"completed_batches": []})
    if expected_metadata is not None and not checkpoint_metadata_matches(
        checkpoint, expected_metadata
    ):
        return False

    completed_batches = set(checkpoint.get("completed_batches", []))
    errors = list(load_json(error_log_file, default=[]))
    expected_batch_names = set(
        build_expected_market_data_batch_names(stock_universe, batch_size)
    )

    if errors or not expected_batch_names:
        return False

    if completed_batches != expected_batch_names:
        return False

    return all(
        _file_has_expected_history_header(chunks_path / f"{batch_name}.csv")
        for batch_name in expected_batch_names
    )


def validate_market_data_download(
    stock_universe: pd.DataFrame,
    batch_size: int,
    chunks_dir: str | Path,
    checkpoint_file: str | Path,
    error_log_file: str | Path,
    expected_metadata: dict[str, str] | None = None,
) -> None:
    chunks_path = Path(chunks_dir)
    checkpoint = load_json(checkpoint_file, default={"completed_batches": []})
    if expected_metadata is not None and not checkpoint_metadata_matches(
        checkpoint, expected_metadata
    ):
        raise RuntimeError(
            "Market-data download metadata is stale. "
            "Rerun the downloader to rebuild the cached history files."
        )

    completed_batches = set(checkpoint.get("completed_batches", []))
    errors = list(load_json(error_log_file, default=[]))
    expected_batch_names = set(
        build_expected_market_data_batch_names(stock_universe, batch_size)
    )

    missing_batches = sorted(expected_batch_names.difference(completed_batches))
    unexpected_batches = sorted(completed_batches.difference(expected_batch_names))
    invalid_batches = sorted(
        batch_name
        for batch_name in expected_batch_names.intersection(completed_batches)
        if not _file_has_expected_history_header(chunks_path / f"{batch_name}.csv")
    )

    if errors or missing_batches or unexpected_batches or invalid_batches:
        error_messages = []

        if errors:
            error_messages.append(
                f"{len(errors)} failed batch download(s) recorded in {error_log_file}"
            )
        if missing_batches:
            error_messages.append(
                f"missing completed batches: {', '.join(missing_batches)}"
            )
        if unexpected_batches:
            error_messages.append(
                f"unexpected completed batches in checkpoint: {', '.join(unexpected_batches)}"
            )
        if invalid_batches:
            error_messages.append(
                f"invalid or outdated chunk files: {', '.join(invalid_batches)}"
            )

        raise RuntimeError(
            "Market-data download is incomplete. "
            + "; ".join(error_messages)
            + ". Fix the failed/missing chunk files and rerun the base pipeline."
        )


def build_expected_market_data_batch_names(
    stock_universe: pd.DataFrame,
    batch_size: int,
) -> list[str]:
    instruments = stock_universe["Instrument"].dropna().unique().tolist()
    batches = list(chunk_list(instruments, batch_size))
    return [f"batch_{idx:05d}" for idx, _ in enumerate(batches)]


def download_market_data_chunks(
    stock_universe,
    market_data_window_start,
    market_data_window_end,
    batch_size,
    chunks_dir,
    checkpoint_file,
    error_log_file,
    completed_batches,
    errors,
    currency,
):
    instruments = stock_universe["Instrument"].dropna().unique().tolist()
    batches = list(chunk_list(instruments, batch_size))
    formation_year = stock_universe["Formation_Year"].iloc[0]

    for idx, instrument_batch in tqdm(
        enumerate(batches),
        total=len(batches),
        desc=f"Downloading market data {formation_year}",
    ):
        batch_name = f"batch_{idx:05d}"
        batch_file = chunks_dir / f"{batch_name}.csv"

        if batch_name in completed_batches and batch_file.exists():
            if _file_has_expected_history_header(batch_file):
                cleaned_errors = remove_batch_errors(errors, batch_name)
                if len(cleaned_errors) != len(errors):
                    errors = cleaned_errors
                    save_json(errors, error_log_file)
                continue

            print(
                f"Re-downloading {batch_name}: cached chunk format is outdated or invalid."
            )
            completed_batches.discard(batch_name)
            save_json(
                {
                    "completed_batches": sorted(completed_batches),
                    **build_market_data_metadata(currency),
                },
                checkpoint_file,
            )

            try:
                batch_file.unlink()
            except FileNotFoundError:
                pass

        try:
            hist, failed_instruments = download_market_data_batch_with_fallback(
                instrument_batch=instrument_batch,
                batch_name=batch_name,
                market_data_window_start=market_data_window_start,
                market_data_window_end=market_data_window_end,
                currency=currency,
            )

            hist.to_csv(batch_file, index=True)

            completed_batches.add(batch_name)
            errors = remove_batch_errors(errors, batch_name)
            save_json(
                {
                    "completed_batches": sorted(completed_batches),
                    **build_market_data_metadata(currency),
                },
                checkpoint_file,
            )
            save_json(errors, error_log_file)

            if failed_instruments:
                failed_instrument_names = ", ".join(
                    error_record["instrument"] for error_record in failed_instruments
                )
                print(
                    f"{batch_name}: continuing with partial batch after isolating "
                    f"failed instrument(s): {failed_instrument_names}"
                )

            time.sleep(SLEEP_BTWN_PULLS)

        except Exception as error:
            errors = replace_batch_error(
                errors,
                {
                    "batch": batch_name,
                    "n_instruments": len(instrument_batch),
                    "instruments": instrument_batch,
                    "error": str(error),
                },
            )
            save_json(errors, error_log_file)
            print(f"FAILED {batch_name}: {error}")


def download_market_data_batch(
    instrument_batch,
    market_data_window_start,
    market_data_window_end,
    currency,
) -> pd.DataFrame:
    hist = fetch_market_data_history(
        instrument_batch=instrument_batch,
        start=market_data_window_start,
        end=market_data_window_end,
        currency=currency,
    )

    if not _has_expected_history_multiindex(hist):
        raise ValueError(
            "Unexpected market data column format "
            f"for instruments {instrument_batch}: {list(hist.columns)[:6]}"
        )

    return hist


def download_market_data_for_instruments(
    instruments: list[str],
    start: str,
    end: str,
    cache_path: str | Path,
    currency: str,
    identifier_options_by_instrument: dict[str, list[str]] | None = None,
    batch_size: int = TARGETED_RETURN_BATCH_SIZE,
    desc: str = "Downloading market data",
) -> tuple[pd.DataFrame, int]:
    cleaned_instruments = sorted(
        {
            str(instrument).strip()
            for instrument in instruments
            if pd.notna(instrument) and str(instrument).strip()
        }
    )
    request_groups = [
        {
            "request_label": f"market_data_{batch_index:03d}",
            "start": start,
            "end": end,
            "instruments": instrument_batch,
        }
        for batch_index, instrument_batch in enumerate(
            chunk_list(cleaned_instruments, batch_size),
            start=1,
        )
    ]

    cache_file = Path(cache_path)
    if cache_file.exists() and market_data_file_has_expected_columns(cache_file):
        cached_market_data = read_market_data_file(cache_file)
        cached_market_data = trim_market_data_to_window(
            market_data=cached_market_data,
            start=start,
            end=end,
        )
        cached_market_data.to_csv(cache_file, index=True)
        return cached_market_data, len(request_groups)

    if not request_groups:
        empty_market_data = pd.DataFrame()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        empty_market_data.to_csv(cache_file, index=True)
        return empty_market_data, 0

    market_data_parts: list[pd.DataFrame] = []
    for request_group in tqdm(
        request_groups,
        total=len(request_groups),
        desc=desc,
    ):
        market_data, failed_instruments = download_market_data_batch_with_fallback(
            instrument_batch=list(cast(list[str], request_group["instruments"])),
            batch_name=str(request_group["request_label"]),
            market_data_window_start=str(request_group["start"]),
            market_data_window_end=str(request_group["end"]),
            currency=currency,
            identifier_options_by_instrument=identifier_options_by_instrument,
        )
        market_data_parts.append(market_data)

        if failed_instruments:
            failed_names = ", ".join(
                error_record["instrument"] for error_record in failed_instruments
            )
            print(
                f"{request_group['request_label']}: continuing after isolating "
                f"failed instrument(s): {failed_names}"
            )

        time.sleep(SLEEP_BTWN_PULLS)

    combined_market_data = pd.concat(market_data_parts, axis=1)
    combined_market_data = combined_market_data.loc[
        :, ~combined_market_data.columns.duplicated()
    ]
    combined_market_data = combined_market_data.sort_index()
    combined_market_data.index.name = "Date"
    combined_market_data = trim_market_data_to_window(
        market_data=combined_market_data,
        start=start,
        end=end,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    combined_market_data.to_csv(cache_file, index=True)
    return combined_market_data, len(request_groups)


def fetch_market_data_history(
    instrument_batch,
    start,
    end,
    currency,
) -> pd.DataFrame:
    raw_market_data = robust_get_data(
        universe=instrument_batch,
        fields=COMBINED_MARKET_DATA_FIELDS,
        parameters={
            "SDate": start,
            "EDate": end,
            "Frq": "D",
            "Curn": currency,
        },
        header_type=ld.HeaderType.NAME,
        max_retries=2,
        base_sleep=1.0,
    )
    return reshape_market_data_history(
        raw_market_data=raw_market_data,
        instrument_batch=instrument_batch,
    )


def reshape_market_data_history(
    raw_market_data: pd.DataFrame | None,
    instrument_batch,
) -> pd.DataFrame:
    empty_columns = pd.MultiIndex.from_arrays(
        [[], []],
        names=["Field", "Instrument"],
    )
    if raw_market_data is None or raw_market_data.empty:
        return pd.DataFrame(columns=empty_columns)

    out = raw_market_data.copy()
    instrument_column = _find_price_history_column(
        out.columns,
        required_label="INSTRUMENT",
    )
    date_column = _find_price_history_column(
        out.columns,
        required_label="DATE",
        exclude_columns={instrument_column},
    )
    price_column = _find_market_data_value_column(
        columns=out.columns,
        include_label="PRICE",
        exclude_columns={instrument_column, date_column},
    )
    total_return_column = _find_market_data_value_column(
        columns=out.columns,
        include_label="TOTALRETURN",
        exclude_columns={instrument_column, date_column, price_column},
    )

    out = out.rename(
        columns={
            instrument_column: "Instrument",
            date_column: "Date",
            price_column: "PriceClose",
            total_return_column: "TotalReturn",
        }
    )
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["PriceClose"] = pd.to_numeric(out["PriceClose"], errors="coerce")
    out["TotalReturn"] = pd.to_numeric(out["TotalReturn"], errors="coerce")
    out["Instrument"] = out["Instrument"].astype("string").str.strip()
    out = out.dropna(subset=["Instrument", "Date"])
    out = out[out["Instrument"].isin(instrument_batch)].copy()

    if out.empty:
        return pd.DataFrame(columns=empty_columns)

    out = (
        out.sort_values(["Instrument", "Date"])
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .set_index(["Date", "Instrument"])
    )
    price_history = out["PriceClose"].unstack("Instrument")
    total_return_history = out["TotalReturn"].unstack("Instrument")
    price_history.columns = price_history.columns.astype(str)
    total_return_history.columns = total_return_history.columns.astype(str)

    return combine_market_history_fields(
        total_return_hist=total_return_history,
        price_hist=price_history_to_multiindex(price_history),
        instrument_batch=instrument_batch,
    )


def price_history_to_multiindex(price_history: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(
            columns=pd.MultiIndex.from_arrays(
                [[], []],
                names=["Field", "Instrument"],
            )
        )

    out = price_history.sort_index().copy()
    out.columns = pd.MultiIndex.from_tuples(
        [("PriceClose", str(column)) for column in out.columns],
        names=["Field", "Instrument"],
    )
    out.index.name = "Date"
    return out


def download_market_data_batch_with_fallback(
    instrument_batch,
    batch_name,
    market_data_window_start,
    market_data_window_end,
    currency,
    identifier_options_by_instrument: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        hist = download_market_data_batch(
            instrument_batch=instrument_batch,
            market_data_window_start=market_data_window_start,
            market_data_window_end=market_data_window_end,
            currency=currency,
        )
        return hist, []
    except Exception as batch_error:
        if len(instrument_batch) <= 1:
            failed_instrument = (
                str(instrument_batch[0]).strip() if instrument_batch else "<empty>"
            )
            print(
                f"{batch_name}: skipping failed single-instrument market-data request "
                f"for {failed_instrument}: {batch_error}"
            )
            return pd.DataFrame(), [
                {
                    "instrument": failed_instrument,
                    "error": str(batch_error),
                }
            ]

        print(
            f"{batch_name}: batch download failed after retries ({batch_error}). "
            "Retrying instruments one by one..."
        )

    partial_histories = []
    failed_instruments = []

    for instrument in instrument_batch:
        identifier_candidates = [instrument]
        if identifier_options_by_instrument is not None:
            identifier_candidates = [
                candidate
                for candidate in identifier_options_by_instrument.get(instrument, [instrument])
                if pd.notna(candidate) and str(candidate).strip()
            ]

        identifier_candidates = list(dict.fromkeys(str(candidate).strip() for candidate in identifier_candidates))
        last_error: Exception | None = None
        instrument_resolved = False
        for candidate in identifier_candidates:
            try:
                instrument_hist = download_market_data_batch(
                    instrument_batch=[candidate],
                    market_data_window_start=market_data_window_start,
                    market_data_window_end=market_data_window_end,
                    currency=currency,
                )
                if candidate != instrument:
                    instrument_hist = rename_market_data_instrument(
                        instrument_hist,
                        current_instrument=candidate,
                        target_instrument=instrument,
                    )
                    print(
                        f"{batch_name}: recovered {instrument} market data using alternate identifier "
                        f"{candidate}"
                    )
                partial_histories.append(instrument_hist)
                instrument_resolved = True
                break
            except Exception as instrument_error:
                last_error = instrument_error

        if not instrument_resolved:
            failed_instruments.append(
                {
                    "instrument": instrument,
                    "error": str(last_error) if last_error is not None else "Unknown error",
                }
            )
            print(
                f"{batch_name}: failed single-instrument retry for "
                f"{instrument}: {last_error}"
            )

    if not partial_histories:
        failed_details = "; ".join(
            f"{error_record['instrument']}: {error_record['error']}"
            for error_record in failed_instruments
        )
        raise RuntimeError(
            f"Batch-level download failed and no single-instrument retry succeeded. "
            f"{failed_details}"
        )

    hist = pd.concat(partial_histories, axis=1)
    hist = hist.loc[:, ~hist.columns.duplicated()]
    hist = hist.sort_index()
    hist.index.name = "Date"
    return hist, failed_instruments


def rename_market_data_instrument(
    market_data: pd.DataFrame,
    *,
    current_instrument: str,
    target_instrument: str,
) -> pd.DataFrame:
    if market_data.empty:
        return market_data
    if not isinstance(market_data.columns, pd.MultiIndex):
        return market_data

    renamed_columns = pd.MultiIndex.from_tuples(
        [
            (field, target_instrument if str(instrument) == current_instrument else instrument)
            for field, instrument in market_data.columns.to_list()
        ],
        names=market_data.columns.names,
    )
    out = market_data.copy()
    out.columns = renamed_columns
    return out


def fetch_price_history(
    instrument_batch,
    start,
    end,
    currency,
) -> pd.DataFrame:
    raw_price_history = robust_get_data(
        universe=instrument_batch,
        fields=["TR.PriceClose.Date", "TR.PriceClose"],
        parameters={
            "SDate": start,
            "EDate": end,
            "Frq": "D",
            "Curn": currency,
        },
        header_type=ld.HeaderType.NAME,
        max_retries=2,
        base_sleep=1.0,
    )
    return reshape_price_history(
        raw_price_history=raw_price_history,
        instrument_batch=instrument_batch,
    )


def remove_batch_errors(errors, batch_name):
    return [error for error in errors if error.get("batch") != batch_name]


def replace_batch_error(errors, error_entry):
    updated_errors = remove_batch_errors(errors, error_entry["batch"])
    updated_errors.append(error_entry)
    return updated_errors


def reshape_price_history(
    raw_price_history: pd.DataFrame | None,
    instrument_batch,
) -> pd.DataFrame:
    empty_columns = pd.MultiIndex.from_arrays(
        [[], []],
        names=["Field", "Instrument"],
    )
    if raw_price_history is None or raw_price_history.empty:
        return pd.DataFrame(columns=empty_columns)

    out = raw_price_history.copy()
    instrument_column = _find_price_history_column(
        out.columns,
        required_label="INSTRUMENT",
    )
    date_column = _find_price_history_column(
        out.columns,
        required_label="DATE",
        exclude_columns={instrument_column},
    )
    value_column = _find_price_history_value_column(
        out.columns,
        exclude_columns={instrument_column, date_column},
    )

    out = out.rename(
        columns={
            instrument_column: "Instrument",
            date_column: "Date",
            value_column: "PriceClose",
        }
    )
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["PriceClose"] = pd.to_numeric(out["PriceClose"], errors="coerce")
    out["Instrument"] = out["Instrument"].astype("string").str.strip()
    out = out.dropna(subset=["Instrument", "Date", "PriceClose"])
    out = out[out["Instrument"].isin(instrument_batch)].copy()

    if out.empty:
        return pd.DataFrame(columns=empty_columns)

    out = (
        out.sort_values(["Instrument", "Date"])
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .pivot(index="Date", columns="Instrument", values="PriceClose")
        .sort_index()
    )
    out.columns = pd.MultiIndex.from_tuples(
        [("PriceClose", str(column)) for column in out.columns],
        names=["Field", "Instrument"],
    )
    out.index.name = "Date"
    return out


def _find_price_history_column(
    columns,
    required_label,
    exclude_columns=None,
):
    excluded = set(exclude_columns or set())
    matches = [
        column
        for column in columns
        if column not in excluded and required_label in str(column).upper()
    ]
    if not matches:
        raise KeyError(f"Missing {required_label.lower()} column in price history data.")
    return matches[0]


def _find_price_history_value_column(columns, exclude_columns=None):
    excluded = set(exclude_columns or set())
    matches = [
        column
        for column in columns
        if column not in excluded and "PRICE" in str(column).upper()
    ]
    if matches:
        return matches[0]

    remaining_columns = [column for column in columns if column not in excluded]
    if len(remaining_columns) == 1:
        return remaining_columns[0]

    raise KeyError("Missing price column in price history data.")


def _find_market_data_value_column(
    columns,
    include_label,
    exclude_columns=None,
):
    excluded = set(exclude_columns or set())
    matches = [
        column
        for column in columns
        if column not in excluded and include_label in str(column).upper()
    ]
    if matches:
        return matches[0]

    raise KeyError(
        f"Missing {include_label.lower()} column in combined market data history."
    )


def normalize_total_return_history(
    total_return_hist: pd.DataFrame | None,
    instrument_batch,
) -> pd.DataFrame:
    empty_columns = pd.MultiIndex.from_arrays(
        [[], []],
        names=["Field", "Instrument"],
    )

    if total_return_hist is None or total_return_hist.empty:
        return pd.DataFrame(columns=empty_columns)

    out = total_return_hist.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].copy()
    out.index.name = "Date"

    if isinstance(out.columns, pd.MultiIndex):
        if out.columns.nlevels >= 2:
            instruments = out.columns.get_level_values(-1).astype(str)
            out.columns = pd.MultiIndex.from_tuples(
                [("TotalReturn", instrument) for instrument in instruments],
                names=["Field", "Instrument"],
            )
            return out

    out.columns = [str(column).strip() for column in out.columns]
    valid_columns = [column for column in out.columns if column in instrument_batch]
    out = out.loc[:, valid_columns].copy()

    out = out.apply(pd.to_numeric, errors="coerce")

    out.columns = pd.MultiIndex.from_tuples(
        [("TotalReturn", column) for column in out.columns],
        names=["Field", "Instrument"],
    )

    return out


def combine_market_history_fields(
    total_return_hist: pd.DataFrame,
    price_hist: pd.DataFrame,
    instrument_batch,
) -> pd.DataFrame:
    total_return_hist = normalize_total_return_history(
        total_return_hist=total_return_hist,
        instrument_batch=instrument_batch,
    )

    combined = pd.concat([total_return_hist, price_hist], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.sort_index()
    combined.index.name = "Date"

    return combined


def combine_market_data_chunks(chunks_dir) -> pd.DataFrame:
    chunk_files = sorted(Path(chunks_dir).glob("batch_*.csv"))

    invalid_chunk_files = []
    market_data_parts = []

    for path in chunk_files:
        part = _read_market_data_chunk(path)

        if part.empty:
            continue

        if not _has_expected_history_multiindex(part):
            invalid_chunk_files.append(path.name)
            continue

        market_data_parts.append(part)

    if invalid_chunk_files:
        raise ValueError(
            "Malformed market data chunk files detected: "
            f"{', '.join(invalid_chunk_files)}. "
            "Delete the listed chunk files or rerun the downloader after upgrading the code."
        )

    if not market_data_parts:
        return pd.DataFrame()

    market_data = pd.concat(market_data_parts, axis=1)
    market_data = market_data.loc[:, ~market_data.columns.duplicated()]
    market_data = market_data.sort_index()
    market_data.index.name = "Date"

    return market_data


def read_market_data_file(path) -> pd.DataFrame:
    return _read_market_data_chunk(path)


def trim_market_data_to_window(
    market_data: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    if market_data.empty:
        return market_data.copy()

    trimmed = market_data.copy()
    trimmed.index = pd.to_datetime(trimmed.index, errors="coerce")
    trimmed = trimmed.loc[trimmed.index.notna()].copy()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    trimmed = trimmed.loc[(trimmed.index >= start_ts) & (trimmed.index <= end_ts)].copy()
    trimmed = trimmed.sort_index()
    trimmed.index.name = "Date"
    return trimmed


def _read_market_data_chunk(path) -> pd.DataFrame:
    chunk = pd.read_csv(path, index_col=0, header=[0, 1], parse_dates=True)

    if isinstance(chunk.columns, pd.MultiIndex) and set(
        chunk.columns.get_level_values(0)
    ).issubset(EXPECTED_HISTORY_FIELDS):
        chunk.columns.names = ["Field", "Instrument"]
        return chunk

    chunk = pd.read_csv(path, index_col=0, parse_dates=True)
    chunk.index.name = "Date"

    restored_columns = restore_history_multiindex(chunk.columns)
    if restored_columns is not None:
        chunk.columns = restored_columns

    return chunk


def _has_expected_history_multiindex(market_data: pd.DataFrame) -> bool:
    if not isinstance(market_data.columns, pd.MultiIndex):
        return False

    field_values = set(market_data.columns.get_level_values(0))
    return bool(field_values) and field_values.issubset(EXPECTED_HISTORY_FIELDS)


def _file_has_expected_history_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first_line = handle.readline().strip()
            second_line = handle.readline().strip()
    except OSError:
        return False

    return first_line.startswith("Field,") and second_line.startswith("Instrument,")


def extract_total_return_history(market_data: pd.DataFrame) -> pd.DataFrame:
    return extract_history_field(
        market_data=market_data,
        field_name="TotalReturn",
        fallback_to_flat_data=True,
    )


def extract_price_history(market_data: pd.DataFrame) -> pd.DataFrame:
    return extract_history_field(
        market_data=market_data,
        field_name="PriceClose",
        fallback_to_flat_data=False,
    )


def subset_market_data_to_instruments(
    market_data: pd.DataFrame,
    instruments: list[str],
) -> pd.DataFrame:
    if market_data.empty:
        return market_data.copy()

    cleaned_instruments = {
        str(instrument).strip()
        for instrument in instruments
        if pd.notna(instrument) and str(instrument).strip()
    }
    if not cleaned_instruments:
        return market_data.iloc[:, 0:0].copy()

    if not isinstance(market_data.columns, pd.MultiIndex):
        available_columns = [
            column for column in market_data.columns if str(column) in cleaned_instruments
        ]
        return market_data.loc[:, available_columns].copy()

    instrument_mask = market_data.columns.get_level_values(-1).astype(str).isin(
        cleaned_instruments
    )
    subset = market_data.loc[:, instrument_mask].copy()
    subset = subset.loc[:, ~subset.columns.duplicated()]
    subset = subset.sort_index()
    subset.index.name = "Date"
    return subset


def extract_history_field(
    market_data: pd.DataFrame,
    field_name: str,
    fallback_to_flat_data: bool,
) -> pd.DataFrame:
    if market_data.empty:
        return pd.DataFrame()

    if not isinstance(market_data.columns, pd.MultiIndex):
        if fallback_to_flat_data:
            return market_data.copy()
        return pd.DataFrame(index=market_data.index)

    market_data_columns = cast(pd.MultiIndex, market_data.columns)
    field_mask = market_data_columns.get_level_values(0) == field_name

    if not field_mask.any():
        return pd.DataFrame(index=market_data.index)

    history = market_data.loc[:, field_mask].copy()
    history.columns = market_data_columns[field_mask].droplevel(0)
    history.columns.name = None

    return history


def price_window_cache_has_expected_columns(path: str | Path) -> bool:
    return _window_cache_has_expected_columns(path, EXPECTED_PRICE_WINDOW_COLUMNS)


def return_window_cache_has_expected_columns(path: str | Path) -> bool:
    return _window_cache_has_expected_columns(path, EXPECTED_RETURN_WINDOW_COLUMNS)


def _window_cache_has_expected_columns(
    path: str | Path,
    expected_columns: list[str],
) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False

    return all(column in columns for column in expected_columns)


def read_price_window_cache(path: str | Path) -> pd.DataFrame:
    cache = pd.read_csv(path)

    for column in EXPECTED_PRICE_WINDOW_COLUMNS:
        if column not in cache.columns:
            raise KeyError(f"price window cache is missing required column: {column}")

    cache["Instrument"] = cache["Instrument"].astype("string").str.strip()
    cache["Date"] = pd.to_datetime(cache["Date"], errors="coerce")
    cache["PriceClose"] = pd.to_numeric(cache["PriceClose"], errors="coerce")
    cache = cache.dropna(subset=["Instrument", "Date", "PriceClose"]).copy()
    return cache.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def read_return_window_cache(path: str | Path) -> pd.DataFrame:
    cache = pd.read_csv(path)

    for column in EXPECTED_RETURN_WINDOW_COLUMNS:
        if column not in cache.columns:
            raise KeyError(f"return window cache is missing required column: {column}")

    cache["Instrument"] = cache["Instrument"].astype("string").str.strip()
    cache["Date"] = pd.to_datetime(cache["Date"], errors="coerce")
    cache["TotalReturn"] = pd.to_numeric(cache["TotalReturn"], errors="coerce")
    cache = cache.dropna(subset=["Instrument", "Date", "TotalReturn"]).copy()
    return cache.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def materialize_price_history_from_cache(price_windows: pd.DataFrame) -> pd.DataFrame:
    return materialize_history_from_long_cache(
        cache=price_windows,
        value_column="PriceClose",
    )


def materialize_total_return_history_from_cache(
    return_windows: pd.DataFrame,
) -> pd.DataFrame:
    return materialize_history_from_long_cache(
        cache=return_windows,
        value_column="TotalReturn",
    )


def materialize_history_from_long_cache(
    cache: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    if cache.empty:
        return pd.DataFrame()

    history = (
        cache.sort_values(["Instrument", "Date"])
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .pivot(index="Date", columns="Instrument", values=value_column)
        .sort_index()
    )
    history.index = pd.to_datetime(history.index, errors="coerce")
    history.index.name = "Date"
    history.columns = history.columns.astype(str)
    history.columns.name = None
    return history


def build_price_window_request_groups(
    events: pd.DataFrame,
    year_context,
    batch_size: int = TARGETED_PRICE_BATCH_SIZE,
    interval_days: int = TARGETED_PRICE_INTERVAL_DAYS,
) -> list[dict[str, object]]:
    if events.empty:
        return []

    request_events = events.loc[:, ["Instrument", "Ann_Date"]].copy()
    request_events["Instrument"] = (
        request_events["Instrument"].astype("string").str.strip()
    )
    request_events["Ann_Date"] = pd.to_datetime(
        request_events["Ann_Date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    request_events = request_events.dropna(subset=["Instrument", "Ann_Date"]).copy()

    if request_events.empty:
        return []

    bucket_origin = pd.Timestamp(year_context.formation_date) + pd.Timedelta(days=1)
    bucket_index = (
        (request_events["Ann_Date"].dt.normalize() - bucket_origin).dt.days // interval_days
    ).clip(lower=0)
    request_events["Bucket_Start"] = bucket_origin + pd.to_timedelta(
        bucket_index * interval_days,
        unit="D",
    )
    request_events["Bucket_End"] = (
        request_events["Bucket_Start"] + pd.Timedelta(days=interval_days - 1)
    )

    request_groups: list[dict[str, object]] = []
    grouped_events = request_events.groupby(["Bucket_Start", "Bucket_End"], sort=True)
    for bucket_number, ((bucket_start, bucket_end), bucket_events) in enumerate(
        grouped_events,
        start=1,
    ):
        instruments = sorted(bucket_events["Instrument"].dropna().unique().tolist())
        request_start = (
            pd.Timestamp(bucket_start) - pd.Timedelta(days=interval_days)
        ).strftime("%Y-%m-%d")
        request_end = pd.Timestamp(bucket_end).strftime("%Y-%m-%d")

        for instrument_batch_index, instrument_batch in enumerate(
            chunk_list(instruments, batch_size),
            start=1,
        ):
            request_groups.append(
                {
                    "request_label": (
                        f"price_window_{bucket_number:03d}_{instrument_batch_index:03d}"
                    ),
                    "start": request_start,
                    "end": request_end,
                    "instruments": instrument_batch,
                }
            )

    return request_groups


def download_price_history_for_events(
    events: pd.DataFrame,
    year_context,
    currency: str,
    batch_size: int = TARGETED_PRICE_BATCH_SIZE,
    interval_days: int = TARGETED_PRICE_INTERVAL_DAYS,
) -> tuple[pd.DataFrame, int]:
    request_groups = build_price_window_request_groups(
        events=events,
        year_context=year_context,
        batch_size=batch_size,
        interval_days=interval_days,
    )
    price_windows = _download_price_window_cache(
        request_groups=request_groups,
        year_context=year_context,
        currency=currency,
    )
    return materialize_price_history_from_cache(price_windows), len(request_groups)


def _download_price_window_cache(
    request_groups: list[dict[str, object]],
    year_context,
    currency: str,
) -> pd.DataFrame:
    cache_parts: list[pd.DataFrame] = []

    for request_group in tqdm(
        request_groups,
        total=len(request_groups),
        desc=f"Downloading price windows {year_context.year}",
    ):
        price_history, failed_instruments = download_price_window_batch_with_fallback(
            instrument_batch=list(cast(list[str], request_group["instruments"])),
            start=str(request_group["start"]),
            end=str(request_group["end"]),
            currency=currency,
        )
        price_windows = wide_history_to_long(
            history=extract_price_history(price_history),
            value_column="PriceClose",
        )
        cache_parts.append(price_windows)

        if failed_instruments:
            failed_names = ", ".join(
                error_record["instrument"] for error_record in failed_instruments
            )
            print(
                f"{request_group['request_label']}: continuing after isolating "
                f"failed instrument(s): {failed_names}"
            )

        time.sleep(SLEEP_BTWN_PULLS)

    price_window_cache = combine_long_window_parts(
        cache_parts,
        expected_columns=EXPECTED_PRICE_WINDOW_COLUMNS,
    )
    save_window_cache(price_window_cache, year_context.price_windows_path)
    return price_window_cache


def download_price_window_batch_with_fallback(
    instrument_batch,
    start,
    end,
    currency,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        price_history = fetch_price_history(
            instrument_batch=instrument_batch,
            start=start,
            end=end,
            currency=currency,
        )
        return price_history, []
    except Exception as batch_error:
        if len(instrument_batch) <= 1:
            raise

        print(
            "Price-window batch download failed after retries "
            f"({batch_error}). Retrying instruments one by one..."
        )

    partial_histories = []
    failed_instruments = []

    for instrument in instrument_batch:
        try:
            instrument_history = fetch_price_history(
                instrument_batch=[instrument],
                start=start,
                end=end,
                currency=currency,
            )
            partial_histories.append(instrument_history)
        except Exception as instrument_error:
            failed_instruments.append(
                {
                    "instrument": instrument,
                    "error": str(instrument_error),
                }
            )
            print(
                "Price-window single-instrument retry failed for "
                f"{instrument}: {instrument_error}"
            )

    if not partial_histories:
        failed_details = "; ".join(
            f"{error_record['instrument']}: {error_record['error']}"
            for error_record in failed_instruments
        )
        raise RuntimeError(
            "Price-window batch download failed and no single-instrument retry "
            f"succeeded. {failed_details}"
        )

    price_history = pd.concat(partial_histories, axis=1)
    price_history = price_history.loc[:, ~price_history.columns.duplicated()]
    price_history = price_history.sort_index()
    price_history.index.name = "Date"
    return price_history, failed_instruments


def build_return_request_groups(
    event_windows: pd.DataFrame,
    batch_size: int = TARGETED_RETURN_BATCH_SIZE,
) -> list[dict[str, object]]:
    if event_windows.empty:
        return []

    instrument_windows = event_windows.copy()
    instrument_windows["Instrument"] = (
        instrument_windows["Instrument"].astype("string").str.strip()
    )
    instrument_windows["Window_Start"] = pd.to_datetime(
        instrument_windows["Window_Start"], errors="coerce"
    )
    instrument_windows["Window_End"] = pd.to_datetime(
        instrument_windows["Window_End"], errors="coerce"
    )
    instrument_windows = instrument_windows.dropna(
        subset=["Instrument", "Window_Start", "Window_End"]
    ).copy()

    if instrument_windows.empty:
        return []

    merged_windows = (
        instrument_windows.groupby("Instrument", as_index=False)
        .agg(Window_Start=("Window_Start", "min"), Window_End=("Window_End", "max"))
        .sort_values(["Window_Start", "Window_End", "Instrument"])
        .reset_index(drop=True)
    )

    request_groups: list[dict[str, object]] = []
    merged_window_records = merged_windows.to_dict("records")
    for batch_index, batch_records in enumerate(
        chunk_list(merged_window_records, batch_size),
        start=1,
    ):
        batch_frame = pd.DataFrame(batch_records)
        request_groups.append(
            {
                "request_label": f"return_window_{batch_index:03d}",
                "start": pd.Timestamp(batch_frame["Window_Start"].min()).strftime(
                    "%Y-%m-%d"
                ),
                "end": pd.Timestamp(batch_frame["Window_End"].max()).strftime(
                    "%Y-%m-%d"
                ),
                "instruments": batch_frame["Instrument"].astype(str).tolist(),
            }
        )

    return request_groups


def download_return_history_for_instruments(
    instruments: list[str],
    start: str,
    end: str,
    cache_path: str | Path,
    batch_size: int = TARGETED_RETURN_BATCH_SIZE,
    desc: str = "Downloading return windows",
) -> tuple[pd.DataFrame, int]:
    cleaned_instruments = sorted(
        {
            str(instrument).strip()
            for instrument in instruments
            if pd.notna(instrument) and str(instrument).strip()
        }
    )
    request_groups = [
        {
            "request_label": f"return_window_{batch_index:03d}",
            "start": start,
            "end": end,
            "instruments": instrument_batch,
        }
        for batch_index, instrument_batch in enumerate(
            chunk_list(cleaned_instruments, batch_size),
            start=1,
        )
    ]

    cache_file = Path(cache_path)
    if cache_file.exists() and return_window_cache_has_expected_columns(cache_file):
        return_windows = read_return_window_cache(cache_file)
        return materialize_total_return_history_from_cache(return_windows), len(
            request_groups
        )

    return_windows = _download_return_window_cache(
        request_groups=request_groups,
        year_context=None,
        cache_path=cache_file,
        desc=desc,
    )
    return materialize_total_return_history_from_cache(return_windows), len(
        request_groups
    )


def download_return_history_for_event_windows(
    event_windows: pd.DataFrame,
    year_context,
    batch_size: int = TARGETED_RETURN_BATCH_SIZE,
) -> tuple[pd.DataFrame, int]:
    request_groups = build_return_request_groups(
        event_windows=event_windows,
        batch_size=batch_size,
    )
    return_windows = _download_return_window_cache(
        request_groups=request_groups,
        year_context=year_context,
    )
    return materialize_total_return_history_from_cache(return_windows), len(
        request_groups
    )


def _download_return_window_cache(
    request_groups: list[dict[str, object]],
    year_context,
    cache_path: str | Path | None = None,
    desc: str | None = None,
) -> pd.DataFrame:
    cache_parts: list[pd.DataFrame] = []
    resolved_desc = desc
    if resolved_desc is None:
        if year_context is None:
            resolved_desc = "Downloading return windows"
        else:
            resolved_desc = f"Downloading return windows {year_context.year}"

    for request_group in tqdm(
        request_groups,
        total=len(request_groups),
        desc=resolved_desc,
    ):
        return_history, failed_instruments = download_return_window_batch_with_fallback(
            instrument_batch=list(cast(list[str], request_group["instruments"])),
            start=str(request_group["start"]),
            end=str(request_group["end"]),
        )
        return_windows = wide_history_to_long(
            history=return_history,
            value_column="TotalReturn",
        )
        cache_parts.append(return_windows)

        if failed_instruments:
            failed_names = ", ".join(
                error_record["instrument"] for error_record in failed_instruments
            )
            print(
                f"{request_group['request_label']}: continuing after isolating "
                f"failed instrument(s): {failed_names}"
            )

        time.sleep(SLEEP_BTWN_PULLS)

    return_window_cache = combine_long_window_parts(
        cache_parts,
        expected_columns=EXPECTED_RETURN_WINDOW_COLUMNS,
    )
    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else year_context.return_windows_path
    )
    save_window_cache(return_window_cache, resolved_cache_path)
    return return_window_cache


def download_return_window_batch_with_fallback(
    instrument_batch,
    start,
    end,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        return_history = fetch_total_return_history(
            instrument_batch=instrument_batch,
            start=start,
            end=end,
        )
        return return_history, []
    except Exception as batch_error:
        if len(instrument_batch) <= 1:
            raise

        print(
            "Return-window batch download failed after retries "
            f"({batch_error}). Retrying instruments one by one..."
        )

    partial_histories = []
    failed_instruments = []

    for instrument in instrument_batch:
        try:
            instrument_history = fetch_total_return_history(
                instrument_batch=[instrument],
                start=start,
                end=end,
            )
            partial_histories.append(instrument_history)
        except Exception as instrument_error:
            failed_instruments.append(
                {
                    "instrument": instrument,
                    "error": str(instrument_error),
                }
            )
            print(
                "Return-window single-instrument retry failed for "
                f"{instrument}: {instrument_error}"
            )

    if not partial_histories:
        failed_details = "; ".join(
            f"{error_record['instrument']}: {error_record['error']}"
            for error_record in failed_instruments
        )
        raise RuntimeError(
            "Return-window batch download failed and no single-instrument retry "
            f"succeeded. {failed_details}"
        )

    return_history = pd.concat(partial_histories, axis=1)
    return_history = return_history.loc[:, ~return_history.columns.duplicated()]
    return_history = return_history.sort_index()
    return_history.index.name = "Date"
    return return_history, failed_instruments


def fetch_total_return_history(
    instrument_batch,
    start,
    end,
) -> pd.DataFrame:
    total_return_history = robust_get_history(
        universe=instrument_batch,
        fields=["TR.TOTALRETURN1D"],
        interval="daily",
        start=start,
        end=end,
        max_retries=2,
        base_sleep=1.0,
    )
    normalized_history = normalize_total_return_history(
        total_return_hist=total_return_history,
        instrument_batch=instrument_batch,
    )
    return extract_total_return_history(normalized_history)


def wide_history_to_long(
    history: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Instrument", "Date", value_column])

    long_history = history.copy()
    long_history.index = pd.to_datetime(long_history.index, errors="coerce")
    long_history = long_history.loc[long_history.index.notna()].copy()
    long_history.index.name = "Date"
    long_history = long_history.reset_index().melt(
        id_vars="Date",
        var_name="Instrument",
        value_name=value_column,
    )
    long_history["Instrument"] = long_history["Instrument"].astype("string").str.strip()
    long_history[value_column] = pd.to_numeric(long_history[value_column], errors="coerce")
    long_history = long_history.dropna(
        subset=["Instrument", "Date", value_column]
    ).copy()
    return long_history.sort_values(["Instrument", "Date"]).reset_index(drop=True)


def combine_long_window_parts(
    cache_parts: list[pd.DataFrame],
    expected_columns: list[str],
) -> pd.DataFrame:
    if not cache_parts:
        return pd.DataFrame(columns=expected_columns)

    combined = pd.concat(cache_parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Instrument", "Date"], keep="last")
    combined = combined.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    return combined.reindex(columns=expected_columns)


def save_window_cache(cache: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)
