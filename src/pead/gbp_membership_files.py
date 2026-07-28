from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

import pandas as pd
try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - local cached workflows do not need LSEG.
    ld = None

from ..core.project_paths import DATA_DIR
from ..core.project_paths import (
    GBP_COMPUSTAT_MONTHLY_RETURNS_PATH,
    year_gbp_constituents_path,
    year_gbp_sedols_path,
)
from ..core.year_context import build_year_context
from ..core.yearly_data_io import merge_and_save_sample_size


EXPECTED_EXCHANGE_CODE = "194"
EXPECTED_JUNE_MONTH = 6
PRIMARY_IDENTIFIER_COLUMN = "isin"
MAPPING_FIELDS = [
    "TR.RIC",
    "TR.RIC(SDate={formation_date})",
    "TR.ISIN",
    "TR.SEDOL",
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeMarketIdCode",
]


def year_gbp_outputs_complete(year: int) -> bool:
    constituents_output_path = year_gbp_constituents_path(year)
    identifiers_output_path = year_gbp_sedols_path(year)
    audit_output_path = constituents_output_path.parent / f"xlon_validation_audit_{int(year)}.csv"
    sample_size_path = build_year_context(int(year), DATA_DIR).sample_size_path
    required_paths = (
        constituents_output_path,
        identifiers_output_path,
        audit_output_path,
        sample_size_path,
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required_paths)


def require_lseg() -> None:
    if ld is None:
        raise ModuleNotFoundError(
            "lseg.data is required to map ISIN identifiers to LSEG RICs in the 10x build."
        )


def clean_gvkey(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(".0", "").zfill(6)


def clean_iid(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def clean_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(".0", "")


def chunk_list(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def build_mapping_fields(formation_date: str) -> list[str]:
    return [
        field.format(formation_date=formation_date)
        for field in MAPPING_FIELDS
    ]


def safe_get_data(universe: list[str], fields: list[str]) -> tuple[bool, pd.DataFrame | str]:
    try:
        df = ld.get_data(
            universe=universe,
            fields=fields,
            header_type=ld.HeaderType.NAME,
        )
        if isinstance(df, pd.DataFrame):
            return True, df
        return False, f"Unexpected return type: {type(df).__name__}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def assert_input_matches_upstream_filters(frame: pd.DataFrame) -> None:
    month_values = set(frame["datadate"].dt.month.dropna().astype(int).unique().tolist())
    if month_values != {EXPECTED_JUNE_MONTH}:
        raise AssertionError(
            "The 10x build expects the upstream June-window file only. "
            f"Observed months: {sorted(month_values)}"
        )

    exchange_values = set(
        frame["exchg"].astype("string").str.strip().dropna().unique().tolist()
    )
    if exchange_values != {EXPECTED_EXCHANGE_CODE}:
        raise AssertionError(
            "The 10x build expects the upstream XLON-filtered file only. "
            f"Observed exchg values: {sorted(exchange_values)}"
        )


def prepare_monthly_returns(monthly_returns_path: Path) -> pd.DataFrame:
    monthly_returns = pd.read_csv(monthly_returns_path, dtype=str)
    monthly_returns.columns = monthly_returns.columns.str.lower().str.strip()

    required_columns = {"gvkey", "iid", "datadate", "conm", "isin", "exchg", "secstat"}
    missing_columns = required_columns - set(monthly_returns.columns)
    if missing_columns:
        raise ValueError(
            "Compustat monthly returns file missing columns: "
            f"{sorted(missing_columns)}"
        )

    monthly_returns["gvkey"] = monthly_returns["gvkey"].map(clean_gvkey)
    monthly_returns["iid"] = monthly_returns["iid"].map(clean_iid)
    monthly_returns["isin"] = monthly_returns["isin"].map(clean_identifier)
    if "sedol" in monthly_returns.columns:
        monthly_returns["sedol"] = monthly_returns["sedol"].map(clean_identifier)
    else:
        monthly_returns["sedol"] = ""
    monthly_returns["secstat"] = monthly_returns["secstat"].astype("string").str.strip().str.upper()
    monthly_returns["datadate"] = pd.to_datetime(monthly_returns["datadate"], errors="coerce")
    monthly_returns = monthly_returns.dropna(subset=["datadate"]).copy()

    assert_input_matches_upstream_filters(monthly_returns)

    monthly_returns["formation_year"] = monthly_returns["datadate"].dt.year.astype(int)
    monthly_returns["junedate"] = monthly_returns["datadate"]
    monthly_returns["lseg_identifier"] = monthly_returns["isin"]
    monthly_returns["compustat_isin_identifier"] = monthly_returns["isin"]
    monthly_returns["compustat_sedol_identifier"] = monthly_returns["sedol"]
    monthly_returns["security_key"] = (
        monthly_returns["gvkey"].fillna("")
        + "::"
        + monthly_returns["iid"].fillna("")
        + "::"
        + monthly_returns["isin"].fillna("")
    )
    return monthly_returns


def dedupe_yearly_window(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["formation_year", "gvkey", "iid", "isin", "datadate"])
        .drop_duplicates(subset=["formation_year", "gvkey", "iid", "isin"], keep="last")
        .reset_index(drop=True)
    )


def parse_mapping_result_row(row: pd.Series) -> dict[str, str]:
    return {
        "resolved_ric": str(row.get("TR.RIC", "")).strip(),
        "historical_ric": str(row.get(next((col for col in row.index if str(col).upper().startswith("TR.RIC(SDATE=")), ""), "")).strip(),
        "resolved_isin": str(row.get("TR.ISIN", "")).strip(),
        "resolved_sedol": str(row.get("TR.SEDOL", "")).strip(),
        "resolved_name": str(row.get("TR.CommonName", "")).strip(),
        "resolved_exchange_name": str(row.get("TR.ExchangeName", "")).strip(),
        "resolved_exchange_code": str(row.get("TR.ExchangeMarketIdCode", "")).strip(),
    }


def resolve_batch(
    identifiers: list[str],
    *,
    formation_date: str,
) -> tuple[bool, dict[str, dict[str, str]], str]:
    ok, result = safe_get_data(identifiers, build_mapping_fields(formation_date))
    if not ok:
        return False, {}, str(result)

    df = result
    info_by_identifier: dict[str, dict[str, str]] = {}
    if not df.empty and "Instrument" in df.columns:
        for _, row in df.iterrows():
            instrument = str(row.get("Instrument", "")).strip()
            if instrument:
                info_by_identifier[instrument] = parse_mapping_result_row(row)
    return True, info_by_identifier, ""


def resolve_identifiers_with_recursive_splitting(
    group: pd.DataFrame,
    *,
    formation_date: str,
    top_level_batch_size: int = 1000000,
    split_factor: int = 5,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if group.empty:
        empty = group.iloc[0:0].copy()
        return empty, empty, {
            "batch_attempts": 0,
            "batch_failures": 0,
            "single_identifier_requests": 0,
        }

    work = group.copy().reset_index(drop=True)
    work["identifier_value"] = work["isin"].astype("string").str.strip()
    work["identifier_available"] = work["identifier_value"].ne("")
    work["mapping_source"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["mapping_message"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_ric"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["historical_ric"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_exchange_name"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_exchange_code"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_name"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_isin"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work["resolved_sedol"] = pd.Series(pd.NA, index=work.index, dtype="string")

    missing_identifier = work.loc[~work["identifier_available"]].copy()
    if not missing_identifier.empty:
        missing_identifier["mapping_source"] = "missing_isin"
        missing_identifier["mapping_message"] = "missing identifier"
        if progress_callback is not None:
            progress_callback(int(len(missing_identifier)))

    available = work.loc[work["identifier_available"]].copy().reset_index(drop=True)
    resolved_rows: list[pd.DataFrame] = []
    unresolved_rows: list[pd.DataFrame] = []
    stats = {
        "batch_attempts": 0,
        "batch_failures": 0,
        "single_identifier_requests": 0,
    }

    def recurse(batch: pd.DataFrame) -> None:
        if batch.empty:
            return

        identifiers = batch["identifier_value"].astype("string").str.strip().tolist()
        stats["batch_attempts"] += 1
        if len(identifiers) == 1:
            stats["single_identifier_requests"] += 1

        ok, info_by_identifier, message = resolve_batch(
            identifiers,
            formation_date=formation_date,
        )
        if ok:
            out = batch.copy()
            out["mapping_source"] = f"isin_to_ric_batch_{len(identifiers)}"
            out["mapping_message"] = ""
            out["resolved_ric"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_ric", "")
            )
            out["historical_ric"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("historical_ric", "")
            )
            out["resolved_exchange_name"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_exchange_name", "")
            )
            out["resolved_exchange_code"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_exchange_code", "")
            )
            out["resolved_name"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_name", "")
            )
            out["resolved_isin"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_isin", "")
            )
            out["resolved_sedol"] = out["identifier_value"].map(
                lambda value: info_by_identifier.get(str(value).strip(), {}).get("resolved_sedol", "")
            )

            resolved_mask = out["historical_ric"].astype("string").str.strip().ne("") | out["resolved_ric"].astype("string").str.strip().ne("")
            resolved_rows.append(out.loc[resolved_mask].copy())

            unresolved = out.loc[~resolved_mask].copy()
            if not unresolved.empty:
                unresolved["mapping_source"] = f"isin_to_ric_batch_{len(identifiers)}_no_ric"
                unresolved["mapping_message"] = unresolved["mapping_message"].fillna("batch ok but no RIC resolved")
                unresolved_rows.append(unresolved)
            if progress_callback is not None:
                progress_callback(int(len(batch)))
            return

        stats["batch_failures"] += 1
        if len(batch) == 1:
            failed = batch.copy()
            failed["mapping_source"] = "isin_to_ric_single_failed"
            failed["mapping_message"] = message
            unresolved_rows.append(failed)
            if progress_callback is not None:
                progress_callback(1)
            return

        smaller_batch_size = max(1, (len(batch) + split_factor - 1) // split_factor)
        for chunk in chunk_list(batch.index.tolist(), smaller_batch_size):
            recurse(batch.loc[chunk].copy())

    active_group = available.loc[available["secstat"].eq("A")].copy()
    inactive_group = available.loc[available["secstat"].ne("A")].copy()

    if not active_group.empty:
        recurse(active_group)
    if not inactive_group.empty:
        recurse(inactive_group)

    accepted = pd.concat(resolved_rows, ignore_index=True) if resolved_rows else work.iloc[0:0].copy()
    rejected_frames = [frame for frame in [missing_identifier, *unresolved_rows] if not frame.empty]
    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else work.iloc[0:0].copy()
    return accepted, rejected, stats


def write_year_outputs(
    *,
    year: int,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    sample_size: dict,
) -> None:
    constituents_output_path = year_gbp_constituents_path(year)
    identifiers_output_path = year_gbp_sedols_path(year)
    audit_output_path = constituents_output_path.parent / f"xlon_validation_audit_{int(year)}.csv"
    year_context = build_year_context(int(year), DATA_DIR)

    constituents_output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(constituents_output_path, index=False)
    rejected.to_csv(audit_output_path, index=False)

    identifiers = (
        accepted["lseg_identifier"]
        .dropna()
        .astype("string")
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    identifiers_output_path.write_text(
        "\n".join(str(value) for value in identifiers) + ("\n" if len(identifiers) else ""),
        encoding="utf-8",
    )
    merge_and_save_sample_size(year_context, sample_size)


def build_year_sample_size(
    *,
    year_group: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    mapping_stats: dict[str, int],
) -> dict[str, int]:
    compustat_sample = int(year_group[["gvkey", "iid"]].drop_duplicates().shape[0])
    nonmissing_isin = int(
        year_group.loc[year_group["isin"].astype("string").str.strip().ne(""), ["gvkey", "iid"]]
        .drop_duplicates()
        .shape[0]
    )
    mapped_to_ric = int(accepted[["gvkey", "iid"]].drop_duplicates().shape[0])
    return {
        "Raw Compustat sample": compustat_sample,
        "Non-missing ISIN": nonmissing_isin,
        "Successfully mapped to RIC": mapped_to_ric,
    }


def build_gbp_universe_files(
    monthly_returns_path: Path = GBP_COMPUSTAT_MONTHLY_RETURNS_PATH,
) -> dict[str, int]:
    require_lseg()
    monthly_returns = prepare_monthly_returns(monthly_returns_path)
    matched_output = dedupe_yearly_window(monthly_returns)

    total_years = int(matched_output["formation_year"].nunique())
    validated_outputs: list[pd.DataFrame] = []
    rejected_outputs: list[pd.DataFrame] = []
    skipped_years = 0

    for year, group in tqdm(matched_output.groupby("formation_year"), desc="Processing years"):
        year = int(year)
        if year_gbp_outputs_complete(year):
            skipped_years += 1
            continue

        year_context = build_year_context(int(year), DATA_DIR)
        year_group = group.copy().reset_index(drop=True)
        year_group = year_group.loc[year_group["isin"].astype("string").str.strip().ne("")].copy()
        with tqdm(
            total=int(len(year_group)),
            desc=f"{year} stocks",
            leave=False,
        ) as year_progress:
            accepted, rejected, mapping_stats = resolve_identifiers_with_recursive_splitting(
                year_group,
                formation_date=year_context.formation_date,
                progress_callback=year_progress.update,
            )

        accepted_output = accepted.copy()
        accepted_output["validated_lseg_identifier"] = accepted_output["historical_ric"].where(
            accepted_output["historical_ric"].astype("string").str.strip().ne(""),
            accepted_output["resolved_ric"],
        )
        accepted_output["validated_lseg_identifier"] = (
            accepted_output["validated_lseg_identifier"].astype("string").str.strip()
        )
        accepted_output["lseg_identifier"] = accepted_output["validated_lseg_identifier"]

        rejected_output = rejected.copy()

        sample_size = build_year_sample_size(
            year_group=group,
            accepted=accepted_output,
            rejected=rejected_output,
            mapping_stats=mapping_stats,
        )
        write_year_outputs(
            year=year,
            accepted=accepted_output,
            rejected=rejected_output,
            sample_size=sample_size,
        )
        validated_outputs.append(accepted_output)
        rejected_outputs.append(rejected_output)

    expected_observations = matched_output.drop_duplicates(["formation_year", "gvkey", "iid"]).shape[0]
    validated_output = pd.concat(validated_outputs, ignore_index=True) if validated_outputs else matched_output.iloc[0:0].copy()
    matched_observations = validated_output.drop_duplicates(["junedate", "gvkey", "iid"]).shape[0]
    rejected_count = int(sum(len(frame) for frame in rejected_outputs))

    return {
        "expected_constituent_security_observations": int(expected_observations),
        "matched_constituent_security_observations": int(matched_observations),
        "missing_after_matching": int(expected_observations - matched_observations),
        "formation_years_written": int(total_years - skipped_years),
        "formation_years_skipped": int(skipped_years),
        "rows_written": int(len(validated_output)),
        "rejected_after_lseg_identifier_mapping": rejected_count,
    }
