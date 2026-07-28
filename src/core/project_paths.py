from __future__ import annotations

import os
from pathlib import Path

from .pipeline_config import (
    BASE_DATA_DIR,
    GBP_COMPUSTAT_MONTHLY_RETURNS_FILE_NAME,
    GBP_COMPUSTAT_MONTHLY_RETURNS_LEGACY_FILE_NAME,
    GBP_CONSTITUENT_FILE_TEMPLATE,
    GBP_CONSTITUENTS_FILE_NAME,
    GBP_IDENTIFIERS_FILE_NAME,
    GBP_MEMBERSHIP_DIRNAME,
    GBP_MEMBERSHIP_LEGACY_DIRNAME,
    GBP_SEDOL_FILE_TEMPLATE,
    GBP_UNIVERSE_SUBDIR_NAME,
    YEARLY_DATA_DIRNAME,
)


ACTIVE_DATA_DIR_ENV_VAR = "BACHELOR_THESIS_DATA_DIR"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_active_data_dir() -> Path:
    override = os.environ.get(ACTIVE_DATA_DIR_ENV_VAR, "").strip()
    if not override:
        return PROJECT_ROOT / BASE_DATA_DIR

    override_path = Path(override).expanduser()
    if not override_path.is_absolute():
        override_path = PROJECT_ROOT / override_path
    return override_path.resolve()


DATA_DIR = _resolve_active_data_dir()
YEARLY_DATA_DIR = DATA_DIR / YEARLY_DATA_DIRNAME
GBP_MEMBERSHIP_DIR = DATA_DIR / GBP_MEMBERSHIP_DIRNAME
GBP_MEMBERSHIP_LEGACY_DIR = DATA_DIR / GBP_MEMBERSHIP_LEGACY_DIRNAME

GBP_CONSTITUENTS_PATH = GBP_MEMBERSHIP_DIR / GBP_CONSTITUENTS_FILE_NAME
GBP_IDENTIFIERS_PATH = GBP_MEMBERSHIP_DIR / GBP_IDENTIFIERS_FILE_NAME


def _resolve_compustat_monthly_returns_path() -> Path:
    candidates = (
        GBP_MEMBERSHIP_DIR / GBP_COMPUSTAT_MONTHLY_RETURNS_FILE_NAME,
        GBP_MEMBERSHIP_DIR / GBP_COMPUSTAT_MONTHLY_RETURNS_LEGACY_FILE_NAME,
        GBP_MEMBERSHIP_LEGACY_DIR / GBP_COMPUSTAT_MONTHLY_RETURNS_FILE_NAME,
        GBP_MEMBERSHIP_LEGACY_DIR / GBP_COMPUSTAT_MONTHLY_RETURNS_LEGACY_FILE_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


GBP_COMPUSTAT_MONTHLY_RETURNS_PATH = _resolve_compustat_monthly_returns_path()


def resolve_yearly_data_dir(base_dir: Path | str | None = None) -> Path:
    if base_dir is None:
        return YEARLY_DATA_DIR

    base_path = Path(base_dir)
    if base_path.name == YEARLY_DATA_DIRNAME:
        return base_path

    yearly_candidate = base_path / YEARLY_DATA_DIRNAME
    if yearly_candidate.exists():
        return yearly_candidate

    return yearly_candidate


def year_dir(year: int | str) -> Path:
    return YEARLY_DATA_DIR / str(year)


def year_gbp_universe_dir(year: int | str) -> Path:
    return year_dir(year) / GBP_UNIVERSE_SUBDIR_NAME


def year_gbp_constituents_path(year: int | str) -> Path:
    year_value = int(year)
    return year_gbp_universe_dir(year_value) / GBP_CONSTITUENT_FILE_TEMPLATE.format(
        year=year_value
    )


def year_gbp_sedols_path(year: int | str) -> Path:
    year_value = int(year)
    return year_gbp_universe_dir(year_value) / GBP_SEDOL_FILE_TEMPLATE.format(
        year=year_value
    )


def available_year_gbp_constituent_paths(base_dir: Path | None = None) -> list[Path]:
    search_root = resolve_yearly_data_dir(base_dir)
    return sorted(
        year_gbp_constituents_path(year_dir.name)
        for year_dir in search_root.glob("[0-9][0-9][0-9][0-9]")
        if year_dir.is_dir()
        and year_dir.name.isdigit()
        and year_gbp_constituents_path(year_dir.name).exists()
    )
