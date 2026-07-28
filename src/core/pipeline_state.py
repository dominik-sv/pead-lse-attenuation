from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Iterable

from ..utils.io_utils import load_json, save_json


def utc_now_z() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_completion_state(path: str | Path) -> dict[str, Any]:
    return load_json(path, default={})


def has_pipeline_version(path: str | Path, expected_version: str) -> bool:
    completion_state = load_completion_state(path)
    return completion_state.get("pipeline_version") == expected_version


def write_stage_completion(
    *,
    path: str | Path,
    year: int,
    stage: str,
    pipeline_version: str,
    outputs: Iterable[Path],
    extra_fields: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "year": int(year),
        "stage": stage,
        "pipeline_version": pipeline_version,
        "completed_at_utc": utc_now_z(),
        "outputs": [output_path.name for output_path in outputs],
    }
    if extra_fields:
        payload.update(extra_fields)
    save_json(payload, path)
