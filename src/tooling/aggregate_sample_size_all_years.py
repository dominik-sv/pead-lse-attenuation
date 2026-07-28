from __future__ import annotations

import math
import sys
from numbers import Number
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import BASE_DATA_DIR
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR, resolve_yearly_data_dir
from src.utils.io_utils import load_json, save_json


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / BASE_DATA_DIR).exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    return (
        PROJECT_DATA_DIR
        if PROJECT_DATA_DIR.exists()
        else resolve_project_root() / BASE_DATA_DIR
    )


def rebuild_yearly_json_aggregate(
    *,
    yearly_filename: str,
    output_filename: str,
) -> Path:
    data_dir = resolve_data_dir()
    yearly_files = sorted(
        resolve_yearly_data_dir(data_dir).glob(f"[0-9][0-9][0-9][0-9]/{yearly_filename}")
    )

    if not yearly_files:
        raise FileNotFoundError(f"No yearly {yearly_filename} files found under {data_dir}.")

    aggregate = {}
    for path in yearly_files:
        aggregate[path.parent.name] = load_json(path)

    output_path = data_dir / output_filename
    save_json(aggregate, output_path)
    return output_path


def rebuild_sample_size_all_years() -> Path:
    yearly_output_path = rebuild_yearly_json_aggregate(
        yearly_filename="sample_size.json",
        output_filename="sample_size_all_years.json",
    )
    yearly_sample_sizes = load_json(yearly_output_path, default={})
    if not isinstance(yearly_sample_sizes, dict):
        raise TypeError(f"Expected {yearly_output_path} to contain a JSON object.")

    aggregate: dict[str, int | float] = {}
    for year, sample_size in sorted(yearly_sample_sizes.items()):
        if not isinstance(sample_size, dict):
            raise TypeError(f"Expected {year} sample-size entry to be an object.")
        for metric, value in sample_size.items():
            if (
                isinstance(value, Number)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                aggregate[metric] = aggregate.get(metric, 0) + value

    save_json(aggregate, resolve_data_dir() / "sample_size_aggregate.json")
    return yearly_output_path


if __name__ == "__main__":
    output_path = rebuild_sample_size_all_years()
    print(f"Rebuilt {output_path}")

