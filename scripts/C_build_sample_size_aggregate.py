from __future__ import annotations

import json
import math
import sys
from numbers import Number
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import BASE_DATA_DIR
from src.core.project_paths import DATA_DIR as PROJECT_DATA_DIR
from src.utils.io_utils import save_json


INPUT_FILENAME = "sample_size_all_years.json"
OUTPUT_FILENAME = "sample_size_aggregate.json"


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / BASE_DATA_DIR).exists():
        return cwd
    return PROJECT_ROOT


def resolve_data_dir() -> Path:
    return (
        PROJECT_DATA_DIR
        if PROJECT_DATA_DIR.exists()
        else resolve_project_root() / BASE_DATA_DIR
    )


def is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Number)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def aggregate_numeric_metrics(sample_size_all_years: dict) -> dict[str, int | float]:
    aggregate: dict[str, int | float] = {}

    for year, sample_size in sorted(sample_size_all_years.items()):
        if not isinstance(sample_size, dict):
            raise TypeError(f"Expected {year} sample-size entry to be an object.")

        for metric, value in sample_size.items():
            if not is_finite_number(value):
                continue

            aggregate[metric] = aggregate.get(metric, 0) + value

    return aggregate


def build_sample_size_aggregate() -> Path:
    data_dir = resolve_data_dir()
    input_path = data_dir / INPUT_FILENAME
    output_path = data_dir / OUTPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}.")

    with input_path.open("r", encoding="utf-8") as file:
        sample_size_all_years = json.load(file)

    if not isinstance(sample_size_all_years, dict):
        raise TypeError(f"Expected {input_path} to contain a JSON object.")

    aggregate = aggregate_numeric_metrics(sample_size_all_years)
    save_json(aggregate, output_path)
    return output_path


if __name__ == "__main__":
    output_path = build_sample_size_aggregate()
    print(f"Built {output_path}")
