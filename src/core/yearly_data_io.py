from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ..utils.io_utils import load_json, save_json


DataFrameNormalizer = Callable[[pd.DataFrame], pd.DataFrame]


def load_sample_size(year_context) -> dict:
    return load_json(year_context.sample_size_path, default={})


def save_sample_size(year_context, sample_size: dict) -> None:
    save_json(sample_size, year_context.sample_size_path)


def merge_and_save_sample_size(year_context, sample_size: dict) -> dict:
    merged_sample_size = dict(load_sample_size(year_context))
    merged_sample_size.update(sample_size)
    save_sample_size(year_context, merged_sample_size)
    return merged_sample_size


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv_if_exists(
    path: str | Path,
    *,
    normalizer: DataFrameNormalizer | None = None,
) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    frame = pd.read_csv(csv_path)
    if normalizer is not None:
        frame = normalizer(frame)
    return frame
