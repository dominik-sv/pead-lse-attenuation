from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Any

import pandas as pd
import sys

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.project_paths import DATA_DIR, PROJECT_ROOT


LOCAL_PACKAGE_DIRS = [
    PROJECT_ROOT / ".python_packages_local",
    PROJECT_ROOT / ".python_packages",
]

for package_dir in LOCAL_PACKAGE_DIRS:
    if package_dir.exists() and str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUTS_DIR = DATA_DIR / "outputs"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "output"


class AnalysisOutputManager:
    def __init__(self, script_path: str | Path):
        script = Path(script_path)
        self.script_stem = script.stem
        self.output_dir = OUTPUTS_DIR / self.script_stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._name_counts: dict[str, int] = {}

    def get_output_dir(self) -> Path:
        return self.output_dir

    def _build_path(self, name: str, suffix: str) -> Path:
        base_name = _slugify(name)
        occurrence = self._name_counts.get(base_name, 0)
        while True:
            filename = (
                f"{base_name}{suffix}"
                if occurrence == 0
                else f"{base_name}_{occurrence + 1}{suffix}"
            )
            path = self.output_dir / filename
            if not path.exists():
                self._name_counts[base_name] = occurrence + 1
                return path
            occurrence += 1

    def save_figure(self, fig: Any, name: str, *, dpi: int = 300) -> Path:
        path = self._build_path(name, ".png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        # Analysis scripts run non-interactively: release the figure after saving
        # so it cannot be displayed and long runs do not accumulate open figures.
        import matplotlib.pyplot as plt

        plt.close(fig)
        return path

    def save_table(self, obj: pd.DataFrame | pd.Series, name: str) -> Path:
        path = self._build_path(name, ".csv")
        if isinstance(obj, pd.Series):
            table = obj.to_frame(name=obj.name if obj.name is not None else "Value")
            table.to_csv(path)
        else:
            obj.to_csv(path, index=True)
        return path

    def save_json(self, obj: Any, name: str) -> Path:
        path = self._build_path(name, ".json")
        with path.open("w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, ensure_ascii=False, default=str)
        return path

    def save_text(self, name: str, content: str) -> Path:
        path = self._build_path(name, ".txt")
        path.write_text(content, encoding="utf-8")
        return path

    def save_text_with_suffix(self, name: str, content: str, suffix: str) -> Path:
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = self._build_path(name, normalized_suffix)
        path.write_text(content, encoding="utf-8")
        return path

    def save_latex(self, name: str, content: str) -> Path:
        return self.save_text_with_suffix(name, content, ".tex")

    def display_and_save(self, obj: Any, name: str, display_fn=None) -> None:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            self.save_table(obj, name)
        elif isinstance(obj, (dict, list, tuple)):
            self.save_json(obj, name)
        else:
            self.save_text(name, str(obj))

        if display_fn is not None:
            display_fn(obj)
