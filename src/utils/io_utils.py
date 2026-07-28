import json
from pathlib import Path


def load_json(filepath, default=None):
    path = Path(filepath)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(obj, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
