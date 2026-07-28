from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.project_paths import DATA_DIR, available_year_gbp_constituent_paths


OUTPUT_DIR = DATA_DIR / "equity"
OUTPUT_GVKEY_TXT = OUTPUT_DIR / "compustat_global_unique_gvkeys.txt"
OUTPUT_GVKEY_CSV = OUTPUT_DIR / "compustat_global_unique_gvkeys.csv"


def load_gvkeys_from_constituent_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"gvkey": "string"})
    if "gvkey" not in frame.columns:
        raise KeyError(f"Missing 'gvkey' column in {path}")

    gvkeys = (
        frame.loc[:, ["gvkey"]]
        .assign(
            gvkey=lambda df: df["gvkey"].astype("string").str.strip(),
            formation_year=int(path.parent.parent.name),
        )
        .loc[lambda df: df["gvkey"].notna() & df["gvkey"].ne("")]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return gvkeys


def build_unique_gvkey_table() -> pd.DataFrame:
    constituent_paths = available_year_gbp_constituent_paths(DATA_DIR)
    if not constituent_paths:
        raise FileNotFoundError(
            f"No yearly GBP constituent files found under {DATA_DIR}"
        )

    gvkey_frames = [
        load_gvkeys_from_constituent_file(path)
        for path in constituent_paths
    ]
    all_gvkeys = pd.concat(gvkey_frames, ignore_index=True)

    grouped = (
        all_gvkeys.groupby("gvkey", as_index=False)
        .agg(
            first_year=("formation_year", "min"),
            last_year=("formation_year", "max"),
            year_count=("formation_year", "nunique"),
            formation_years=(
                "formation_year",
                lambda years: ",".join(str(year) for year in sorted(set(years))),
            ),
        )
        .sort_values("gvkey")
        .reset_index(drop=True)
    )
    return grouped


def write_outputs(unique_gvkeys: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unique_gvkeys.to_csv(OUTPUT_GVKEY_CSV, index=False)
    OUTPUT_GVKEY_TXT.write_text(
        "\n".join(unique_gvkeys["gvkey"].astype(str).tolist()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    unique_gvkeys = build_unique_gvkey_table()
    write_outputs(unique_gvkeys)

    print(f"Unique gvkeys: {len(unique_gvkeys):,}")
    print(f"Wrote WRDS upload list to {OUTPUT_GVKEY_TXT}")
    print(f"Wrote gvkey summary table to {OUTPUT_GVKEY_CSV}")


if __name__ == "__main__":
    main()
