from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.project_paths import DATA_DIR, available_year_gbp_constituent_paths


EQUITY_PATH = DATA_DIR / "equity" / "compustat_equity.csv"
BOOK_EQUITY_COLUMN = "Book_Equity_Last_Fiscal_Year"
BOOK_EQUITY_DATADATE_COLUMN = "Book_Equity_Datadate"
BOOK_EQUITY_SOURCE_FIELD_COLUMN = "Book_Equity_Source_Field"
TARGET_COLUMNS = [
    "Compustat_CEQ",
    BOOK_EQUITY_COLUMN,
    BOOK_EQUITY_DATADATE_COLUMN,
    BOOK_EQUITY_SOURCE_FIELD_COLUMN,
]


def clean_gvkey(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(".0", "").zfill(6)


def choose_book_equity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Compustat_CEQ"] = pd.to_numeric(out.get("ceq"), errors="coerce")
    out[BOOK_EQUITY_COLUMN] = out["Compustat_CEQ"]
    out[BOOK_EQUITY_SOURCE_FIELD_COLUMN] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[out["Compustat_CEQ"].notna(), BOOK_EQUITY_SOURCE_FIELD_COLUMN] = "ceq"

    out[BOOK_EQUITY_DATADATE_COLUMN] = pd.to_datetime(
        out.get("datadate"),
        errors="coerce",
    )
    return out


def load_compustat_equity() -> pd.DataFrame:
    if not EQUITY_PATH.exists():
        raise FileNotFoundError(f"Missing Compustat equity file: {EQUITY_PATH}")

    equity = pd.read_csv(EQUITY_PATH, dtype=str)
    equity.columns = equity.columns.str.lower().str.strip()
    required_columns = {"gvkey", "datadate", "ceq"}
    missing_columns = required_columns.difference(equity.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(
            "Compustat equity file is missing required columns: "
            f"{missing}. File: {EQUITY_PATH}"
        )

    equity["gvkey"] = equity["gvkey"].map(clean_gvkey)
    equity = choose_book_equity(equity)
    equity = equity.loc[
        equity["gvkey"].ne("")
        & equity[BOOK_EQUITY_DATADATE_COLUMN].notna()
        & equity[BOOK_EQUITY_COLUMN].notna()
    ].copy()
    equity = equity.sort_values(["gvkey", BOOK_EQUITY_DATADATE_COLUMN]).reset_index(drop=True)
    return equity


def equity_snapshot_for_fiscal_year(
    equity: pd.DataFrame,
    fiscal_year: int,
) -> pd.DataFrame:
    exact_year = equity.loc[
        equity[BOOK_EQUITY_DATADATE_COLUMN].dt.year.eq(fiscal_year)
    ].copy()
    if exact_year.empty:
        return exact_year.loc[:, ["gvkey", *TARGET_COLUMNS]]

    selected = exact_year.drop_duplicates(subset=["gvkey"], keep="last").copy()
    return selected.loc[:, ["gvkey", *TARGET_COLUMNS]].reset_index(drop=True)


def populate_constituent_file(
    path: Path,
    equity: pd.DataFrame,
) -> dict[str, int]:
    constituents = pd.read_csv(path, dtype=str)
    if "gvkey" not in constituents.columns:
        raise KeyError(f"Missing 'gvkey' column in {path}")

    formation_year = int(path.parent.parent.name)
    equity_snapshot = equity_snapshot_for_fiscal_year(equity, formation_year - 1)

    constituents["gvkey"] = constituents["gvkey"].map(clean_gvkey)
    constituents = constituents.drop(columns=TARGET_COLUMNS, errors="ignore")
    populated = constituents.merge(equity_snapshot, on="gvkey", how="left")
    populated.to_csv(path, index=False)

    matched_rows = int(populated[BOOK_EQUITY_COLUMN].notna().sum())
    return {
        "formation_year": formation_year,
        "rows": int(len(populated)),
        "matched_rows": matched_rows,
    }


def main() -> None:
    constituent_paths = available_year_gbp_constituent_paths(DATA_DIR)
    if not constituent_paths:
        raise FileNotFoundError(
            f"No yearly GBP constituent files found under {DATA_DIR}"
        )

    equity = load_compustat_equity()
    summaries = [
        populate_constituent_file(path, equity)
        for path in constituent_paths
    ]

    total_rows = sum(item["rows"] for item in summaries)
    total_matched = sum(item["matched_rows"] for item in summaries)
    print(f"Updated yearly constituent files: {len(summaries):,}")
    print(f"Rows with Compustat equity populated: {total_matched:,} of {total_rows:,}")
    for summary in summaries:
        print(
            f"{summary['formation_year']}: "
            f"{summary['matched_rows']:,}/{summary['rows']:,} rows populated"
        )


if __name__ == "__main__":
    main()
