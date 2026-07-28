from pathlib import Path
import sys
import warnings

import pandas as pd
import lseg.data as ld

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.project_paths import DATA_DIR
from src.pead.gbp_membership_files import build_gbp_universe_files
from src.tooling.aggregate_sample_size_all_years import rebuild_sample_size_all_years


# LSEG's dataframe helper still triggers a pandas FutureWarning on replace().
# Opt into the future behavior so the 00-stage stays quiet without changing results.
pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"lseg\.data\._tools\._dataframe",
)


def main() -> None:
    ld.open_session()
    filtered_source_path = DATA_DIR / "XLON_membership" / "compustat_filtered_june.csv"
    summary = build_gbp_universe_files(monthly_returns_path=filtered_source_path)
    aggregate_sample_size_path = rebuild_sample_size_all_years()

    print(
        "Built yearly GBP universe files from "
        f"{filtered_source_path.resolve()}"
    )
    print(
        "Matched "
        f"{summary['matched_constituent_security_observations']:,} of "
        f"{summary['expected_constituent_security_observations']:,} "
        "constituent-security observations."
    )
    print(f"Missing after matching: {summary['missing_after_matching']:,}")
    print(
        "Rejected after LSEG ISIN-to-RIC mapping: "
        f"{summary['rejected_after_lseg_identifier_mapping']:,}"
    )
    print(f"Yearly GBP universe folders written: {summary['formation_years_written']:,}")
    print(f"Yearly GBP universe folders skipped: {summary['formation_years_skipped']:,}")
    print(f"Rebuilt aggregate sample-size file: {aggregate_sample_size_path}")


if __name__ == "__main__":
    main()
