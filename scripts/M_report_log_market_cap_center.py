"""Report the market-capitalisation value underlying the regression size centering.

The H3 size specifications centre each firm's average natural-log
pre-announcement market capitalisation at the cross-firm mean. Exponentiating
that mean gives the equivalent value in the original market-capitalisation
units. It is therefore a geometric mean, rather than an arithmetic mean.

The regression-dataset pickle is deliberately not used because pandas pickle
files are not reliably portable across pandas versions. The script instead
reconstructs the event-level input used by the regression suite.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.analysis.regression_suite import (
    PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN,
    build_regression_suite_dataset,
)
from src.core.project_paths import DATA_DIR
from src.core.pead_sample_variants import MAIN_PEAD_SAMPLE


FIRM_IDENTIFIER_COLUMN = "Instrument"
ANALYST_COUNT_COLUMN = "Forecast_Analyst_Count"


def main() -> None:
    """Print the original-unit value corresponding to the log-market-cap centre."""
    dataset = build_regression_suite_dataset(
        DATA_DIR,
        abnormal_returns_filename=MAIN_PEAD_SAMPLE.abnormal_returns_filename,
    )
    analyst_count = pd.to_numeric(dataset[ANALYST_COUNT_COLUMN], errors="coerce")
    market_cap = pd.to_numeric(dataset[PRE_ANNOUNCEMENT_MARKET_CAP_COLUMN], errors="coerce")
    eligible = analyst_count.ge(MAIN_PEAD_SAMPLE.min_analyst_forecasts) & market_cap.gt(0)
    sample = dataset.loc[eligible, [FIRM_IDENTIFIER_COLUMN]].copy()
    sample["log_market_cap"] = np.log(market_cap.loc[eligible])

    firm_average_log_market_cap = sample.groupby(FIRM_IDENTIFIER_COLUMN)["log_market_cap"].mean()
    log_center = firm_average_log_market_cap.mean()
    market_cap_center = float(np.exp(log_center))

    print(f"Firm count used for centering: {len(firm_average_log_market_cap):,}")
    print(f"Log-market-cap centre (natural log): {log_center:.6f}")
    print(f"Market-cap centre (original units): {market_cap_center:,.2f}")
    print("This is the geometric mean of firms' geometric-mean pre-announcement market caps.")


if __name__ == "__main__":
    main()
