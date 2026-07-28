from pathlib import Path
import sys

import pandas as pd
from IPython.display import display


PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _analysis_shared import AnalysisOutputManager
from src.core.pipeline_config import SUE_COMPUTATION_GROUP_COUNT, SUE_PLOT_GROUP_COUNT
from src.analysis.time_varying_analysis import (
    FIRM_IDENTIFIER_COLUMN,
    collapse_to_event_level,
    load_abnormal_returns_with_groups,
    prepare_regression_frame,
)
from src.pead.sue_groups import SUE_GROUP_COLUMN, SUE_PLOT_GROUP_COLUMN, add_plot_group_column


OUTPUTS = AnalysisOutputManager(__file__)

# Match the complete-case event-level sample used by the baseline firm-FE model.
event_level = collapse_to_event_level(load_abnormal_returns_with_groups(PROJECT_ROOT / "data"))
if SUE_PLOT_GROUP_COLUMN not in event_level.columns:
    event_level = add_plot_group_column(
        event_level,
        group_column=SUE_GROUP_COLUMN,
        plot_group_column=SUE_PLOT_GROUP_COLUMN,
        computation_group_count=SUE_COMPUTATION_GROUP_COUNT,
        plot_group_count=SUE_PLOT_GROUP_COUNT,
    )
regression_sample = prepare_regression_frame(event_level, regressor_column=SUE_PLOT_GROUP_COLUMN)

firm_summary = regression_sample.groupby(FIRM_IDENTIFIER_COLUMN)[SUE_PLOT_GROUP_COLUMN].agg(
    announcement_count="size",
    ranked_sue_values="nunique",
)
multi_announcement = firm_summary[firm_summary["announcement_count"] > 1]
varying_ranked_sue = multi_announcement[multi_announcement["ranked_sue_values"] > 1]
constant_ranked_sue = multi_announcement[multi_announcement["ranked_sue_values"] == 1]

total_observations = len(regression_sample)
total_firms = len(firm_summary)
repeat_observations = int(multi_announcement["announcement_count"].sum())
repeat_firms = len(multi_announcement)

summary = pd.DataFrame(
    {
        "Observations": [
            total_observations,
            repeat_observations,
            int(varying_ranked_sue["announcement_count"].sum()),
            int(constant_ranked_sue["announcement_count"].sum()),
            int((firm_summary["announcement_count"] == 1).sum()),
        ],
        "Firms": [
            total_firms,
            repeat_firms,
            len(varying_ranked_sue),
            len(constant_ranked_sue),
            int((firm_summary["announcement_count"] == 1).sum()),
        ],
    },
    index=[
        "All eligible regression observations",
        "Firms with multiple eligible announcements",
        "Firms with varying ranked SUE (contribute to firm-FE identification)",
        "Firms with constant ranked SUE (do not contribute to firm-FE identification)",
        "Firms with one eligible announcement",
    ],
)
summary["Observation share (%)"] = 100 * summary["Observations"] / total_observations
summary["Firm share (%)"] = 100 * summary["Firms"] / total_firms
summary["Share of repeat-firm observations (%)"] = pd.NA
summary["Share of repeat firms (%)"] = pd.NA
summary.loc[summary.index[1:4], "Share of repeat-firm observations (%)"] = (
    100 * summary.loc[summary.index[1:4], "Observations"] / repeat_observations
)
summary.loc[summary.index[1:4], "Share of repeat firms (%)"] = (
    100 * summary.loc[summary.index[1:4], "Firms"] / repeat_firms
)
summary = summary.round(
    {
        "Observation share (%)": 1,
        "Firm share (%)": 1,
        "Share of repeat-firm observations (%)": 1,
        "Share of repeat firms (%)": 1,
    }
)

display(summary)
OUTPUTS.save_table(summary, "firm_fe_identification_summary")
