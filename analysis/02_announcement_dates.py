import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _analysis_shared import AnalysisOutputManager, DATA_DIR


YEARLY_DATA_DIR = DATA_DIR / "yearly"
OUTPUTS = AnalysisOutputManager(__file__)
EARNINGS_ANNOUNCEMENT_COLOR = "#0072B2"


def load_earnings_events():
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_abnormal_returns.csv")
    )

    earnings_events = pd.concat(
        (
            pd.read_csv(
                path,
                usecols=lambda column: column in {
                    "Event_ID",
                    "Instrument",
                    "Ann_Date",
                },
                parse_dates=["Ann_Date"],
            )
            for path in earnings_files
        ),
        ignore_index=True,
    )
    if "Event_ID" in earnings_events.columns:
        earnings_events = earnings_events.drop_duplicates(subset=["Event_ID"])
    else:
        earnings_events = earnings_events.drop_duplicates(
            subset=["Instrument", "Ann_Date"]
        )

    return earnings_events


earnings_events = load_earnings_events()

announcement_plot_df = earnings_events.dropna(subset=["Ann_Date"]).copy()

calendar_day = announcement_plot_df["Ann_Date"].dt.dayofyear.clip(upper=365)
announcement_plot_df["Calendar_Week"] = ((calendar_day - 1) // 7 + 1).astype(int)

fig, ax = plt.subplots(figsize=(7, 4.2))

ax.hist(
    announcement_plot_df["Calendar_Week"],
    bins=np.arange(0.5, 53.5, 1),
    edgecolor="black",
    color=EARNINGS_ANNOUNCEMENT_COLOR,
    alpha=0.85,
)

month_starts = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
month_tick_positions = ((month_starts.dayofyear - 1) // 7 + 1).astype(int)

ax.set_xlabel("Calendar week")
ax.set_ylabel("Earnings announcements")
ax.set_xticks(month_tick_positions)
ax.set_xticklabels(month_starts.strftime("%b"))
ax.set_xlim(0.5, 52.5)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
OUTPUTS.save_figure(fig, "weekly_distribution_of_earnings_announcement_dates")
plt.close(fig)
