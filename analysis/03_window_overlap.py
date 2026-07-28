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
all_events_path = DATA_DIR / "earnings_release_diagnostics" / "all_filtered_earnings_events.csv"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
OUTPUTS = AnalysisOutputManager(__file__)

pead_window_start_day = 2
pead_window_end_day = 60

abnormal_return_files = sorted(
    YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_abnormal_returns.csv")
)

pead_overlap_rows = pd.concat(
    (
        pd.read_csv(path, parse_dates=["Ann_Date", "Trading_Date"])
        for path in abnormal_return_files
    ),
    ignore_index=True,
)

pead_overlap_source = (
    "Concatenated data/yearly/<year>/earnings_abnormal_returns.csv "
    f"(Relative_Day {pead_window_start_day}..{pead_window_end_day})"
)

pead_overlap_rows["Relative_Day"] = pd.to_numeric(
    pead_overlap_rows["Relative_Day"], errors="coerce"
)

if "Event_ID" not in pead_overlap_rows.columns:
    pead_overlap_rows["Event_ID"] = (
        pead_overlap_rows[["Instrument", "Ann_Date", "Report_Frequency"]]
        .astype(str)
        .agg("|".join, axis=1)
    )

pead_window_rows = (
    pead_overlap_rows.loc[
        pead_overlap_rows["Relative_Day"].between(pead_window_start_day, pead_window_end_day),
        ["Event_ID", "Instrument", "Ann_Date", "Trading_Date"],
    ]
    .dropna(subset=["Event_ID", "Instrument", "Ann_Date", "Trading_Date"])
    .copy()
)

pead_window_rows["Trading_Date"] = pead_window_rows["Trading_Date"].dt.normalize()

pead_overlap_events = (
    pead_window_rows.groupby(["Event_ID", "Instrument", "Ann_Date"], as_index=False)
    .agg(
        Window_Start=("Trading_Date", "min"),
        Window_End=("Trading_Date", "max"),
        Window_Length_Trading_Days=("Trading_Date", "nunique"),
    )
    .sort_values(["Window_Start", "Ann_Date", "Instrument"])
    .reset_index(drop=True)
)

trade_days_by_event = {
    event_id: frozenset(trading_days)
    for event_id, trading_days in pead_window_rows.groupby("Event_ID")["Trading_Date"]
}

pead_overlap_events["Overlap_Count"] = 0
pead_overlap_events["Overlap_Trading_Days"] = 0

starts = pead_overlap_events["Window_Start"].to_numpy(dtype="datetime64[ns]")
ends = pead_overlap_events["Window_End"].to_numpy(dtype="datetime64[ns]")
event_ids = pead_overlap_events["Event_ID"].tolist()

sorted_starts = np.sort(starts)
sorted_ends = np.sort(ends)

pead_overlap_events["Overlap_Count"] = (
    np.searchsorted(sorted_starts, ends, side="right")
    - np.searchsorted(sorted_ends, starts, side="left")
    - 1
).astype(int)

pead_overlap_events["Has_Overlap"] = pead_overlap_events["Overlap_Count"] > 0

trading_calendar = pd.DatetimeIndex(
    sorted(pead_window_rows["Trading_Date"].drop_duplicates())
)
calendar_positions = pd.Index(trading_calendar)

start_positions = calendar_positions.get_indexer(pd.DatetimeIndex(pead_overlap_events["Window_Start"]))
end_positions = calendar_positions.get_indexer(pd.DatetimeIndex(pead_overlap_events["Window_End"]))

if (start_positions < 0).any() or (end_positions < 0).any():
    raise ValueError("Missing event boundary dates in trading calendar.")

daily_changes = np.zeros(len(trading_calendar) + 1, dtype=int)
np.add.at(daily_changes, start_positions, 1)
np.add.at(daily_changes, end_positions + 1, -1)

pead_active_windows = pd.Series(
    daily_changes[:-1].cumsum(),
    index=trading_calendar,
    name="Active PEAD Windows",
)

pead_overlap_events["Overlap_Share"] = (
    pead_overlap_events["Overlap_Trading_Days"]
    / pead_overlap_events["Window_Length_Trading_Days"]
)

# Plot concurrent active PEAD windows through time using the trading-day calendar
fig, ax = plt.subplots(figsize=(7, 3.5))

ax.plot(
    pead_active_windows.index,
    pead_active_windows.values, # type: ignore
    linewidth=1.5,
    color="#4C78A8",
)

ax.axhline(
    pead_active_windows.median(),
    color="#E45756",
    linestyle="--",
    linewidth=2,
    label=f"Median: {pead_active_windows.median():.0f}",
)

ax.set_xlabel("Trading date")
ax.set_ylabel("Active PEAD event windows")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)

plt.tight_layout()
OUTPUTS.save_figure(fig, "concurrent_active_pead_windows_over_time")
plt.show()

overlap_summary_text = "\n".join(
    [
        f"Maximum concurrent PEAD windows: {pead_active_windows.max():,.0f}",
        f"Median concurrent PEAD windows: {pead_active_windows.median():,.0f}",
        f"PEAD event source: {pead_overlap_source}",
        f"Unique announcement windows analysed: {len(pead_overlap_events):,}",
        "Overlap count method: all-window starts/ends with numpy.searchsorted",
        "Overlap share method: exact all-instrument trading-day intersections within each event window",
        f"Share of windows overlapping another window across all instruments: {pead_overlap_events['Has_Overlap'].mean():.1%}",
        f"Median overlap count: {pead_overlap_events['Overlap_Count'].median():.0f}",
        f"Median overlap share: {pead_overlap_events['Overlap_Share'].median():.1%}",
    ]
)
print(overlap_summary_text)
OUTPUTS.save_text("overlap_summary", overlap_summary_text)

# pead_overlap_events[
#     [
#         "Instrument",
#         "Ann_Date",
#         "Window_Start",
#         "Window_End",
#         "Window_Length_Trading_Days",
#         "Overlap_Count",
#         "Overlap_Trading_Days",
#         "Overlap_Share",
#         "Has_Overlap",
#     ]
# ].head()
