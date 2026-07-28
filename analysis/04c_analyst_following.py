from pathlib import Path
import sys

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd

from _analysis_shared import AnalysisOutputManager


PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
OUTPUTS = AnalysisOutputManager(__file__)


def load_earnings_events() -> pd.DataFrame:
    earnings_files = sorted(
        YEARLY_DATA_DIR.glob("[0-9][0-9][0-9][0-9]/earnings_events.csv")
    )

    if not earnings_files:
        raise FileNotFoundError(
            "No yearly earnings_events.csv files found under data/yearly/<year>/. "
            "Run scripts/03x_build_earnings_and_sue.py first."
        )

    frames: list[pd.DataFrame] = []
    for path in earnings_files:
        frame = pd.read_csv(path)
        frame["Formation_Year"] = path.parent.name
        frames.append(frame)

    earnings_events = pd.concat(frames, ignore_index=True)
    earnings_events["Forecast_Analyst_Count"] = pd.to_numeric(
        earnings_events["Forecast_Analyst_Count"],
        errors="coerce",
    )
    return earnings_events


earnings_events = load_earnings_events()

plot_data = earnings_events.dropna(subset=["Forecast_Analyst_Count"]).copy()
plot_data["Forecast_Analyst_Count"] = plot_data["Forecast_Analyst_Count"].astype(int)
plot_data = plot_data.loc[plot_data["Forecast_Analyst_Count"] >= 0].copy()

if plot_data.empty:
    raise ValueError("No valid analyst-following observations were found.")

counts = (
    plot_data["Forecast_Analyst_Count"]
    .value_counts()
    .sort_index()
)
all_analyst_counts = np.arange(int(counts.index.min()), int(counts.index.max()) + 1)
counts = counts.reindex(all_analyst_counts, fill_value=0)
cumulative_counts = counts.cumsum()

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(
    all_analyst_counts,
    cumulative_counts.values,
    width=0.9,
    color="#9ecae1",
    edgecolor="#3182bd",
    linewidth=0.7,
)
ax.set_xlabel("Analysts following")
ax.set_ylabel("Cumulative events with that many analysts or fewer")
ax.set_xticks(all_analyst_counts)
ax.grid(axis="y", alpha=0.2)

if len(all_analyst_counts) > 25:
    tick_step = max(1, int(np.ceil(len(all_analyst_counts) / 20)))
    ax.set_xticks(all_analyst_counts[::tick_step])

fig.tight_layout()
OUTPUTS.save_figure(fig, "cumulative_analyst_following_histogram")
plt.close(fig)
