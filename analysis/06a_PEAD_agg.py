from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
LOCAL_PACKAGE_DIRS = [
    PROJECT_ROOT / ".python_packages_local",
    PROJECT_ROOT / ".python_packages",
]
for package_dir in LOCAL_PACKAGE_DIRS:
    if package_dir.exists() and str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from IPython.display import display
from matplotlib.ticker import PercentFormatter

from src.core.pipeline_config import COLOR_PALETTE, SUE_COMPUTATION_GROUP_COUNT
from src.pead.sue_groups import SUE_GROUP_COLUMN
from src.analysis.time_varying_analysis import collapse_to_event_level, load_abnormal_returns_with_groups, summarize_by_group
from _analysis_shared import AnalysisOutputManager


DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS = AnalysisOutputManager(__file__)
GROUP_COLUMN = SUE_GROUP_COLUMN
GROUP_COUNT = SUE_COMPUTATION_GROUP_COUNT
GROUP_LABEL = "SUE decile"

abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(abnormal_returns)


group_summary = summarize_by_group(event_level, group_column=GROUP_COLUMN)
display(group_summary)
OUTPUTS.save_table(group_summary, "group_summary")


fig, ax = plt.subplots(figsize=(7, 4.2))

colors = plt.get_cmap(COLOR_PALETTE)(np.linspace(0.08, 0.92, len(group_summary)))
ax.bar(
    group_summary[GROUP_COLUMN],
    group_summary["PEAD_Mean"],
    yerr=1.96 * group_summary["PEAD_SE"],
    capsize=4,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel(GROUP_LABEL)
ax.set_ylabel("Mean BHAR(2,60)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.set_xticks(range(1, GROUP_COUNT + 1))
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
OUTPUTS.save_figure(fig, "mean_pead_by_sue_decile")
plt.close(fig)


