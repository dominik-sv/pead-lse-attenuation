import matplotlib.pyplot as plt
plt.style.use("ggplot")
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.pipeline_config import SUE_COMPUTATION_GROUP_COUNT
from src.pead.sue_groups import (
    SUE_GROUP_COLUMN,
    get_extreme_group_values,
)
from src.analysis.time_varying_analysis import collapse_to_event_level, load_abnormal_returns_with_groups

from _analysis_shared import AnalysisOutputManager, DATA_DIR
OUTPUTS = AnalysisOutputManager(__file__)


abnormal_returns = load_abnormal_returns_with_groups(DATA_DIR)
event_level = collapse_to_event_level(abnormal_returns)

available_years = sorted(event_level["Formation_Year"].dropna().astype(int).unique().tolist())

bottom_group, top_group = get_extreme_group_values(SUE_COMPUTATION_GROUP_COUNT)
normality_samples = {
    f"Bottom group ({bottom_group})": event_level.loc[event_level[SUE_GROUP_COLUMN] == bottom_group, "PEAD"].dropna().astype(float),
    f"Top group ({top_group})": event_level.loc[event_level[SUE_GROUP_COLUMN] == top_group, "PEAD"].dropna().astype(float),
}

fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))

for row_index, (label, sample) in enumerate(normality_samples.items()):
    hist_ax = axes[row_index, 0]
    qq_ax = axes[row_index, 1]

    hist_ax.hist(sample, bins=50, range=(-100, 200), color="#4c78a8", edgecolor="white", alpha=0.9)
    hist_ax.axvline(sample.mean(), color="#d62728", linestyle="--", linewidth=1.5, label="Mean")
    hist_ax.set_xlabel("BHAR(2,60)")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_xlim(-100, 200)
    hist_ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    hist_ax.legend(frameon=False)

    stats.probplot(sample, dist="norm", plot=qq_ax)

fig.tight_layout()
OUTPUTS.save_figure(fig, "pead_distribution_and_qq_plots")
plt.show()


def run_normality_test(sample: pd.Series, label: str) -> dict[str, object]:
    skew_val = stats.skew(sample, nan_policy="omit")
    kurt_val = stats.kurtosis(sample, fisher=True, nan_policy="omit")

    shapiro_res = stats.shapiro(sample)
    dag_res = stats.normaltest(sample, nan_policy="omit")

    return {
        "Sample": label,
        "N": int(len(sample)),
        "Mean": round(sample.mean(), 3),
        "Std": round(sample.std(ddof=1), 3),
        "Skewness": round(float(skew_val), 3),
        "Excess Kurtosis": round(float(kurt_val), 3),
        "Shapiro-Wilk p-val": round(shapiro_res.pvalue, 3),
        "D'Agostino p-val": round(dag_res.pvalue, 3),
    }

normality_summary = pd.DataFrame(
    [run_normality_test(sample, label) for label, sample in normality_samples.items()]
)
normality_summary
OUTPUTS.save_table(normality_summary, "normality_summary")


