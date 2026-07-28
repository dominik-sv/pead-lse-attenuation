from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PEADEventSampleVariant:
    key: str
    label: str
    min_analyst_forecasts: int
    earnings_events_filename: str
    abnormal_returns_filename: str
    abnormal_returns_drop_missing_filename: str
    abnormal_returns_terminal_loss_filename: str
    missing_return_fill_summary_filename: str
    abnormal_return_failures_filename: str
    sample_size_suffix: str

    def earnings_events_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.earnings_events_filename

    def abnormal_returns_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.abnormal_returns_filename

    def abnormal_returns_drop_missing_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.abnormal_returns_drop_missing_filename

    def abnormal_returns_terminal_loss_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.abnormal_returns_terminal_loss_filename

    def missing_return_fill_summary_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.missing_return_fill_summary_filename

    def abnormal_return_failures_path(self, year_dir: Path) -> Path:
        return Path(year_dir) / self.abnormal_return_failures_filename


MAIN_PEAD_SAMPLE = PEADEventSampleVariant(
    key="main",
    label="Main PEAD sample",
    min_analyst_forecasts=3,
    earnings_events_filename="earnings_events.csv",
    abnormal_returns_filename="earnings_abnormal_returns.csv",
    abnormal_returns_drop_missing_filename="earnings_abnormal_returns_drop_missing.csv",
    abnormal_returns_terminal_loss_filename="earnings_abnormal_returns_terminal_loss.csv",
    missing_return_fill_summary_filename="missing_return_fill_summary.json",
    abnormal_return_failures_filename="abnormal_return_failures.json",
    sample_size_suffix="",
)

MIN1_PEAD_SAMPLE = PEADEventSampleVariant(
    key="min1",
    label="One-forecast PEAD sample",
    min_analyst_forecasts=1,
    earnings_events_filename="earnings_events_min1.csv",
    abnormal_returns_filename="earnings_abnormal_returns_min1.csv",
    abnormal_returns_drop_missing_filename="earnings_abnormal_returns_drop_missing_min1.csv",
    abnormal_returns_terminal_loss_filename="earnings_abnormal_returns_terminal_loss_min1.csv",
    missing_return_fill_summary_filename="missing_return_fill_summary_min1.json",
    abnormal_return_failures_filename="abnormal_return_failures_min1.json",
    sample_size_suffix=" (min1)",
)

PEAD_EVENT_SAMPLE_VARIANTS = (
    MAIN_PEAD_SAMPLE,
    MIN1_PEAD_SAMPLE,
)


def get_pead_event_sample_variant(key: str) -> PEADEventSampleVariant:
    normalized_key = str(key).strip().lower()
    for variant in PEAD_EVENT_SAMPLE_VARIANTS:
        if variant.key == normalized_key:
            return variant
    raise KeyError(f"Unsupported PEAD event sample variant: {key!r}")
