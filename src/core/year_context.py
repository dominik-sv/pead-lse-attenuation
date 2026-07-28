from dataclasses import dataclass
from pathlib import Path
import datetime as dt

from .pipeline_config import PRE_ANNOUNCEMENT_WINDOW_LENGTH
from .project_paths import (
    resolve_yearly_data_dir,
    year_gbp_constituents_path,
    year_gbp_sedols_path,
    year_gbp_universe_dir,
)


@dataclass(frozen=True)
class YearContext:
    year: int
    year_dir: Path
    formation_date: str
    universe_window_start: str
    universe_window_end: str
    last_fiscal_year_end: str
    market_data_window_start: str
    market_data_window_end: str
    earnings_window_end: str

    @property
    def stock_universe_path(self) -> Path:
        return self.year_dir / "stock_universe.csv"

    @property
    def earnings_events_path(self) -> Path:
        return self.year_dir / "earnings_events.csv"

    @property
    def earnings_events_full_path(self) -> Path:
        return self.year_dir / "earnings_events_full.csv"

    @property
    def earnings_abnormal_returns_path(self) -> Path:
        return self.year_dir / "earnings_abnormal_returns.csv"

    @property
    def earnings_abnormal_returns_drop_missing_path(self) -> Path:
        return self.year_dir / "earnings_abnormal_returns_drop_missing.csv"

    @property
    def earnings_abnormal_returns_terminal_loss_path(self) -> Path:
        return self.year_dir / "earnings_abnormal_returns_terminal_loss.csv"

    @property
    def missing_return_fill_summary_path(self) -> Path:
        return self.year_dir / "missing_return_fill_summary.json"

    @property
    def abnormal_return_failures_path(self) -> Path:
        return self.year_dir / "abnormal_return_failures.json"

    @property
    def market_data_path(self) -> Path:
        return self.year_dir / "market_data.csv"

    @property
    def price_windows_path(self) -> Path:
        return self.cache_dir / "price_windows.csv"

    @property
    def return_windows_path(self) -> Path:
        return self.cache_dir / "return_windows.csv"

    @property
    def benchmark_returns_path(self) -> Path:
        return self.year_dir / "benchmark_portfolio_returns.csv"

    @property
    def benchmark_breakpoints_path(self) -> Path:
        return self.year_dir / "benchmark_breakpoints.json"

    @property
    def enriched_gbp_universe_path(self) -> Path:
        return self.cache_dir / "enriched_gbp_universe.csv"

    @property
    def benchmark_constituents_path(self) -> Path:
        return self.cache_dir / "benchmark_portfolio_constituents.csv"

    @property
    def benchmark_return_windows_path(self) -> Path:
        return self.cache_dir / "benchmark_return_windows.csv"

    @property
    def benchmark_market_data_path(self) -> Path:
        return self.cache_dir / "benchmark_market_data.csv"

    @property
    def shared_post_cleaning_universe_path(self) -> Path:
        return self.cache_dir / "shared_post_cleaning_universe.csv"

    @property
    def shared_market_data_path(self) -> Path:
        return self.cache_dir / "shared_market_data.csv"

    @property
    def sample_size_path(self) -> Path:
        return self.year_dir / "sample_size.json"

    @property
    def final_exchange_validation_audit_path(self) -> Path:
        return self.year_dir / "final_exchange_validation_audit.json"

    @property
    def benchmark_output_paths(self) -> tuple[Path, ...]:
        return (
            self.benchmark_returns_path,
            self.benchmark_breakpoints_path,
            self.enriched_gbp_universe_path,
            self.shared_post_cleaning_universe_path,
            self.shared_market_data_path,
            self.benchmark_constituents_path,
            self.benchmark_return_windows_path,
            self.benchmark_market_data_path,
        )

    @property
    def base_output_paths(self) -> tuple[Path, ...]:
        return (
            self.stock_universe_path,
            self.market_data_path,
            self.sample_size_path,
        )

    @property
    def sue_output_paths(self) -> tuple[Path, ...]:
        return (
            self.earnings_events_path,
            self.earnings_events_full_path,
            self.sample_size_path,
            self.price_windows_path,
        )

    @property
    def abnormal_returns_output_paths(self) -> tuple[Path, ...]:
        return (
            self.earnings_abnormal_returns_path,
            self.earnings_abnormal_returns_drop_missing_path,
            self.earnings_abnormal_returns_terminal_loss_path,
            self.missing_return_fill_summary_path,
            self.abnormal_return_failures_path,
            self.return_windows_path,
        )

    @property
    def earnings_output_paths(self) -> tuple[Path, ...]:
        return (
            self.earnings_events_path,
            self.earnings_abnormal_returns_path,
            self.missing_return_fill_summary_path,
            self.sample_size_path,
            self.price_windows_path,
            self.return_windows_path,
        )

    @property
    def cache_dir(self) -> Path:
        return self.year_dir / "_cache"

    @property
    def gbp_universe_dir(self) -> Path:
        return year_gbp_universe_dir(self.year)

    @property
    def gbp_constituents_path(self) -> Path:
        return year_gbp_constituents_path(self.year)

    @property
    def gbp_sedols_path(self) -> Path:
        return year_gbp_sedols_path(self.year)

    @property
    def market_data_chunks_dir(self) -> Path:
        return self.cache_dir / "market_data_chunks"

    @property
    def market_data_checkpoint_path(self) -> Path:
        return self.cache_dir / "market_data_checkpoint.json"

    @property
    def market_data_errors_path(self) -> Path:
        return self.cache_dir / "market_data_errors.json"

    @property
    def earnings_release_events_path(self) -> Path:
        return self.cache_dir / "earnings_release_events.csv"

    @property
    def base_complete_path(self) -> Path:
        return self.cache_dir / "base_complete.json"

    @property
    def benchmark_complete_path(self) -> Path:
        return self.cache_dir / "benchmark_complete.json"

    @property
    def sue_complete_path(self) -> Path:
        return self.cache_dir / "sue_complete.json"

    @property
    def sue_groups_complete_path(self) -> Path:
        return self.cache_dir / "sue_groups_complete.json"

    @property
    def earnings_complete_path(self) -> Path:
        return self.cache_dir / "earnings_complete.json"


def build_year_context(year: int, base_data_dir: str | Path) -> YearContext:
    formation_dt = dt.datetime(year, 6, 30)
    benchmark_cycle_start_dt = dt.datetime(year, 7, 1)
    # Shared market-data window must cover pre-announcement lagged-price lookups
    # plus late-cycle earnings announcements and the full 90-trading-day
    # post-announcement BHAR window.
    pre_announcement_buffer_days = max(int(PRE_ANNOUNCEMENT_WINDOW_LENGTH), 120)
    post_announcement_buffer_days = 150
    yearly_data_dir = resolve_yearly_data_dir(base_data_dir)

    return YearContext(
        year=year,
        year_dir=yearly_data_dir / str(year),
        formation_date=formation_dt.strftime("%Y-%m-%d"),
        universe_window_start=formation_dt.strftime("%Y-%m-%d"),
        universe_window_end=formation_dt.strftime("%Y-%m-%d"),
        last_fiscal_year_end=dt.datetime(year - 1, 12, 31).strftime("%Y-%m-%d"),
        market_data_window_start=(
            benchmark_cycle_start_dt - dt.timedelta(days=pre_announcement_buffer_days)
        ).strftime("%Y-%m-%d"),
        market_data_window_end=(
            dt.datetime(year + 1, 7, 1)
            + dt.timedelta(days=post_announcement_buffer_days)
        ).strftime("%Y-%m-%d"),
        earnings_window_end=dt.datetime(year + 1, 7, 1).strftime("%Y-%m-%d"),
    )
