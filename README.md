# PEAD on the London Stock Exchange

This repository contains the data-construction and empirical-analysis code for a bachelor thesis examining post-earnings-announcement drift (PEAD) among London Stock Exchange ordinary shares over 1990–2024. Earnings surprises are measured using standardized unexpected earnings (SUE), and post-announcement buy-and-hold abnormal returns (BHARs) are evaluated against locally constructed Fama–French-style 2×3 size–book-to-market benchmark portfolios, with additional analysis of time variation, firm size, analyst following, and robustness.

## Repository overview

The repository separates data construction from empirical analysis:

- `scripts/` contains the ordered data pipeline and additional standalone processes.
- `analysis/` contains thesis analyses grouped by numbered themes; it is not one sequential pipeline.
- `src/` contains reusable implementation used by scripts and analyses.
- `data/` contains the redistributable derived benchmark returns and placeholders for omitted proprietary inputs and generated outputs.
- `figures/` contains selected figures used in the thesis.
- `requirements.txt` records the Python environment used for the project.

The full workflow is not reproducible from this deposit alone. It requires authorised Compustat Global data obtained through WRDS and a running LSEG Workspace application with a correctly entitled user logged in so that `lseg.data` requests work. The underlying proprietary data are intentionally omitted. See [DATA_SOURCES.md](DATA_SOURCES.md) and [data/README.md](data/README.md).

Within `scripts/`, filenames beginning with numbers are main-pipeline stages and must be run in numerical order. Lettered filenames are separate supporting processes; files sharing a letter belong to the same process and are ordered by their following number. The `preprocessing_data/` scripts run before the main pipeline and are also ordered numerically. An `x` in a script stage name indicates that the stage requires an open LSEG Workspace session for data requests.

## Main data workflow

After installing the dependencies in `requirements.txt`, authorised users run the preprocessing stages followed by main stages `01`–`05`. `scripts/98_run_pipeline.py` is a retained draft runner and is currently commented out, so it is not the operational entry point.

### `scripts/preprocessing_data/`

- `01_XLON_ordinary_shares.py` — filters the Compustat Global security file to XLON ordinary shares.
- `02_june_window.py` — restricts Compustat observations to the annual June formation window.
- `10x_build_gbp_stock_universes.py` — maps identifiers through LSEG and writes formation-year GBP universe files.
- `11_collect_unique_gvkeys.py` — exports the unique GVKEY list needed for the WRDS equity query.
- `12_populate_gbp_constituents_with_equity.py` — joins the authorised Compustat book-equity extract to yearly constituents.

### `scripts/` — numbered main pipeline

- `01x_build_french_benchmarks.py` — constructs the six size–book-to-market benchmark portfolios and their daily value-weighted returns.
- `02_build_universe_and_market_data.py` — builds the filtered analysis universe and saves its market-data panel.
- `03x_build_earnings_and_sue.py` — requests earnings announcements and analyst forecasts and calculates SUE.
- `04_build_sue_groups.py` — assigns SUE groups using prior-year information.
- `05_build_abnormal_returns.py` — creates event windows and BHARs under the configured missing-return treatments.
- `98_run_pipeline.py` — disabled draft orchestration script retained for reference.

### `scripts/` — lettered standalone processes

- `A1x_compare_current_us_vs_xlon_thresholds.py` — requests current US and XLON samples and compares price and market-cap thresholds.
- `A2_current_exchange_threshold_comparison_from_csv.py` — produces the second-stage threshold comparison from saved A1 outputs.
- `Bx_inspect_earnings_release_frequencies.py` — requests and diagnoses earnings-report frequencies.
- `C_build_sample_size_aggregate.py` — aggregates formation-year sample-size records.
- `D1x_build_compustat_datastream_sample_sizes.py` — requests Datastream availability and constructs Compustat–Datastream sample-size inputs.
- `D2_compustat_datastream_sample_size_comparison.py` — summarizes and plots the D1 comparison.
- `J_outlier_stock_returns_diagnostics.py` — diagnoses extreme security returns and price-return inconsistencies.
- `K_outlier_portfolio_returns_diagnostics.py` — traces extreme benchmark returns to portfolio constituents.
- `L_outlier_BHAR_diagnostics.py` — inspects extreme event-level BHAR observations and their drivers.
- `M_report_log_market_cap_center.py` — reports the central value of log market capitalization used in interpretation.

## Analysis files

The number prefixes group related analyses; they do not prescribe a complete execution order.

- `_analysis_shared.py` — common output paths, naming, and table/figure writers.
- `00_export_return_price_mismatch_removals.py` — exports observations removed by the return-versus-price validation rule.
- `01a_universe.py` — plots sample construction and yearly attrition.
- `01b_universe_descriptives.py` — produces universe and earnings-sample descriptive statistics.
- `01c_benchmark_portfolios.py` — describes benchmark composition, breakpoints, and portfolio coverage.
- `02_announcement_dates.py` — analyzes the timing and weekly distribution of earnings announcements.
- `03_window_overlap.py` — measures overlap between event windows.
- `04a_SUE.py` — examines the distribution and time variation of SUE.
- `04b_SUE_group.py` — summarizes observations across SUE groups.
- `04c_analyst_following.py` — relates analyst coverage to firm and event characteristics.
- `05a_BHAR_normality.py` — evaluates BHAR normality and produces diagnostic plots/tests.
- `05b_BHAR_distribution_by_SUE_decile.py` — compares BHAR distributions across SUE deciles.
- `05c_BHAR_distributions.py` — summarizes BHAR distributions across event horizons.
- `05d_BHAR_horizon_summary_by_SUE_quintile.py` — reports horizon-specific BHAR results by SUE quintile.
- `06a_PEAD_agg.py` — produces aggregate PEAD estimates and plots.
- `07x_remediate_pre_announcement_market_caps.py` — requests missing pre-announcement market caps and updates the local cache.
- `08_regression_suite.py` — runs the main, alternative, and robustness regression specifications.
- `09a_time_period_diagnostics.py` — analyzes time-period heterogeneity and attenuation.
- `10_firm_fe_identification.py` — diagnoses identification in firm-fixed-effects specifications.

## Source package

### `src/core/`

- `__init__.py` — marks the core package.
- `pipeline_config.py` — central years, filters, thresholds, fields, and pipeline-version settings.
- `project_paths.py` — resolves repository and data locations, including `BACHELOR_THESIS_DATA_DIR`.
- `year_context.py` — defines all formation-year dates and expected file paths.
- `pipeline_state.py` — reads and writes versioned stage-completion metadata.
- `yearly_data_io.py` — shared loading and saving of yearly CSV/JSON data.
- `pead_sample_variants.py` — defines the supported earnings-event sample variants.

### `src/pead/`

- `__init__.py` — marks the PEAD package.
- `gbp_membership_files.py` — builds yearly GBP membership files and performs identifier mapping.
- `gbp_benchmark_builder.py` — cleans the GBP universe and prepares benchmark inputs.
- `universe_filters.py` — applies exchange, security, accounting, and data-quality filters.
- `market_data_fetch.py` — downloads and checkpoints LSEG price and return histories.
- `market_data_repairs.py` — detects and removes implausible return/price observations.
- `daily_market_cap_benchmarks.py` — obtains daily market caps and builds value-weighted benchmark returns.
- `french_benchmarks.py` — calculates breakpoints, assigns 2×3 portfolios, and constructs benchmark returns.
- `market_cap_splits.py` — creates market-cap breakpoints and size groups.
- `earnings_events.py` — requests earnings announcements and forecasts and calculates event-level SUE.
- `sue_groups.py` — assigns prior-year SUE groups and plotting groups.
- `abnormal_returns.py` — aligns event windows and calculates abnormal returns and BHAR inputs.

### `src/analysis/`

- `__init__.py` — marks the analysis package.
- `regression_suite.py` — regression specifications, estimation helpers, diagnostics, and table formatting.
- `time_varying_analysis.py` — shared sample filters and time-varying analysis utilities.
- `bhar_outlier_policy.py` — identifies, records, and applies BHAR outlier treatments.

### `src/tooling/` and `src/utils/`

- `src/tooling/__init__.py` — marks the tooling package.
- `src/tooling/aggregate_sample_size_all_years.py` — rebuilds the aggregate sample-size JSON from yearly files.
- `src/utils/__init__.py` — marks the utilities package.
- `src/utils/io_utils.py` — lightweight JSON input/output helpers.
- `src/utils/pandas_utils.py` — batching and LSEG history-column normalization helpers.
- `src/__init__.py` — marks `src` as the project source package.

## Data and figures

`data/benchmark_portfolio_returns_by_formation_year/` contains 35 annual CSV files, `benchmark_portfolio_returns_1990.csv` through `benchmark_portfolio_returns_2024.csv`. Each contains `Date` and the six Fama–French-style size–book-to-market portfolio return series (`SG`, `SN`, `SV`, `BG`, `BN`, and `BV`). Other expected data paths are documented in [data/README.md](data/README.md) but are absent because they contain proprietary inputs or outputs derived at a level unsuitable for redistribution.

### `figures/Cover/`

- `logouk.pdf` — cover-page institutional logo asset.

### `figures/Data/`

- `analyst_following_per_announcement_before_filtering_histogram.png` — analyst-coverage distribution before final filtering.
- `compustat_vs_datastream_historical_sample_sizes.png` — historical provider sample-size comparison.
- `cumulative_distribution_plot.png` — cumulative threshold-distribution comparison.
- `distribution_of_earnings_announcements_per_eligible_firm.png` — announcements per eligible firm.
- `final_earnings_sample_book_to_market_at_formation_histogram.png` — formation-date book-to-market distribution.
- `final_earnings_sample_firm_size_by_year_boxplots.png` — yearly firm-size distributions.
- `final_earnings_sample_market_cap_at_formation_histogram.png` — formation-date market-cap distribution.
- `final_earnings_sample_price_at_formation_histogram.png` — formation-date price distribution.
- `firm_size_by_analyst_following_boxplots.png` — firm size by analyst-following group.
- `weekly_distribution_of_earnings_announcement_dates.png` — weekly announcement-date distribution.
- `yearly_sample_size_plot.png` — sample size by formation year.

### `figures/Results/`

- `benchmark_portfolio_constituent_dot_range_plot.png` — annual constituent-count ranges by benchmark portfolio.
- `bhar_2_20_distribution_p5_p95.png` — central BHAR(2,20) distribution.
- `bhar_2_20_normal_qq_plot.png` — BHAR(2,20) normal Q–Q diagnostic.
- `distribution_of_sue.png` — SUE distribution.
- `log_market_cap_by_sue_quintile_boxplots.png` — log market capitalization by SUE quintile.
- `median_market_cap_vs_bm_by_french_benchmark_portfolio.png` — benchmark portfolio size and book-to-market characteristics.
- `sue_by_analyst_following.png` — SUE by analyst coverage.
- `sue_by_formation_year.png` — SUE over formation years.

Generated `__pycache__/` directories and `.pyc` files are not part of the documented research workflow.
