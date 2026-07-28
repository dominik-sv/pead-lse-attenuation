# Data directory

Most of this directory is intentionally absent from the submitted repository because it contains proprietary Compustat Global or LSEG Workspace/Datastream source data and detailed derived records. Only the annual Fama–French-style 2×3 size–book-to-market benchmark portfolio returns are included.

See [../DATA_SOURCES.md](../DATA_SOURCES.md) for source and access details.

## Included data

`benchmark_portfolio_returns_by_formation_year/` contains one CSV for each formation year from 1990 through 2024. Files are named `benchmark_portfolio_returns_<year>.csv` and contain:

- `Date` — trading date.
- `SG`, `SN`, `SV` — small-growth, small-neutral, and small-value portfolio returns.
- `BG`, `BN`, `BV` — big-growth, big-neutral, and big-value portfolio returns.

## Expected working structure

The following abbreviated tree shows the principal licensed inputs and generated outputs used by the code. It is documentation, not a claim that these files are included.

```text
data/
├── XLON_membership/
│   ├── compustat_returns.csv                 # licensed WRDS/Compustat input
│   ├── compustat_filtered.csv                # preprocessing stage 01 output
│   └── compustat_filtered_june.csv           # preprocessing stage 02 output
├── equity/
│   ├── compustat_global_unique_gvkeys.csv    # stage 11 WRDS query list
│   ├── compustat_global_unique_gvkeys.txt    # stage 11 WRDS upload list
│   └── compustat_equity.csv                  # licensed WRDS/Compustat input
├── yearly/
│   └── <formation_year>/
│       ├── gbp_universe/
│       │   ├── constituents_<year>.csv
│       │   └── sedols_<year>.txt
│       ├── stock_universe.csv
│       ├── market_data.csv
│       ├── benchmark_portfolio_returns.csv
│       ├── benchmark_breakpoints.json
│       ├── earnings_events.csv
│       ├── earnings_events_full.csv
│       ├── earnings_abnormal_returns.csv
│       ├── earnings_abnormal_returns_drop_missing.csv
│       ├── earnings_abnormal_returns_terminal_loss.csv
│       ├── missing_return_fill_summary.json
│       ├── abnormal_return_failures.json
│       ├── sample_size.json
│       └── _cache/                            # checkpoints and detailed vendor-derived data
├── outputs/                                   # analysis tables, figures, and diagnostics
├── sample_size_all_years.json
└── benchmark_portfolio_returns_by_formation_year/
    └── benchmark_portfolio_returns_<1990-2024>.csv
```

## Recreation by an authorised reviewer

1. Obtain the required Compustat Global security data through an authorised WRDS account and save the initial extract as `data/XLON_membership/compustat_returns.csv`.
2. Run `scripts/preprocessing_data/01_XLON_ordinary_shares.py` and `02_june_window.py`.
3. Start LSEG Workspace, log in with the required entitlements, verify that `lseg.data` can open a session, and run `10x_build_gbp_stock_universes.py`.
4. Run `11_collect_unique_gvkeys.py`, use its identifiers to obtain authorised Compustat common-equity data, save that extract as `data/equity/compustat_equity.csv`, and run stage `12`.
5. From the repository root, run main pipeline scripts `01x`, `02`, `03x`, `04`, and `05` in that order. Stages marked `x` require LSEG Workspace.
6. Run the relevant files in `analysis/` to recreate analysis outputs. These files are grouped by theme rather than forming a single ordered pipeline.

The data root can be redirected by setting `BACHELOR_THESIS_DATA_DIR`; otherwise the code uses this `data/` directory.
