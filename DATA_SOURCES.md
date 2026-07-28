# Data sources and access

This repository intentionally does **not** redistribute the underlying LSEG, Compustat Global, or WRDS source files. They were accessed for academic research under institutional subscriptions and must be obtained independently by an authorised reviewer under the terms of the reviewer's own institution. No credentials, API tokens, raw vendor extracts, or account configuration are included.

## Source inventory

| Source | Use in the thesis | Access required | Retrieval record |
| --- | --- | --- | --- |
| LSEG Workspace / Datastream, accessed with the `lseg.data` Python library | Historical prices and returns, market capitalization, security and identifier information, earnings announcements, and detailed analyst EPS forecasts | A running LSEG Workspace application; a logged-in, correctly entitled account; and working Python-library access | Exact request dates are not preserved in this distributable repository. Responses were collected during thesis data construction and cached only in the non-distributed working data directory. |
| Compustat Global through WRDS | Exchange/security classification, monthly security observations, identifiers, and annual common equity (`ceq`) used for book-to-market ratios | An institutional WRDS subscription with Compustat Global entitlement; authorised manual queries/downloads | Exact WRDS download dates are not preserved in this distributable repository. The required extracts must be recreated by an authorised user. |
| Research-derived 2×3 size–book-to-market benchmark returns | Daily returns for six locally constructed Fama–French-style portfolios, by formation year from 1990 through 2024 | Included in `data/benchmark_portfolio_returns_by_formation_year/`; further use remains subject to the applicable source-data agreements | Packaged for thesis submission on 28 July 2026. This packaging date is not necessarily the original LSEG retrieval date. |

## Non-redistribution statement

The files needed to reconstruct the security universe, firm-level market-data panels, earnings events, analyst forecasts, book equity, and event-level abnormal returns are proprietary and are not included. The repository provides code describing the transformations, expected paths, and selected derived benchmark return series. Inclusion of a derived file does not grant permission for further redistribution; users remain responsible for complying with LSEG, WRDS/Compustat, and institutional licence terms.

Authorised reviewers can recreate the omitted structure by following [data/README.md](data/README.md), supplying their own licensed inputs, opening an entitled LSEG Workspace session for stages marked `x`, and running the preprocessing and main pipeline stages in numerical order.
