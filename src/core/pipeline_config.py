FORMATION_YEARS = list(range(1990, 2025))

# -----------------------------
# Market / universe settings
# -----------------------------
CURRENCY = "USD"
ANALYSIS_EXCHANGE = "XLON"

# Universe construction starts from one explicit interim candidate source:
# a broad non-active Workspace exchange snapshot around formation date, followed
# by security-definition and PEAD accounting/market filters downstream. This is
# intentionally isolated so it can later be replaced by a proper historical
# exchange-population endpoint without redesigning the rest of the pipeline.
UNIVERSE_SOURCE = "00_gbp_constituent_universe_enriched_via_lseg"

# -----------------------------
# Sample filters
# -----------------------------
MARKET_CAP_THRESHOLD = 0.5      # millions
STOCK_PRICE_THRESHOLD = 0.0025     # USD
MARKET_CAP_SIZE_SPLIT_PERCENTILE = 0.50
IMPLAUSIBLE_SAME_SIGN_SHARE_THRESHOLD = 0.98
ZERO_RETURN_SHARE_THRESHOLD = 0.95
# Daily return histories are stored in percentage-point units throughout the
# pipeline (for example, +1.5 means +1.5%). These volatility thresholds must
# therefore also be expressed in percentage-point units.
HIGH_VOLATILITY_STD_THRESHOLD_PCT = 40.0
LOW_VOLATILITY_STD_THRESHOLD_PCT = 1e-4
# Used by the section-10 visual analysis notebooks. With the currently saved
# base metadata, choose one of the recorded deciles: 0.10, 0.20, ..., 0.90.
MARKET_CAP_ANALYSIS_SPLIT_PERCENTILE = MARKET_CAP_SIZE_SPLIT_PERCENTILE

ORDINARY_SHARE_TYPES = (
    "Ordinary Shares",
    "Fully Paid Ordinary Shares",
)

# -----------------------------
# API / download settings
# -----------------------------
INSTRUMENT_BATCH_SIZE = 5
SLEEP_BTWN_PULLS = 0.25
UNIVERSE_ENRICHMENT_BATCH_SIZE = 250
EARNINGS_REQUEST_INSTRUMENT_LIMIT = 400

# -----------------------------
# Data paths
# -----------------------------
BASE_DATA_DIR = "data"
YEARLY_DATA_DIRNAME = "yearly"
GBP_MEMBERSHIP_DIRNAME = "XLON_membership"
GBP_MEMBERSHIP_LEGACY_DIRNAME = "xlon_membership"
GBP_UNIVERSE_SUBDIR_NAME = "gbp_universe"
GBP_CONSTITUENTS_FILE_NAME = "gbp_constituents.csv"
GBP_IDENTIFIERS_FILE_NAME = "WRDS_data.csv"
GBP_COMPUSTAT_MONTHLY_RETURNS_FILE_NAME = "compustat_monthly_returns.csv"
GBP_COMPUSTAT_MONTHLY_RETURNS_LEGACY_FILE_NAME = "WRDS_data.csv"
GBP_CONSTITUENT_FILE_TEMPLATE = "constituents_{year}.csv"
GBP_SEDOL_FILE_TEMPLATE = "sedols_{year}.txt"

# -----------------------------
# Pipeline versions
# -----------------------------
# Incremented because the base-universe stage now:
# 1. uses the post-cleaning GBP universe produced by the benchmark preprocessing
#    stage instead of repeating the canonical venue/security/accounting filters,
# 2. keeps the canonical universe-screen sample-size labels owned by that
#    preprocessing stage,
# 3. applies only analysis-specific return-data and threshold filters in the
#    base analysis stage.
BASE_PIPELINE_VERSION = (
    "pead_base_universe_and_market_data_v19_cleaned_gbp_universe_analysis_filters"
)
SUE_BASE_PIPELINE_VERSION = "earnings_and_sue_base_v6_parallel_main_and_min1_outputs"
SUE_GROUPS_PIPELINE_VERSION = "earnings_sue_groups_v2_parallel_main_and_min1_outputs"
ABNORMAL_RETURNS_PIPELINE_VERSION = (
    "earnings_abnormal_returns_v19_post_announcement_20_40_60_outputs"
)

# -----------------------------
# Earnings event settings
# -----------------------------
EARNINGS_RELEASE_FREQUENCIES = {
    "FY": "Annual earnings report",
}

SUE_EARNINGS_FREQUENCIES = {
    "FY": "Annual earnings report",
}

# -----------------------------
# Analyst forecast / SUE settings
# -----------------------------
FORECAST_LOOKBACK_DAYS = 90
MIN_ANALYST_FORECASTS_FOR_SUE = 3
DETAILED_EPS_ESTIMATE_FIELD = "TR.EPSEstValue"

# Configurable absolute forecast-period templates by earnings frequency.
# The earnings pipeline tries candidate periods in the listed order and keeps
# the first one that yields usable in-window analyst forecasts.
FORECAST_PERIOD_TEMPLATES_BY_FREQUENCY = {
    "FY": (
        "FY{announcement_year_minus_1}",
        "FY{announcement_year}",
    )
}

SUE_COMPUTATION_GROUP_COUNT = 5
SUE_PLOT_GROUP_COUNT = 5

# -----------------------------
# BHAR analysis filters
# -----------------------------
BHAR_EARNINGS_FREQUENCIES = tuple(SUE_EARNINGS_FREQUENCIES.keys())
PRE_ANNOUNCEMENT_WINDOW_LENGTH = 20
POST_ANNOUNCEMENT_MISSING_RETURN_FILL_VALUE = 0.0

# Regression-suite BHAR / SUE settings
MAIN_REGRESSION_BHAR_WINDOW = (2, 20)
ALTERNATIVE_REGRESSION_BHAR_WINDOWS = (
    (2, 40),
    (2, 60),
)
ALTERNATIVE_REGRESSION_SUE_GROUP_COUNTS = (
    10,
    2,
)
UNBIASEDNESS_ANNOUNCEMENT_WINDOW = (0, 1)
UNBIASEDNESS_FULL_WINDOW = (0, 20)

# -----------------------------
# Targeted market-data settings
# -----------------------------
TARGETED_PRICE_BATCH_SIZE = 40
TARGETED_PRICE_INTERVAL_DAYS = 15
TARGETED_RETURN_BATCH_SIZE = 40

# -----------------------------
# Market-data repair settings
# -----------------------------
POSITIVE_OUTLIER_RETURN_THRESHOLD_PCT = 100.0
NEGATIVE_OUTLIER_RETURN_THRESHOLD_PCT = -50.0
PRICE_RETURN_MISMATCH_TOLERANCE_PCT_POINTS = 50.0


# -----------------------------
# Plotting settings
# -----------------------------
COLOR_PALETTE = "plasma"
