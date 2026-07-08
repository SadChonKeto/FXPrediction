"""
FXPRED Configuration
====================
Non-sensitive project parameters for GEL/USD exchange rate forecasting.
All API keys / secrets belong in .env (see .env.example).
"""
import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "thesis_figures"

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# Date range for data collection
DATA_START_DATE = "2000-01-01"
DATA_END_DATE = "2025-06-30"  # Update as needed

# NBG API base URL (no authentication required)
NBG_API_BASE = "https://nbg.gov.ge/gw/api/ct/monetarypolicy"

# LBMA precious metals price endpoints (public, no auth)
LBMA_GOLD_URL = "https://prices.lbma.org.uk/json/gold_am.json"
LBMA_SILVER_URL = "https://prices.lbma.org.uk/json/silver.json"

# FRED series codes
FRED_SERIES = ["DFF", "TB3MS", "DTB3", "DGS3MO"]
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# yfinance tickers for daily data
YFINANCE_TICKERS = {
    "gold_futures": "GC=F",
    "oil_futures": "CL=F",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "us_10y_yield": "^TNX",
    "move_index": "^MOVE",
}

# Currencies scraped from NBG
CURRENCY_CODES = ["USD", "EUR", "RUB", "TRY", "GBP"]

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# Train/test split ratio
TRAIN_RATIO = 0.80

# Random seed for reproducibility
SEED = 42
TSMIXER_SEED = 11  # Used in Stage 1 (TSMixer)

# TSMixer (Stage 1)
TSMIXER_WINDOW_SIZES = [12, 24, 36]
TSMIXER_HORIZONS = [16, 24, 32, 48, 64]
TSMIXER_EPOCHS = 150
TSMIXER_BATCH_SIZE = 64
TSMIXER_LR = 1e-3
TSMIXER_DROPOUT = 0.2
TSMIXER_N_BLOCKS = 4
TSMIXER_D_FF = 128

# XGBoost (Stage 2)
XGB_N_FOLDS = 5
XGB_HYPEROPT_EVALS = 200

# Probit Direction Model (Stage 3)
PROBIT_FEATURES = [
    "direction_lag1",
    "direction_streak3",
    "tibR3M_last_lag1",
    "dollarization_ratio_lag1",
    "gold_24_usd_oz_mean_lag1",
    "shock",
]

# ARIMA Magnitude (Stage 3)
ARIMA_P_RANGE = range(4)
ARIMA_D_RANGE = range(2)
ARIMA_Q_RANGE = range(4)

# Ridge Regression (Stage 4)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Feature specifications for Stage 4 robustness analysis
FEATURES_4 = [
    "direction_lag1",
    "direction_streak3",
    "dollarization_ratio_lag1",
    "gold_24_usd_oz_mean_lag1",
]

FEATURES_6 = FEATURES_4 + [
    "tibR3M_last_lag1",
    "shock",
]

FEATURES_14 = FEATURES_6 + [
    "direction_lag2",
    "direction_streak5",
    "fx_rate_formatted_mean_USD_diff_lag1",
    "fx_rate_formatted_mean_USD_diff_lag2",
    "rate_diff_lag1",
    "nbg_interventions_usd_lag1",
    "Total_rem_lag1",
    "headline_yoy_lag1",
]

# Multi-Horizon (Stage 5)
FORECAST_HORIZONS = [1, 3, 6, 9, 12]
