# FXPRED — Forecasting GEL/USD Exchange Rate Using Machine Learning

**Bachelor Thesis Project**  
Author: Ketevan Machavariani  
Institution: ISET
Platform: Databricks (Azure), Runtime 15.4

---

## Overview

This project investigates whether machine learning and econometric methods can
forecast the Georgian Lari (GEL) to US Dollar exchange rate out-of-sample,
benchmarked against the random walk (Meese & Rogoff, 1983).

The research proceeds through **five progressive stages**, each building on
the lessons of the previous:

| Stage | Method | Result |
|-------|--------|--------|
| 1 | TSMixer (deep learning, daily) | Does NOT beat random walk |
| 2 | XGBoost (monthly level & diff) | Does NOT beat random walk |
| 3 | Direction-Magnitude Decomposition | 75% directional accuracy but break-even = 82% |
| 4 | Direct Ridge on signed Δy | CW p=0.058 (4-feat), marginal significance |
| 5 | Multi-horizon iterated (h=1–12) | **CW p=0.016 (h=9), p=0.0085 (h=12)** |

**Key finding**: Macro fundamentals (dollarization, gold, Georgian rates) predict
GEL/USD at horizons of 9–12 months with strong statistical significance, consistent
with Engel & West (2005).

---

## Directory Structure

```
FXPRED/
├── README.md                 ← This file
├── requirements.txt          ← Python dependencies
├── .env.example              ← Template for API keys (FRED)
├── config.py                 ← All non-sensitive configuration
├── data/
│    ├── dt.zip
│       ├── data/
│           ├── raw/                  ← Output from scraping notebooks
│           └── processed/            ← xgboost_monthly_data.csv (model input)
├── scraping/
│   ├── 01_scrape_nbg         ← NBG exchange rates & monetary data
│   ├── 02_scrape_lbma        ← Gold & silver prices (LBMA)
│   ├── 03_scrape_fred        ← US interest rates (FRED API)
│   ├── 04_scrape_yfinance    ← Daily financial data (Yahoo Finance)
│   └── 05_build_monthly_dataset ← Merge all sources → monthly panel
└── notebooks/
    ├── 01_data_preparation   ← Feature engineering & category normalization
    ├── 02_stage1_tsmixer_daily     ← TSMixer deep learning (fails)
    ├── 03_stage2_xgboost_monthly  ← XGBoost regression (fails)
    ├── 04_stage3_direction_magnitude ← Probit + ARIMA decomposition
    ├── 05_stage4_direct_regression  ← Ridge on signed Δy + CW test
    └── 06_stage5_multihorizon       ← Iterated multi-horizon forecasting
```

---

## Replication Instructions

### Prerequisites

- Databricks workspace (Azure) with Runtime 15.4+
- A cluster with at least 1 driver node (Standard_DS13_v2 or equivalent)
- Python 3.12+
- (Optional) FRED API key for automated data download

### Step 1: Install dependencies

```python
%pip install -r requirements.txt
```

### Step 2: Set up environment

```python
# Copy .env.example to .env and fill in your FRED API key
# Or set directly:
import os
os.environ["FRED_API_KEY"] = "your_key_here"
```

### Step 3: Run scraping (or use provided data)

Run notebooks in `scraping/` in order (01 through 05).  
Alternatively, place `xgboost_monthly_data.csv` directly in `data/processed/`.

### Step 4: Run modeling notebooks

Each modeling notebook in `notebooks/` is self-contained.  
Run `01_data_preparation` first, then any stage notebook:

```python
%run ./01_data_preparation
```

---

## Data Sources

| Source | Series | Frequency | Auth Required |
|--------|--------|-----------|---------------|
| NBG API | Exchange rates (all currencies) | Daily | No |
| NBG Statistics | M2, M3, deposits, dollarization, TIBR, CPI | Monthly | No |
| LBMA | Gold (24k/18k/14k), Silver | Daily | No |
| FRED | DFF, TB3MS, DTB3, DGS3MO | Daily/Monthly | Yes (free key) |
| Yahoo Finance | VIX, S&P500, TNX, gold/oil futures, MOVE | Daily | No |
| World Bank | GDP, macro indicators (Georgia) | Annual | No |

---

## Key Features Used

| Feature | Description | Source |
|---------|-------------|--------|
| `direction_lag1` | Previous month: 1=depreciation, 0=appreciation | Derived |
| `direction_streak3` | 3-month rolling mean of direction | Derived |
| `tibR3M_last_lag1` | Georgian 3-month T-bill rate (lagged) | NBG |
| `dollarization_ratio_lag1` | FX deposits / total deposits (lagged) | NBG |
| `gold_24_usd_oz_mean_lag1` | Gold price USD/oz (lagged) | LBMA |
| `shock` | Deviation of last Δ from expanding mean | Derived |

---

## Statistical Tests

- **Clark-West (2006)**: Tests nested model vs RW, adjusts for parameter estimation noise
- **Pesaran-Timmermann (1992)**: Directional accuracy significance
- **Diebold-Mariano (1995)** with HLN correction: Equal predictive accuracy
- **ADF**: Stationarity of magnitude series

---

## Important Notes

1. **No API keys are stored in code**. Use `.env` file or environment variables.
2. **All models use walk-forward (expanding window)** evaluation to prevent look-ahead bias.
3. **Z-score normalization uses training-only statistics** at each step.
4. **Random seed is fixed** (SEED=42, TSMixer SEED=11) for reproducibility.
5. **The processed CSV** (`xgboost_monthly_data.csv`) contains all features pre-computed.
   If you cannot run scraping, use this file directly.

---

## References

- Meese, R. & Rogoff, K. (1983). Empirical exchange rate models of the seventies.
- Clark, T. & West, K. (2006). Using OOS mean squared prediction errors to test the martingale difference hypothesis.
- Pesaran, M. & Timmermann, A. (1992). A simple nonparametric test of predictive performance.
- Engel, C. & West, K. (2005). Exchange rates and fundamentals.
- Mark, N. (1995). Exchange rates and fundamentals: Evidence on long-horizon predictability.
- Chen, S. et al. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting.
