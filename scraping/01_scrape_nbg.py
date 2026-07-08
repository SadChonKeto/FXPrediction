# Databricks notebook source
# DBTITLE 1,NBG Exchange Rates & Monetary Data
"""
01_scrape_nbg.py — Scrape National Bank of Georgia (NBG) data
================================================================
Sources:
  1. Daily exchange rates (GEL vs multiple currencies) via NBG public API
  2. Monetary aggregates (M2, M3, deposits, dollarization, loans-to-deposits,
     remittances, TIBR rates, official rate, CPI/inflation) via NBG statistics

Outputs:
  - ../data/raw/nbg_exchange_rates_daily.csv
  - ../data/raw/nbg_monetary_monthly.csv

No API key required — NBG endpoints are fully public.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import NBG_API_BASE, DATA_START_DATE, DATA_END_DATE, CURRENCY_CODES, RAW_DATA_DIR

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Fetch daily exchange rates
# =============================================================================
# 1. DAILY EXCHANGE RATES
# =============================================================================

def get_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_exchange_rates(startdate: str, enddate: str, currency_code: str = "USD") -> pd.DataFrame:
    """
    Fetch daily exchange rates for a single currency from NBG API.
    
    Parameters
    ----------
    startdate : str  — format 'YYYY-MM-DD'
    enddate : str    — format 'YYYY-MM-DD'
    currency_code : str — ISO 4217 code (e.g., 'USD', 'EUR')
    
    Returns
    -------
    pd.DataFrame with columns: [date, currency_code, rate_formatted, diff]
    """
    session = get_session()
    rows = []
    date_range = pd.date_range(startdate, enddate, freq="D")

    for temp_date in tqdm(date_range, desc=f"Fetching {currency_code}"):
        url = (
            f"{NBG_API_BASE}/currencies/"
            f"?date={temp_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
            f"&currencies={currency_code}"
        )
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and data[0].get("currencies"):
                for curr in data[0]["currencies"]:
                    rows.append({
                        "date": temp_date.date(),
                        "currency_code": curr.get("code", currency_code),
                        "rate_formatted": curr.get("rateFormated") or curr.get("rate"),
                        "diff": curr.get("diffFormated") or curr.get("diff"),
                    })
        except Exception as e:
            pass  # Skip failed dates silently

    return pd.DataFrame(rows)


def get_all_exchange_rates(startdate: str, enddate: str) -> pd.DataFrame:
    """
    Fetch daily exchange rates for ALL currencies from NBG API.
    Returns wide-format DataFrame pivoted by currency code.
    """
    session = get_session()
    rows = []
    date_range = pd.date_range(startdate, enddate, freq="D")

    for temp_date in tqdm(date_range, desc="Fetching ALL currencies"):
        url = (
            f"{NBG_API_BASE}/currencies/"
            f"?date={temp_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
        )
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and data[0].get("currencies"):
                for curr in data[0]["currencies"]:
                    rows.append({
                        "date": temp_date.date(),
                        "currency_code": curr.get("code"),
                        "rate_formatted": curr.get("rateFormated") or curr.get("rate"),
                        "quantity": curr.get("quantity", 1),
                    })
        except Exception:
            pass

    return pd.DataFrame(rows)

# COMMAND ----------

# DBTITLE 1,Execute NBG exchange rate scraping
# Scrape GEL/USD (primary target) + other key currencies
print("Scraping NBG exchange rates...")
print(f"Date range: {DATA_START_DATE} to {DATA_END_DATE}")
print(f"Currencies: {CURRENCY_CODES}")
print()

frames = []
for code in CURRENCY_CODES:
    df_curr = get_exchange_rates(DATA_START_DATE, DATA_END_DATE, currency_code=code)
    frames.append(df_curr)
    print(f"  {code}: {len(df_curr)} observations")

df_fx = pd.concat(frames, ignore_index=True)
df_fx["date"] = pd.to_datetime(df_fx["date"])
df_fx["rate_formatted"] = pd.to_numeric(df_fx["rate_formatted"], errors="coerce")

# Save
outpath = RAW_DATA_DIR / "nbg_exchange_rates_daily.csv"
df_fx.to_csv(outpath, index=False)
print(f"\n✓ Saved {len(df_fx)} rows to {outpath}")

# COMMAND ----------

# DBTITLE 1,NBG Monetary Aggregates (manual note)
# =============================================================================
# 2. NBG MONETARY & MACRO DATA (Monthly)
# =============================================================================
# The following monthly series are used in the model:
#   - M2_total, M3_total: Monetary aggregates
#   - deposits_total: Total banking deposits
#   - dollarization_ratio: FX deposits / total deposits
#   - loans_to_deposits: Banking system leverage
#   - Total_rem: Total remittances to Georgia
#   - tibr, tibR1M, tibR3M, tibR6M: Interbank lending rates
#   - official_exchange_rate_gel_usd: NBG official rate
#   - headline_yoy: Year-over-year CPI inflation
#   - nbg_interventions_usd: FX market interventions
#
# These are scraped from NBG's statistical portal or downloaded as CSV.
# The NBG statistics API endpoint:
#   https://nbg.gov.ge/gw/api/ct/monetarypolicy/statistics/
#
# Note: Some series require manual download from:
#   https://nbg.gov.ge/en/monetary-policy/monetary-statistics
#   https://nbg.gov.ge/en/monetary-policy/inflation
#
# The processed monthly dataset (xgboost_monthly_data.csv) already contains
# these features merged and aligned by year_month.

print("NBG monetary data note: see data/processed/xgboost_monthly_data.csv")
print("Manual download sources:")
print("  - Monetary statistics: https://nbg.gov.ge/en/monetary-policy/monetary-statistics")
print("  - Inflation data: https://nbg.gov.ge/en/monetary-policy/inflation")
print("  - Interventions: https://nbg.gov.ge/en/monetary-policy/currency-interventions")
