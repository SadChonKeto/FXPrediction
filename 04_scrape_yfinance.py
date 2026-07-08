# Databricks notebook source
# DBTITLE 1,FRED US Interest Rates
"""
03_scrape_fred.py — Download FRED (Federal Reserve Economic Data) series
====================================================================
Series used:
  - DFF:    Federal Funds Effective Rate (daily)
  - TB3MS:  3-Month Treasury Bill Secondary Market Rate (monthly)
  - DTB3:   3-Month Treasury Bill Secondary Market Rate (daily)
  - DGS3MO: 3-Month Treasury Constant Maturity Rate (daily)

Outputs:
  - ../data/raw/fred_{series_id}.csv for each series

Requires FRED API key. Set FRED_API_KEY in your .env file.
Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html

Alternative: Download CSV directly from the FRED website:
  https://fred.stlouisfed.org/series/{SERIES_ID}
  (Click "Download" -> CSV)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import FRED_SERIES, FRED_API_BASE, DATA_START_DATE, DATA_END_DATE, RAW_DATA_DIR

import requests
import pandas as pd

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load API key from environment
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("WARNING: FRED_API_KEY not set. Set it via:")
    print('  os.environ["FRED_API_KEY"] = "your_key_here"')
    print("  Or set in .env file and load with python-dotenv")
    print("\nAlternatively, download CSVs manually from https://fred.stlouisfed.org/")

# COMMAND ----------

# DBTITLE 1,Fetch FRED series via API
def fetch_fred_series(series_id: str, api_key: str,
                      start: str = "2000-01-01",
                      end: str = "2025-12-31") -> pd.DataFrame:
    """
    Fetch a single FRED series via the API.
    
    Parameters
    ----------
    series_id : str — FRED series code (e.g., 'DFF')
    api_key : str — Your FRED API key
    start, end : str — Date range in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.DataFrame with columns: [date, value]
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    observations = data.get("observations", [])
    rows = [
        {"date": obs["date"], "value": obs["value"]}
        for obs in observations
    ]
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# Fetch all FRED series
if FRED_API_KEY:
    for series_id in FRED_SERIES:
        print(f"Fetching {series_id}...", end=" ")
        df_series = fetch_fred_series(
            series_id, FRED_API_KEY,
            start=DATA_START_DATE, end=DATA_END_DATE
        )
        outpath = RAW_DATA_DIR / f"fred_{series_id}.csv"
        df_series.to_csv(outpath, index=False)
        print(f"{len(df_series)} obs → {outpath.name}")
    print("\n✓ All FRED series saved.")
else:
    print("Skipping API fetch (no key). Place manual CSVs in data/raw/:")
    for s in FRED_SERIES:
        print(f"  - fred_{s}.csv  (download from https://fred.stlouisfed.org/series/{s})")