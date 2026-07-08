# Databricks notebook source
# DBTITLE 1,yfinance Daily Financial Data
"""
04_scrape_yfinance.py — Download daily financial data via yfinance
===================================================================
Tickers:
  - GC=F:  COMEX Gold Futures (USD/troy oz)
  - CL=F:  WTI Crude Oil Futures
  - ^VIX:  CBOE Volatility Index
  - ^GSPC: S&P 500 Index
  - ^TNX:  10-Year Treasury Yield
  - ^MOVE: ICE BofA MOVE Index (bond volatility)

Outputs:
  - ../data/raw/yfinance_daily.csv

No API key required (yfinance uses Yahoo Finance public data).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import YFINANCE_TICKERS, DATA_START_DATE, DATA_END_DATE, RAW_DATA_DIR

import yfinance as yf
import pandas as pd

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Download all tickers
# =============================================================================
# Download daily closing prices for all tickers
# =============================================================================

def download_yfinance_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """
    Download daily closing prices for a dictionary of tickers.
    
    Parameters
    ----------
    tickers : dict — {friendly_name: yfinance_ticker}
    start, end : str — Date range
    
    Returns
    -------
    pd.DataFrame with date index and one column per ticker (friendly name + '_close')
    """
    frames = []
    
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 0:
                col_name = f"{ticker}_close"
                series = df["Close"].rename(col_name)
                frames.append(series)
                print(f"{len(df)} obs")
            else:
                print("NO DATA")
        except Exception as e:
            print(f"FAILED: {e}")
    
    if frames:
        result = pd.concat(frames, axis=1)
        result.index.name = "date"
        return result.reset_index()
    return pd.DataFrame()


print(f"Downloading yfinance data ({DATA_START_DATE} to {DATA_END_DATE})...")
df_yf = download_yfinance_data(YFINANCE_TICKERS, DATA_START_DATE, DATA_END_DATE)

outpath = RAW_DATA_DIR / "yfinance_daily.csv"
df_yf.to_csv(outpath, index=False)
print(f"\n✓ Saved {len(df_yf)} rows to {outpath}")
print(f"  Columns: {list(df_yf.columns)}")
