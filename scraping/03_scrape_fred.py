# Databricks notebook source
# DBTITLE 1,LBMA Gold and Silver Prices
"""
02_scrape_lbma.py — Scrape LBMA precious metals prices
=======================================================
Sources:
  - Gold AM fix (24k, 18k, 14k USD/oz) from LBMA JSON endpoint
  - Silver fix (fine, britannia, sterling) from LBMA JSON endpoint

Outputs:
  - ../data/raw/lbma_gold_daily.csv
  - ../data/raw/lbma_silver_daily.csv

No API key required — LBMA endpoints are public.
Data available from 1968-01-02 onwards.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import LBMA_GOLD_URL, LBMA_SILVER_URL, RAW_DATA_DIR

import requests
import pandas as pd

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Scrape gold prices
# =============================================================================
# 1. GOLD PRICES (LBMA AM Fix)
# =============================================================================

def scrape_lbma_gold() -> pd.DataFrame:
    """
    Fetch daily gold prices from LBMA AM fix JSON endpoint.
    Returns DataFrame with columns:
      date, gold_24_usd_oz, gold_18_usd_oz, gold_14_usd_oz
    """
    resp = requests.get(LBMA_GOLD_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        rows.append({
            "date": item["d"],
            "gold_24_usd_oz": item["v"][0],
            "gold_18_usd_oz": item["v"][1],
            "gold_14_usd_oz": item["v"][2],
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


df_gold = scrape_lbma_gold()
print(f"Gold prices: {len(df_gold)} observations")
print(f"Date range: {df_gold['date'].min()} to {df_gold['date'].max()}")

outpath = RAW_DATA_DIR / "lbma_gold_daily.csv"
df_gold.to_csv(outpath, index=False)
print(f"\n✓ Saved to {outpath}")

# COMMAND ----------

# DBTITLE 1,Scrape silver prices
# =============================================================================
# 2. SILVER PRICES (LBMA Fix)
# =============================================================================

def scrape_lbma_silver() -> pd.DataFrame:
    """
    Fetch daily silver prices from LBMA JSON endpoint.
    Returns DataFrame with columns:
      date, fine_silver, britannia_silver, sterling_silver
    """
    resp = requests.get(LBMA_SILVER_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        rows.append({
            "date": item["d"],
            "fine_silver": item["v"][0],
            "britannia_silver": item["v"][1],
            "sterling_silver": item["v"][2],
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


df_silver = scrape_lbma_silver()
print(f"Silver prices: {len(df_silver)} observations")
print(f"Date range: {df_silver['date'].min()} to {df_silver['date'].max()}")

outpath = RAW_DATA_DIR / "lbma_silver_daily.csv"
df_silver.to_csv(outpath, index=False)
print(f"\n✓ Saved to {outpath}")
