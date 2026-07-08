# Databricks notebook source
# DBTITLE 1,Setup and Imports
"""
01_data_preparation.py — Feature Engineering for GEL/USD Forecasting
====================================================================
Loads the processed monthly dataset and engineers all features used
across modeling stages 1–5:
  - Category normalization (CPI contributors)
  - Direction/momentum indicators
  - Shock features
  - Rate differentials
  - Lag selection

Input:  ../data/processed/xgboost_monthly_data.csv
Output: Cleaned DataFrame `df` ready for all subsequent notebooks
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import PROCESSED_DATA_DIR, TRAIN_RATIO, SEED, PROBIT_FEATURES

import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(SEED)

# COMMAND ----------

# DBTITLE 1,Load data and normalize categories
# =============================================================================
# 1. LOAD DATA
# =============================================================================
data_path = PROCESSED_DATA_DIR / "xgboost_monthly_data.csv"
df = pd.read_csv(data_path)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

# =============================================================================
# 2. NORMALIZE CPI CONTRIBUTOR CATEGORIES
# =============================================================================
# Maps Georgian/English category labels to standardized English names

def normalize_category(val):
    """Map CPI contributor category labels to standardized names."""
    if pd.isna(val):
        return 'nan'
    val = str(val).strip().lower()
    mapping = [
        (r'fruit|xili|ხილი|grape|ყურძენი|yurzeni', 'fruit_grapes'),
        (r'bread|puri|პური', 'bread'),
        (r'meat|xorci|ხორცი', 'meat'),
        (r'vegetable|bostneuli|ბოსტნეული|garden|melons|kartofili|potato|ბაღჩეული|bolqvovan', 'vegetables'),
        (r'oil|zeTi|ზეთი|fat|fats|cximi|sunflower', 'oil_fats'),
        (r'milk|რძე|cheese|კვერცხი|imeretian', 'dairy'),
        (r'alcohol|alkohol|wine|თამბაქო|cigarette|სასმელები|tobacco', 'alcohol_tobacco'),
        (r'health|janmr|ჯანდაცვა|ჯანმრთელობის|dacva', 'health'),
        (r'transport|ტრანსპორტ|gasoline|ბენზინი|საწვავი|დიზელის', 'transport'),
        (r'service|მომსახურება|bank|ბანკ|communications|კავშირგაბმულობა|internet', 'services'),
        (r'housing|სახლი|electricity|ელექტროენერგია|აირი|gas|სათბობის|water|wyali', 'housing'),
        (r'clothing|footwear|ტანსაცმელი|ფეხსაცმელი', 'clothing'),
        (r'miscellaneous|სხვადასხვა|დანარჩენი', 'misc'),
        (r'education|ganaTleba', 'education'),
        (r'recreation|entertainment|culture|rest|სასტუმრო|hotel|restaurant|კაფე|sastumroebi', 'recreation'),
        (r'food|სურსათი|non-alcoholic', 'food'),
    ]
    for pattern, label in mapping:
        if re.search(pattern, val):
            return label
    return val


# Apply normalization to CPI contributor columns
cat_cols = [
    'contributor_1_category_lag1', 'contributor_2_category_lag1', 'contributor_3_category_lag1',
    'contributor_1_category_lag3', 'contributor_2_category_lag3', 'contributor_3_category_lag3'
]
existing_cat_cols = [c for c in cat_cols if c in df.columns]

for col in existing_cat_cols:
    df[col] = df[col].apply(normalize_category)

# Integer-encode categories
unique_cats = sorted(set(
    val for col in existing_cat_cols for val in df[col].unique() if val != 'nan'
))
cat_mapping = {cat: idx + 1 for idx, cat in enumerate(unique_cats)}
cat_mapping['nan'] = 0

for col in existing_cat_cols:
    df[col] = df[col].map(cat_mapping)

print(f"Normalized {len(existing_cat_cols)} category columns")
print(f"Unique categories: {len(unique_cats)}")

# COMMAND ----------

# DBTITLE 1,Drop leaky columns and engineer features
# =============================================================================
# 3. DROP LEAKY / REDUNDANT COLUMNS
# =============================================================================

cols_to_drop = (
    [c for c in df.columns if 'unnamed' in c.lower()]
    + [c for c in ['year', 'year_lag1', 'year_lag3', 'month', 'month_lag1', 'month_lag3'] if c in df.columns]
    + [c for c in df.columns if c.startswith('fx_rate_formatted') 
       and c != 'fx_rate_formatted_mean_USD'
       and 'lag' not in c.lower()]
    + [c for c in ['fx_rate_formatted_std_USD_lag1', 'fx_rate_formatted_std_USD_lag3',
                   'fx_rate_formatted_last_USD_lag1', 'fx_rate_formatted_last_USD_lag3',
                   'official_exchange_rate_gel_usd_lag1', 'official_exchange_rate_gel_usd_lag3']
       if c in df.columns]
)
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)

# =============================================================================
# 4. ENGINEER FEATURES
# =============================================================================

# Rate differential (Georgian 3M - US 3M)
if 'tibR3M_mean' in df.columns and 'DGS3MO_mean' in df.columns:
    df['rate_diff'] = df['tibR3M_mean'] - df['DGS3MO_mean']

# Lagged FX rate for risk premium
df['fx_rate_formatted_mean_USD_lag1'] = df['fx_rate_formatted_mean_USD'].shift(1)
if 'rate_diff' in df.columns:
    df['risk_premium'] = df['rate_diff'] - df['fx_rate_formatted_mean_USD_lag1']

# Generate missing lags
for col in [c for c in df.columns if 'lag' not in c.lower() 
            and c not in ['year_month', 'fx_rate_formatted_mean_USD']]:
    for lag in [1, 3]:
        lag_col = f"{col}_lag{lag}"
        if lag_col not in df.columns:
            df[lag_col] = df[col].shift(lag)

# Keep only lagged features + target + time index
cols_to_keep = [col for col in df.columns if 'lag' in col.lower()] + \
               ['year_month', 'fx_rate_formatted_mean_USD']
df = df[cols_to_keep]

# Remove duplicate exchange rate lags (prevent leakage)
for drop_col in ['official_exchange_rate_gel_usd_lag1', 'official_exchange_rate_gel_usd_lag3']:
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])

print(f"After feature engineering: {df.shape}")

# COMMAND ----------

# DBTITLE 1,Direction and momentum features
# =============================================================================
# 5. DIRECTION, DIFF LAGS, AND SHOCK FEATURES
# =============================================================================

df = df.sort_values('year_month').reset_index(drop=True)

# First difference of target
df['fx_rate_formatted_mean_USD_diff'] = df['fx_rate_formatted_mean_USD'].diff()

# Diff lags (1, 2, 3 months)
for lag in [1, 2, 3]:
    df[f'fx_rate_formatted_mean_USD_diff_lag{lag}'] = \
        df['fx_rate_formatted_mean_USD_diff'].shift(lag)
df[['fx_rate_formatted_mean_USD_diff_lag1',
    'fx_rate_formatted_mean_USD_diff_lag2',
    'fx_rate_formatted_mean_USD_diff_lag3']] = \
    df[['fx_rate_formatted_mean_USD_diff_lag1',
        'fx_rate_formatted_mean_USD_diff_lag2',
        'fx_rate_formatted_mean_USD_diff_lag3']].fillna(0)

# Binary direction: 1 = depreciation (GEL weakens), 0 = appreciation
df['direction'] = (df['fx_rate_formatted_mean_USD_diff'] > 0).astype(int)
for lag in [1, 2, 3]:
    df[f'direction_lag{lag}'] = df['direction'].shift(lag)

# Rolling direction streaks
df['direction_streak3'] = df['direction_lag1'].rolling(3).mean()
df['direction_streak5'] = df['direction_lag1'].rolling(5).mean()

# Shock: deviation of last period's Δ from expanding mean
y_diff = df['fx_rate_formatted_mean_USD_diff']
running_mean = y_diff.expanding(min_periods=3).mean()
df['shock'] = y_diff.shift(1) - running_mean.shift(1)
df['shock'] = df['shock'].fillna(0)

print(f"Final dataset: {df.shape}")
print(f"Target: fx_rate_formatted_mean_USD (level), fx_rate_formatted_mean_USD_diff (change)")
print(f"Direction: {df['direction'].value_counts().to_dict()}")
print(f"\n✓ DataFrame 'df' ready for modeling.")
print(f"  Use TRAIN_RATIO = {TRAIN_RATIO} for temporal split.")
