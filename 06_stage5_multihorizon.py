# Databricks notebook source
# DBTITLE 1,Stage 4: Direct Ridge Regression on Signed Change
"""
05_stage4_direct_regression.py — Ridge on Signed Δy(t)
=======================================================
Stage 4 of the thesis pipeline.

Insight from Stage 3: the forced binary direction decision requires 82%
accuracy to break even. Instead, predict Δy = y(t) - y(t-1) directly.
The model can predict SMALL changes when uncertain, avoiding catastrophic
error from wrong-direction large predictions.

Approach:
  - RidgeCV regression (L2 regularization)
  - Walk-forward expanding window
  - Three feature specifications (4, 6, 14 features)
  - Clark-West (2006) test vs Random Walk
  - Diebold-Mariano (1995) test with HLN correction

Key Results:
  - 4-feature: MAE +2.2% vs RW, CW p=0.058 (10% significant)
  - 6-feature: MAE +10.0% vs RW, CW p=0.161
  - 14-feature: MAE +4.3% vs RW, CW p=0.080 (10% significant)
  - Model generates positive MSPE-adjusted values → population-level
    predictive content exists even if sample MAE doesn't beat RW

Input: Run 01_data_preparation first
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import (SEED, TRAIN_RATIO, RIDGE_ALPHAS,
                    FEATURES_4, FEATURES_6, FEATURES_14)

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(SEED)

# NOTE: Run 01_data_preparation first
# %run ./01_data_preparation

# COMMAND ----------

# DBTITLE 1,Walk-Forward Ridge Regression
# =============================================================================
# 1. WALK-FORWARD RIDGE REGRESSION
# =============================================================================

def walk_forward_ridge(df_all, split_idx, features, alphas=RIDGE_ALPHAS):
    """
    Walk-forward expanding-window Ridge regression on Δy(t).
    
    Returns:
        preds: np.array of predicted Δy for each test observation
    """
    n_test = len(df_all) - split_idx
    preds = []
    
    for t in range(n_test):
        train_end = split_idx + t
        df_tr = df_all.iloc[:train_end]
        df_te = df_all.iloc[[train_end]]
        
        X_tr = df_tr[features].fillna(df_tr[features].median())
        y_tr = df_tr['fx_rate_formatted_mean_USD_diff']
        X_te = df_te[features].fillna(df_tr[features].median())
        
        # Leakage-free scaling
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # Ridge with cross-validated alpha
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_tr_s, y_tr)
        pred = float(ridge.predict(X_te_s)[0])
        preds.append(pred)
    
    return np.array(preds)


# Prepare data
df_reg = df.dropna(subset=['fx_rate_formatted_mean_USD_diff']).copy()
df_reg = df_reg.sort_values('year_month').reset_index(drop=True)
split_idx = int(len(df_reg) * TRAIN_RATIO)
T_test = len(df_reg) - split_idx

# Run all three specifications
specs = {
    '4-feature': [c for c in FEATURES_4 if c in df_reg.columns],
    '6-feature': [c for c in FEATURES_6 if c in df_reg.columns],
    '14-feature': [c for c in FEATURES_14 if c in df_reg.columns],
}

results = {}
for name, feats in specs.items():
    print(f"Computing {name} ({len(feats)} features)...", end=" ")
    preds = walk_forward_ridge(df_reg, split_idx, feats)
    results[name] = preds
    print("done.")

# Actual values
y_actual_diff = df_reg['fx_rate_formatted_mean_USD_diff'].iloc[split_idx:].values
actual_rates = df_reg['fx_rate_formatted_mean_USD'].iloc[split_idx:].values
y_prev = np.concatenate([
    [df_reg['fx_rate_formatted_mean_USD'].iloc[split_idx - 1]],
    actual_rates[:-1]
])

# COMMAND ----------

# DBTITLE 1,Clark-West (2006) Test
# =============================================================================
# 2. CLARK-WEST (2006) TEST vs RANDOM WALK
# =============================================================================
# Reference: Clark & West (2006), JBES
# Designed for nested model comparisons where DM test is undersized.
# H0: Random walk is at least as good
# H1: Model has population-level predictive content

def clark_west_test(y_actual, model_preds, rw_preds=None):
    """
    Clark-West MSPE-adjusted test.
    
    Parameters
    ----------
    y_actual : actual Δy values
    model_preds : model-predicted Δy values
    rw_preds : random walk predictions (default: all zeros for Δy)
    
    Returns
    -------
    cw_stat, p_value
    """
    if rw_preds is None:
        rw_preds = np.zeros_like(y_actual)  # RW predicts Δy = 0
    
    e_rw = y_actual - rw_preds      # RW errors
    e_model = y_actual - model_preds  # Model errors
    
    # MSPE-adjusted statistic
    f_t = e_rw**2 - (e_model**2 - (rw_preds - model_preds)**2)
    
    # t-test on f_t (one-sided: model is better)
    t_stat = np.mean(f_t) / (np.std(f_t, ddof=1) / np.sqrt(len(f_t)))
    p_value = 1 - stats.t.cdf(t_stat, df=len(f_t) - 1)
    
    return t_stat, p_value


def diebold_mariano_test(y_actual, preds_1, preds_2, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    H0: Equal predictive accuracy
    Uses absolute error loss.
    """
    e1 = np.abs(y_actual - preds_1)
    e2 = np.abs(y_actual - preds_2)
    d = e1 - e2  # Loss differential
    
    T = len(d)
    d_bar = np.mean(d)
    
    # HAC variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0
        gamma_sum += 2 * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / T
    DM_stat = d_bar / np.sqrt(var_d) if var_d > 0 else 0
    
    # HLN correction
    hln_factor = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    DM_adj = DM_stat * hln_factor
    
    p_value = 2 * (1 - stats.t.cdf(abs(DM_adj), df=T - 1))
    return DM_adj, p_value


# Run tests for all specifications
rw_diff_preds = np.zeros(T_test)  # RW predicts no change

print("\n" + "=" * 70)
print("STAGE 4 RESULTS: Ridge Δy Specifications vs Random Walk")
print("=" * 70)
print(f"{'Spec':<12} {'MAE Model':<12} {'MAE RW':<10} {'Ratio':<8} {'CW stat':<10} {'CW p':<8} {'DM stat':<10} {'DM p':<8}")
print("-" * 70)

for name, preds in results.items():
    # Level reconstruction for MAE
    level_pred = y_prev + preds
    level_rw = y_prev
    
    mae_model = mean_absolute_error(actual_rates, level_pred)
    mae_rw = mean_absolute_error(actual_rates, level_rw)
    
    # Clark-West on Δy
    cw_stat, cw_p = clark_west_test(y_actual_diff, preds)
    
    # Diebold-Mariano on levels
    dm_stat, dm_p = diebold_mariano_test(actual_rates, level_pred, level_rw)
    
    sig = '***' if cw_p < 0.01 else '**' if cw_p < 0.05 else '*' if cw_p < 0.10 else ''
    print(f"{name:<12} {mae_model:<12.6f} {mae_rw:<10.6f} {mae_model/mae_rw:<8.4f} "
          f"{cw_stat:<10.4f} {cw_p:<8.4f}{sig} {dm_stat:<10.4f} {dm_p:<8.4f}")

print("\nInterpretation:")
print("  - CW p < 0.10: population-level predictive content vs RW (10% level)")
print("  - MAE ratio > 1: model doesn't beat RW in sample MAE")
print("  - These are NOT contradictory: CW adjusts for parameter estimation noise")
print("\n→ Proceeding to Stage 5: multi-horizon forecasting.")