# Databricks notebook source
# DBTITLE 1,Stage 3: Direction-Magnitude Decomposition
"""
04_stage3_direction_magnitude.py — Probit Direction + ARIMA Magnitude
======================================================================
Stage 3 of the thesis pipeline. The key methodological contribution.

Approach:
  1. DIRECTION: Probit model predicts P(depreciation) using 6 features:
     - direction_lag1, direction_streak3 (momentum)
     - tibR3M_last_lag1 (Georgian interest rate)
     - dollarization_ratio_lag1 (financial depth)
     - gold_24_usd_oz_mean_lag1 (commodity safe haven)
     - shock (mean-reversion signal)

  2. MAGNITUDE: ARIMA(p,d,q) walk-forward on |Δy(t)|
     - AIC grid search over p∈[0,3], d∈[0,1], q∈[0,3]
     - Walk-forward expanding window

  3. RECONSTRUCTION: ŷ(t) = y(t-1) + direction_sign × magnitude

Key Results:
  - Fixed-split directional accuracy: 75.0% (PT test: p=0.0076)
  - Walk-forward directional accuracy: 57.5%
  - Break-even threshold: 82.1% (model doesn't reach it)
  - Conclusion: Decomposition doesn't beat RW at h=1 but provides
    theoretical insight for Stage 4

Input: Run 01_data_preparation first
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import SEED, TRAIN_RATIO, PROBIT_FEATURES, ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(SEED)

# NOTE: Run 01_data_preparation first to get `df`
# %run ./01_data_preparation

# COMMAND ----------

# DBTITLE 1,Probit Direction Model (Walk-Forward)
# =============================================================================
# 1. PROBIT DIRECTION MODEL — Walk-Forward Expanding Window
# =============================================================================

df_dir = df.dropna(subset=['fx_rate_formatted_mean_USD_diff']).copy()
df_dir = df_dir.sort_values('year_month').reset_index(drop=True)

# Temporal split
split_idx = int(len(df_dir) * TRAIN_RATIO)
T_test = len(df_dir) - split_idx

print(f"Train: {split_idx} months, Test: {T_test} months")
print(f"Probit features: {PROBIT_FEATURES}")

# Walk-forward probit
probit_wf_preds = []
probit_wf_proba = []

for t in range(T_test):
    train_end = split_idx + t
    df_train = df_dir.iloc[:train_end]
    df_test_row = df_dir.iloc[[train_end]]
    
    # Features
    avail_cols = [c for c in PROBIT_FEATURES if c in df_dir.columns]
    X_tr = df_train[avail_cols].fillna(df_train[avail_cols].median())
    y_tr = df_train['direction']
    X_te = df_test_row[avail_cols].fillna(df_train[avail_cols].median())
    
    # Leakage-free Z-score normalization
    tr_mean = X_tr.mean()
    tr_std = X_tr.std().replace(0, 1)
    X_tr_z = (X_tr - tr_mean) / tr_std
    X_te_z = (X_te - tr_mean) / tr_std
    
    X_tr_c = sm.add_constant(X_tr_z)
    X_te_c = sm.add_constant(X_te_z)
    
    try:
        model = sm.Probit(y_tr, X_tr_c).fit(disp=0)
        prob = float(model.predict(X_te_c).iloc[0])
    except Exception:
        prob = 0.5  # Fallback if convergence fails
    
    probit_wf_proba.append(prob)
    probit_wf_preds.append(int(prob > 0.5))

probit_wf_preds = np.array(probit_wf_preds)
probit_wf_proba = np.array(probit_wf_proba)
y_test_dir = df_dir['direction'].iloc[split_idx:].values

# Evaluation
acc_wf = accuracy_score(y_test_dir, probit_wf_preds)
auc_wf = roc_auc_score(y_test_dir, probit_wf_proba)

print(f"\n{'=' * 58}")
print("   Probit Direction — Walk-Forward Results")
print(f"{'=' * 58}")
print(f"  Accuracy: {acc_wf*100:.1f}%")
print(f"  AUC:      {auc_wf:.4f}")

# COMMAND ----------

# DBTITLE 1,Pesaran-Timmermann Directional Test
# =============================================================================
# 2. PESARAN-TIMMERMANN DIRECTIONAL ACCURACY TEST
# =============================================================================
# Reference: Pesaran & Timmermann (1992), JBES 10(4)
# H0: predictions independent of outcomes (no directional skill)
# H1: model has significant directional predictive power

def pesaran_timmermann_test(y_true, y_pred):
    """Compute PT statistic and p-value for directional accuracy."""
    T = len(y_true)
    P_hat = np.mean(y_true == y_pred)
    P_y = np.mean(y_true)
    P_f = np.mean(y_pred)
    P_star = P_y * P_f + (1 - P_y) * (1 - P_f)
    
    var_PT = (1 / T) * (
        P_y * (1 - P_y) * (2 * P_f - 1) ** 2 +
        P_f * (1 - P_f) * (2 * P_y - 1) ** 2
    )
    
    PT_stat = (P_hat - P_star) / np.sqrt(var_PT)
    p_one_sided = 1 - stats.norm.cdf(PT_stat)
    return PT_stat, p_one_sided, P_hat, P_star


PT_stat, p_val, P_hat, P_star = pesaran_timmermann_test(y_test_dir, probit_wf_preds)

print(f"\n{'=' * 58}")
print("   Pesaran–Timmermann Test (walk-forward)")
print(f"{'=' * 58}")
print(f"  Correct calls: {int(round(P_hat * len(y_test_dir)))} / {len(y_test_dir)} ({P_hat*100:.1f}%)")
print(f"  Expected (H0): {P_star*100:.1f}%")
print(f"  PT statistic:  {PT_stat:.4f}")
print(f"  p-value:       {p_val:.4f}")

if p_val < 0.01:
    print("  Reject H0 *** (p < 0.01)")
elif p_val < 0.05:
    print("  Reject H0 **  (p < 0.05)")
elif p_val < 0.10:
    print("  Reject H0 *   (p < 0.10)")
else:
    print("  Fail to reject H0")

# COMMAND ----------

# DBTITLE 1,ARIMA Magnitude (Walk-Forward)
# =============================================================================
# 3. ARIMA MAGNITUDE MODEL — Walk-Forward
# =============================================================================

df_mag = df_dir.copy()
y_mag_full = df_mag['fx_rate_formatted_mean_USD_diff'].abs().values

# Stationarity check
adf_stat, adf_p, *_ = adfuller(y_mag_full[:split_idx], autolag='AIC')
print(f"\nADF on train magnitude: stat={adf_stat:.4f}, p={adf_p:.4f}")
print(f"  → {'Stationary' if adf_p < 0.05 else 'Non-stationary (d=1 recommended)'}")

# AIC grid search on training data
y_train_mag = y_mag_full[:split_idx]
best_aic, best_order = np.inf, (1, 0, 1)

for p, d, q in itertools.product(ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE):
    if p + q == 0:
        continue
    try:
        res = ARIMA(y_train_mag, order=(p, d, q)).fit()
        if res.aic < best_aic:
            best_aic, best_order = res.aic, (p, d, q)
    except Exception:
        pass

print(f"Best ARIMA order: {best_order} (AIC={best_aic:.2f})")

# Walk-forward predictions
arima_preds = []
for t in range(T_test):
    train_end = split_idx + t
    y_history = y_mag_full[:train_end]
    try:
        model = ARIMA(y_history, order=best_order).fit()
        pred = float(model.forecast(steps=1)[0])
    except Exception:
        pred = y_history[-1]  # Fallback: persistence
    arima_preds.append(max(pred, 0))  # Magnitude is non-negative

arima_preds = np.array(arima_preds)
y_test_mag = y_mag_full[split_idx:]

# Baseline: mean magnitude
baseline_mag = np.mean(y_train_mag)
mae_arima = mean_absolute_error(y_test_mag, arima_preds)
mae_baseline = mean_absolute_error(y_test_mag, [baseline_mag] * T_test)

print(f"\nARIMA Magnitude MAE: {mae_arima:.6f}")
print(f"Baseline (mean) MAE: {mae_baseline:.6f}")
print(f"Improvement: {(1 - mae_arima/mae_baseline)*100:.1f}%")

# COMMAND ----------

# DBTITLE 1,Reconstruction and Break-Even Analysis
# =============================================================================
# 4. RECONSTRUCTION: direction × magnitude vs Random Walk
# =============================================================================

actual_rates = df_dir['fx_rate_formatted_mean_USD'].iloc[split_idx:].values
last_train_rate = df_dir['fx_rate_formatted_mean_USD'].iloc[split_idx - 1]
y_prev = np.concatenate([[last_train_rate], actual_rates[:-1]])

# Predicted sign from probit
predicted_sign = np.where(probit_wf_preds == 1, +1, -1)

# Reconstructed FX rate
reconstructed = y_prev + predicted_sign * arima_preds
rw_pred = y_prev  # Random walk: no change

mae_hybrid = mean_absolute_error(actual_rates, reconstructed)
mae_rw = mean_absolute_error(actual_rates, rw_pred)

print(f"\n{'=' * 60}")
print("STAGE 3 RESULTS: Direction-Magnitude vs Random Walk")
print(f"{'=' * 60}")
print(f"Hybrid (Probit+ARIMA) MAE: {mae_hybrid:.6f}")
print(f"Random Walk MAE:           {mae_rw:.6f}")
print(f"Ratio: {mae_hybrid/mae_rw:.4f}")
print(f"Beats RW: {'YES' if mae_hybrid < mae_rw else 'NO'}")

# =============================================================================
# 5. BREAK-EVEN DIRECTIONAL ACCURACY
# =============================================================================
# What hit rate is needed to beat RW given ARIMA's magnitude predictions?

actual_mag = df_dir['fx_rate_formatted_mean_USD_diff'].abs().iloc[split_idx:].values

# Error when direction correct vs wrong:
#   Correct: |actual_mag - pred_mag|     (only magnitude error matters)
#   Wrong:   actual_mag + pred_mag       (adds in wrong direction)
error_correct = np.abs(actual_mag - arima_preds)
error_wrong = actual_mag + arima_preds
error_rw = actual_mag  # RW error = actual magnitude

# Break-even: p * error_correct + (1-p) * error_wrong = error_rw
# Solving: p = (error_wrong - error_rw) / (error_wrong - error_correct)
with np.errstate(divide='ignore', invalid='ignore'):
    p_breakeven = (error_wrong - error_rw) / (error_wrong - error_correct)
    p_breakeven = np.clip(p_breakeven, 0, 1)

mean_breakeven = np.nanmean(p_breakeven)
print(f"\nBreak-even directional accuracy: {mean_breakeven*100:.1f}%")
print(f"Model achieves: {acc_wf*100:.1f}% (walk-forward)")
print(f"Gap: {(mean_breakeven - acc_wf)*100:.1f} percentage points")
print("\nConclusion: Model accuracy < break-even threshold.")
print("The magnitude model's errors are too large for decomposition to work.")
print("\n→ Proceeding to Stage 4: predict signed Δy directly.")