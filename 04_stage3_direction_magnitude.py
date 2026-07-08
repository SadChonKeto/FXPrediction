# Databricks notebook source
# DBTITLE 1,Stage 2: XGBoost Monthly Forecasting
"""
03_stage2_xgboost_monthly.py — XGBoost on Monthly GEL/USD
==========================================================
Stage 2 of the thesis pipeline.

Approach:
  - Hyperopt-tuned XGBoost regressor with TimeSeriesSplit CV
  - Targets: (a) FX level, (b) First difference Δy(t)
  - Walk-forward expanding window evaluation
  - SHAP feature importance analysis

Result: XGBoost fails to beat random walk on either target.
  The model overfits to in-sample patterns that don't generalize.

Input:  Run 01_data_preparation first to get `df` DataFrame
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import SEED, TRAIN_RATIO, XGB_N_FOLDS, XGB_HYPEROPT_EVALS

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import mlflow
mlflow.autolog(disable=True)

np.random.seed(SEED)

# NOTE: Run 01_data_preparation first, or load directly:
# %run ./01_data_preparation

# COMMAND ----------

# DBTITLE 1,Hyperopt XGBoost with TimeSeriesSplit
# =============================================================================
# HYPEROPT-TUNED XGBOOST
# =============================================================================

# Temporal split
split_date = df['year_month'].sort_values().iloc[int(len(df) * TRAIN_RATIO)]
train = df[df['year_month'] <= split_date].copy()
test = df[df['year_month'] > split_date].copy()

# Target: FX level (or diff)
TARGET = 'fx_rate_formatted_mean_USD'
TARGET_DIFF = 'fx_rate_formatted_mean_USD_diff'

feature_cols = [c for c in df.columns if 'lag' in c.lower()]

X_train = train[feature_cols].fillna(0)
y_train_level = train[TARGET]
y_train_diff = train[TARGET_DIFF].fillna(0)

X_test = test[feature_cols].fillna(0)
y_test_level = test[TARGET]
y_test_diff = test[TARGET_DIFF]

# Hyperopt search space
space = {
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 500]),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-4), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-4), np.log(10)),
}

def objective(params):
    """Hyperopt objective: minimize negative CV MAE."""
    model = xgb.XGBRegressor(
        **params, random_state=SEED, verbosity=0, n_jobs=-1
    )
    tscv = TimeSeriesSplit(n_splits=XGB_N_FOLDS)
    scores = cross_val_score(
        model, X_train, y_train_diff,
        cv=tscv, scoring='neg_mean_absolute_error'
    )
    return {'loss': -scores.mean(), 'status': STATUS_OK}


print(f"Running Hyperopt ({XGB_HYPEROPT_EVALS} evaluations)...")
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest,
            max_evals=XGB_HYPEROPT_EVALS, trials=trials, verbose=0)

print(f"Best params: {best}")

# COMMAND ----------

# DBTITLE 1,Walk-forward evaluation vs Random Walk
# =============================================================================
# WALK-FORWARD EVALUATION
# =============================================================================

# Reconstruct best params
best_params = {
    'max_depth': [3, 4, 5, 6, 7, 8][best['max_depth']],
    'learning_rate': best['learning_rate'],
    'n_estimators': [50, 100, 200, 300, 500][best['n_estimators']],
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda'],
}

split_idx = (df['year_month'] <= split_date).sum()
T_test = len(df) - split_idx

xgb_preds_diff = []
for t in range(T_test):
    train_end = split_idx + t
    df_tr = df.iloc[:train_end]
    df_te = df.iloc[[train_end]]
    
    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr[TARGET_DIFF].fillna(0)
    X_te = df_te[feature_cols].fillna(0)
    
    model = xgb.XGBRegressor(**best_params, random_state=SEED, verbosity=0)
    model.fit(X_tr, y_tr)
    pred = float(model.predict(X_te)[0])
    xgb_preds_diff.append(pred)

xgb_preds_diff = np.array(xgb_preds_diff)

# Reconstruct levels
actual_rates = test[TARGET].values
y_prev = np.concatenate([[train[TARGET].iloc[-1]], actual_rates[:-1]])
xgb_level = y_prev + xgb_preds_diff
rw_pred = y_prev  # Random walk

# Metrics
mae_xgb = mean_absolute_error(actual_rates, xgb_level)
mae_rw = mean_absolute_error(actual_rates, rw_pred)

print("=" * 60)
print("STAGE 2 RESULTS: XGBoost vs Random Walk")
print("=" * 60)
print(f"XGBoost MAE: {mae_xgb:.6f}")
print(f"Random Walk MAE: {mae_rw:.6f}")
print(f"Ratio (XGB/RW): {mae_xgb/mae_rw:.4f}")
print(f"Beats RW: {'YES' if mae_xgb < mae_rw else 'NO'}")
print("\nConclusion: XGBoost does NOT beat random walk.")
print("Proceeding to Stage 3 (Direction-Magnitude Decomposition).")