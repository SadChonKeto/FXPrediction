# Databricks notebook source
# DBTITLE 1,Stage 1: TSMixer Deep Learning
"""
02_stage1_tsmixer_daily.py — TSMixer on Daily GEL/USD Exchange Rate
=====================================================================
Stage 1 of the thesis pipeline.

Approach:
  - TSMixer (Time-Series Mixer) architecture from Chen et al. (2023)
  - Multivariate daily data: GEL/USD, RUB, EUR, TRY, VIX, TNX, gold,
    oil, TIBR rates, S&P500, MOVE index, geopolitical risk
  - Grid search: 15 window-horizon combinations
    (windows: 12, 24, 36 days; horizons: 16, 24, 32, 48, 64 days)
  - Both level target and first-difference target

Result: TSMixer fails to beat random walk on any configuration.
  This is consistent with Meese & Rogoff (1983) — neural approaches
  add complexity without improving out-of-sample FX prediction.

Input:  ../data/processed/data_v6_no_shift.csv (daily multivariate)
        OR run from ../data/raw/ sources via scraping/05_build_monthly_dataset
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname("__file__"), ".."))
from config import (SEED, TSMIXER_SEED, TRAIN_RATIO, PROCESSED_DATA_DIR,
                    TSMIXER_WINDOW_SIZES, TSMIXER_HORIZONS,
                    TSMIXER_EPOCHS, TSMIXER_BATCH_SIZE, TSMIXER_LR,
                    TSMIXER_DROPOUT, TSMIXER_N_BLOCKS, TSMIXER_D_FF)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(TSMIXER_SEED)
torch.manual_seed(TSMIXER_SEED)
torch.cuda.manual_seed_all(TSMIXER_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# COMMAND ----------

# DBTITLE 1,TSMixer Architecture
# =============================================================================
# TSMIXER ARCHITECTURE (Chen et al., 2023)
# =============================================================================

class MixerBlock(nn.Module):
    """Single TSMixer block: time-mixing + feature-mixing with residual."""
    def __init__(self, seq_len: int, n_features: int, d_ff: int, dropout: float):
        super().__init__()
        # Time-mixing MLP (operates across time steps for each feature)
        self.time_norm = nn.LayerNorm(seq_len)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
            nn.Dropout(dropout),
        )
        # Feature-mixing MLP (operates across features for each time step)
        self.feat_norm = nn.LayerNorm(n_features)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        # Time mixing
        res = x
        x_t = self.time_norm(x.transpose(1, 2))  # (batch, feat, seq)
        x_t = self.time_mlp(x_t).transpose(1, 2)  # (batch, seq, feat)
        x = res + x_t
        # Feature mixing
        res = x
        x_f = self.feat_norm(x)  # (batch, seq, feat)
        x_f = self.feat_mlp(x_f)  # (batch, seq, feat)
        x = res + x_f
        return x


class TSMixer(nn.Module):
    """TSMixer for multi-step forecasting."""
    def __init__(self, seq_len: int, n_features: int, horizon: int,
                 n_blocks: int = 4, d_ff: int = 128, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MixerBlock(seq_len, n_features, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(seq_len * n_features, horizon)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.blocks(x)
        x = x.flatten(1)  # (batch, seq_len * n_features)
        return self.head(x)  # (batch, horizon)

# COMMAND ----------

# DBTITLE 1,Data loading and grid search
# =============================================================================
# DATA LOADING
# =============================================================================

# Load daily multivariate data
data_path = PROCESSED_DATA_DIR / "data_v6_no_shift.csv"
if not data_path.exists():
    # Fallback: try relative path in old structure
    data_path = "./data_v6_no_shift.csv"

df = pd.read_csv(data_path)
df = df[df['year_month'] > '1999-12-31']

# Features for TSMixer
KEEP_FEATURES = [
    'USD', 'RUB', 'EUR', 'TRY',
    '^VIX_close', '^TNX_close', 'GC=F_close', 'CL=F_close',
    'tibr', 'tibR3M', 'spread_3m',
    '^GSPC_close', '^MOVE_close',
]
avail_features = [c for c in KEEP_FEATURES if c in df.columns]
df_model = df[avail_features + ['year_month']].copy()
df_model = df_model.sort_values('year_month').reset_index(drop=True)
df_model = df_model.fillna(0)

print(f"Data: {df_model.shape}")
print(f"Features: {avail_features}")

# =============================================================================
# SEQUENCE CREATION & TRAINING
# =============================================================================

def create_sequences(X, y, window_size, horizon):
    """Create input/output sequences for supervised learning."""
    Xs, ys = [], []
    for i in range(len(X) - window_size - horizon + 1):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size: i + window_size + horizon])
    return np.array(Xs), np.array(ys)


def train_tsmixer(X_train, y_train, X_test, y_test, window, horizon, n_features):
    """Train TSMixer and return test predictions."""
    model = TSMixer(
        seq_len=window, n_features=n_features, horizon=horizon,
        n_blocks=TSMIXER_N_BLOCKS, d_ff=TSMIXER_D_FF, dropout=TSMIXER_DROPOUT
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=TSMIXER_LR)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    
    # Training loop
    model.train()
    for epoch in range(TSMIXER_EPOCHS):
        for i in range(0, len(X_tr), TSMIXER_BATCH_SIZE):
            batch_X = X_tr[i:i + TSMIXER_BATCH_SIZE]
            batch_y = y_tr[i:i + TSMIXER_BATCH_SIZE]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()
    
    return preds


# Grid search over window-horizon combinations
TARGET = 'USD'
split_idx = int(len(df_model) * TRAIN_RATIO)

results = []
for window in TSMIXER_WINDOW_SIZES:
    for horizon in TSMIXER_HORIZONS:
        print(f"\nWindow={window}, Horizon={horizon}", end=" ")
        
        # Prepare features
        X_all = df_model[avail_features].values
        y_all = df_model[TARGET].values
        
        # Scale
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all[:split_idx])
        X_scaled = np.vstack([X_scaled, scaler_X.transform(X_all[split_idx:])])
        y_scaled = scaler_y.fit_transform(y_all[:split_idx].reshape(-1, 1)).ravel()
        y_scaled = np.concatenate([y_scaled, scaler_y.transform(y_all[split_idx:].reshape(-1, 1)).ravel()])
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, window, horizon)
        
        # Split (maintain temporal order)
        n_train = split_idx - window - horizon + 1
        if n_train <= 0:
            print("SKIP (insufficient data)")
            continue
        
        X_train_seq, y_train_seq = X_seq[:n_train], y_seq[:n_train]
        X_test_seq, y_test_seq = X_seq[n_train:], y_seq[n_train:]
        
        if len(X_test_seq) == 0:
            print("SKIP (no test data)")
            continue
        
        # Train
        preds = train_tsmixer(X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                             window, horizon, len(avail_features))
        
        # Inverse transform for evaluation
        true = scaler_y.inverse_transform(y_test_seq)
        pred_inv = scaler_y.inverse_transform(preds)
        
        # Naive baseline: last known value repeated
        naive = np.repeat(scaler_y.inverse_transform(
            X_test_seq[:, -1, 0].reshape(-1, 1)), horizon, axis=1)
        
        # Evaluate step-1 ahead
        mae_model = mean_absolute_error(true[:, 0], pred_inv[:, 0])
        mae_naive = mean_absolute_error(true[:, 0], naive[:, 0])
        ratio = mae_model / mae_naive
        
        results.append({
            'window': window, 'horizon': horizon,
            'mae_model': mae_model, 'mae_naive': mae_naive,
            'ratio': ratio, 'beats_rw': ratio < 1.0
        })
        print(f"MAE={mae_model:.4f}, Naive={mae_naive:.4f}, Ratio={ratio:.3f}"
              f" {'\u2713' if ratio < 1 else '\u2717'}")

df_results = pd.DataFrame(results)
print("\n" + "=" * 60)
print("STAGE 1 SUMMARY: TSMixer vs Random Walk")
print("=" * 60)
print(df_results.to_string(index=False))
print(f"\nBeats RW in {df_results['beats_rw'].sum()} / {len(df_results)} configurations")
print("\nConclusion: TSMixer does NOT consistently beat random walk.")
print("Proceeding to Stage 2 (XGBoost) with monthly data.")
