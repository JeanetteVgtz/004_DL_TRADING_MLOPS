# data_preprocessing.py

"""
Download and preprocess historical price data from Yahoo Finance.
Ensures no look-ahead bias using a chronological 60%/20%/20% split.
"""

import pandas as pd
import yfinance as yf
import os

# -----------------------------
# 1. Download historical data
# -----------------------------
data = yf.download("DIS", period="15y", progress=False, auto_adjust=False)
data = data.reset_index()

# Flatten MultiIndex if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# Standardize column names
data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]
print(f"Columns downloaded: {list(data.columns)}")

# -----------------------------
# 2. Select required columns
# -----------------------------
required_cols = ["date", "open", "high", "low", "close", "volume"]

# Verify column presence
for col in required_cols:
    if col not in data.columns:
        print(f"Error: Missing required column '{col}'")
        print(f"Available columns: {list(data.columns)}")
        exit(1)

# Keep only OHLCV
data = data[required_cols]
data = data.sort_values("date").reset_index(drop=True)

print(f"Data contains {len(data)} daily observations")
print(f"From: {data['date'].min()}  To: {data['date'].max()}")

# -----------------------------
# 3. Handle missing values
# -----------------------------
data = data.fillna(method="ffill")  # Forward fill
data = data.fillna(method="bfill")  # Backward fill
missing = data.isnull().sum().sum()
print(f"Remaining missing values: {missing}")

# -----------------------------
# 4. Chronological split
# -----------------------------
n_rows = len(data)
train_end = int(n_rows * 0.6)
test_end = int(n_rows * 0.8)

train = data.iloc[:train_end].copy()
test = data.iloc[train_end:test_end].copy()
val = data.iloc[test_end:].copy()

print(f"Train: {len(train)} rows  ({train['date'].min()} → {train['date'].max()})")
print(f"Test:  {len(test)} rows  ({test['date'].min()} → {test['date'].max()})")
print(f"Val:   {len(val)} rows  ({val['date'].min()} → {val['date'].max()})")

# -----------------------------
# 5. Save to CSV
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

train.to_csv("data/processed/train_raw.csv", index=False)
test.to_csv("data/processed/test_raw.csv", index=False)
val.to_csv("data/processed/val_raw.csv", index=False)

print(" Data successfully saved in 'data/processed/'")
