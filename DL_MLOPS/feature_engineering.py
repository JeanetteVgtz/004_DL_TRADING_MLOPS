# feature_engineering.py
"""
Generate technical indicators for time series analysis.
Includes momentum, volatility, and volume-based features (20+ indicators).
"""

import pandas as pd
import numpy as np


# =========================================================
# Technical Indicator Functions
# =========================================================

def calculate_rsi(prices, period=14):
    """
    Relative Strength Index (RSI)
    Momentum indicator: High RSI (>70) = overbought, Low RSI (<30) = oversold
    """
    delta = prices.diff()
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gains).rolling(window=period).mean()
    avg_loss = pd.Series(losses).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_moving_average(prices, window):
    """Simple Moving Average (SMA): trend indicator."""
    return prices.rolling(window=window).mean()


def calculate_returns(prices, days):
    """Price returns over N days (percentage change)."""
    return prices.pct_change(days) * 100


def calculate_volatility(prices, window):
    """Rolling standard deviation of returns (volatility measure)."""
    return prices.rolling(window=window).std()


# =========================================================
# Feature Generation Function
# =========================================================

def add_all_indicators(df):
    """
    Add all technical indicators to the dataframe.
    Produces 30+ features across momentum, volatility, and volume.
    """
    print("  Calculating technical indicators...")
    df = df.copy()

    # 1. RSI (Momentum)
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["rsi_30"] = calculate_rsi(df["close"], 30)

    # 2. Moving Averages (Trend)
    for window in [5, 10, 20, 50]:
        df[f"ma_{window}"] = calculate_moving_average(df["close"], window)

    # 3. Distance from Moving Averages (%)
    df["dist_ma_5"] = (df["close"] - df["ma_5"]) / df["ma_5"] * 100
    df["dist_ma_20"] = (df["close"] - df["ma_20"]) / df["ma_20"] * 100

    # 4. Returns (Momentum)
    for days in [1, 5, 10, 20]:
        df[f"return_{days}d"] = calculate_returns(df["close"], days)

    # 5. Volatility (Rolling Std)
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = calculate_volatility(df["close"], window)

    # 6. Daily Range and Percentage Range
    df["daily_range"] = df["high"] - df["low"]
    df["range_pct"] = (df["daily_range"] / df["close"]) * 100

    # 7. Volume Indicators
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # 8. Momentum (Price Difference)
    for period in [5, 10, 20]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)

    # 9. Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 10. Average True Range (ATR)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["close"].shift())
    df["tr3"] = abs(df["low"] - df["close"].shift())
    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr_14"] = df["true_range"].rolling(window=14).mean()
    df["atr_pct"] = (df["atr_14"] / df["close"]) * 100

    # Drop temporary columns
    df = df.drop(["tr1", "tr2", "tr3", "true_range", "bb_std"], axis=1)

    # Count number of features generated
    n_features = len([c for c in df.columns if c not in ["date", "open", "high", "low", "close", "volume"]])
    print(f"  ✅ {n_features} features generated.")

    return df


# =========================================================
# Main Execution Script
# =========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load raw datasets
    print("\n[1/4] Loading raw data...")
    train = pd.read_csv("data/processed/train_raw.csv", parse_dates=["date"])
    test = pd.read_csv("data/processed/test_raw.csv", parse_dates=["date"])
    val = pd.read_csv("data/processed/val_raw.csv", parse_dates=["date"])

    # 2. Generate features
    print("\n[2/4] Generating technical indicators...")
    train = add_all_indicators(train)
    test = add_all_indicators(test)
    val = add_all_indicators(val)

    # 3. Clean NaN rows (from rolling calculations)
    print("\n[3/4] Cleaning NaN values...")
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)

    print(f"   Train: {len(train)} rows")
    print(f"   Test:  {len(test)} rows")
    print(f"   Val:   {len(val)} rows")

    # 4. Save processed files
    print("\n[4/4] Saving feature datasets...")
    train.to_csv("data/processed/train_features.csv", index=False)
    test.to_csv("data/processed/test_features.csv", index=False)
    val.to_csv("data/processed/val_features.csv", index=False)

    print(" Feature datasets saved to 'data/processed/'")
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
