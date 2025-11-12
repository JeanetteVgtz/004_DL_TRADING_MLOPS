# feature_engineering.py
"""
Generate technical indicators for time series analysis.
Includes momentum, volatility, trend, and volume-based features.
Optimized version: ~30 key indicators for stable model performance.
"""

import pandas as pd
import numpy as np

# =========================================================
# Technical Indicator Functions
# =========================================================

def calculate_rsi(prices, period=14):
    """Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_moving_average(prices, window):
    """Simple Moving Average (SMA)."""
    return prices.rolling(window=window).mean()


def calculate_returns(prices, days):
    """Percentage returns over N days."""
    return prices.pct_change(days) * 100


def calculate_volatility(prices, window):
    """Rolling standard deviation of returns (volatility)."""
    return prices.rolling(window=window).std()


# =========================================================
# Feature Generation Function
# =========================================================

def add_all_indicators(df):
    """
    Add technical indicators to the dataframe.
    Balanced mix of momentum, trend, volatility, and volume (≈30 features).
    """
    df = df.copy()

    # 1. RSI (Momentum)
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["rsi_30"] = calculate_rsi(df["close"], 30)

    # 2. Moving Averages (Trend)
    for window in [5, 20, 50]:
        df[f"ma_{window}"] = calculate_moving_average(df["close"], window)

    # 3. Distance from MA (%)
    df["dist_ma_5"] = (df["close"] - df["ma_5"]) / df["ma_5"] * 100
    df["dist_ma_20"] = (df["close"] - df["ma_20"]) / df["ma_20"] * 100

    # 4. Returns (Momentum)
    for days in [1, 5, 20]:
        df[f"return_{days}d"] = calculate_returns(df["close"], days)

    # 5. Volatility
    for window in [5, 20]:
        df[f"volatility_{window}"] = calculate_volatility(df["close"], window)

    # 6. Daily range
    df["daily_range"] = df["high"] - df["low"]
    df["range_pct"] = (df["daily_range"] / df["close"]) * 100

    # 7. Volume indicators
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # 8. Momentum (price difference)
    for period in [5, 10]:
        df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)

    # 9. Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 10. ATR (Average True Range)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["close"].shift())
    df["tr3"] = abs(df["low"] - df["close"].shift())
    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr_14"] = df["true_range"].rolling(window=14).mean()
    df["atr_pct"] = (df["atr_14"] / df["close"]) * 100

    # 11. MACD (Moving Average Convergence Divergence)
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 12. Relative price ratios
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]

    # 13. Slope and acceleration (momentum change)
    df["slope_5"] = df["close"].diff(5)
    df["accel_5"] = df["slope_5"].diff()

    # =========================================================
    # Final cleanup
    # =========================================================
    df = df.drop([
        "tr1", "tr2", "tr3", "true_range", "bb_std",
        "ema_12", "ema_26"
    ], axis=1)

    n_features = len([c for c in df.columns if c not in ["date", "open", "high", "low", "close", "volume"]])
    print(f"    {n_features} features generated.")
    return df


# =========================================================
# Main Execution Script
# =========================================================

if __name__ == "__main__":

    # 1. Load raw datasets
    train = pd.read_csv("data/processed/train_raw.csv", parse_dates=["date"])
    test = pd.read_csv("data/processed/test_raw.csv", parse_dates=["date"])
    val = pd.read_csv("data/processed/val_raw.csv", parse_dates=["date"])

    # 2. Generate features
    train = add_all_indicators(train)
    test = add_all_indicators(test)
    val = add_all_indicators(val)

    # 3. Clean NaN rows
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)

    print(f"   Train: {len(train)} rows")
    print(f"   Test:  {len(test)} rows")
    print(f"   Val:   {len(val)} rows")

    # 4. Save processed datasets
    train.to_csv("data/processed/train_features.csv", index=False)
    test.to_csv("data/processed/test_features.csv", index=False)
    val.to_csv("data/processed/val_features.csv", index=False)
