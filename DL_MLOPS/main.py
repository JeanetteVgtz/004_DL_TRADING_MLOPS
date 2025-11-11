# main.py
"""
Run full trading pipeline:
- Load processed validation data
- Apply trained CNN model to generate trading signals
- Execute backtest using generated signals
- Display and evaluate performance metrics
"""

import pandas as pd
import numpy as np
import pickle
from tensorflow import keras

from backtest import execute_backtest
from feature_engineering import add_all_indicators
from model_cnn import create_sequences
from evaluation import compute_full_metrics, plot_equity_curve


def probs_to_signals(pred_probs, long_thr=0.55, short_thr=0.55):
    """
    Convert predicted probabilities into trading signals.
    
    Rules:
        - If prob(long) >= long_thr → signal = +1 (LONG)
        - If prob(short) >= short_thr → signal = -1 (SHORT)
        - Else → signal = 0 (HOLD)
    """
    signals = []
    for prob in pred_probs:
        p_short, p_hold, p_long = prob
        if p_long >= long_thr:
            signals.append(1)
        elif p_short >= short_thr:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 6: FULL BACKTESTING PIPELINE")
    print("=" * 70)

    # 1️⃣ LOAD VALIDATION DATA
    print("\n[1/7] Loading validation data...")
    df = pd.read_csv("data/processed/val_raw.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  ✅ {len(df)} rows loaded")

    # 2️⃣ FEATURE ENGINEERING
    print("\n[2/7] Generating technical indicators...")
    df = add_all_indicators(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  ✅ {len(df)} rows after indicator generation")

    # 3️⃣ FEATURE PREPARATION
    print("\n[3/7] Preparing features...")
    exclude_cols = ["date", "open", "high", "low", "close", "volume"]
    features = [col for col in df.columns if col not in exclude_cols]
    print(f"  ✅ Using {len(features)} features")

    # 4️⃣ NORMALIZATION
    print("\n[4/7] Loading and applying scaler...")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    df[features] = scaler.transform(df[features])
    print("  ✅ Scaler applied")

    # 5️⃣ CREATE SEQUENCES AND PREDICT
    print("\n[5/7] Creating input windows and predicting...")
    X_val, valid_idx = create_sequences(df[features].values, window_size=30)

    model = keras.models.load_model("models/best_model.h5")
    pred_probs = model.predict(X_val, verbose=0)
    print(f"  ✅ {len(pred_probs)} predictions generated")

    # 6️⃣ CONVERT TO SIGNALS
    print("\n[6/7] Converting predictions to trading signals...")
    signals = probs_to_signals(pred_probs, long_thr=0.55, short_thr=0.55)

    # Attach signals to dataframe
    df["signal"] = 0
    df.loc[valid_idx, "signal"] = signals

    # Signal distribution
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_hold = (signals == 0).sum()
    total = len(signals)
    print(f"  LONG:  {n_long} ({n_long/total*100:.1f}%)")
    print(f"  SHORT: {n_short} ({n_short/total*100:.1f}%)")
    print(f"  HOLD:  {n_hold} ({n_hold/total*100:.1f}%)")

    # 7️⃣ RUN BACKTEST
    print("\n" + "=" * 70)
    print("EXECUTING BACKTEST")
    print("=" * 70)
    results, final_cash = execute_backtest(df)

    # 💰 Summary
    start_cash = 1_000_000
    net_return = (final_cash / start_cash - 1) * 100
    print(f"\n✅ Final capital: {final_cash:,.2f}")
    print(f"💹 Net return: {net_return:.2f}%")

    # OPTIONAL: Evaluate and plot
    try:
        compute_full_metrics(results, initial_cash=start_cash)
        plot_equity_curve(results)
    except Exception as e:
        print(f"⚠️ Skipping metrics/plots: {e}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✅")
    print(f"Final Capital: {final_cash:,.2f}")
    print(f"Net Return: {net_return:.2f}%")
    print("=" * 70)
