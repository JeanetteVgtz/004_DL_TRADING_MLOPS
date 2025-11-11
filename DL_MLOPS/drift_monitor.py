# drift_monitor.py
"""
Feature drift detection using the Kolmogorov–Smirnov (KS) test.
Compares the distribution of each technical indicator between
train, test, and validation datasets to detect potential drift.
"""

import pandas as pd
from scipy.stats import ks_2samp
import os


def calculate_drift(train_df, test_df, val_df, indicators):
    """Run KS-test for each feature to detect data drift."""
    results = []
    
    for indicator in indicators:
        train_vals = train_df[indicator].dropna()
        test_vals = test_df[indicator].dropna()
        val_vals = val_df[indicator].dropna()
        
        # KS test: train vs test
        _, p_test = ks_2samp(train_vals, test_vals)
        
        # KS test: train vs validation
        _, p_val = ks_2samp(train_vals, val_vals)
        
        results.append({
            "Feature": indicator,
            "P-Value (Train vs Test)": p_test,
            "P-Value (Train vs Val)": p_val,
            "Drift Test": "Yes" if p_test < 0.05 else "No",
            "Drift Val": "Yes" if p_val < 0.05 else "No"
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 7: FEATURE DRIFT DETECTION")
    print("=" * 70)
    
    # 1️⃣ Load processed datasets
    print("\n[1/3] Loading processed feature datasets...")
    train = pd.read_csv("data/processed/train_features.csv", parse_dates=["date"])
    test = pd.read_csv("data/processed/test_features.csv", parse_dates=["date"])
    val = pd.read_csv("data/processed/val_features.csv", parse_dates=["date"])
    
    # 2️⃣ Define indicators to evaluate
    exclude_cols = ["date", "open", "high", "low", "close", "volume", "next_return"]
    indicators = [col for col in train.columns if col not in exclude_cols]
    
    print(f"  ✅ Analyzing {len(indicators)} technical indicators.")
    
    # 3️⃣ Run drift analysis
    print("\n[2/3] Running KS-test for drift detection...")
    drift_df = calculate_drift(train, test, val, indicators)
    
    # Sort by smallest p-value (strongest drift first)
    drift_df = drift_df.sort_values("P-Value (Train vs Val)")
    
    # Save report
    print("\n[3/3] Saving drift report...")
    os.makedirs("results", exist_ok=True)
    drift_df.to_csv("results/drift_metrics.csv", index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("DRIFT SUMMARY")
    print("=" * 70)
    
    n_drift_test = (drift_df["Drift Test"] == "Yes").sum()
    n_drift_val = (drift_df["Drift Val"] == "Yes").sum()
    
    print(f"\nTotal indicators:        {len(drift_df)}")
    print(f"Drift detected (Test):   {n_drift_test} ({n_drift_test/len(drift_df)*100:.1f}%)")
    print(f"Drift detected (Val):    {n_drift_val} ({n_drift_val/len(drift_df)*100:.1f}%)")
    
    print(f"\nTop 5 indicators with strongest drift:")
    print(drift_df[["Feature", "P-Value (Train vs Val)", "Drift Val"]].head())
    
    print("\n✅ Results saved to: results/drift_metrics.csv")
    print("=" * 70)
