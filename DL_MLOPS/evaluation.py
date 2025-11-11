# evaluation.py
"""
Model and strategy evaluation utilities.
Includes classification metrics, performance ratios, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


# ==========================
# CLASSIFICATION METRICS
# ==========================

def evaluate_classification(y_true, y_pred):
    """Evaluate CNN classification performance."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)
    
    class_names = ['Short', 'Hold', 'Long']
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix and save to file."""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Short', 'Hold', 'Long'],
                yticklabels=['Short', 'Hold', 'Long'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion matrix saved at {save_path}")
    plt.close()


# ==========================
# PERFORMANCE METRICS
# ==========================

def compute_sharpe(returns, risk_free_rate=0.02):
    """Compute annualized Sharpe Ratio."""
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    if annual_volatility == 0:
        return 0
    
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    return sharpe


def compute_max_drawdown(values):
    """Compute Maximum Drawdown."""
    peaks = values.cummax()
    drawdowns = (values - peaks) / peaks
    max_dd = drawdowns.min()
    return max_dd


def compute_calmar(returns, values):
    """Compute Calmar Ratio."""
    annual_return = returns.mean() * 252
    max_dd = abs(compute_max_drawdown(values))
    
    if max_dd == 0:
        return np.inf
    
    calmar = annual_return / max_dd
    return calmar


# ==========================
# VISUALIZATION
# ==========================

def plot_equity_curve(result, save_path='results/equity_curve.png'):
    """Plot portfolio equity curve."""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(result['portfolio_value'], linewidth=2, label='Portfolio')
    
    start_value = result['portfolio_value'].iloc[0]
    plt.axhline(y=start_value, color='r', linestyle='--', 
                label=f'Initial Capital: ${start_value:,.0f}', alpha=0.7)
    
    plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Equity curve saved at {save_path}")
    plt.close()


# ==========================
# FULL TRADING METRICS
# ==========================

def compute_full_metrics(result, initial_cash=1_000_000):
    """Compute complete set of trading performance metrics."""
    print("\n" + "=" * 70)
    print("TRADING PERFORMANCE METRICS")
    print("=" * 70)
    
    # Capital evolution
    final_capital = result['portfolio_value'].iloc[-1]
    total_return = (final_capital / initial_cash - 1) * 100
    
    print(f"\nInitial Capital:  ${initial_cash:,.2f}")
    print(f"Final Capital:    ${final_capital:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    
    # Daily returns
    returns = result['portfolio_value'].pct_change().dropna()
    
    # Sharpe
    sharpe = compute_sharpe(returns)
    print(f"\nSharpe Ratio:     {sharpe:.3f}")
    
    # Max Drawdown
    max_dd = compute_max_drawdown(result['portfolio_value'])
    print(f"Max Drawdown:     {max_dd:.2%}")
    
    # Calmar
    calmar = compute_calmar(returns, result['portfolio_value'])
    print(f"Calmar Ratio:     {calmar:.3f}")
    
    # Win rate and trade stats
    if 'trade_pnl' in result.columns:
        trades = result[result['trade_pnl'] != 0]['trade_pnl']
        if len(trades) > 0:
            wins = (trades > 0).sum()
            win_rate = wins / len(trades) * 100
            
            print(f"\nNumber of Trades: {len(trades)}")
            print(f"Win Rate:         {win_rate:.2f}%")
            
            if wins > 0:
                avg_win = trades[trades > 0].mean()
                print(f"Average Win:      ${avg_win:,.2f}")
            
            losses = len(trades) - wins
            if losses > 0:
                avg_loss = trades[trades < 0].mean()
                print(f"Average Loss:     ${avg_loss:,.2f}")
    
    metrics = {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'final_capital': final_capital
    }
    
    return metrics
