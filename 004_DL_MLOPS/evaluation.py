# evaluation.py
"""
Funciones para evaluar el modelo y la estrategia.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def evaluar_clasificacion(y_true, y_pred):
    """Evaluar métricas de clasificación"""
    print("\n" + "="*60)
    print("MÉTRICAS DE CLASIFICACIÓN")
    print("="*60)
    
    # Reporte
    nombres_clases = ['Short', 'Hold', 'Long']
    print(classification_report(y_true, y_pred, target_names=nombres_clases))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    return cm


def graficar_confusion_matrix(cm, guardar='results/confusion_matrix.png'):
    """Graficar matriz de confusión"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Short', 'Hold', 'Long'],
                yticklabels=['Short', 'Hold', 'Long'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(guardar, dpi=300)
    print(f"Matriz guardada en {guardar}")
    plt.close()


def calcular_sharpe(retornos, tasa_libre_riesgo=0.02):
    """Calcular Sharpe Ratio anualizado"""
    retorno_anual = retornos.mean() * 252
    volatilidad_anual = retornos.std() * np.sqrt(252)
    
    if volatilidad_anual == 0:
        return 0
    
    sharpe = (retorno_anual - tasa_libre_riesgo) / volatilidad_anual
    return sharpe


def calcular_max_drawdown(valores):
    """Calcular máximo drawdown"""
    picos = valores.cummax()
    drawdowns = (valores - picos) / picos
    max_dd = drawdowns.min()
    return max_dd


def calcular_calmar(retornos, valores):
    """Calcular Calmar Ratio"""
    retorno_anual = retornos.mean() * 252
    max_dd = abs(calcular_max_drawdown(valores))
    
    if max_dd == 0:
        return np.inf
    
    calmar = retorno_anual / max_dd
    return calmar


def graficar_equity_curve(resultado, guardar='results/equity_curve.png'):
    """Graficar curva de capital"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(resultado['portfolio_value'], linewidth=2, label='Portfolio')
    
    inicio = resultado['portfolio_value'].iloc[0]
    plt.axhline(y=inicio, color='r', linestyle='--', 
                label=f'Capital Inicial: ${inicio:,.0f}', alpha=0.7)
    
    plt.title('Valor del Portafolio', fontsize=14, fontweight='bold')
    plt.xlabel('Día de Trading')
    plt.ylabel('Valor ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(guardar, dpi=300)
    print(f"Equity curve guardada en {guardar}")
    plt.close()


def calcular_metricas_completas(resultado, capital_inicial=1_000_000):
    """Calcular todas las métricas de trading"""
    print("\n" + "="*60)
    print("MÉTRICAS DE TRADING")
    print("="*60)
    
    # Capital final
    capital_final = resultado['portfolio_value'].iloc[-1]
    retorno_total = (capital_final / capital_inicial - 1) * 100
    
    print(f"\nCapital Inicial:  ${capital_inicial:,.2f}")
    print(f"Capital Final:    ${capital_final:,.2f}")
    print(f"Retorno Total:    {retorno_total:.2f}%")
    
    # Retornos diarios
    retornos = resultado['portfolio_value'].pct_change().dropna()
    
    # Sharpe
    sharpe = calcular_sharpe(retornos)
    print(f"\nSharpe Ratio:     {sharpe:.3f}")
    
    # Max Drawdown
    max_dd = calcular_max_drawdown(resultado['portfolio_value'])
    print(f"Max Drawdown:     {max_dd:.2%}")
    
    # Calmar
    calmar = calcular_calmar(retornos, resultado['portfolio_value'])
    print(f"Calmar Ratio:     {calmar:.3f}")
    
    # Win rate
    if 'trade_pnl' in resultado.columns:
        trades = resultado[resultado['trade_pnl'] != 0]['trade_pnl']
        if len(trades) > 0:
            wins = (trades > 0).sum()
            win_rate = wins / len(trades) * 100
            
            print(f"\nNúmero de Trades: {len(trades)}")
            print(f"Win Rate:         {win_rate:.2f}%")
            
            if wins > 0:
                avg_win = trades[trades > 0].mean()
                print(f"Ganancia Promedio: ${avg_win:,.2f}")
            
            losses = len(trades) - wins
            if losses > 0:
                avg_loss = trades[trades < 0].mean()
                print(f"Pérdida Promedio:  ${avg_loss:,.2f}")
    
    metricas = {
        'retorno_total': retorno_total,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'capital_final': capital_final
    }
    
    return metricas