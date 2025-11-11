# drift_monitor.py
"""
Detectar drift en features usando test de Kolmogorov-Smirnov.
"""

import pandas as pd
from scipy.stats import ks_2samp
import os


def calcular_drift(train_df, test_df, val_df, indicadores):
    """Calcular KS-test para cada indicador"""
    resultados = []
    
    for indicador in indicadores:
        train_vals = train_df[indicador].dropna()
        test_vals = test_df[indicador].dropna()
        val_vals = val_df[indicador].dropna()
        
        # KS test: train vs test
        _, p_test = ks_2samp(train_vals, test_vals)
        
        # KS test: train vs val
        _, p_val = ks_2samp(train_vals, val_vals)
        
        resultados.append({
            'Indicador': indicador,
            'P-Value (Train vs Test)': p_test,
            'P-Value (Train vs Val)': p_val,
            'Drift Test': 'Sí' if p_test < 0.05 else 'No',
            'Drift Val': 'Sí' if p_val < 0.05 else 'No'
        })
    
    return pd.DataFrame(resultados)


if __name__ == "__main__":
    print("="*60)
    print("DETECCIÓN DE DRIFT")
    print("="*60)
    
    # Cargar datos
    print("\n[1/3] Cargando datos...")
    train = pd.read_csv('data/processed/train_features.csv', parse_dates=['date'])
    test = pd.read_csv('data/processed/test_features.csv', parse_dates=['date'])
    val = pd.read_csv('data/processed/val_features.csv', parse_dates=['date'])
    
    # Obtener indicadores
    columnas_quitar = ['date', 'open', 'high', 'low', 'close', 'volume', 'retorno_siguiente']
    indicadores = [col for col in train.columns if col not in columnas_quitar]
    
    print(f"  Analizando {len(indicadores)} indicadores")
    
    # Calcular drift
    print("\n[2/3] Calculando drift (KS-test)...")
    drift_df = calcular_drift(train, test, val, indicadores)
    
    # Ordenar por mayor drift
    drift_df = drift_df.sort_values('P-Value (Train vs Val)')
    
    # Guardar resultados
    print("\n[3/3] Guardando resultados...")
    os.makedirs('results', exist_ok=True)
    drift_df.to_csv('results/drift_metrics.csv', index=False)
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN DE DRIFT")
    print("="*60)
    
    n_drift_test = (drift_df['Drift Test'] == 'Sí').sum()
    n_drift_val = (drift_df['Drift Val'] == 'Sí').sum()
    
    print(f"\nTotal de indicadores:     {len(drift_df)}")
    print(f"Con drift (Test):         {n_drift_test} ({n_drift_test/len(drift_df)*100:.1f}%)")
    print(f"Con drift (Val):          {n_drift_val} ({n_drift_val/len(drift_df)*100:.1f}%)")
    
    print(f"\nTop 5 indicadores con más drift:")
    print(drift_df[['Indicador', 'P-Value (Train vs Val)', 'Drift Val']].head())
    
    print(f"\nResultados guardados en results/drift_metrics.csv")
    print("="*60)