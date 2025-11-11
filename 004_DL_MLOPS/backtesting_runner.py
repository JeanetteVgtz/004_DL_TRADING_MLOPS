# backtesting_runner.py
"""
Generar señales y ejecutar backtest (adaptado a run_backtest).
"""

import pandas as pd
import numpy as np
import pickle
from tensorflow import keras

from backtest import run_backtest
from feature_engineering import agregar_indicadores
from model_cnn import crear_ventanas
from evaluation import calcular_metricas_completas, graficar_equity_curve


def convertir_probabilidades_a_senales(probabilidades, umbral_long=0.55, umbral_short=0.55):
    """
    Convertir probabilidades a señales de trading.
    
    Reglas:
    - Si prob_long >= umbral_long → señal = +1 (LONG)
    - Si prob_short >= umbral_short → señal = -1 (SHORT)
    - Caso contrario → señal = 0 (HOLD)
    """
    senales = []
    
    for prob in probabilidades:
        p_short = prob[0]
        p_hold = prob[1]
        p_long = prob[2]
        
        if p_long >= umbral_long:
            senales.append(1)
        elif p_short >= umbral_short:
            senales.append(-1)
        else:
            senales.append(0)
    
    return np.array(senales)


if __name__ == "__main__":
    print("="*60)
    print("BACKTESTING")
    print("="*60)
    
    # 1. CARGAR DATOS
    print("\n[1/7] Cargando datos de validación...")
    datos = pd.read_csv('data/processed/val_raw.csv', parse_dates=['date'])
    datos = datos.sort_values('date').reset_index(drop=True)
    print(f"  {len(datos)} filas cargadas")
    
    # 2. AGREGAR INDICADORES
    print("\n[2/7] Calculando indicadores...")
    datos = agregar_indicadores(datos)
    datos = datos.dropna().reset_index(drop=True)
    print(f"  {len(datos)} filas después de indicadores")
    
    # 3. PREPARAR INDICADORES
    print("\n[3/7] Preparando features...")
    columnas_quitar = ['date', 'open', 'high', 'low', 'close', 'volume', 'retorno_siguiente']
    indicadores = [col for col in datos.columns if col not in columnas_quitar]
    print(f"  {len(indicadores)} indicadores")
    
    # 4. NORMALIZAR
    print("\n[4/7] Normalizando...")
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    datos[indicadores] = scaler.transform(datos[indicadores])
    
    # 5. CREAR VENTANAS
    print("\n[5/7] Creando ventanas...")
    X, indices_validos = crear_ventanas(datos[indicadores].values, tamano_ventana=30)
    print(f"  {len(X)} ventanas creadas")
    
    # 6. PREDECIR
    print("\n[6/7] Prediciendo con CNN...")
    modelo = keras.models.load_model('models/best_model.h5')
    probabilidades = modelo.predict(X, verbose=0)
    print(f"  {len(probabilidades)} predicciones generadas")
    
    # 7. CONVERTIR A SEÑALES
    print("\n[7/7] Generando señales...")
    senales = convertir_probabilidades_a_senales(
        probabilidades, 
        umbral_long=0.55, 
        umbral_short=0.55
    )
    
    # Contar señales
    n_long = (senales == 1).sum()
    n_short = (senales == -1).sum()
    n_hold = (senales == 0).sum()
    total = len(senales)
    
    print(f"  Señales generadas:")
    print(f"    LONG:  {n_long} ({n_long/total*100:.1f}%)")
    print(f"    SHORT: {n_short} ({n_short/total*100:.1f}%)")
    print(f"    HOLD:  {n_hold} ({n_hold/total*100:.1f}%)")
    
    # 8. AGREGAR SEÑALES AL DATAFRAME
    datos['signal'] = 0
    datos.loc[indices_validos, 'signal'] = senales
    
    # 9. EJECUTAR TU BACKTEST (con los parámetros correctos)
    print("\n" + "="*60)
    print("EJECUTANDO BACKTEST")
    print("="*60)
    
    resultado, capital_final = run_backtest(
        df=datos,                    # ← Cambio: df en lugar de data
        stop_loss=0.02,              # ← Cambio: stop_loss en lugar de stop_thr
        take_profit=0.04,            # ← Cambio: take_profit en lugar de tp_thr
        n_shares=1,                  # ← Cambio: n_shares en lugar de lot_size
        com=0.125/100,               # ← Cambio: com en lugar de comision
        borrow_rate=0.25/100,        # ← Parámetro adicional de tu backtest
        price_col='close',           # ← Cambio: price_col en lugar de col_price
        initial_cash=1_000_000       # ← Cambio: initial_cash en lugar de start_cap
    )
    
    # 10. CALCULAR MÉTRICAS
    metricas = calcular_metricas_completas(resultado, capital_inicial=1_000_000)
    
    # 11. GRAFICAR
    graficar_equity_curve(resultado)
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETO")
    print("="*60)