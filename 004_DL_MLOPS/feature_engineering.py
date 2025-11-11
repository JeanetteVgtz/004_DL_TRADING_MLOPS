# feature_engineering.py
"""
Calcular indicadores técnicos (RSI, medias móviles, volatilidad, etc.)
"""

import pandas as pd
import numpy as np

def calcular_rsi(precios, periodo=14):
    """RSI: si está alto (>70) = caro, si está bajo (<30) = barato"""
    cambios = precios.diff()
    
    ganancias = cambios.copy()
    ganancias[ganancias < 0] = 0
    
    perdidas = cambios.copy()
    perdidas[perdidas > 0] = 0
    perdidas = abs(perdidas)
    
    promedio_ganancias = ganancias.rolling(window=periodo).mean()
    promedio_perdidas = perdidas.rolling(window=periodo).mean()
    
    rs = promedio_ganancias / promedio_perdidas
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calcular_media_movil(precios, ventana):
    """Media móvil: promedio de los últimos N días"""
    return precios.rolling(window=ventana).mean()


def calcular_retornos(precios, dias):
    """Retorno: cuánto cambió el precio en N días (en %)"""
    return precios.pct_change(dias) * 100


def calcular_volatilidad(precios, ventana):
    """Volatilidad: qué tanto varía el precio"""
    return precios.rolling(window=ventana).std()


def agregar_indicadores(df):
    """Agregar todos los indicadores técnicos"""
    print("  Calculando indicadores...")
    
    df = df.copy()
    
    # 1. RSI (14 y 30 días)
    df['rsi_14'] = calcular_rsi(df['close'], 14)
    df['rsi_30'] = calcular_rsi(df['close'], 30)
    
    # 2. Medias móviles
    df['ma_5'] = calcular_media_movil(df['close'], 5)
    df['ma_10'] = calcular_media_movil(df['close'], 10)
    df['ma_20'] = calcular_media_movil(df['close'], 20)
    df['ma_50'] = calcular_media_movil(df['close'], 50)
    
    # 3. Distancia a la media
    df['dist_ma_5'] = (df['close'] - df['ma_5']) / df['ma_5'] * 100
    df['dist_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20'] * 100
    
    # 4. Retornos
    df['retorno_1d'] = calcular_retornos(df['close'], 1)
    df['retorno_5d'] = calcular_retornos(df['close'], 5)
    df['retorno_10d'] = calcular_retornos(df['close'], 10)
    df['retorno_20d'] = calcular_retornos(df['close'], 20)
    
    # 5. Volatilidad
    df['volatilidad_5'] = calcular_volatilidad(df['close'], 5)
    df['volatilidad_10'] = calcular_volatilidad(df['close'], 10)
    df['volatilidad_20'] = calcular_volatilidad(df['close'], 20)
    
    # 6. Rango diario
    df['rango_diario'] = df['high'] - df['low']
    df['rango_pct'] = (df['rango_diario'] / df['close']) * 100
    
    # 7. Volumen
    df['volumen_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volumen_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volumen_ratio'] = df['volume'] / df['volumen_ma_20']
    
    # 8. Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # 9. Bollinger Bands
    df['bb_media'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_superior'] = df['bb_media'] + (2 * df['bb_std'])
    df['bb_inferior'] = df['bb_media'] - (2 * df['bb_std'])
    df['bb_posicion'] = (df['close'] - df['bb_inferior']) / (df['bb_superior'] - df['bb_inferior'])
    
    # 10. ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    
    # Limpiar columnas temporales
    df = df.drop(['tr1', 'tr2', 'tr3', 'true_range', 'bb_std'], axis=1)
    
    num_indicadores = len([col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']])
    print(f"{num_indicadores} indicadores agregados")
    
    return df


if __name__ == "__main__":
    print("="*60)
    print("INGENIERÍA DE CARACTERÍSTICAS")
    print("="*60)
    
    # Leer datos crudos
    print("\n[1/4] Cargando datos...")
    train = pd.read_csv('data/processed/train_raw.csv', parse_dates=['date'])
    test = pd.read_csv('data/processed/test_raw.csv', parse_dates=['date'])
    val = pd.read_csv('data/processed/val_raw.csv', parse_dates=['date'])
    
    # Agregar indicadores
    print("\n[2/4] Generando features...")
    train = agregar_indicadores(train)
    test = agregar_indicadores(test)
    val = agregar_indicadores(val)
    
    # Quitar filas con NaN
    print("\n[3/4] Limpiando NaN...")
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    
    print(f"  Train: {len(train)} filas")
    print(f"  Test:  {len(test)} filas")
    print(f"  Val:   {len(val)} filas")
    
    # Guardar
    print("\n[4/4] Guardando archivos...")
    train.to_csv('data/processed/train_features.csv', index=False)
    test.to_csv('data/processed/test_features.csv', index=False)
    val.to_csv('data/processed/val_features.csv', index=False)
    
    print("Features guardados en data/processed/")
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETO")
    print("="*60)