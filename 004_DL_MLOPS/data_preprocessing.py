# data_preprocessing.py
"""
Descargar datos de Yahoo Finance y dividirlos en 3 partes.
"""

import pandas as pd
import yfinance as yf
import os

print("="*60)
print("DESCARGA Y PREPROCESAMIENTO DE DATOS")
print("="*60)

# Descargar datos
print("\n[1/4] Descargando datos de SPY...")
data = yf.download('SPY', period='15y', progress=False, auto_adjust=False)

# Resetear index para obtener la columna 'Date'
data = data.reset_index()

# IMPORTANTE: Aplanar MultiIndex ANTES de intentar usar .str
if isinstance(data.columns, pd.MultiIndex):
    # Convertir MultiIndex a Index simple
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# AHORA SÍ podemos usar list comprehension para minúsculas
data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]

print(f"   Columnas descargadas: {list(data.columns)}")

# Seleccionar columnas necesarias
# Nota: yfinance puede devolver 'adj_close' o simplemente no tenerla
columnas_disponibles = data.columns.tolist()

# Mapeo flexible de columnas
columnas_finales = ['date', 'open', 'high', 'low', 'close', 'volume']

# Verificar que tengamos lo necesario
for col in columnas_finales:
    if col not in columnas_disponibles:
        print(f"Error: Columna '{col}' no encontrada")
        print(f"   Columnas disponibles: {columnas_disponibles}")
        exit(1)

# Seleccionar solo columnas necesarias
data = data[columnas_finales]

# Ordenar por fecha
data = data.sort_values('date').reset_index(drop=True)

print(f"Descargados {len(data)} días de datos")
print(f"   Desde: {data['date'].min()}")
print(f"   Hasta: {data['date'].max()}")

# Rellenar valores faltantes
print("\n[2/4] Limpiando datos...")
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

missing = data.isnull().sum().sum()
print(f"Valores faltantes: {missing}")

# Dividir en 3 partes
print("\n[3/4] Dividiendo datos...")
total_filas = len(data)
fin_train = int(total_filas * 0.6)
fin_test = int(total_filas * 0.8)

train = data.iloc[:fin_train].copy()
test = data.iloc[fin_train:fin_test].copy()
val = data.iloc[fin_test:].copy()

print(f"Train: {len(train)} filas ({train['date'].min()} a {train['date'].max()})")
print(f"Test:  {len(test)} filas ({test['date'].min()} a {test['date'].max()})")
print(f"Val:   {len(val)} filas ({val['date'].min()} a {val['date'].max()})")

# Guardar archivos
print("\n[4/4] Guardando archivos...")
os.makedirs('data/processed', exist_ok=True)

train.to_csv('data/processed/train_raw.csv', index=False)
test.to_csv('data/processed/test_raw.csv', index=False)
val.to_csv('data/processed/val_raw.csv', index=False)

print("Datos guardados en data/processed/")
print("\n" + "="*60)
print("PREPROCESAMIENTO COMPLETO")
print("="*60)