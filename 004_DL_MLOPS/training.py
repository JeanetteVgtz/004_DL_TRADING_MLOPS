# training.py
"""
Entrenar el modelo CNN con MLFlow.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import mlflow
import mlflow.keras

from model_cnn import crear_ventanas, crear_etiquetas, crear_modelo

print("="*60)
print("ENTRENAMIENTO CON MLFLOW")
print("="*60)

# 1. CARGAR DATOS
print("\n[1/7] Cargando datos...")
train = pd.read_csv('data/processed/train_features.csv', parse_dates=['date'])
val = pd.read_csv('data/processed/val_features.csv', parse_dates=['date'])

print(f"  Train: {len(train)} filas")
print(f"  Val:   {len(val)} filas")

# 2. PREPARAR INDICADORES
print("\n[2/7] Preparando indicadores...")
columnas_quitar = ['date', 'open', 'high', 'low', 'close', 'volume', 'retorno_siguiente']
indicadores = [col for col in train.columns if col not in columnas_quitar]
print(f"  Usando {len(indicadores)} indicadores")

# 3. NORMALIZAR
print("\n[3/7] Normalizando datos...")
scaler = StandardScaler()
scaler.fit(train[indicadores])

train_scaled = train.copy()
val_scaled = val.copy()
train_scaled[indicadores] = scaler.transform(train[indicadores])
val_scaled[indicadores] = scaler.transform(val[indicadores])

# Guardar scaler
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✅ Scaler guardado")

# 4. CREAR VENTANAS
print("\n[4/7] Creando ventanas...")
VENTANA = 30

X_train, indices_train = crear_ventanas(train_scaled[indicadores].values, VENTANA)
X_val, indices_val = crear_ventanas(val_scaled[indicadores].values, VENTANA)

y_train = crear_etiquetas(train_scaled, indices_train)
y_val = crear_etiquetas(val_scaled, indices_val)

# 5. CREAR MODELO
print("\n[5/7] Creando modelo CNN...")
modelo = crear_modelo(ventana=VENTANA, num_indicadores=len(indicadores))

# Calcular pesos de clases
clases = np.unique(y_train)
pesos = compute_class_weight('balanced', classes=clases, y=y_train)
pesos_dict = {i: peso for i, peso in enumerate(pesos)}
print(f"  Pesos de clases: {pesos_dict}")

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 6. ENTRENAR CON MLFLOW
print("\n[6/7] Entrenando con MLFlow...")
print("="*60)

mlflow.set_experiment("trading_cnn")

with mlflow.start_run():
    
    # Registrar parámetros
    mlflow.log_param("ventana", VENTANA)
    mlflow.log_param("num_indicadores", len(indicadores))
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("architecture", "CNN_2layers")
    
    # Entrenar
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=pesos_dict,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Registrar métricas finales
    acc_train = historial.history['accuracy'][-1]
    acc_val = historial.history['val_accuracy'][-1]
    loss_train = historial.history['loss'][-1]
    loss_val = historial.history['val_loss'][-1]
    
    mlflow.log_metric("accuracy_train", acc_train)
    mlflow.log_metric("accuracy_val", acc_val)
    mlflow.log_metric("loss_train", loss_train)
    mlflow.log_metric("loss_val", loss_val)
    
    # Guardar modelo
    modelo.save('models/best_model.h5')
    mlflow.keras.log_model(modelo, "model")
    mlflow.log_artifact('models/best_model.h5')
    
    print(f"\nModelo guardado en models/best_model.h5")

# 7. RESULTADOS
print(f"\n{'='*60}")
print("RESULTADOS FINALES")
print(f"{'='*60}")
print(f"Accuracy Train: {acc_train:.2%}")
print(f"Accuracy Val:   {acc_val:.2%}")
print(f"Loss Train:     {loss_train:.4f}")
print(f"Loss Val:       {loss_val:.4f}")
print(f"\nVer experimentos en MLFlow:")
print(f"   mlflow ui")
print(f"   http://localhost:5000")
print("="*60)