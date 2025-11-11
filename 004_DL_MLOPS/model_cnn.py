# model_cnn.py
"""
Funciones para crear ventanas, etiquetas y el modelo CNN.
"""
import warnings
warnings.filterwarnings('ignore', category=Warning)  # Silenciar warnings

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def crear_ventanas(datos, tamano_ventana=30):
    """
    Convertir datos en ventanas deslizantes.
    
    Ejemplo: Si tenemos 100 filas y ventana=30
    - Ventana 1: filas 0-29
    - Ventana 2: filas 1-30
    - etc.
    """
    X = []
    indices_validos = []
    
    for i in range(len(datos) - tamano_ventana):
        ventana = datos[i:i + tamano_ventana]
        X.append(ventana)
        indices_validos.append(i + tamano_ventana)
    
    X = np.array(X)
    print(f"  Creadas {len(X)} ventanas de forma {X.shape[1:]}")
    
    return X, indices_validos


def crear_etiquetas(df, indices_validos, umbral=0.01):
    """
    Crear etiquetas:
    - 0 = SHORT (bajará)
    - 1 = HOLD (se mantendrá)
    - 2 = LONG (subirá)
    """
    df = df.copy()
    df['retorno_siguiente'] = df['close'].pct_change().shift(-1)
    
    etiquetas = []
    for idx in indices_validos:
        retorno = df.iloc[idx]['retorno_siguiente']
        
        if retorno > umbral:
            etiquetas.append(2)  # LONG
        elif retorno < -umbral:
            etiquetas.append(0)  # SHORT
        else:
            etiquetas.append(1)  # HOLD
    
    etiquetas = np.array(etiquetas)
    
    # Mostrar distribución
    n_short = (etiquetas == 0).sum()
    n_hold = (etiquetas == 1).sum()
    n_long = (etiquetas == 2).sum()
    total = len(etiquetas)
    
    print("Etiquetas creadas:")
    print(f"    SHORT: {n_short} ({n_short/total*100:.1f}%)")
    print(f"    HOLD:  {n_hold} ({n_hold/total*100:.1f}%)")
    print(f"    LONG:  {n_long} ({n_long/total*100:.1f}%)")
    
    return etiquetas


def crear_modelo(ventana, num_indicadores):
    """Crear red neuronal CNN"""
    modelo = keras.Sequential([
        layers.Input(shape=(ventana, num_indicadores)),
        
        # Capa convolucional 1
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        
        # Capa convolucional 2
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Salida: 3 clases
        layers.Dense(3, activation='softmax')
    ])
    
    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo


def cargar_modelo(ruta):
    """Cargar modelo entrenado"""
    modelo = keras.models.load_model(ruta)
    print(f"Modelo cargado desde {ruta}")
    return modelo