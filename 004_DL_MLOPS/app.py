# api/app.py
"""
API REST con FastAPI para servir predicciones.
Ejecutar con: uvicorn api.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow import keras
from typing import List
import uvicorn

app = FastAPI(
    title="Trading Signal API",
    description="API para predecir señales de trading",
    version="1.0"
)

# Rutas
MODEL_PATH = '../models/best_model.h5'
SCALER_PATH = '../models/scaler.pkl'

modelo = None
scaler = None


@app.on_event("startup")
def cargar_recursos():
    """Cargar modelo y scaler al iniciar"""
    global modelo, scaler
    
    try:
        modelo = keras.models.load_model(MODEL_PATH)
        print(f"Modelo cargado")
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler cargado")
        
    except Exception as e:
        print(f"Error: {e}")


# Modelos de datos
class SecuenciaInput(BaseModel):
    secuencia: List[List[float]]
    umbral_long: float = 0.55
    umbral_short: float = 0.55


class PrediccionOutput(BaseModel):
    senal: str
    probabilidades: dict
    confianza: float


# Endpoints
@app.get("/")
def root():
    """Health check"""
    return {
        "mensaje": "API funcionando",
        "estado": "OK" if modelo is not None else "Error"
    }


@app.post("/predict", response_model=PrediccionOutput)
def predecir(input_data: SecuenciaInput):
    """
    Predecir señal de trading.
    
    Entrada:
    {
        "secuencia": [[val1, val2, ...], [val3, val4, ...], ...],
        "umbral_long": 0.55,
        "umbral_short": 0.55
    }
    
    Salida:
    {
        "senal": "long" | "short" | "hold",
        "probabilidades": {"short": 0.2, "hold": 0.3, "long": 0.5},
        "confianza": 0.5
    }
    """
    
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        # Convertir a numpy
        X = np.array([input_data.secuencia])
        
        if len(X.shape) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Forma incorrecta: {X.shape}"
            )
        
        # Predecir
        probs = modelo.predict(X, verbose=0)[0]
        
        p_short = float(probs[0])
        p_hold = float(probs[1])
        p_long = float(probs[2])
        
        # Determinar señal
        if p_long >= input_data.umbral_long:
            senal = "long"
        elif p_short >= input_data.umbral_short:
            senal = "short"
        else:
            senal = "hold"
        
        return PrediccionOutput(
            senal=senal,
            probabilidades={
                "short": p_short,
                "hold": p_hold,
                "long": p_long
            },
            confianza=float(max(probs))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predecir_lote(secuencias: List[List[List[float]]], 
                  umbral_long: float = 0.55,
                  umbral_short: float = 0.55):
    """Predecir múltiples secuencias"""
    
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        X = np.array(secuencias)
        probs = modelo.predict(X, verbose=0)
        
        resultados = []
        for prob in probs:
            p_short, p_hold, p_long = prob
            
            if p_long >= umbral_long:
                senal = "long"
            elif p_short >= umbral_short:
                senal = "short"
            else:
                senal = "hold"
            
            resultados.append({
                "senal": senal,
                "probabilidades": {
                    "short": float(p_short),
                    "hold": float(p_hold),
                    "long": float(p_long)
                },
                "confianza": float(max(prob))
            })
        
        return {"predicciones": resultados, "total": len(resultados)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def info_modelo():
    """Info del modelo cargado"""
    if modelo is None:
        return {"estado": "Modelo no cargado"}
    
    return {
        "estado": "Modelo cargado",
        "arquitectura": "1D CNN",
        "input_shape": str(modelo.input_shape),
        "num_parametros": modelo.count_params()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)