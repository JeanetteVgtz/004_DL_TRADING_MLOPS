# api/app.py
"""
REST API using FastAPI to serve trading signal predictions.
Run with: uvicorn api.app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
from tensorflow import keras
import os
import uvicorn

# ------------------------------------------------------------
# APP INITIALIZATION
# ------------------------------------------------------------
app = FastAPI(
    title="Trading Signal API",
    description="API to predict trading signals from the trained CNN model",
    version="1.0"
)

# ------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

model = None
scaler = None


# ------------------------------------------------------------
# STARTUP EVENT
# ------------------------------------------------------------
@app.on_event("startup")
def load_resources():
    """Load model and scaler when starting the API"""
    global model, scaler
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading resources: {e}")


# ------------------------------------------------------------
# INPUT AND OUTPUT SCHEMAS
# ------------------------------------------------------------
class SequenceInput(BaseModel):
    sequence: List[List[float]]
    long_threshold: float = 0.55
    short_threshold: float = 0.55


class PredictionOutput(BaseModel):
    signal: str
    probabilities: dict
    confidence: float


# ------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "API running correctly",
        "status": "OK" if model is not None else "Model not loaded"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_signal(input_data: SequenceInput):
    """
    Predict trading signal from a single sequence.
    
    Input example:
    {
        "sequence": [[val1, val2, ...], [val3, val4, ...], ...],
        "long_threshold": 0.55,
        "short_threshold": 0.55
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array([input_data.sequence])

        if len(X.shape) != 3:
            raise HTTPException(status_code=400, detail=f"Invalid shape: {X.shape}")

        # Predict
        probs = model.predict(X, verbose=0)[0]
        p_short, p_hold, p_long = map(float, probs)

        # Determine trading signal
        if p_long >= input_data.long_threshold:
            signal = "LONG"
        elif p_short >= input_data.short_threshold:
            signal = "SHORT"
        else:
            signal = "HOLD"

        return PredictionOutput(
            signal=signal,
            probabilities={
                "SHORT": p_short,
                "HOLD": p_hold,
                "LONG": p_long
            },
            confidence=float(max(probs))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predict_batch(
    sequences: List[List[List[float]]],
    long_threshold: float = 0.55,
    short_threshold: float = 0.55
):
    """
    Predict trading signals for multiple sequences.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array(sequences)
        probs = model.predict(X, verbose=0)

        results = []
        for prob in probs:
            p_short, p_hold, p_long = map(float, prob)

            if p_long >= long_threshold:
                signal = "LONG"
            elif p_short >= short_threshold:
                signal = "SHORT"
            else:
                signal = "HOLD"

            results.append({
                "signal": signal,
                "probabilities": {
                    "SHORT": p_short,
                    "HOLD": p_hold,
                    "LONG": p_long
                },
                "confidence": float(max(prob))
            })

        return {"predictions": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def model_info():
    """Return model information"""
    if model is None:
        return {"status": "Model not loaded"}

    return {
        "status": "Model loaded",
        "architecture": "1D CNN",
        "input_shape": str(model.input_shape),
        "total_parameters": model.count_params()
    }

@app.get("/health")
def health_check():
    """Health check for Streamlit connectivity"""
    return {"status": "OK", "message": "API running correctly"}

# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
