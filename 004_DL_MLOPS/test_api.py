# test_api.py
"""
Script para probar la API.
Ejecutar DESPUÉS de iniciar: uvicorn api.app:app --reload
"""

import requests
import numpy as np

BASE_URL = "http://localhost:8000"

print("="*60)
print("PROBANDO API")
print("="*60)

# Test 1: Health check
print("\n[1/3] Health check...")
try:
    r = requests.get(f"{BASE_URL}/")
    print(f"✅ {r.json()}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("¿Está corriendo la API?")
    exit()

# Test 2: Info del modelo
print("\n[2/3] Info del modelo...")
try:
    r = requests.get(f"{BASE_URL}/model_info")
    print(f"✅ {r.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Predicción
print("\n[3/3] Predicción de prueba...")
try:
    # Datos aleatorios (30 días, 35 indicadores)
    secuencia = np.random.randn(30, 35).tolist()
    
    datos = {
        "secuencia": secuencia,
        "umbral_long": 0.55,
        "umbral_short": 0.55
    }
    
    r = requests.post(f"{BASE_URL}/predict", json=datos)
    
    if r.status_code == 200:
        resultado = r.json()
        print(f"✅ Señal: {resultado['senal'].upper()}")
        print(f"   Probabilidades:")
        print(f"     Short: {resultado['probabilidades']['short']:.3f}")
        print(f"     Hold:  {resultado['probabilidades']['hold']:.3f}")
        print(f"     Long:  {resultado['probabilidades']['long']:.3f}")
        print(f"   Confianza: {resultado['confianza']:.3f}")
    else:
        print(f"❌ Error: {r.status_code}")
        print(r.json())

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("PRUEBAS COMPLETAS")
print("Para docs interactivos: http://localhost:8000/docs")
print("="*60)