# DL_MLOPS/streamlit_drift.py
"""
Streamlit Dashboard - Data Drift Monitoring (Versión Final)
Carga automática de datasets desde data/processed/
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------
st.set_page_config(
    page_title="Data Drift Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Data Drift Monitoring Dashboard")
st.markdown("Visualización automática del drift entre los periodos de entrenamiento, prueba y validación.")

# ------------------------------------------------------------
# RUTAS DE ARCHIVOS
# ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(_file_), "data", "processed")

TRAIN_PATH = os.path.join(DATA_DIR, "train_features.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_features.csv")
VAL_PATH = os.path.join(DATA_DIR, "val_features.csv")

# ------------------------------------------------------------
# CARGA AUTOMÁTICA DE DATOS
# ------------------------------------------------------------
if all(os.path.exists(p) for p in [TRAIN_PATH, TEST_PATH, VAL_PATH]):
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    val_df = pd.read_csv(VAL_PATH)

    st.success("✅ Archivos cargados automáticamente desde data/processed/")
    st.write(f"- Train: {train_df.shape[0]} filas × {train_df.shape[1]} columnas")
    st.write(f"- Test: {test_df.shape[0]} filas × {test_df.shape[1]} columnas")
    st.write(f"- Validation: {val_df.shape[0]} filas × {val_df.shape[1]} columnas")

    # --------------------------------------------------------
    # DETECTAR COLUMNAS NUMÉRICAS
    # --------------------------------------------------------
    features = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # --------------------------------------------------------
    # CÁLCULO DE KS-TEST
    # --------------------------------------------------------
    st.subheader("📊 Drift Statistics Table")

    drift_results = []
    for col in features:
        ks_train_test = ks_2samp(train_df[col].dropna(), test_df[col].dropna()).pvalue
        ks_train_val = ks_2samp(train_df[col].dropna(), val_df[col].dropna()).pvalue
        drift_detected = (ks_train_test < 0.05) or (ks_train_val < 0.05)
        drift_results.append([col, ks_train_test, ks_train_val, drift_detected])

    drift_df = pd.DataFrame(
        drift_results,
        columns=["Feature", "KS p-value (Train vs Test)", "KS p-value (Train vs Val)", "Drift Detected"]
    )

    drift_df["Drift Detected"] = drift_df["Drift Detected"].apply(lambda x: "⚠ YES" if x else "✅ NO")
    st.dataframe(drift_df, use_container_width=True)

    # --------------------------------------------------------
    # TOP 5 FEATURES CON MAYOR DRIFT
    # --------------------------------------------------------
    st.subheader("🔥 Top 5 Features con Mayor Drift")
    top_drift = drift_df.sort_values("KS p-value (Train vs Test)").head(5)
    st.table(top_drift)

    # --------------------------------------------------------
    # VISUALIZACIÓN DE DISTRIBUCIONES
    # --------------------------------------------------------
    st.subheader("📈 Feature Distribution by Period")

    selected_feature = st.selectbox("Selecciona una feature para visualizar:", features)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_df[selected_feature], bins=40, alpha=0.6, label="Train")
    ax.hist(test_df[selected_feature], bins=40, alpha=0.6, label="Test")
    ax.hist(val_df[selected_feature], bins=40, alpha=0.6, label="Validation")
    ax.legend()
    ax.set_title(f"Distribución de '{selected_feature}' por periodo")
    st.pyplot(fig)

    # --------------------------------------------------------
    # INTERPRETACIÓN
    # --------------------------------------------------------
    st.subheader("🧠 Interpretación")
    st.markdown("""
    - *KS-test p-value < 0.05* indica un cambio estadísticamente significativo.  
    - Las features marcadas con ⚠ pueden reflejar:
        - Cambios de régimen de mercado  
        - Incrementos de volatilidad  
        - Cambios estructurales en la serie
    - Antes de reentrenar el modelo, revisa las variables más afectadas.
    """)

else:
    st.error("❌ No se encontraron los archivos 'train_features.csv', 'test_features.csv' y 'val_features.csv' en /data/processed/.")
    st.info("Verifica que el preprocesamiento haya generado los datasets correctamente antes de abrir el dashboard.")