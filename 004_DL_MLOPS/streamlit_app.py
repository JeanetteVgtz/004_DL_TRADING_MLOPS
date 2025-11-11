# streamlit_app.py
"""
Dashboard de Streamlit para monitorear drift.
Ejecutar con: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(page_title="Drift Monitor", layout="wide")

st.title("Monitor de Drift de Features")
st.markdown("Detecta cambios en la distribución de indicadores técnicos")

# Cargar datos
@st.cache_data
def cargar_datos():
    train = pd.read_csv('data/processed/train_features.csv', parse_dates=['date'])
    test = pd.read_csv('data/processed/test_features.csv', parse_dates=['date'])
    val = pd.read_csv('data/processed/val_features.csv', parse_dates=['date'])
    return train, test, val

try:
    train_df, test_df, val_df = cargar_datos()
    
    # Obtener indicadores
    columnas_quitar = ['date', 'open', 'high', 'low', 'close', 'volume', 'retorno_siguiente']
    indicadores = [col for col in train_df.columns if col not in columnas_quitar]
    
    # Sidebar
    st.sidebar.header("Configuración")
    alpha = st.sidebar.slider("Nivel de significancia", 0.01, 0.10, 0.05, 0.01)
    
    # Calcular drift
    @st.cache_data
    def calcular_drift_todos(train, test, val, feats):
        resultados = []
        for feat in feats:
            _, p_test = ks_2samp(train[feat].dropna(), test[feat].dropna())
            _, p_val = ks_2samp(train[feat].dropna(), val[feat].dropna())
            
            resultados.append({
                'Indicador': feat,
                'P-Value Test': p_test,
                'P-Value Val': p_val,
                'Drift Test': 'Sí' if p_test < alpha else 'No',
                'Drift Val': 'Sí' if p_val < alpha else 'No'
            })
        
        return pd.DataFrame(resultados)
    
    drift_df = calcular_drift_todos(train_df, test_df, val_df, indicadores)
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Indicadores", len(drift_df))
    
    with col2:
        n_drift_test = (drift_df['Drift Test'] == 'Sí').sum()
        st.metric("Drift en Test", f"{n_drift_test} ({n_drift_test/len(drift_df)*100:.0f}%)")
    
    with col3:
        n_drift_val = (drift_df['Drift Val'] == 'Sí').sum()
        st.metric("Drift en Val", f"{n_drift_val} ({n_drift_val/len(drift_df)*100:.0f}%)")
    
    # Tabla de resultados
    st.subheader("Resultados del Test KS")
    
    drift_sorted = drift_df.sort_values('P-Value Val')
    
    def highlight_drift(row):
        if row['Drift Val'] == 'Sí':
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        drift_sorted.style.apply(highlight_drift, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Top 5 con drift
    st.subheader("Top 5 Indicadores con Mayor Drift")
    
    top5 = drift_sorted.head(5)
    
    for idx, row in top5.iterrows():
        with st.expander(f"**{row['Indicador']}** (p-value: {row['P-Value Val']:.6f})"):
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(train_df[row['Indicador']].dropna(), bins=30, alpha=0.5, label='Train', density=True)
            ax.hist(test_df[row['Indicador']].dropna(), bins=30, alpha=0.5, label='Test', density=True)
            ax.hist(val_df[row['Indicador']].dropna(), bins=30, alpha=0.5, label='Val', density=True)
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            ax.set_title(f'Distribución de {row["Indicador"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    # Explicación
    st.subheader("¿Qué significa drift?")
    st.markdown("""
    **Drift** significa que la distribución de un indicador cambió entre entrenamientos y validación.
    
    **Posibles causas:**
    - Cambio de régimen de mercado (bull → bear)
    - Aumento/disminución de volatilidad
    - Eventos económicos importantes
    - Efectos estacionales
    
    **P-value < 0.05** indica drift significativo.
    """)
    
    # Descargar
    st.subheader("Descargar Resultados")
    csv = drift_sorted.to_csv(index=False)
    st.download_button(
        "Descargar CSV",
        csv,
        "drift_report.csv",
        "text/csv"
    )

except FileNotFoundError:
    st.error("Archivos no encontrados. Ejecuta primero:")
    st.code("python data_preprocessing.py")
    st.code("python feature_engineering.py")