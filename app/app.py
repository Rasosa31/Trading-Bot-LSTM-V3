import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os

# 1. CONFIGURACIÓN DE ESTABILIDAD (Crucial para Mac y Cloud)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass # Ya inicializado

# 2. CARGA DEL COMITÉ DE MODELOS
MODELS_DIR = 'models'
model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
model_committee = []

for name in model_names:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        try:
            model_committee.append(tf.keras.models.load_model(path))
        except Exception as e:
            st.error(f"Error cargando {name}: {e}")

# 3. FUNCIONES DE DATOS
def get_data(ticker, period="5y", interval="1d"): # Aumentamos a 5y para asegurar indicadores
    df = yf.download(ticker, period=period, interval=interval)
    
    # Limpieza profunda de columnas para Streamlit Cloud
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]
    
    if df.empty:
        return pd.DataFrame()

    # Cálculo de indicadores (asegurando que existan datos suficientes)
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# 4. INTERFAZ
st.set_page_config(page_title="StockAI V4", layout="wide")
st.title("🤖 StockAI V4: Multi-Model Committee")

with st.sidebar:
    if len(model_committee) == 5:
        st.success("✅ Comité V4 Activo (5 Motores)")
    else:
        st.warning(f"⚠️ Comité Incompleto ({len(model_committee)}/5)")
    
    ticker = st.text_input("Símbolo (Ticker):", value="AAPL").upper()
    timeframe = st.selectbox("Temporalidad:", ["Daily", "Weekly"])
    days_to_show = st.slider("Días a visualizar:", 30, 365, 180)

# 5. LÓGICA PRINCIPAL
df = get_data(ticker)

if not df.empty:
    df_filtered = df.tail(days_to_show)
    
    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_filtered.index, open=df_filtered['Open'], 
                                 high=df_filtered['High'], low=df_filtered['Low'], 
                                 close=df_filtered['Close'], name='Precio'))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['SMA_100'], name='SMA 100', line=dict(color='orange')))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTÓN DE PREDICCIÓN V4 ---
    if st.button("🚀 Ejecutar Predicción con Comité (5 Motores)"):
        if len(model_committee) < 5:
            st.error("No se detectaron los 5 modelos en /models. Ejecuta train_v4.py primero.")
        else:
            with st.spinner("Consultando al comité de expertos..."):
                # Preparación de datos
                scaler = RobustScaler()
                full_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                scaled_data = scaler.fit_transform(df[full_features].values)
                
                last_window = scaled_data[-60:].reshape(1, 60, 8)
                
                # Inferencia del comité
                preds_raw = []
                for model in model_committee:
                    p = model.predict(last_window, verbose=0)
                    preds_raw.append(p[0][0])
                
                # Promedio Ponderado
                avg_pred_raw = np.mean(preds_raw)
                
                # Des-escalado manual para Close (índice 3)
                current_price = float(df['Close'].iloc[-1])
                mean_c, std_c = np.mean(scaled_data[:, 3]), np.std(scaled_data[:, 3])
                vol = df['Close'].pct_change().std()
                fuerza = 0.40 # Sensibilidad de predicción
                
                z_score = (avg_pred_raw - mean_c) / (std_c + 1e-9)
                pred_final = current_price * (1 + (z_score * vol * fuerza))
                
                # Métricas
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"${current_price:.2f}")
                m2.metric("Target Consenso", f"${pred_final:.2f}", f"{pred_final - current_price:+.2f}")
                
                # Cálculo de acuerdo (Acuerdo = 100% - Coeficiente de variación)
                std_preds = np.std(preds_raw)
                agreement = max(0, 100 - (std_preds * 1000))
                m3.metric("Acuerdo del Comité", f"{agreement:.1f}%")

                # Desglose
                with st.expander("🔍 Ver veredicto individual"):
                    nombres = ["M1 (Puro)", "M2 (Volátil)", "M3 (Tendencia)", "M4 (Memoria)", "M5 (Agresivo)"]
                    precios_ind = [current_price * (1 + (((p - mean_c) / (std_c + 1e-9)) * vol * fuerza)) for p in preds_raw]
                    
                    st.table(pd.DataFrame({
                        "Motor": nombres,
                        "Predicción": [f"${p:.2f}" for p in precios_ind],
                        "Diferencia %": [f"{((p/current_price)-1)*100:+.2f}%" for p in precios_ind]
                    }))
                    
                    if agreement < 70:
                        st.warning("⚠️ Alta divergencia entre modelos. El mercado está indeciso.")
                    else:
                        st.success("✅ Consenso sólido entre expertos.")

else:
    st.error("No se pudieron cargar datos para este ticker.")