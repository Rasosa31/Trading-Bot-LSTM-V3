import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os

# 1. CONFIGURACIÓN DE ESTABILIDAD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except: pass

# 2. CARGA DEL COMITÉ
MODELS_DIR = 'models'
model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
model_committee = []

for name in model_names:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        try: model_committee.append(tf.keras.models.load_model(path))
        except: pass

# 3. FUNCIONES DE DATOS
def get_data(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]
    if df.empty: return pd.DataFrame()
    
    # Indicadores
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

# 4. INTERFAZ
st.set_page_config(page_title="StockAI V4 Pro", layout="wide")
tab1, tab2 = st.tabs(["📈 Análisis en Vivo", "🧪 Backtesting V4"])

with st.sidebar:
    st.header("Configuración")
    ticker = st.text_input("Símbolo:", value="EURUSD=X").upper()
    tf_choice = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    days_to_show = st.slider("Velas visibles:", 30, 500, 150)
    st.divider()
    fuerza = st.slider("Sensibilidad (Fuerza):", 0.1, 1.0, 0.4)

df = get_data(ticker, interval=interval_map[tf_choice])

# --- TAB 1: ANÁLISIS EN VIVO ---
with tab1:
    if not df.empty:
        df_f = df.tail(days_to_show)
        precision = 4 if df_f['Close'].iloc[-1] < 10 else 2
        
        fig = go.Figure(data=[go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'])])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Consultar Comité de Expertos"):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
            scaled = scaler.fit_transform(df[features].values)
            
            # Inferencia
            last_window = scaled[-60:].reshape(1, 60, 8)
            preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
            
            # Des-escalado manual
            curr_p = float(df['Close'].iloc[-1])
            mean_c, std_c = np.mean(scaled[:, 3]), np.std(scaled[:, 3])
            vol = df['Close'].pct_change().std()
            
            z_score = (np.mean(preds_raw) - mean_c) / (std_c + 1e-9)
            pred_final = curr_p * (1 + (z_score * vol * fuerza))
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Precio Actual", f"{curr_p:.{precision}f}")
            m2.metric("Target Comité", f"{pred_final:.{precision}f}", f"{pred_final-curr_p:+.{precision}f}")
            m3.metric("Acuerdo", f"{max(0, 100-(np.std(preds_raw)*1000)):.1f}%")

# --- TAB 2: BACKTESTING V4 ---
with tab2:
    st.header("🧪 Evaluación de Desempeño")
    test_days = st.number_input("Días de prueba:", 10, 100, 20)
    
    if st.button("📊 Iniciar Backtest"):
        with st.spinner("Simulando operaciones pasadas..."):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
            scaled = scaler.fit_transform(df[features].values)
            
            results = []
            # Simulamos los últimos 'test_days'
            for i in range(len(scaled) - test_days, len(scaled)):
                window = scaled[i-60:i].reshape(1, 60, 8)
                preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                
                # Predicción promediada
                avg_p_raw = np.mean(preds)
                real_p_raw = scaled[i, 3] # Lo que realmente pasó
                
                # ¿El bot dijo que subía y subió? ¿O dijo que bajaba y bajó?
                dir_pred = 1 if avg_p_raw > scaled[i-1, 3] else -1
                dir_real = 1 if real_p_raw > scaled[i-1, 3] else -1
                
                results.append(1 if dir_pred == dir_real else 0)
            
            accuracy = (sum(results) / len(results)) * 100
            st.success(f"Efectividad del Comité: {accuracy:.2f}%")
            
            # Gráfico de Aciertos
            st.line_chart(pd.Series(results).rolling(5).mean(), help="Media móvil de aciertos (1=Acierto, 0=Fallo)")