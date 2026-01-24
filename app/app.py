import streamlit as st
import tensorflow as tf
import os

# 1. CONFIGURACIÓN DE ESTABILIDAD (Protección contra re-inicialización)
try:
    if not tf.config.list_physical_devices('GPU'):
        # Solo intentamos configurar si no se ha inicializado el contexto
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    # Si TensorFlow ya se inició, ignoramos la configuración silenciosamente
    pass

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import random
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# 2. CONFIGURACIÓN DE SEMILLAS
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

st.set_page_config(page_title="StockAI V3 Pro", layout="wide")

# 3. RUTAS Y CARGA DE MODELO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

@st.cache_resource
def load_v4_ensemble():
    ensemble_models = []
    names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
    
    try:
        for name in names:
            path = os.path.join(BASE_DIR, 'models', f'{name}.keras')
            if os.path.exists(path):
                ensemble_models.append(load_model(path, compile=False))
        return ensemble_models
    except Exception as e:
        st.error(f"Error cargando el comité de expertos: {e}")
    return []

model_committee = load_v4_ensemble()

# 4. INTERFAZ Y SIDEBAR
# 4. INTERFAZ Y SIDEBAR
st.title("🤖 StockAI V4: Multi-Model Committee")

with st.sidebar:
    # Cambiamos la verificación: ahora comprobamos si la lista de modelos tiene algo
    if 'model_committee' in globals() and len(model_committee) > 0:
        st.success(f"✅ Comité V4 Activo ({len(model_committee)} Motores)")
    else:
        st.warning("⚠️ Modo Entrenamiento (Esperando archivos .keras)")
    
    ticker = st.text_input("Símbolo (Ticker):", value="AAPL").upper()
    timeframe = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

    st.divider()
    show_backtest = st.checkbox("Habilitar Backtesting")
    st.info("V4: Bagging de 5 modelos independientes + Robust Scaling.")

# 5. OBTENCIÓN DE DATOS
@st.cache_data(ttl=3600)
def get_data(symbol, interval):
    if interval == "1mo":
        p = "max"
    elif interval == "1wk":
        p = "5y"
    else:
        p = "2y"
    d = yf.download(symbol, period=p, interval=interval)
    if isinstance(d.columns, pd.MultiIndex): 
        d.columns = d.columns.get_level_values(0)
    return d

data = get_data(ticker, interval_map[timeframe])

# 6. LÓGICA PRINCIPAL (Aquí es donde la indentación es crítica)
if not data.empty and len(data) > 60:
    df = data.copy()
    
    # Ingeniería de Atributos (8 columnas exactas para V3)
    df['SMA_100'] = df['Close'].rolling(window=min(len(df), 100)).mean()
    df['SMA_200'] = df['Close'].rolling(window=min(len(df), 200)).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    df_filtered = df[features].bfill().ffill().dropna()

    # --- BOTÓN DE PREDICCIÓN ---
    if st.button("🚀 Ejecutar Predicción con Ensemble"):
        with st.spinner("IA realizando Ensemble Forecasting..."):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(df_filtered.values)
            last_window = scaled_data[-60:].reshape(1, 60, 8)
            
            ensemble_preds = []
            for i in range(3):
                noise = np.random.normal(0, 0.0001, last_window.shape)
                p = model_v3.predict(last_window + noise, verbose=0)
                ensemble_preds.append(p[0][0])
            
            avg_pred_raw = np.mean(ensemble_preds)
            current_price = float(df_filtered['Close'].iloc[-1])
            es_forex_ticker = ticker.endswith('=X')
            precision = 4 if (current_price < 5.0 or es_forex_ticker) else 2
            
            mean_s, std_s = np.mean(scaled_data[:, 3]), np.std(scaled_data[:, 3])
            z_score = (avg_pred_raw - mean_s) / (std_s + 1e-9)
            vol = df_filtered['Close'].pct_change().std()
            fuerza = 0.20 if es_forex_ticker else 0.45 
            pred_final = current_price * (1 + (z_score * vol * fuerza))

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Precio Actual", f"${current_price:.{precision}f}")
            c2.metric("IA Target (Ensemble)", f"${pred_final:.{precision}f}", f"{pred_final - current_price:.{precision}f}")
            c3.metric("Movimiento Esperado", f"{((pred_final/current_price)-1)*100:.2f}%")

            with st.expander("🔍 Ver desglose de opiniones de la IA"):
                individual_prices = [current_price * (1 + (((p - mean_s) / (std_s + 1e-9)) * vol * fuerza)) for p in ensemble_preds]
                df_ens = pd.DataFrame({
                    "Pasada": ["IA 1", "IA 2", "IA 3"],
                    "Predicción": [f"${p:.{precision}f}" for p in individual_prices],
                    "Cambio": [f"{((p/current_price)-1)*100:+.2f}%" for p in individual_prices]
                })
                st.table(df_ens)

            # Fecha Futura Dinámica
            delta_f = pd.Timedelta(days=1) if timeframe == "Daily" else pd.Timedelta(weeks=1) if timeframe == "Weekly" else pd.Timedelta(days=30)
            future_date = df_filtered.index[-1] + delta_f

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered.index[-40:], y=df_filtered['Close'][-40:], name="Histórico", line=dict(color='#00d1ff')))
            fig.add_trace(go.Scatter(x=[df_filtered.index[-1], future_date], y=[current_price, pred_final], name="Proyección", line=dict(color='orange', dash='dash')))
            fig.update_layout(template="plotly_dark", height=400, yaxis=dict(tickformat=f".{precision}f"))
            st.plotly_chart(fig, width='stretch')

    # --- SECCIÓN DE BACKTESTING ---
    if show_backtest:
        st.divider()
        if st.button("🔄 Ejecutar Test de Precisión"):
            with st.spinner("Analizando historial..."):
                current_price = float(df_filtered['Close'].iloc[-1])
                es_forex_ticker = ticker.endswith('=X')
                precision = 4 if (current_price < 5.0 or es_forex_ticker) else 2
                test_len, window = 15, 60
                from sklearn.preprocessing import RobustScaler
                b_scaler = RobustScaler()
                b_scaled = b_scaler.fit_transform(df_filtered.values)
                real_p, pred_p = [], []
                dates = df_filtered.index[-test_len:]
                
                for i in range(test_len, 0, -1):
                    idx = len(b_scaled) - i
                    input_win = b_scaled[idx-window:idx].reshape(1, window, 8)
                    p_r = model_v3.predict(input_win, verbose=0)[0][0]
                    z = (p_r - np.mean(b_scaled[:, 3])) / (np.std(b_scaled[:, 3]) + 1e-9)
                    p_f = df_filtered['Close'].iloc[idx-1] * (1 + (z * df_filtered['Close'].pct_change().std() * (0.20 if es_forex_ticker else 0.45)))
                    pred_p.append(p_f)
                    real_p.append(df_filtered['Close'].iloc[idx])

                rmse = np.sqrt(np.mean((np.array(pred_p) - np.array(real_p))**2))
                mape = np.mean(np.abs((np.array(real_p) - np.array(pred_p)) / np.array(real_p))) * 100
                
                m1, m2 = st.columns(2)
                m1.metric("Error (RMSE)", f"${rmse:.{precision}f}")
                m2.metric("Precisión", f"{100 - mape:.2f}%")
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=dates, y=real_p, name="Real", line=dict(color='#00ff88')))
                fig_b.add_trace(go.Scatter(x=dates, y=pred_p, name="IA", line=dict(color='orange', dash='dot')))
                fig_b.update_layout(template="plotly_dark", height=400, yaxis=dict(tickformat=f".{precision}f"))
                st.plotly_chart(fig_b, width='stretch')
else:
    st.error("No hay suficientes datos para este activo/temporalidad. Se requieren al menos 60 periodos.")