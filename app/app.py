import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os
from datetime import datetime
from plotly.subplots import make_subplots
import pandas_ta as ta

# 1. INICIALIZACIÓN DE SESIÓN
if 'bitacora' not in st.session_state:
    st.session_state.bitacora = []

# 2. CONFIGURACIÓN DE ESTABILIDAD TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except: pass

# 3. CARGA DEL COMITÉ (V6)
MODELS_DIR = 'models_v6'
model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
model_committee = []

for name in model_names:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        try: 
            model = tf.keras.models.load_model(path)
            model_committee.append(model)
        except: pass

# 4. FUNCIONES DE DATOS
def get_data(ticker, timeframe):
    period_map = {"Daily": "5y", "Weekly": "max", "Monthly": "max"}
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    try:
        df = yf.download(ticker, period=period_map[timeframe], interval=interval_map[timeframe], progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty or len(df) < 10: return pd.DataFrame()
        
        # Indicadores V6
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_data is not None:
            df['ADX'] = adx_data['ADX_14']
        
        df.dropna(inplace=True)
        return df.bfill().ffill()
    except: return pd.DataFrame()

# 5. INTERFAZ MAESTRA
st.set_page_config(page_title="StockAI V6 Ultra", layout="wide")
st.title("🤖 StockAI Committee V6")

tab1, tab2, tab3 = st.tabs(["📈 Análisis Individual", "🧪 Backtesting Pro", "🚀 Escaneo Maestro"])

# --- SIDEBAR COMÚN ---
with st.sidebar:
    st.header("⚙️ Configuración Global")
    ticker_main = st.text_input("Símbolo Principal:", value="NQ=F").upper()
    tf_main = st.selectbox("Temporalidad Base:", ["Daily", "Weekly", "Monthly"])
    fuerza_global = st.slider("Sensibilidad (Fuerza):", 0.1, 1.0, 0.4)
    st.divider()
    st.caption("V6: LSTM Ensemble + ADX Filter")

df = get_data(ticker_main, tf_main)

# --- TAB 1: ANÁLISIS INDIVIDUAL ---
with tab1:
    if not df.empty:
        df_f = df.tail(150)
        curr_p = float(df['Close'].iloc[-1])
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_f.index, open=df_f['Open'], high=df_f['High'], 
            low=df_f['Low'], close=df_f['Close'], name=ticker_main
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Consultar Comité"):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
            scaled_data = scaler.fit_transform(df[features].values)
            
            if len(scaled_data) >= 60:
                last_window = scaled_data[-60:].reshape(1, 60, len(features))
                preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
                
                vol = df['Close'].pct_change().std()
                z_score = (np.mean(preds_raw) - np.mean(scaled_data[:, 3])) / (np.std(scaled_data[:, 3]) + 1e-9)
                pred_final = curr_p * (1 + (z_score * vol * fuerza_global))
                acuerdo_val = max(0, 100-(np.std(preds_raw)*1000))
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"{curr_p:.2f}")
                m2.metric("Target", f"{pred_final:.2f}", f"{pred_final-curr_p:+.2f}")
                m3.metric("Confianza", f"{acuerdo_val:.1f}%")

# --- TAB 2: BACKTESTING ---
with tab2:
    if not df.empty:
        test_days = st.number_input("Velas de prueba:", 5, 200, 30)
        if st.button("📊 Correr Backtest"):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
            scaled = scaler.fit_transform(df[features].values)
            
            hits, pips_step = [], []
            df_precios = df['Close'].iloc[-test_days:].values
            df_precios_prev = df['Close'].iloc[-test_days-1:-1].values
            
            for i in range(len(scaled) - test_days, len(scaled)):
                window = scaled[i-60:i].reshape(1, 60, len(features))
                preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                res = 1 if (np.mean(preds) > scaled[i-1, 3]) == (scaled[i, 3] > scaled[i-1, 3]) else 0
                hits.append(res)
            
            for i in range(len(hits)):
                cambio = abs(df_precios[i] - df_precios_prev[i])
                if "=X" in ticker_main: cambio *= 10000
                pips_step.append(cambio if hits[i] == 1 else -cambio)

            pips_acum = np.cumsum(pips_step)
            
            fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_bt.add_trace(go.Candlestick(x=df.index[-test_days:], open=df['Open'].iloc[-test_days:], high=df['High'].iloc[-test_days:], low=df['Low'].iloc[-test_days:], close=df['Close'].iloc[-test_days:]), row=1, col=1)
            fig_bt.add_trace(go.Scatter(x=df.index[-test_days:], y=pips_acum, mode='lines', fill='tozeroy', line=dict(color='#00ccff')), row=2, col=1)
            fig_bt.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, showlegend=False)
            fig_bt.update_yaxes(autorange=True, row=2, col=1)
            st.plotly_chart(fig_bt, use_container_width=True)

# --- TAB 3: ESCANEO MAESTRO (EL NUEVO PODER) ---
with tab3:
    st.subheader("🚀 Escaneo Multiactivo de Alta Velocidad")
    lista_raw = st.text_area("Lista (Tickers separados por espacio o coma):", value="AAPL, NVDA, BTC-USD, GC=F, NQ=F, EURUSD=X, SPY, TSLA", height=100)
    
    if st.button("🔍 Iniciar Escaneo"):
        tickers = [t.strip().replace(",", "").upper() for t in lista_raw.replace("\n", " ").split() if t.strip()]
        resultados = []
        progreso = st.progress(0)
        
        for idx, t in enumerate(tickers):
            df_t = get_data(t, tf_main)
            if not df_t.empty and len(df_t) >= 65:
                sc = RobustScaler()
                feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
                scaled_t = sc.fit_transform(df_t[feats].values)
                win = scaled_t[-60:].reshape(1, 60, len(feats))
                preds = [m.predict(win, verbose=0)[0][0] for m in model_committee]
                
                cp = float(df_t['Close'].iloc[-1])
                vol = df_t['Close'].pct_change().std()
                zs = (np.mean(preds) - np.mean(scaled_t[:, 3])) / (np.std(scaled_t[:, 3]) + 1e-9)
                pf = cp * (1 + (zs * vol * fuerza_global))
                ac = max(0, 100 - (np.std(preds) * 1000))
                
                resultados.append({
                    "Activo": t, "Precio": f"{cp:.2f}", "Target": f"{pf:.2f}",
                    "Potencial %": round(((pf/cp)-1)*100, 2),
                    "Señal": "🚀 COMPRA" if pf > cp else "📉 VENTA",
                    "Confianza": f"{ac:.1f}%"
                })
            progreso.progress((idx + 1) / len(tickers))
        
        if resultados:
            df_res = pd.DataFrame(resultados).sort_values(by="Confianza", ascending=False)
            st.table(df_res) # Usamos table para una vista rápida y limpia