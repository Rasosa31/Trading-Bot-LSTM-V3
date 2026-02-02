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

# 1. INICIALIZACIÓN DE SESIÓN (BITÁCORAS CORTAS)
if 'bitacora_individual' not in st.session_state:
    st.session_state.bitacora_individual = []

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
    period_map = {"Daily": "10y", "Weekly": "max", "Monthly": "max"}
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    try:
        df = yf.download(ticker, period=period_map[timeframe], interval=interval_map[timeframe], progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty or len(df) < 10: return pd.DataFrame()
        
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_data is not None:
            df['ADX'] = adx_data['ADX_14']
        
        df.dropna(inplace=True)
        return df.bfill().ffill()
    except: return pd.DataFrame()

# 5. INTERFAZ
st.set_page_config(page_title="StockAI V6 Ultra Pro", layout="wide")
st.title("🤖 StockAI Committee V6 - Intelligence Terminal")

tab1, tab2, tab3 = st.tabs(["📈 Análisis Individual", "🧪 Backtesting Pro", "🚀 Escaneo Maestro"])

with st.sidebar:
    st.header("⚙️ Configuración")
    ticker_main = st.text_input("Símbolo Principal:", value="NQ=F").upper()
    tf_main = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    fuerza_global = st.slider("Sensibilidad:", 0.1, 1.0, 0.4)

df = get_data(ticker_main, tf_main)

# --- TAB 1: ANÁLISIS INDIVIDUAL ---
with tab1:
    if not df.empty:
        df_f = df.tail(150)
        curr_p = float(df['Close'].iloc[-1])
        fig = go.Figure(data=[go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'], name=ticker_main)])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Consultar Comité"):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
            scaled_data = scaler.fit_transform(df[features].values)
            last_window = scaled_data[-60:].reshape(1, 60, len(features))
            preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
            
            vol = df['Close'].pct_change().std()
            z_score = (np.mean(preds_raw) - np.mean(scaled_data[:, 3])) / (np.std(scaled_data[:, 3]) + 1e-9)
            pred_final = curr_p * (1 + (z_score * vol * fuerza_global))
            acuerdo_val = max(0, 100-(np.std(preds_raw)*1000))
            
            registro = {
                "Fecha Consulta": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Activo": ticker_main,
                "Temporalidad": tf_main,
                "Precio Cierre": round(curr_p, 4),
                "Predicción": round(pred_final, 4),
                "Dirección": "⬆️ ALZA" if pred_final > curr_p else "⬇️ BAJA",
                "Acuerdo %": f"{acuerdo_val:.1f}%"
            }
            st.session_state.bitacora_individual.append(registro)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Precio Actual", f"{curr_p:.2f}")
            m2.metric("Target", f"{pred_final:.2f}", f"{pred_final-curr_p:+.2f}")
            m3.metric("Confianza", f"{acuerdo_val:.1f}%")

        if st.session_state.bitacora_individual:
            st.divider()
            st.subheader("📋 Consultas Recientes")
            df_individual = pd.DataFrame(st.session_state.bitacora_individual)
            st.dataframe(df_individual, use_container_width=True)
            st.download_button("📥 Descargar Consultas (CSV)", df_individual.to_csv(index=False), "consultas_individuales.csv", "text/csv")

# --- TAB 2: BACKTESTING ---
with tab2:
    if not df.empty:
        test_days = st.number_input("Velas de prueba:", 5, 200, 30)
        if st.button("📊 Correr Backtest"):
            scaler = RobustScaler()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
            scaled = scaler.fit_transform(df[features].values)
            
            hits, bt_log = [], []
            df_precios = df['Close'].iloc[-test_days:].values
            df_precios_prev = df['Close'].iloc[-test_days-1:-1].values
            
            for i in range(len(scaled) - test_days, len(scaled)):
                window = scaled[i-60:i].reshape(1, 60, len(features))
                preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                pred_dir = 1 if (np.mean(preds) > scaled[i-1, 3]) else -1
                real_dir = 1 if (scaled[i, 3] > scaled[i-1, 3]) else -1
                hit = 1 if pred_dir == real_dir else 0
                hits.append(hit)
                
                bt_log.append({
                    "Fecha": df.index[i].strftime("%Y-%m-%d"),
                    "Activo": ticker_main,
                    "Dirección Pred": "ALZA" if pred_dir == 1 else "BAJA",
                    "Dirección Real": "ALZA" if real_dir == 1 else "BAJA",
                    "Resultado": "✅" if hit == 1 else "❌"
                })

            df_bt_results = pd.DataFrame(bt_log)
            st.success(f"Backtest completado. Efectividad: {(sum(hits)/len(hits))*100:.1f}%")
            st.dataframe(df_bt_results, use_container_width=True)
            st.download_button("📥 Descargar Reporte Backtesting", df_bt_results.to_csv(index=False), f"backtest_{ticker_main}.csv", "text/csv")

# --- TAB 3: ESCANEO MAESTRO ---
with tab3:
    st.subheader("🚀 Escaneo Multiactivo")
    lista_raw = st.text_area("Tickers (ej: AAPL, NVDA, BTC-USD):", value="AAPL, NVDA, BTC-USD, GC=F, NQ=F, EURUSD=X", height=100)
    
    if st.button("🔍 Iniciar Escaneo"):
        tickers = [t.strip().replace(",", "").upper() for t in lista_raw.replace("\n", " ").split() if t.strip()]
        resultados_escaneo = []
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
                
                resultados_escaneo.append({
                    "Fecha Consulta": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Activo": t,
                    "Temporalidad": tf_main,
                    "Precio Cierre": round(cp, 4),
                    "Predicción": round(pf, 4),
                    "Dirección": "🚀 COMPRA" if pf > cp else "📉 VENTA",
                    "Acuerdo %": f"{ac:.1f}%"
                })
            progreso.progress((idx + 1) / len(tickers))
        
        if resultados_escaneo:
            df_res = pd.DataFrame(resultados_escaneo).sort_values(by="Acuerdo %", ascending=False)
            st.subheader("📊 Resultados del Escaneo")
            st.dataframe(df_res, use_container_width=True)
            st.download_button("📥 Descargar Escaneo Maestro", df_res.to_csv(index=False), f"escaneo_maestro_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")