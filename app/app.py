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

# 1. INICIALIZACIÓN DE SESIÓN (DEBE IR PRIMERO QUE NADA)
if 'bitacora' not in st.session_state:
    st.session_state.bitacora = []

# Importación diferida para estabilidad
try:
    from streamlit_gsheets import GSheetsConnection
except ImportError:
    st.error("Librería 'st-gsheets-connection' no encontrada. Revisa requirements.txt")

# 2. FUNCIÓN DE GUARDADO OPTIMIZADA
def guardar_en_sheets(registro):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        try:
            existing_data = conn.read(worksheet="Consultas")
            if existing_data is not None:
                existing_data = existing_data.dropna(how="all")
        except:
            existing_data = pd.DataFrame()

        new_row = pd.DataFrame([registro])
        
        if not existing_data.empty:
            updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        else:
            updated_df = new_row
            
        conn.update(worksheet="Consultas", data=updated_df)
        return True
    except Exception as e:
        st.error(f"Error técnico real: {e}") 
        return False

# 3. CONFIGURACIÓN DE ESTABILIDAD TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except: pass

# 4. CARGA DEL COMITÉ (V6)
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

# 5. FUNCIONES DE DATOS
def get_data(ticker, timeframe):
    period_map = {"Daily": "5y", "Weekly": "max", "Monthly": "max"}
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    try:
        df = yf.download(ticker, period=period_map[timeframe], interval=interval_map[timeframe])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty or len(df) < 10: return pd.DataFrame()
        
        # Cálculo de Indicadores
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Cálculo Seguro de ADX
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_data is not None:
            df['ADX'] = adx_data['ADX_14']
        
        df.dropna(inplace=True)
        return df.bfill().ffill()
    except: return pd.DataFrame()

# 6. INTERFAZ
st.set_page_config(page_title="StockAI V6 Pro", layout="wide")
tab1, tab2 = st.tabs(["📈 Análisis en Vivo", "🧪 Backtesting V6"])

with st.sidebar:
    st.header("Configuración")
    ticker = st.text_input("Símbolo:", value="EURUSD=X").upper()
    tf_choice = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    days_to_show = st.slider("Velas visibles:", 30, 500, 150)
    st.divider()
    fuerza = st.slider("Sensibilidad (Fuerza):", 0.1, 1.0, 0.4)

df = get_data(ticker, tf_choice)

# --- TAB 1: ANÁLISIS EN VIVO ---
with tab1:
    if not df.empty:
        df_f = df.tail(days_to_show)
        curr_p = float(df['Close'].iloc[-1])
        precision = 4 if curr_p < 10 else 2
        formato = f".{precision}f"
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_f.index, open=df_f['Open'], high=df_f['High'], 
            low=df_f['Low'], close=df_f['Close'], name="Precio"
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Consultar Comité de Expertos"):
            with st.spinner("Los expertos están deliberando..."):
                scaler = RobustScaler()
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
                
                # Preparamos los datos escalados (los 9 features)
                scaled_data = scaler.fit_transform(df[features].values)
                
                if len(scaled_data) >= 60:
                    last_window = scaled_data[-60:].reshape(1, 60, len(features))
                    preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
                    
                    mean_c, std_c = np.mean(scaled_data[:, 3]), np.std(scaled_data[:, 3])
                    vol = df['Close'].pct_change().std()
                    z_score = (np.mean(preds_raw) - mean_c) / (std_c + 1e-9)
                    pred_final = curr_p * (1 + (z_score * vol * fuerza))
                    
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precio Actual", f"{curr_p:{formato}}")
                    m2.metric("Target Comité", f"{pred_final:{formato}}", f"{pred_final-curr_p:+.{precision}f}")
                    acuerdo_val = max(0, 100-(np.std(preds_raw)*1000))
                    m3.metric("Acuerdo", f"{acuerdo_val:.1f}%")

                    registro = {
                        "Fecha Consulta": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Activo": ticker,
                        "Temporalidad": tf_choice,
                        "Precio Cierre": f"{curr_p:{formato}}",
                        "Predicción": f"{pred_final:{formato}}",
                        "Dirección": "⬆️ ALZA" if pred_final > curr_p else "⬇️ BAJA",
                        "Acuerdo %": f"{acuerdo_val:.1f}%"
                    }
                    st.session_state.bitacora.append(registro)
                else:
                    st.warning("No hay suficientes datos para una ventana de 60 velas.")

        st.subheader("📋 Bitácora de Consultas")
        if len(st.session_state.bitacora) > 0:
            log_df = pd.DataFrame(st.session_state.bitacora)
            st.dataframe(log_df, use_container_width=True)
            csv_log = log_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV", csv_log, "trading_log.csv", "text/csv")

# --- TAB 2: BACKTESTING ---
with tab2:
    st.header("🧪 Evaluación de Desempeño V6")
    if df.empty or len(df) < 65:
        st.error(f"⚠️ Datos insuficientes. Se requieren al menos 65 velas.")
    else:
        max_posible = len(df) - 62
        test_days = st.number_input("Velas de prueba:", 5, min(200, max_posible), min(30, max_posible))
        
        if st.button("📊 Iniciar Backtest Profesional"):
            with st.spinner(f"Simulando {test_days} decisiones..."):
                try:
                    scaler = RobustScaler()
                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
                    scaled = scaler.fit_transform(df[features].values)
                    
                    hits = []
                    # Primero obtenemos los aciertos/fallos
                    for i in range(len(scaled) - test_days, len(scaled)):
                        window = scaled[i-60:i].reshape(1, 60, len(features))
                        preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                        avg_p_raw = np.mean(preds)
                        
                        dir_pred = 1 if avg_p_raw > scaled[i-1, 3] else -1
                        dir_real = 1 if scaled[i, 3] > scaled[i-1, 3] else -1
                        hits.append(1 if dir_pred == dir_real else 0)
                    
                    # Luego calculamos pips acumulados correctamente
                    pips_step = []
                    df_precios = df['Close'].iloc[-test_days:].values
                    df_precios_prev = df['Close'].iloc[-test_days-1:-1].values
                    
                    for i in range(len(hits)):
                        cambio_real = abs(df_precios[i] - df_precios_prev[i])
                        if "=X" in ticker:
                            cambio_real *= 10000
                        pips_step.append(cambio_real if hits[i] == 1 else -cambio_real)

                    pips_acum = np.cumsum(pips_step)
                    win_rate = (sum(hits)/len(hits)) * 100
                    total_pips = pips_acum[-1]

                    # Panel de Métricas
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Efectividad", f"{win_rate:.1f}%")
                    m2.metric("Total Pips/Puntos", f"{total_pips:+,.2f}")
                    m3.metric("Balance ✅ / ❌", f"{sum(hits)} / {len(hits)-sum(hits)}")
                    
                    # Gráfico Espejo
                    df_bt = df.iloc[-test_days:].copy()
                    fig_bt = make_subplots(
                        rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=("Análisis Forense (Señales)", "Curva de Equidad Acumulada"),
                        row_heights=[0.7, 0.3]
                    )

                    fig_bt.add_trace(go.Candlestick(
                        x=df_bt.index, open=df_bt['Open'], high=df_bt['High'],
                        low=df_bt['Low'], close=df_bt['Close'], name="Precio"
                    ), row=1, col=1)

                    # Subplot 2: Curva de Pips
                    fig_bt.add_trace(go.Scatter(
                        x=df_bt.index, y=pips_acum,
                        mode='lines', fill='tozeroy',
                        line=dict(color='#00ccff', width=3),
                        name="Rendimiento"
                    ), row=2, col=1)

                    # Ajustes de Ejes (AQUÍ ESTÁ LO QUE ME PEDISTE)
                    fig_bt.update_yaxes(title_text="Precio", row=1, col=1)
                    fig_bt.update_yaxes(
                        title_text="Pips/Puntos Acum.", 
                        row=2, col=1, 
                        autorange=True, 
                        fixedrange=False
                    )

                    fig_bt.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=800, showlegend=False)
                    st.plotly_chart(fig_bt, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en Backtest: {e}")