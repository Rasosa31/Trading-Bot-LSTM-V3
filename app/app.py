import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os
from datetime import datetime
from plotly.subplots import make_subplots

# 1. INICIALIZACIÓN DE SESIÓN (DEBE IR PRIMERO QUE NADA)
if 'bitacora' not in st.session_state:
    st.session_state.bitacora = []

#Importación diferida para estabilidad
try:
    from streamlit_gsheets import GSheetsConnection
except ImportError:
    st.error("Librería 'st-gsheets-connection' no encontrada. Revisa requirements.txt")

#2. FUNCIÓN DE GUARDADO OPTIMIZADA
def guardar_en_sheets(registro):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Intentar leer; si falla o está vacío, crear DF nuevo
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
        # Cambia el 'print' por un 'st.error' para ver el problema real en pantalla
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

# 4. CARGA DEL COMITÉ
MODELS_DIR = 'models'
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
        
        df['SMA_100'] = df['Close'].rolling(window=min(100, len(df)//2)).mean()
        df['SMA_200'] = df['Close'].rolling(window=min(200, len(df)//2)).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        return df.bfill().ffill()
    except: return pd.DataFrame()

# 6. INTERFAZ
st.set_page_config(page_title="StockAI V5 Pro", layout="wide")
tab1, tab2 = st.tabs(["📈 Análisis en Vivo", "🧪 Backtesting V5"])

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
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                df_clean = df[features].ffill().bfill()
                scaled = scaler.fit_transform(df_clean.values)
                
                input_data = scaled[-60:] if len(scaled) >= 60 else np.vstack([np.zeros((60-len(scaled), 8)), scaled])
                last_window = input_data.reshape(1, 60, 8)
                
                preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
                mean_c, std_c = np.mean(scaled[:, 3]), np.std(scaled[:, 3])
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
                
                # Intentar guardado en nube
                # if guardar_en_sheets(registro):
                #     st.success("✅ Registro exitoso en Google Sheets")
                
                # Guardado en sesión local (con protección)
                st.session_state.bitacora.append(registro)

        # SECCIÓN DE BITÁCORA SEGURA
        st.subheader("📋 Bitácora de Consultas")
        if len(st.session_state.bitacora) > 0:
            log_df = pd.DataFrame(st.session_state.bitacora)
            st.dataframe(log_df, use_container_width=True)
            csv_log = log_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV", csv_log, "trading_log.csv", "text/csv")
        else:
            st.info("Sin consultas en esta sesión.")

# --- TAB 2: BACKTESTING ---
# --- SUSTITUIR EN EL TAB 2 (BACKTESTING) ---
with tab2:
    st.header("🧪 Evaluación de Desempeño Adaptativa")
    if df.empty or len(df) < 65:
        st.error(f"⚠️ Datos insuficientes. Se requieren al menos 65 velas.")
    else:
        max_posible = len(df) - 62
        test_days = st.number_input("Velas de prueba:", 5, min(200, max_posible), min(30, max_posible))
        
        if st.button("📊 Iniciar Backtest Profesional"):
            with st.spinner(f"Simulando {test_days} decisiones..."):
                try:
                    scaler = RobustScaler()
                    # Nota: Para V6 mañana recuerda añadir 'ADX' aquí
                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                    df_clean = df[features].ffill().bfill()
                    scaled = scaler.fit_transform(df_clean.values)
                    
                    hits, dates, pips_step = [], [], []

                    for i in range(len(scaled) - test_days, len(scaled)):
                        window = scaled[i-60:i].reshape(1, 60, len(features))
                        preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                        avg_p_raw = np.mean(preds)
                        
                        dir_pred = 1 if avg_p_raw > scaled[i-1, 3] else -1
                        dir_real = 1 if scaled[i, 3] > scaled[i-1, 3] else -1
                        
                        resultado = 1 if dir_pred == dir_real else 0
                        hits.append(resultado)
                        
                        # CÁLCULO DE PIPS REFINADO (Anclado al precio real)
                        cambio_real = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1])
                        pips_step.append(cambio_real if resultado == 1 else -cambio_real)

                        pips_step = []
                        for i in range(len(hits)):
                            idx_actual = len(df) - test_days + i
                            # Calculamos la diferencia de precio real
                            cambio_real = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1])
                        
                            # APLICAR MULTIPLICADOR SI ES FOREX
                            # (Si el ticker tiene "=X", multiplicamos por 10,000 para ver pips reales)
                            if "=X" in ticker_input:
                                cambio_real = cambio_real * 10000
                        
                            pips_step.append(cambio_real if hits[i] == 1 else -cambio_real)

                    # 2. Lógica de Métricas y Gráfico
                    df_bt = df.iloc[len(df)-test_days:].copy()
                    pips_acum = np.cumsum(pips_step)
                    win_rate = (sum(hits)/len(hits)) * 100
                    total_pips = pips_acum[-1]

                    # --- PANEL DE MÉTRICAS VISUALES ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Efectividad", f"{win_rate:.1f}%")
                    # Usamos f"{total_pips:+,.2f}" para que ponga el signo + y separe miles
                    m2.metric("Pips/Puntos Totales", f"{total_pips:+,.2f}")
                    m3.metric("Balance ✅ / ❌", f"{sum(hits)} / {len(hits)-sum(hits)}")

                    # --- NUEVO PANEL DE MÉTRICAS VISUALES ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Efectividad (Win Rate)", f"{win_rate:.1f}%")
                    m2.metric("Total Pips/Puntos", f"{total_pips:.2f}")
                    m3.metric("Balance ✅ / ❌", f"{sum(hits)} / {len(hits)-sum(hits)}")
                    
                    # 3. CONSTRUCCIÓN DEL GRÁFICO ESPEJO
                    fig_bt = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=("Análisis Forense (Señales)", "Curva de Equidad Acumulada"),
                        row_heights=[0.7, 0.3]
                    )

                    # Subplot 1: Velas
                    fig_bt.add_trace(go.Candlestick(
                        x=df_bt.index, open=df_bt['Open'], high=df_bt['High'],
                        low=df_bt['Low'], close=df_bt['Close'], name="Precio"
                    ), row=1, col=1)

                    # Subplot 1: Marcadores ✅/❌
                    offset = (df_bt['High'] - df_bt['Low']).mean() * 0.8
                    fig_bt.add_trace(go.Scatter(
                        x=df_bt.index, 
                        y=[row['High'] + offset if hits[i] == 1 else row['Low'] - offset for i, (idx, row) in enumerate(df_bt.iterrows())],
                        mode='markers',
                        marker=dict(
                            symbol=['triangle-up' if h == 1 else 'x' for h in hits],
                            color=['#00ff00' if h == 1 else '#ff4b4b' for h in hits],
                            size=12
                        ),
                        name="Resultado"
                    ), row=1, col=1)

                    # Subplot 2: Curva de Pips
                    fig_bt.add_trace(go.Scatter(
                        x=df_bt.index, y=pips_acum,
                        mode='lines', fill='tozeroy',
                        line=dict(color='#00ccff', width=3),
                        name="Rendimiento"
                    ), row=2, col=1)

                    fig_bt.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=800, showlegend=False)
                    st.plotly_chart(fig_bt, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en Backtest: {e}")