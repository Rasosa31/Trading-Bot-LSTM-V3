import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os

from streamlit_gsheets import GSheetsConnection

# Función para guardar en Google Sheets
def guardar_en_sheets(registro):
    try:
        # 1. Conexión limpia (Toma la URL directamente de Secrets)
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # 2. Leer sin forzar columnas (Esto evita el 404)
        # Quitamos usecols para que el bot simplemente lea lo que haya
        existing_data = conn.read(worksheet="Consultas")
        
        # 3. Limpieza de filas vacías
        if existing_data is not None:
            existing_data = existing_data.dropna(how="all")
        
        # 4. Crear nuevo DataFrame con la fila actual
        new_row = pd.DataFrame([registro])
        
        # 5. Concatenar y actualizar
        if existing_data is not None and not existing_data.empty:
            updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        else:
            updated_df = new_row
            
        conn.update(worksheet="Consultas", data=updated_df)
        return True
    except Exception as e:
        # Este mensaje nos dirá exactamente qué pasa si falla
        st.error(f"Error al guardar en Google Sheets: {e}")
        return False

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
def get_data(ticker, timeframe):
    period_map = {"Daily": "5y", "Weekly": "max", "Monthly": "max"}
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    
    df = yf.download(ticker, period=period_map[timeframe], interval=interval_map[timeframe])
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]
    
    if df.empty or len(df) < 10:
        return pd.DataFrame()

    length = len(df)
    df['SMA_100'] = df['Close'].rolling(window=min(100, length//2)).mean()
    df['SMA_200'] = df['Close'].rolling(window=min(200, length//2)).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    return df.bfill().ffill()

# 4. INTERFAZ
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
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Consultar Comité de Expertos"):
            with st.spinner("Los expertos están deliberando..."):
                scaler = RobustScaler()
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                df_clean = df[features].ffill().bfill()
                scaled = scaler.fit_transform(df_clean.values)

                disponibles = len(scaled)
                ventana_tam = min(60, disponibles - 1) 
                
                if ventana_tam < 10:
                    st.error("Realmente no hay datos suficientes para predecir.")
                else:
                    input_data = scaled[-ventana_tam:]
                    if ventana_tam < 60:
                        padding = np.zeros((60 - ventana_tam, 8))
                        input_data = np.vstack([padding, input_data])
                    
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
                    m3.metric("Acuerdo", f"{max(0, 100-(np.std(preds_raw)*1000)):.1f}%")

                    # --- GUARDAR EN BITÁCORA ---
                    registro = {
                        "Fecha Consulta": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        "Activo": ticker,
                        "Temporalidad": tf_choice,
                        "Precio Cierre": f"{curr_p:{formato}}",
                        "Predicción": f"{pred_final:{formato}}",
                        "Dirección": "⬆️ ALZA" if pred_final > curr_p else "⬇️ BAJA",
                        "Acuerdo %": f"{max(0, 100-(np.std(preds_raw)*1000)):.1f}%"
                    }
                    
                    # Guardar en la nube automáticamente
                    if guardar_en_sheets(registro):
                        st.success("✅ Predicción registrada en Google Sheets!")
                    
                    st.session_state.bitacora.append(registro)

                    st.markdown("### 🗣️ Veredictos Individuales")
                    perfiles = {
                        "m1_puro": {"emoji": "⚖️", "nick": "El Purista"},
                        "m2_volatilidad": {"emoji": "🌪️", "nick": "Cazador"},
                        "m3_tendencia": {"emoji": "📈", "nick": "Trend-Follower"},
                        "m4_memoria": {"emoji": "🧠", "nick": "Analista"},
                        "m5_agresivo": {"emoji": "⚡", "nick": "Agresivo"}
                    }

                    cols = st.columns(5)
                    for i, name in enumerate(model_names):
                        with cols[i]:
                            p_ind_raw = preds_raw[i]
                            z_ind = (p_ind_raw - mean_c) / (std_c + 1e-9)
                            p_final_ind = curr_p * (1 + (z_ind * vol * fuerza))
                            diff = p_final_ind - curr_p
                            color = "#00ff00" if diff > 0 else "#ff4b4b"
                            flecha = "🔼" if diff > 0 else "🔽"
                            
                            st.markdown(f"""
                            <div style="border: 1px solid #444; border-radius: 10px; padding: 10px; text-align: center; background-color: #1e1e1e;">
                                <h2 style="margin:0;">{perfiles[name]['emoji']}</h2>
                                <b style="font-size: 0.7em; color: #aaa;">{perfiles[name]['nick']}</b><br>
                                <span style="color:{color}; font-weight:bold; font-size: 0.9em;">{flecha} {p_final_ind:{formato}}</span>
                            </div>
                            """, unsafe_allow_html=True)

            st.divider()
        st.subheader("📋 Bitácora de Consultas (Sesión Actual)")
        if st.session_state.bitacora:
            log_df = pd.DataFrame(st.session_state.bitacora)
            st.dataframe(log_df, use_container_width=True)
            
            # Botón para descargar la bitácora completa
            csv_log = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Diario de Predicciones",
                csv_log,
                "diario_trading.csv",
                "text/csv",
                key='download-csv-log'
            )
        else:
            st.info("Aún no hay consultas en esta sesión. Haz clic en 'Consultar Comité' para empezar el registro.")
            
# --- TAB 2: BACKTESTING V4 ---
with tab2:
    st.header("🧪 Evaluación de Desempeño Adaptativa")
    if df.empty or len(df) < 65:
        st.error(f"⚠️ Datos insuficientes. Se requieren al menos 65 velas (disponibles: {len(df)}).")
    else:
        max_posible = len(df) - 62
        test_days = st.number_input("Velas de prueba:", 5, min(200, max_posible), min(30, max_posible))
        
        if st.button("📊 Iniciar Backtest Profesional"):
            with st.spinner(f"Simulando {test_days} decisiones..."):
                try:
                    scaler = RobustScaler()
                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                    df_clean = df[features].ffill().bfill()
                    scaled = scaler.fit_transform(df_clean.values)
                    
                    hits, dates = [], []
                    progreso = st.progress(0)

                    for i in range(len(scaled) - test_days, len(scaled)):
                        progreso.progress((i - (len(scaled) - test_days)) / test_days)
                        window = scaled[i-60:i].reshape(1, 60, 8)
                        preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                        avg_p_raw = np.mean(preds)
                        
                        dir_pred = 1 if avg_p_raw > scaled[i-1, 3] else -1
                        dir_real = 1 if scaled[i, 3] > scaled[i-1, 3] else -1
                        hits.append(1 if dir_pred == dir_real else 0)
                        dates.append(df.index[i])

                    progreso.empty()
                    acc_series = pd.Series(hits, index=dates)
                    accuracy = acc_series.mean() * 100
                
                    # Cálculo de Ganancia Estimada (Pips o Puntos)
                    # Tomamos la diferencia de precio entre el cierre de hoy y ayer para cada acierto
                    pips_totales = 0
                    for i in range(len(scaled) - test_days, len(scaled)):
                        idx_hit = i - (len(scaled) - test_days)
                        if hits[idx_hit] == 1: # Si acertó la dirección
                            cambio = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1])
                            pips_totales += cambio
                        else: # Si falló, restamos el movimiento
                            cambio = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1])
                            pips_totales -= cambio

                    # Detector de unidad
                    es_forex = ticker.endswith("=X") or precision > 2
                    if es_forex:
                        factor = 10000 if "JPY" not in ticker else 100
                        valor_pips = pips_totales * factor
                        etiqueta_pips = "Pips Ganados"
                    else:
                        valor_pips = pips_totales
                        etiqueta_pips = "Puntos/USD"

                    # MÉTRICAS EN PANTALLA
                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Efectividad", f"{accuracy:.2f}%")
                    c2.metric(etiqueta_pips, f"{valor_pips:+.2f}")
                    c3.metric("Aciertos ✅", sum(hits))
                    c4.metric("Fallos ❌", len(hits) - sum(hits))
                    
                    st.subheader("Consistencia del Comité")
                    v_rolling = min(5, len(hits))
                    chart_data = acc_series.rolling(window=v_rolling).mean().fillna(acc_series.mean())
                    st.area_chart(chart_data)

                    with st.expander("📄 Registro detallado"):
                        res_df = pd.DataFrame({"Resultado": ["✅ ACIERTO" if x == 1 else "❌ FALLO" for x in hits]}, index=dates)
                        st.dataframe(res_df, use_container_width=True)
                        
                        # --- BOTÓN DE DESCARGA ---
                        csv = res_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Reporte de Backtest (CSV)",
                            data=csv,
                            file_name=f"backtest_{ticker}_{tf_choice}.csv",
                            mime='text/csv',
                        )
                        

                except Exception as e:
                    st.error(f"Error crítico: {e}")