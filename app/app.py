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
def get_data(ticker, timeframe):
    # Mapeo de periodos para asegurar suficiente historia
    period_map = {"Daily": "5y", "Weekly": "max", "Monthly": "max"}
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    
    df = yf.download(ticker, period=period_map[timeframe], interval=interval_map[timeframe])
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]
    
    if df.empty or len(df) < 10: # Si es extremadamente nuevo
        return pd.DataFrame()

    # INDICADORES ADAPTATIVOS: Si no hay datos para 200, usa lo que haya
    length = len(df)
    df['SMA_100'] = df['Close'].rolling(window=min(100, length//2)).mean()
    df['SMA_200'] = df['Close'].rolling(window=min(200, length//2)).mean()
    
    # RSI estándar
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # Rellenar nulos iniciales para no perder filas preciosas en activos cortos
    return df.bfill().ffill()

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

df = get_data(ticker, tf_choice)

# --- TAB 1: ANÁLISIS EN VIVO ---
with tab1:
    if not df.empty:
        df_f = df.tail(days_to_show)
        
        # 1. Definir precisión global
        curr_p = float(df['Close'].iloc[-1])
        precision = 4 if curr_p < 10 else 2
        formato = f".{precision}f" # <--- Aquí se define la variable faltante
        
        # 2. Gráfico
        fig = go.Figure(data=[go.Candlestick(
            x=df_f.index, open=df_f['Open'], high=df_f['High'], 
            low=df_f['Low'], close=df_f['Close'], name="Precio"
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 3. Botón de Predicción
        if st.button("🚀 Consultar Comité de Expertos"):
            with st.spinner("Los expertos están deliberando..."):
                scaler = RobustScaler()
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                df_clean = df[features].ffill().bfill()
                scaled = scaler.fit_transform(df_clean.values)

                # VENTANA DINÁMICA: Si hay menos de 60, usa el máximo disponible
            disponibles = len(scaled)
            ventana_tam = min(60, disponibles - 1) 
            
            if ventana_tam < 10:
                st.error("Realmente no hay datos suficientes para predecir.")
            else:
                # Tomamos las últimas 'ventana_tam' velas y las rellenamos con ceros si faltan para llegar a 60
                input_data = scaled[-ventana_tam:]
                if ventana_tam < 60:
                    # Pad (relleno) con ceros al principio para que el modelo reciba 60
                    padding = np.zeros((60 - ventana_tam, 8))
                    input_data = np.vstack([padding, input_data])
                
                last_window = input_data.reshape(1, 60, 8)
                # ... (procede con el predict)
                
                # Inferencia
                last_window = scaled[-60:].reshape(1, 60, 8)
                preds_raw = [m.predict(last_window, verbose=0)[0][0] for m in model_committee]
                
                # Des-escalado manual
                mean_c, std_c = np.mean(scaled[:, 3]), np.std(scaled[:, 3])
                vol = df['Close'].pct_change().std()
                
                # Predicción Consenso
                z_score = (np.mean(preds_raw) - mean_c) / (std_c + 1e-9)
                pred_final = curr_p * (1 + (z_score * vol * fuerza))
                
                # MÉTRICAS PRINCIPALES
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"{curr_p:{formato}}")
                m2.metric("Target Comité", f"{pred_final:{formato}}", f"{pred_final-curr_p:+.{precision}f}")
                m3.metric("Acuerdo", f"{max(0, 100-(np.std(preds_raw)*1000)):.1f}%")

                # --- SECCIÓN: PERSONALIDAD DE LOS EXPERTOS ---
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
                    
            
# --- TAB 2: BACKTESTING V4 ---
with tab2:
    st.header("🧪 Evaluación de Desempeño Adaptativa")
    
    # 1. Validación de datos mínimos antes de empezar
    if df.empty or len(df) < 65:
        st.error(f"⚠️ Datos insuficientes para {ticker} en esta temporalidad. Se requieren al menos 70 velas (disponibles: {len(df)}).")
    else:
        # Ajustamos el máximo de días de prueba según lo que hay disponible
        max_posible = len(df) - 62
        test_days = st.number_input("Velas de prueba:", 5, min(200, max_posible), min(30, max_posible))
        
        if st.button("📊 Iniciar Backtest Profesional"):
            with st.spinner(f"Simulando {test_days} decisiones del comité..."):
                try:
                    scaler = RobustScaler()
                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
                    
                    # Aseguramos que no haya NaNs antes de escalar
                    df_clean = df[features].ffill().bfill()
                    scaled = scaler.fit_transform(df_clean.values)
                    
                    hits = []
                    dates = []
                    progreso = st.progress(0)

                    # 2. Bucle de simulación con control de límites
                    for i in range(len(scaled) - test_days, len(scaled)):
                        # Actualizamos barra de progreso
                        progreso.progress((i - (len(scaled) - test_days)) / test_days)
                        
                        # Ventana de 60 velas para el LSTM
                        window = scaled[i-60:i].reshape(1, 60, 8)
                        
                        # Predicción del promedio del comité
                        preds = [m.predict(window, verbose=0)[0][0] for m in model_committee]
                        avg_p_raw = np.mean(preds)
                        
                        # Lógica de acierto: Comparar dirección predicha vs Realidad
                        # ¿El comité dijo que el cierre de hoy (i) sería mayor al de ayer (i-1)?
                        dir_pred = 1 if avg_p_raw > scaled[i-1, 3] else -1
                        dir_real = 1 if scaled[i, 3] > scaled[i-1, 3] else -1
                        
                        hits.append(1 if dir_pred == dir_real else 0)
                        dates.append(df.index[i])

                    progreso.empty()

                    # 3. Visualización de Resultados
                    acc_series = pd.Series(hits, index=dates)
                    accuracy = acc_series.mean() * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Efectividad", f"{accuracy:.2f}%")
                    c2.metric("Aciertos", sum(hits))
                    c3.metric("Fallos", len(hits) - sum(hits))
                    
                    # Gráfico de estabilidad (Moving Accuracy)
                    st.subheader("Consistencia del Comité en el tiempo")
                    # Usamos ventana de 5 o menos si el test es muy corto
                    v_rolling = min(5, len(hits))
                    chart_data = acc_series.rolling(window=v_rolling).mean().fillna(acc_series.mean())
                    st.area_chart(chart_data)

                    with st.expander("📄 Registro detallado de señales"):
                        res_df = pd.DataFrame({
                            "Fecha": dates,
                            "Resultado": ["✅ ACIERTO" if x == 1 else "❌ FALLO" for x in hits]
                        }).set_index("Fecha")
                        st.dataframe(res_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error crítico en el cálculo: {e}")