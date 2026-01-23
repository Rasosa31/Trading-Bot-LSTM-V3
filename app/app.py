import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import random
import tensorflow as tf
import os

# Lógica de Persistencia V3
MODEL_PATH = 'models/lstm_model.keras'
SCALER_PATH = 'models/scaler.pkl'

import joblib
from tensorflow.keras.models import load_model

model_v3 = None
scaler_v3 = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model_v3 = load_model(MODEL_PATH)
    scaler_v3 = joblib.load(SCALER_PATH)
    st.sidebar.success("✅ Modelo V3 pre-entrenado cargado")
    
# ==========================================
# 1. MEJORA V3: CONGELADOR DE ALEATORIEDAD
# ==========================================
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# Configuración de página
st.set_page_config(page_title="StockAI V3 Pro - Elite Edition", layout="wide")

# ==========================================
# 2. MOTOR DE IA (Adaptado a N features)
# ==========================================
def train_multivariate_model(X, y):
    # X.shape[1] = ventana (60), X.shape[2] = features (8)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=12, batch_size=32, verbose=0)
    return model

# Interface y Sidebar
st.title("🤖 StockAI V3: Multi-Indicator Intelligence")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    ticker = st.text_input(
        "Enter Stock Ticker:", 
        value="AAPL", 
        help="Use Yahoo Finance symbols (e.g., 'AAPL', 'BTC-USD')."
    ).upper()

    interval_label = st.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly"])
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    interval_code = interval_map[interval_label]
    
    st.divider()
    st.header("🛠️ Extra Tools")
    show_backtest = st.checkbox("Enable Backtesting Analysis")

# ==========================================
# 3. MOTOR DE DATOS V3 (8 FEATURES)
# ==========================================
data = yf.download(ticker, period="max", interval=interval_code)

# Limpieza de MultiIndex si existe
if isinstance(data.columns, pd.MultiIndex): 
    data.columns = data.columns.get_level_values(0)

if not data.empty and len(data) > 30:
    df = data.copy()
    
    # Cálculos Técnicos
    total_candles = len(df)
    w100 = 100 if total_candles >= 100 else max(2, total_candles // 2)
    w200 = 200 if total_candles >= 200 else max(2, total_candles - 1)
    
    df['SMA_100'] = df['Close'].rolling(window=w100).mean()
    df['SMA_200'] = df['Close'].rolling(window=w200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # Evitar división por cero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # LISTA DE FEATURES V3 (8 COLUMNAS)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    
    # Limpiamos nulos causados por las SMAs y RSI
    df_filtered = df[features].bfill().ffill().dropna()
    
    # Escalador de 8 columnas
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_matrix = scaler.fit_transform(df_filtered)

    # 4. Lógica de Predicción
    if st.button(f"🚀 Run {interval_label} Projection"):
        with st.spinner(f"Analyzing {ticker} with 8 indicators..."):
            
            window = 60 if len(scaled_matrix) > 60 else len(scaled_matrix) // 2
            X, y = [], []
            
            # El índice de 'Close' en nuestra lista de features es 3
            close_idx = features.index('Close')
            
            for i in range(window, len(scaled_matrix)):
                X.append(scaled_matrix[i-window:i, :]) # Todas las columnas
                y.append(scaled_matrix[i, close_idx]) # Solo predecimos el cierre
            
            X, y = np.array(X), np.array(y)
            
            # Entrenamos
            model = train_multivariate_model(X, y)
            
            # Preparamos los últimos datos para la predicción de mañana
            last_window = scaled_matrix[-window:].reshape(1, window, len(features))
            pred_scaled = model.predict(last_window)
            
            # Inversión del escalado (Para volver a dólares)
            # Creamos una fila "dummy" de 8 columnas para que el scaler no proteste
            dummy = np.zeros((1, len(features)))
            dummy[0, close_idx] = pred_scaled[0][0] 
            pred_final = float(scaler.inverse_transform(dummy)[0][close_idx])
            
            # Cálculos de métricas
            current_price = float(df_filtered['Close'].iloc[-1])
            diff = pred_final - current_price
            pct = (diff / current_price) * 100
            precision = "4f" if current_price < 2 else "2f"
            
            # Visualización
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${current_price:,.{precision}}")
            m2.metric(f"AI Prediction (Next)", f"${pred_final:,.{precision}}", f"{diff:,.{precision}}")
            m3.metric("Expected Movement", f"{pct:.2f}%")
            
            # Gráfico Principal
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered.index[-120:], y=df_filtered['Close'][-120:], name="Historical Price"))
            
            delta_time = pd.Timedelta(days=1 if interval_code=="1d" else 7 if interval_code=="1wk" else 30)
            fig.add_trace(go.Scatter(
                x=[df_filtered.index[-1], df_filtered.index[-1] + delta_time], 
                y=[current_price, pred_final], 
                name="AI Future Path", 
                line=dict(color='orange', dash='dash', width=3)
            ))
            fig.update_layout(template="plotly_dark", title=f"{ticker} Performance Analysis ({interval_label})")
            st.plotly_chart(fig, use_container_width=True)

    # 5. Backtesting (Ajustado a 8 features)
    if show_backtest:
        st.divider()
        st.subheader(f"📊 Historical Backtesting ({interval_label})")
        if st.button("🔄 Run Accuracy Test"):
            with st.spinner("Testing model stability..."):
                test_days = 20
                window = 60 if len(scaled_matrix) > 80 else 10
                close_idx = features.index('Close')

                X_b, y_b = [], []
                train_data_b = scaled_matrix[:-test_days]
                for i in range(window, len(train_data_b)):
                    X_b.append(train_data_b[i-window:i, :])
                    y_b.append(train_data_b[i, close_idx])
                
                model_b = train_multivariate_model(np.array(X_b), np.array(y_b))
                
                X_test = []
                test_segment = scaled_matrix[-(test_days + window):]
                for i in range(window, len(test_segment)):
                    X_test.append(test_segment[i-window:i, :])
                
                preds_b_scaled = model_b.predict(np.array(X_test))
                
                # Des-escalar múltiples predicciones
                dummy_b = np.zeros((len(preds_b_scaled), len(features)))
                dummy_b[:, close_idx] = preds_b_scaled.flatten()
                preds_b = scaler.inverse_transform(dummy_b)[:, close_idx]
                real_b = df_filtered['Close'].values[-test_days:]
                
                rmse = np.sqrt(np.mean((preds_b - real_b)**2))
                st.info(f"Model Accuracy (RMSE): {rmse:.4f}")
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(y=real_b, name="Actual Data", line=dict(color='blue')))
                fig_b.add_trace(go.Scatter(y=preds_b, name="AI Prediction", line=dict(color='orange', dash='dot')))
                fig_b.update_layout(template="plotly_dark", height=350, title="Backtesting Reality vs Prediction")
                st.plotly_chart(fig_b, use_container_width=True)
else:
    st.error("Not enough data to run V3 Model. Please try a different asset or timeframe.")