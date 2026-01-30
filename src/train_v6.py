import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Nueva librería para ADX
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import os

# 1. CONFIGURACIÓN DE FEATURES (Ahora son 9)
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
TICKER = "YM=F" # Puedes cambiarlo o hacerlo dinámico
EPOCHS = 50
BATCH_SIZE = 32

def preparar_datos_v6(ticker):
    print(f"📥 Descargando datos para {ticker}...")
    df = yf.download(ticker, period="5y", interval="1d")
    
    # Cálculos Técnicos
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df.dropna(inplace=True)
    return df[FEATURES]

def crear_modelo_v6(n_features):
    model = Sequential([
        Bidirectional(LSTM(70, return_sequences=True, input_shape=(60, n_features))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 2. PROCESO DE ENTRENAMIENTO
data = preparar_datos_v6(TICKER)
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i, 3]) # Predecimos el 'Close'

X, y = np.array(X), np.array(y)

# Crear carpeta para modelos V6 para no sobreescribir los V5
if not os.path.exists('models_v6'):
    os.makedirs('models_v6')

print("🚀 Entrenando Comité V6 (Equilibrado con ADX)...")
model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]

for name in model_names:
    print(f"--- Entrenando Experto: {name} ---")
    model = crear_modelo_v6(len(FEATURES))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    model.save(f'models_v6/{name}.keras')
    print(f"✅ {name} guardado en models_v6/")

print("\n✨ ¡Comité V6 completado con éxito!")