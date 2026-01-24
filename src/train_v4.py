import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Configuración de entorno
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

def get_data_v4(ticker):
    df = yf.download(ticker, period="10y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores V3/V4
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    return df[features].dropna()

def build_model_v4(units=50, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(60, 8)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- INICIO DEL ENTRENAMIENTO ---
ticker = "AAPL" # Usaremos Apple como "pista de pruebas" base
data = get_data_v4(ticker)
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data.values)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i, 3]) # Target: Close
X, y = np.array(X), np.array(y)

configs = [
    {"name": "m1_puro", "units": 50, "dropout": 0.2},
    {"name": "m2_volatilidad", "units": 64, "dropout": 0.3}, # Más neuronas y dropout para ruido
    {"name": "m3_tendencia", "units": 32, "dropout": 0.1},    # Más conservador
    {"name": "m4_memoria", "units": 80, "dropout": 0.2},      # Mayor capacidad de patrón
    {"name": "m5_agresivo", "units": 100, "dropout": 0.4}     # Explora variaciones fuertes
]

for conf in configs:
    print(f"\n🚀 Entrenando Motor: {conf['name']}...")
    model = build_model_v4(units=conf['units'], dropout=conf['dropout'])
    model.fit(X, y, batch_size=32, epochs=5, verbose=1) # 5 épocas para prueba rápida
    model.save(os.path.join(MODELS_DIR, f"{conf['name']}.keras"))
    print(f"✅ {conf['name']} guardado.")

print("\n🎯 ¡Los 5 motores están listos en la carpeta /models!")