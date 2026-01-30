import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os

# 1. CONFIGURACIÓN
TICKER = "EURUSD=X" # O el activo que prefieras como base
START_DATE = "2019-01-01" 
END_DATE = "2025-12-31"

def prepare_data(df):
    # INDICADORES TÉCNICOS V5
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # ATR (Volatilidad)
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(window=14).mean()
    
    # Retornos logarítmicos
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 3]) # Target: Close
        
    return np.array(X), np.array(y), scaler

# 2. CREADOR DE MODELOS (EL COMITÉ)
def create_model(model_type):
    model = Sequential([Input(shape=(60, 8))])
    
    if model_type == "m1_puro":
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
    elif model_type == "m5_agresivo":
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
    else: # Estándar para los demás
        model.add(LSTM(64))
    
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3. EJECUCIÓN DEL ENTRENAMIENTO
print(f"Descargando datos para {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

X, y, scaler = prepare_data(data)

# Crear carpeta si no existe
if not os.path.exists('models'): os.makedirs('models')

model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]

for name in model_names:
    print(f"\n--- Entrenando Motor: {name} ---")
    model = create_model(name)
    
    # Early Stopping: Detiene si no mejora en 10 epochs
    monitor = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    model.fit(X, y, batch_size=32, epochs=50, callbacks=[monitor], verbose=1)
    model.save(f"models/{name}.keras")
    print(f"✅ {name} guardado correctamente.")

print("\n🚀 ¡Comité V5 completado!")