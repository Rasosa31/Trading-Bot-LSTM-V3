import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

# 1. ESTABILIDAD PARA MAC (Lo que hizo funcionar el test)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Forzar CPU y deshabilitar optimizaciones conflictivas
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("📡 Descargando datos y procesando indicadores...")
df = yf.download("AAPL", period="5y", interval="1d")
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

# Indicadores Técnicos
df['SMA_100'] = df['Close'].rolling(window=100).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-9)
df['RSI'] = 100 - (100 / (1 + rs))
data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']].dropna()

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data.values)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i, 3])
X, y = np.array(X), np.array(y)

# Configuración del Comité
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

configs = [
    {"name": "m1_puro", "units": 32, "drop": 0.1},
    {"name": "m2_volatilidad", "units": 48, "drop": 0.2},
    {"name": "m3_tendencia", "units": 32, "drop": 0.1},
    {"name": "m4_memoria", "units": 64, "drop": 0.1},
    {"name": "m5_agresivo", "units": 32, "drop": 0.3}
]

print(f"🧠 Entrenando Comité de 5 Motores...")

for conf in configs:
    print(f"\n🚀 Entrenando {conf['name']}...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 8)),
        tf.keras.layers.LSTM(conf['units'], return_sequences=False),
        tf.keras.layers.Dropout(conf['drop']),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenamos con los parámetros que sabemos que funcionan
    model.fit(X, y, epochs=3, batch_size=128, verbose=1)
    model.save(os.path.join(MODELS_DIR, f"{conf['name']}.keras"))
    print(f"✅ {conf['name']} Guardado con éxito.")

print("\n🏆 ¡PROCESO COMPLETADO! Los 5 motores están listos en /models.")