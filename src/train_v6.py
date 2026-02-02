import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import os

# 1. CONFIGURACIÓN
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
PATH_DATA = "data/multi_stock_data.csv"
EPOCHS = 30 # Bajamos un poco los epochs porque ahora hay MUCHOS datos
BATCH_SIZE = 64

def procesar_dataset_global(path):
    print(f"📖 Cargando dataset maestro: {path}")
    df_raw = pd.read_csv(path, index_col=0, parse_dates=True)
    
    X_global, y_global = [], []
    tickers = df_raw['Ticker'].unique()
    
    scaler = RobustScaler()

    for t in tickers:
        print(f"🧠 Procesando patrones de: {t}")
        df_t = df_raw[df_raw['Ticker'] == t].copy()
        
        # Calcular ADX que faltaba en el downloader
        adx_df = ta.adx(df_t['High'], df_t['Low'], df_t['Close'], length=14)
        if adx_df is not None:
            df_t['ADX'] = adx_df['ADX_14']
        
        df_t.dropna(inplace=True)
        
        if len(df_t) < 100: continue
        
        # Escalar datos del activo
        scaled_t = scaler.fit_transform(df_t[FEATURES])
        
        # Crear ventanas (60 días para predecir el siguiente)
        for i in range(60, len(scaled_t)):
            X_global.append(scaled_t[i-60:i])
            y_global.append(scaled_t[i, 3]) # Columna 3 es 'Close'
            
    return np.array(X_global), np.array(y_global)

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

# --- EJECUCIÓN ---
if not os.path.exists(PATH_DATA):
    print(f"❌ Error: No se encuentra {PATH_DATA}. Ejecuta primero data_downloader.py")
else:
    X, y = procesar_dataset_global(PATH_DATA)
    print(f"📊 Dataset listo. Total de ejemplos de aprendizaje: {len(X)}")

    if not os.path.exists('models_v6'): 
        os.makedirs('models_v6')

    model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]

    # 1. Definición del EarlyStopping (El freno inteligente)
    callback_parada = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=5,          # Si en 5 épocas no mejora, se detiene
        restore_best_weights=True  # Mantiene la mejor versión encontrada
    )

    # 2. Bucle de entrenamiento del Comité
    for name in model_names:
        print(f"\n🚀 Iniciando entrenamiento del Experto: {name}")
        model = crear_modelo_v6(len(FEATURES))
        
        # Entrenamos con la configuración optimizada
        model.fit(
            X, y, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=1,
            callbacks=[callback_parada] 
        )
        
        model.save(f'models_v6/{name}.keras')
        print(f"✅ {name} guardado con éxito en models_v6/")

    print("\n✨ ¡Comité V6 GLOBAL completado y optimizado!")