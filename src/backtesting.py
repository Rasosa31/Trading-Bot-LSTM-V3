import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

# ==========================================
# 1. MEJORA V3: CONSISTENCIA ALEATORIA
# ==========================================
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

def run_backtest(ticker="AAPL", days_to_test=60):
    print(f"\n--- 📊 Iniciando Backtesting V3 para {ticker} ---")
    
    # 1. Descarga de datos
    data = yf.download(ticker, period="3y", interval="1d")
    if data.empty: 
        print("❌ No se pudieron obtener datos.")
        return
    
    # Limpieza de MultiIndex por si acaso (Yahoo Finance API update)
    if isinstance(data.columns, pd.MultiIndex): 
        data.columns = data.columns.get_level_values(0)

    # 2. Preparación de indicadores V3
    df = data.copy()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EL ESTÁNDAR V3: 8 FEATURES
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    close_idx = features.index('Close') # Dinámico para evitar errores
    
    df_filtered = df[features].bfill().ffill().dropna()
    
    # 3. Escalado Multivariado (8 dimensiones)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_filtered.values)
    
    # 4. División de Datos para Validación
    window = 60
    train_size = len(scaled_data) - days_to_test
    train_data = scaled_data[:train_size]
    # Necesitamos los últimos 'window' días antes del test para la primera predicción
    test_data = scaled_data[train_size - window:] 
    
    # Preparar X_train, y_train
    X_train, y_train = [], []
    for i in range(window, len(train_data)):
        X_train.append(train_data[i-window:i, :])
        y_train.append(train_data[i, close_idx]) # Predecimos el Cierre
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 5. Modelo de Evaluación V3
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"Entrenando motor de validación con {len(X_train)} velas...")
    model.fit(X_train, y_train, epochs=12, batch_size=32, verbose=0)
    
    # 6. Ejecución del Test (Predicciones fuera de la muestra)
    X_test = []
    for i in range(window, len(test_data)):
        X_test.append(test_data[i-window:i, :])
    X_test = np.array(X_test)
    
    predictions_scaled = model.predict(X_test)
    
    # MEJORA V3: Des-escalado correcto de 8 dimensiones
    dummy = np.zeros((len(predictions_scaled), len(features)))
    dummy[:, close_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, close_idx]
    
    # Precios Reales para comparar
    actual_prices = df_filtered['Close'].values[-days_to_test:]
    
    # 7. Cálculo de Métricas y Gráfica
    rmse = np.sqrt(np.mean((predictions - actual_prices)**2))
    mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
    
    print(f"✅ Backtest Finalizado.")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - Precisión Promedio (MAPE): {100 - mape:.2f}%")
    
    # Visualización
    plt.style.use('dark_background')
    plt.figure(figsize=(14,7))
    plt.plot(actual_prices, label="Precio Real (Mercado)", color="#00ffcc", linewidth=2)
    plt.plot(predictions, label="Predicción IA (V3)", color="#ff9900", linestyle="--", linewidth=2)
    plt.fill_between(range(len(actual_prices)), actual_prices, predictions, color='gray', alpha=0.2)
    
    plt.title(f"StockAI V3 Backtesting: {ticker} (Last {days_to_test} days)")
    plt.xlabel("Días de Prueba")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.savefig('backtest_result_v3.png')
    print("📈 Gráfica guardada como 'backtest_result_v3.png'")
    plt.show()

if __name__ == "__main__":
    # Puedes cambiar el activo aquí para probar tu nueva V3
    run_backtest(ticker="BTC-USD", days_to_test=30)