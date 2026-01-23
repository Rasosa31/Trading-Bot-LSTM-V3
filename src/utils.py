import yfinance as yf
import pandas as pd
import numpy as np

def load_data_with_indicators(ticker, period='2y', interval='1d'):
    """
    Descarga y procesa datos asegurando la estructura completa de indicadores
    para el ecosistema StockAI V3.
    """
    print(f"--- 🛠️ Procesando indicadores para: {ticker} ---")
    
    try:
        # Descarga de datos base
        df = yf.download(ticker, period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No se obtuvieron datos para {ticker}")

        # Limpieza de MultiIndex (Yahoo Finance API 2024/2025)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        df = df.copy()

    except Exception as e:
        print(f"Error en descarga: {e}")
        raise ValueError(f"Error al obtener datos de {ticker}")

    # 1. CÁLCULO DE INDICADORES TÉCNICOS
    # Medias Móviles (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # 1e-9 evita la división por cero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bandas de Bollinger
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    
    # 2. FILTRADO Y ORGANIZACIÓN DE COLUMNAS (Total: 12)
    # Definimos el orden exacto para mantener la consistencia en el Scaler
    cols_to_keep = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 
        'RSI', 'BB_Mid', 'BB_Std'
    ]
    
    # Verificación de existencia y relleno de seguridad (bfill/ffill)
    # Esto asegura que no perdamos filas valiosas por el retraso de las SMA
    df_clean = df.reindex(columns=cols_to_keep)
    df_clean = df_clean.bfill().ffill().dropna()
    
    print(f"✅ Pipeline completado: {len(df_clean)} velas procesadas con {len(df_clean.columns)} indicadores.")
    
    return df_clean

def download_data(ticker, period='2y', interval='1d'):
    """Función simplificada de descarga."""
    return yf.download(ticker, period=period, interval=interval)