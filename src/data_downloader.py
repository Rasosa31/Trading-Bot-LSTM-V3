import yfinance as yf
import pandas as pd
import os
import numpy as np

def download_multitoken_data():
    # Lista profesional diversificada
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
        'AMD', 'INTC', 'PYPL', 'NFLX', 'ADBE',
        'SPY', 'QQQ', 'DIA', 'BTC-USD', 'ETH-USD',
        'GLD', 'VTI', 'TLT', '^TNX', 'CL=F',
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X', 'CNHUSD=X',
        'AUDUSD=X', 'USDCHF=X', 'EURJPY=X', 'DX-Y.NYB', 'OTLY', 'PL', 'RKLB',
        'XLE', 'XLF', 'XLK', 'XLY', 'XLI', 'XLB', 'XLV', 'XLU'
        'KO', 'DIS', 'PYPL', 'CRM', 'ABNB', 'UBER', 'RKLB'
    ]
    
    all_data = []
    print(f"🚀 StockAI V3: Iniciando descarga de {len(tickers)} activos...")

    for ticker in tickers:
        try:
            print(f"📥 Descargando {ticker}...")
            # Descargamos 10 años para tener una base sólida de aprendizaje
            df = yf.download(ticker, period="10y", interval="1d")
            
            if df.empty:
                continue

            # --- LIMPIEZA DE MULTIINDEX (Vital para evitar errores) ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # --- CÁLCULO DE INDICADORES (Sincronizado con V3) ---
            # 1. Medias Móviles
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # 2. RSI (14 periodos)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. IDENTIFICADOR
            df['Ticker'] = ticker

            # 4. SELECCIÓN DE COLUMNAS (El estándar de 8 features + Ticker)
            # Solo guardamos lo que el modelo realmente va a usar para entrenar
            cols_v3 = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'Ticker']
            df = df[cols_v3]

            # Limpiar filas iniciales (los primeros 200 días de cada activo)
            df = df.dropna()
            
            all_data.append(df)
            
        except Exception as e:
            print(f"⚠️ Error con {ticker}: {e}")

    # Combinación y Guardado
    if all_data:
        final_df = pd.concat(all_data)
        
        os.makedirs('data', exist_ok=True)
        # index=True para conservar la fecha, que es útil para depurar
        final_df.to_csv('data/multi_stock_data.csv', index=True) 
        
        print("\n" + "="*30)
        print("✅ DATASET V3 CREADO CON ÉXITO")
        print(f"📂 Archivo: data/multi_stock_data.csv")
        print(f"📊 Registros totales: {len(final_df)}")
        print(f"🛡️ Columnas: {list(final_df.columns)}")
        print("="*30)
    else:
        print("❌ No se pudo descargar ningún dato.")

if __name__ == "__main__":
    download_multitoken_data()