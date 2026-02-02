import yfinance as yf
import pandas as pd
import os
import numpy as np
import pandas_ta as ta

def download_multitoken_data():
    # Lista profesional diversificada (Cesta de Mercado)
    tickers = [
        '399001.SZ', 'AAPL', 'ABBV', 'ABT', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALNY', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANET', 'ARM', 'ASML', 'AUDUSD=X', 'AVGO', 'AW=F', 'AXP', 'AZN', 'BA', 'BAC', 'BIDU', 'BIIB', 'BKNG', 'BKR', 'BRK.B', 'BSX', 'BX', 'BZ=F', 'CAT', 'CB', 'CCEP', 'CDNS', 'CEG', 'CHTR', 'CL=F', 'CMCSA', 'COST', 'CPRT', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CVX', 'DASH', 'DDOG', 'DE', 'DHR', 'DIS', 'DLTR', 'DOW', 'DX-Y.NYB', 'DXCM', 'EA', 'ETN', 'EURCHF=X', 'EURGBP=X', 'EURJPY=X', 'EURUSD=X', 'EXC', 'FANG', 'FAST','FTNT', 'GC=F', 'GE', 'GILD', 'GOOG', 'GS', 'HD', 'HG=F', 'HON', 'IBM','IDXX', 'IHI', 'ILMN', 'INTC', 'INTU', 'ISRG', 'IWN', 'IWO', 'JNJ', 'JPM', 'KC=F', 'KDP', 'KE=F', 'KHC', 'KLAC', 'KO', 'LIN', 'LLY', 'LMT', 'LRCX', 'LULU', 'MA', 'MAR', 'MCD', 'MCHP', 'MDLZ', 'MDT', 'MELI', 'META', 'MMC', 'MMM', 'MNST', 'MRK', 'MRVL', 'MSFT', 'MU', 'NEE', 'NFLX', 'NG=F', 'NKE', 'NOW', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORCL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PFE', 'PG', 'PH', 'PLD', 'PM', 'PSCE', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'RTX', 'SB=F', 'SBUX', 'SI=F', 'SIRI', 'SNPS', 'SPGI', 'SPLK', 'SYK', 'TEAM', 'TJX', 'TMUS', 'TRV', 'TSLA', 'TXN', 'UBER', 'UNH', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X', 'V', 'VRSK', 'VRTX', 'VZ', 'WBD', 'WDAY', 'WMT', 'XEL','XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV','XLY', 'XOM', 'ZC=F', 'ZS', 'ZS=F', 'ZW=F', '^AXJO', '^BCOM', '^BVSP', '^COLCAP', '^DJI', '^DJUSFN', '^DJUSRT', '^FCHI', '^FTSE', '^GDAXI' '^GSPC', '^GSPTSE', '^HSI', '^IBEX', '^IPSA', '^IXIC', '^KS11', '^MERV', '^MSCI', '^MXX', '^N225', '^NYA', '^R2FIN', '^R2HC','^R2ICBBAN', '^R2ICBINTR', '^R2RGSENG', '^R2TECH', '^RGUSHSBT', '^RUJ','^RUO', '^RUT', '^SP500-10', '^SP500-15', '^SP500-20', '^SP500-30','^SP500-35', '^SP500-40', '^SP500-45', '^SP500-50', '^SP500-55', '^SP500-60','^SP600', '^SP600-20', '^SP600-40', '^SSEC', '^STOXX50E', '^VIX'
    ]
    
    all_data = []
    print(f"🚀 StockAI V6: Iniciando descarga y pre-procesamiento de {len(tickers)} activos...")

    for ticker in tickers:
        try:
            print(f"📥 Procesando {ticker}...")
            # Descargamos 10 años para un entrenamiento profundo
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            
            if df.empty or len(df) < 250:
                continue

            # --- LIMPIEZA DE MULTIINDEX ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # --- CÁLCULO DE INDICADORES V6 ---
            # 1. ADX (Filtro de Fuerza)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx_df is not None:
                df['ADX'] = adx_df['ADX_14']
            
            # 2. RSI (Momento)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 3. Medias Móviles (Tendencia)
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # 4. Identificador
            df['Ticker'] = ticker

            # --- SELECCIÓN DE FEATURES (El estándar de 9 columnas) ---
            cols_v6 = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX', 'Ticker']
            df = df[cols_v6]

            # Limpiar filas con valores nulos (por el cálculo de medias y ADX)
            df = df.dropna()
            
            all_data.append(df)
            
        except Exception as e:
            print(f"⚠️ Error con {ticker}: {e}")

    # --- UNIFICACIÓN Y GUARDADO ---
    if all_data:
        final_df = pd.concat(all_data)
        
        os.makedirs('data', exist_ok=True)
        # Guardamos con el nombre que espera train_v6.py
        final_df.to_csv('data/multi_stock_data.csv', index=True) 
        
        print("\n" + "="*40)
        print("✅ DATASET MAESTRO V6 CREADO")
        print(f"📂 Ubicación: data/multi_stock_data.csv")
        print(f"📊 Muestras de entrenamiento: {len(final_df)}")
        print(f"🛡️ Activos procesados: {len(all_data)}")
        print("="*40)
    else:
        print("❌ Fallo crítico: No se pudieron recolectar datos.")

if __name__ == "__main__":
    download_multitoken_data()