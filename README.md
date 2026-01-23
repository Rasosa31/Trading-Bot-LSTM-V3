📈 <h1>StockAI V2: Adaptive Multi-Indicator Intelligence</h1>

StockAI V2 is a financial predictive analysis platform that merges Deep Learning with traditional technical analysis. It utilizes a Long Short-Term Memory (LSTM) Recurrent Neural Network architecture to process time-series data and project price trends in global financial markets.

🔗 <h2>Quick Links</h2>

Web Deployment: https://stockai-predictor-rasosa.streamlit.app

Project Status: Production / Stable (Python 3.11).

📝 <h2>1. Problem Description</h2>

Predicting financial markets is a complex challenge due to high volatility and the non-linear nature of data. Many traditional models fail because they ignore momentum index(RSI) or long-term trends (Moving Averages).

<h2>StockAI V2 solves this through:</h2>

Multivariate Analysis: The model integrates the Relative Strength Index (RSI) and Simple Moving Averages (SMA 100/200) as input features for the neural network.

Dynamic Training: The app trains a neural network in real-time using the latest data from Yahoo Finance, adapting to current market conditions.

⚙️ <h2>2. Internal Mechanics and Structure
AI Architecture</h2>

The "brain" of the app is an LSTM network designed to remember long-term historical patterns.

Extraction: Downloads data via yfinance.

Feature Engineering: Calculates technical indicators (RSI, SMA) in real-time.

Scaling: Normalizes data using MinMaxScaler for optimal learning.

Prediction: Projects the value for the next period based on a sliding observation window.

```
StockAI-Predictor-V2/ Structure

├── .python-version          # Forces Python 3.11 for cloud stability
├── requirements.txt         # Core dependencies (TensorFlow-CPU, Streamlit, etc.)
├── app/
│   └── app.py               # Main Entry point for the Streamlit Dashboard
├── src/                     # Source code logic
│   ├── model.py             # LSTM Architecture
│   ├── data_downloader.py   # Yahoo Finance API integration
│   ├── strategy.py          # Technical indicators calculation
│   ├── train.py             # Model training and scaling logic
│   └── backtesting.py       # Validation and RMSE calculation engine
├── models/                  # Saved .h5 models and scalers
└── notebooks/               # EDA (Exploratory Data Analysis) and research
```

🚀 <h2>3. Local Execution Guide
Prerequisites</h2>

Python 3.11 installed.

Active internet connection.

Installation Steps

Clone the Project:

```Bash
git clone https://github.com/Rasosa31/StockAI-Predictor-V2.git

cd StockAI-Predictor-V2

```
Install Dependencies:

```Bash
pip install -r requirements.txt
```
Launch the Application:

Bash
Run in the terminal (be sure to stay inside the carpet project) and run the follow:
```
python3 -m streamlit run app/app.py
```

This action opens a box or message that allows you to launch the app's web interface.


Steps to make a prediction ( You must use Yahoo Finance tickets).

- Select the ticket you want to predict and press Enter.

- Choose the timeframe you want to predict for the asset: daily, weekly, or monthly.

- In the right-hand window, the "Run Projection" button will appear. Wait a few seconds, and you will get a chart with the asset's current price, the projected price, and the expected percentage change.

Note: If you predict on a daily timeframe, the predicted price will be the asset's closing price at the end of the day. If you predict on a weekly timeframe, the price will be the closing price one week later. The same applies to the monthly timeframe. 

<h2>Backtesting</h2>

You can also backtest any available asset. This involves testing how well the model predicts using historical data and comparing it to the asset's price during the same period. In addition to the visual tool, an RMSE calculation is also provided, comparing the projection to the actual data during the backtesting period.

📊 <h2>4. Technical Findings and EDA</h2>

Data Engineering & Preprocessing Pipeline

The integration with Yahoo Finance required a sophisticated data pipeline to transform raw market data into a format digestible by the LSTM architecture:

Multi-Step Data Cleaning: Raw data from yfinance often contains gaps or missing values during market holidays; the system implements a robust cleaning process to ensure temporal continuity.

Feature Alignment: To enable multivariate analysis, technical indicators (RSI and SMAs) are calculated and then merged with the price action, ensuring all features are perfectly aligned by timestamp.

Sequential Windowing: The data is reshaped from a flat table into a 3D structure (samples, time steps, features), allowing the LSTM to look back at the last 60 periods to predict the next movement.

Normalization & Scaling: We utilized MinMaxScaler to compress all features (Price, RSI, SMA) into a range between 0 and 1, preventing high-value features (like Price) from overshadowing smaller-range features (like RSI) during model training.

Dynamic Handling of Young Assets: The pipeline includes a logic check to handle assets with short histories, automatically adjusting the window size for indicators like SMA 200 to prevent data truncation.

Metric Reliability: The model uses RMSE (Root Mean Square Error). A low RMSE indicates the AI tracked the actual price trends closely during backtesting.

Optimization: The system was optimized for Streamlit Cloud by using tensorflow-cpu to manage memory constraints effectively.
