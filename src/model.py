import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM input.
    X: (samples, seq_length, features)
    y: (samples, target)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Predicting the first column (Close price)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    """
    Build LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Regression output
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_save_model(df, seq_length=60, epochs=20, batch_size=32, model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    """
    Full pipeline: Preprocess, Train, Save.
    Assumes df has 'Close' as the first column or we select it.
    """
    # Use Close price for prediction target, but use all features for input
    # Ensure 'Close' is at index 0 or handle scaling appropriately
    # For simplicity, let's scale all features
    
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for inference
    joblib.dump(scaler, scaler_path)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split Data (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training shape: {X_train.shape}")
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history, scaler
