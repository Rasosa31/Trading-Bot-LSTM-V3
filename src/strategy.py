import numpy as np

def get_signal(current_price, predicted_price, threshold=0.01):
    """
    Generate BUY/SELL/HOLD signal based on price difference.
    threshold: Minimum percentage change to trigger signal.
    """
    pct_change = (predicted_price - current_price) / current_price
    
    if pct_change > threshold:
        return "BUY"
    elif pct_change < -threshold:
        return "SELL"
    else:
        return "HOLD"

def calculate_sl_tp(current_price, signal, volatility, risk_reward_ratio=2.0):
    """
    Calculate Stop Loss and Take Profit levels.
    volatility: Standard deviation or ATR value.
    risk_reward_ratio: Target reward relative to risk.
    """
    # Simple strategy: Risk = 2 * Volatility
    risk = 2 * volatility
    
    stop_loss = 0.0
    take_profit = 0.0
    
    if signal == "BUY":
        stop_loss = current_price - risk
        take_profit = current_price + (risk * risk_reward_ratio)
    elif signal == "SELL":
        stop_loss = current_price + risk
        take_profit = current_price - (risk * risk_reward_ratio)
    
    return stop_loss, take_profit
