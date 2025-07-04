import pandas as pd
import ta  # Technical Analysis library for Python

def calculate_technical_indicators(df: pd.DataFrame):
    # Calculate technical indicators
    # Stochastic Oscillator (K and D values)
    df['k_9'] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3
    ).stoch()
    df['d_9'] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3
    ).stoch_signal()
    
    # Moving Averages
    df['ma_6'] = ta.trend.sma_indicator(df['close'], window=6)
    df['ma_9'] = ta.trend.sma_indicator(df['close'], window=9)
    df['ma_12'] = ta.trend.sma_indicator(df['close'], window=12)
    
    # Bias indicators (difference between price and MA)
    df['bias_6'] = (df['close'] - df['ma_6']) / df['ma_6'] * 100
    df['bias_3'] = (df['close'] - ta.trend.sma_indicator(df['close'], window=3)) / ta.trend.sma_indicator(df['close'], window=3) * 100
    df['bias_3_minus_bias_6'] = df['bias_3'] - df['bias_6']
    
    # RSI
    df['rsi_6'] = ta.momentum.rsi(df['close'], window=6)
    
    # Williams %R
    df['williams_r_12'] = ta.momentum.WilliamsRIndicator(
        high=df['high'], low=df['low'], close=df['close'], lbp=12
    ).williams_r()
    
    # Momentum
    df['momentum_6'] = ta.momentum.roc(df['close'], window=6)
    # Momentum Moving Average
    df['momentum_6_ma'] = ta.trend.sma_indicator(df['momentum_6'], window=6)
    
    # MACD
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_9'] = macd.macd()
    df['macd_signal_9'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()  # This is the difference value (12-day - 26-day)

    return df