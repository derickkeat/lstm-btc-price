import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from datetime import datetime

def plot_data(df: pd.DataFrame):
    os.makedirs("plots", exist_ok=True)
    # plot the close price and volume over time
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["open"], label="open", color="blue")
    plt.plot(df.index, df["close"], label="close", color="red")
    plt.legend()
    plt.title("Open/Close Price over Time")
    plt.savefig("plots/open_close_price.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["volume"], label="volume", color="orange")
    plt.title("Volume over Time")
    plt.savefig("plots/volume.png")
    plt.close()

def normalize(frame: pd.DataFrame):
    return frame.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

def denormalize(normalized_values, original_values):
    min_val = np.min(original_values)
    max_val = np.max(original_values)
    return normalized_values * (max_val - min_val) + min_val

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

def build_train_data(df, past_days: int = 20, future_days: int = 1):
    x_train, y_train = [], []

    # sliding window
    for i in range(df.shape[0] - past_days - future_days):
        x_train.append(np.array(df.iloc[i:i+past_days]))
        y_train.append(np.array(df.iloc[i+past_days:i+past_days+future_days]["close"]))
    
    return np.array(x_train), np.array(y_train)

def split_data(x, y, test_size: float = 0.2):
    # Calculate the split point - use the last test_size portion for testing
    split_idx = int(x.shape[0] * (1 - test_size))
    
    # Training data is the older data (beginning to split point)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    
    # Test data is the more recent data (split point to end)
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    
    return x_train, y_train, x_test, y_test

def train_model(x_train: np.array, y_train: np.array):
    # build the model
    model = tf.keras.Sequential()

    # add the layers
    model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(units=128, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=32, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation="linear"))

    # compile the model
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    model.fit(x_train, y_train, epochs=20, batch_size=32)
    model.summary()

    # save the model
    os.makedirs("models_backup", exist_ok=True) # create the models directory if it doesn't exist
    model.save(f"models_backup/lstm_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")

def predict_model(model_path: str, x_test: np.array):
    # load the model
    model = tf.keras.models.load_model(model_path)

    # predict the data
    predictions = model.predict(x_test)
    return predictions

def save_results(frame: pd.DataFrame):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    frame.to_csv(f"results/{timestamp}/results.csv", sep=";", index=True)

    # plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(original_test.index, original_test["close"], label="close", color="red")
    plt.plot(original_test.index, original_test["Predictions"], label="predictions", color="blue")
    plt.legend()
    plt.savefig(f"results/{timestamp}/predictions.png")
    plt.close()

    print(frame)
    print(f"Results saved to /results/{timestamp}/")

if __name__ == "__main__":
    # read the csv file and drop the columns that are not needed
    df = pd.read_csv("new.csv", sep=";", index_col="timestamp")
    df = df.drop(["timeOpen", "timeClose", "timeHigh", "timeLow", "name"], axis=1)
    
    plot_data(df) # NOTE: only run this if you want to see the data

    # add the technical indicators to the dataframe
    df = calculate_technical_indicators(df)

    # drop the rows that have NaN values
    df = df.dropna()

    # store the original dataframe
    original_df = df.copy()
    
    # normalize the data
    df = normalize(df)

    x_train, y_train = build_train_data(df)

    # split the data into train and test
    x_train, y_train, x_test, y_test = split_data(x_train, y_train)

    # train the model
    train_model(x_train, y_train) # NOTE: only run this if you want to train the model

    # predict the test data
    predictions = predict_model("models/lstm_paper.keras", x_test)

    # get the test start index
    test_start_idx = df.shape[0] - predictions.shape[0]
    
    # get the original close price values after dropping NaN rows
    original_close = original_df['close'].copy()
    original_close_test = original_close[test_start_idx:]
    
    # denormalize predictions using the original test data values
    predictions = denormalize(predictions, original_close_test)

    # seperate the data into train and test
    original_train = original_df[:test_start_idx]
    original_test = original_df[test_start_idx:].copy()

    # add the predictions to the test data
    original_test["Predictions"] = predictions

    # drop the columns that are not needed
    original_test = original_test.drop(["open", "high", "low", "k_9", "d_9", "ma_6", "ma_9", "ma_12", "bias_6", "bias_3", "bias_3_minus_bias_6", "rsi_6", "williams_r_12", "momentum_6", "momentum_6_ma", "macd_9", "macd_signal_9", "macd_diff"], axis=1)

    original_test["diff"] = original_test["Predictions"] - original_test["close"]
    original_test["diff_percentage"] = original_test["diff"] / original_test["close"]

    save_results(original_test) # NOTE: only run this if you want to save the results