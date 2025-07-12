import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from datetime import datetime
from typing import Optional
import random

from src.utils.plotting import plot_data
from src.data.preprocess import normalize, denormalize, build_train_data, split_data
from src.features.indicators import calculate_technical_indicators
from src.models.lstm import train_model, predict_model
from src.metrics import evaluate_mse
from src.bias.stbc import STBC
from src.optim.ssga import SSGA
from src.utils.reproducibility import set_seed
from src.data.saver import save_results

if __name__ == "__main__":
    set_seed(42)  # Set seed for reproducibility

    # read the csv file and drop the columns that are not needed
    df = pd.read_csv("data/processed/new.csv", sep=";", index_col="timestamp")
    df = df.drop(["timeOpen", "timeClose", "timeHigh", "timeLow", "name"], axis=1)
    
    # plot_data(df) # NOTE: only run this if you want to see the data

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
    # train_model(x_train, y_train) # NOTE: only run this if you want to train the model

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

    save_results(original_test, "Predictions") # NOTE: only run this if you want to save the results without calibration

    print("MSE without STBC:" + str(evaluate_mse(original_test)))
    print("Average prediction error without STBC:" + str(np.mean(np.abs(original_test["diff_percentage"]))))

    # calibrate the predictions
    stbc = STBC(original_test, 0.05)
    stbc.calibrate()
    print("MSE before optimizing STBC:" + str(stbc.evaluate_mse()))
    print("Average prediction error before optimizing STBC:" + str(stbc.evaluate_accuracy()))

    ssga = SSGA(
        df=original_test,
        pop_size=100, # not mentioned in the paper
        num_generations=15,
        mutation_rate=0.01,
        crossover_rate=1, # not mentioned in the paper
        tournament_size=5,
        random_seed=42
    )

    print("STBC Calibration using SSGA...")
    best_individual, best_fitness = ssga()
    print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")

    stbc = STBC(original_test, best_individual)
    stbc.calibrate()
    print("MSE after optimizing STBC:" + str(stbc.evaluate_mse()))
    print("Average prediction error after optimizing STBC:" + str(stbc.evaluate_accuracy()))

    save_results(stbc.df, "STBC Predictions") # NOTE: only run this if you want to save the results after calibration