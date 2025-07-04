import pandas as pd
import numpy as np

def evaluate_mse(frame: pd.DataFrame):
    # Calculate the mean squared error between actual close price and predictions
    mse = np.mean((frame['close'] - frame['Predictions']) ** 2)
    return mse