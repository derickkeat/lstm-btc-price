import pandas as pd
import numpy as np

def normalize(frame: pd.DataFrame):
    return frame.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

def denormalize(normalized_values, original_values):
    min_val = np.min(original_values)
    max_val = np.max(original_values)
    return normalized_values * (max_val - min_val) + min_val

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