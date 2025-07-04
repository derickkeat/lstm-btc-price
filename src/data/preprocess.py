import pandas as pd
import numpy as np

def normalize(frame: pd.DataFrame):
    return frame.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

def denormalize(normalized_values, original_values):
    min_val = np.min(original_values)
    max_val = np.max(original_values)
    return normalized_values * (max_val - min_val) + min_val