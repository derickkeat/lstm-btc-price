import pandas as pd
import numpy as np

# Short-term bias compensation from the paper
class STBC:
    def __init__(self, df: pd.DataFrame, calibration_threshold: float):
        self.df = df.copy()
        self.previous_calibration = 0
        self.calibration_threshold = calibration_threshold

    def calibrate(self):
        # insert a new column for calibrated predictions
        self.df.insert(6, "STBC Predictions", 0.0)
        self.df.insert(7, "STBC Predictions diff", 0.0)
        
        #calibrate the predictions row by row
        for i in range(self.df.shape[0]):
            calibrated_price = self.df.iloc[i, 3] + self.previous_calibration
            calibration_today = self.df.iloc[i, 0] - calibrated_price
            if abs(calibration_today) > self.df.iloc[i, 0] * self.calibration_threshold:
                self.previous_calibration = calibration_today
            self.df.iloc[i, 6] = calibrated_price
            self.df.iloc[i, 7] = (self.df.iloc[i, 6] - self.df.iloc[i, 0]) / self.df.iloc[i, 0]
            
        return self.df

    def evaluate_accuracy(self):
        # take the mean of the absolute percentage difference
        return np.mean(np.abs(self.df.iloc[:, 7]))

    # calculate the mean squared error of the STBC value
    def evaluate_mse(self):
        # Mean squared error between actual close price and STBC predictions
        return np.mean((self.df.iloc[:, 0] - self.df.iloc[:, 6]) ** 2)