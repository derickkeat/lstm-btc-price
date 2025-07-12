import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
import os

def save_results(frame: pd.DataFrame, prediction_column: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"results/{timestamp}", exist_ok=True)
    frame.to_csv(f"results/{timestamp}/results.csv", sep=";", index=True)

    # plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(frame.index, frame["close"], label="close", color="red")
    plt.plot(frame.index, frame[prediction_column], label=prediction_column, color="blue")
    plt.legend()
    plt.savefig(f"results/{timestamp}/prediction_plot.png")
    plt.close()

    print(frame)
    print(f"Results saved to /results/{timestamp}/")