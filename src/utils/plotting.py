import matplotlib.pyplot as plt
import pandas as pd

import os

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