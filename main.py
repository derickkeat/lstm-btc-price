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
from src.data.preprocess import normalize, denormalize
from src.features.indicators import calculate_technical_indicators

SEED = 42

# Set seeds for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set TensorFlow session determinism for reproducibility
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

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
    model.save("models/lstm_paper.keras") # update the used model with the new one

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

def evaluate_mse(frame: pd.DataFrame):
    # Calculate the mean squared error between actual close price and predictions
    mse = np.mean((frame['close'] - frame['Predictions']) ** 2)
    return mse

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

# Steady state genetic algorithm
class SSGA:
    def __init__(
        self,
        df: pd.DataFrame,
        pop_size: int,
        num_generations: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int,
        random_seed: Optional[int],
    ):
        self.df = df.copy()
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        if random_seed is not None:
            np.random.seed(random_seed)

    def _initialize_population(self):
        population = []
        
        for _ in range(self.pop_size):
            individual = np.random.random()  # Generate a random float between 0 and 1
            population.append(individual)
            
        return population

    def _fitness(self, df: pd.DataFrame, individual: int):
        stbc = STBC(df, individual)
        stbc.calibrate()
        return 1 / stbc.evaluate_accuracy()
    
    def _selection(self, population: list[int], fitness_scores: list[float]):
        # randomly get the indexes of population
        tournament = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]

        winner = tournament[tournament_fitness.index(max(tournament_fitness))]
        return population[winner]
    
    def _crossover(self, parent1: float, parent2: float):
        # One-point crossover implementation for floats
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # For floats, we can do a weighted average (linear interpolation)
        # Choose a random weight between 0 and 1
        alpha = random.random()
        
        # Create two children using complementary weights
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        # Ensure the values stay in the [0,1] range
        child1 = max(0.0, min(1.0, child1))
        child2 = max(0.0, min(1.0, child2))
        
        return child1, child2
        
    def _mutate(self, individual: int):
        mutated = individual

        if random.random() < self.mutation_rate:
            # Randomly flip a bit in the integer representation
            bit_to_flip = random.randint(0, 31)
            mutated ^= (1 << bit_to_flip)  # Flip the bit at the position

        return mutated

    def __call__(self):
        population = self._initialize_population()
        best_individual = None
        best_fitness = float('-inf')

        fitness_scores = [self._fitness(self.df, individual) for individual in population]

        for _ in range(self.num_generations):
            parent1 = self._selection(population, fitness_scores)
            parent2 = self._selection(population, fitness_scores)

            child1, child2 = self._crossover(parent1, parent2)

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            child1_fitness = self._fitness(self.df, child1)
            child2_fitness = self._fitness(self.df, child2)

            if child1_fitness > min(fitness_scores):
                index_to_replace = fitness_scores.index(min(fitness_scores))
                population[index_to_replace] = child1
                fitness_scores[index_to_replace] = child1_fitness
            if child2_fitness > min(fitness_scores):
                index_to_replace = fitness_scores.index(min(fitness_scores))
                population[index_to_replace] = child2
                fitness_scores[index_to_replace] = child2_fitness

            if max(fitness_scores) > best_fitness:
                best_fitness = max(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

        return best_individual, best_fitness

if __name__ == "__main__":
    # read the csv file and drop the columns that are not needed
    df = pd.read_csv("new.csv", sep=";", index_col="timestamp")
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

    # save_results(original_test) # NOTE: only run this if you want to save the results without calibration

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

    # save_results(stbc.df) # NOTE: only run this if you want to save the results after calibration