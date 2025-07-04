import pandas as pd
import numpy as np
import tensorflow as tf

import os
from datetime import datetime

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