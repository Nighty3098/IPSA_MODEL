import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

home_dir = os.path.expanduser("~")


class StockModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.model = None
        self.scaler = None
        self.model_path = "stock_model.h5"
        self.scaler_path = "scaler.save"
        # self.model_path = os.path.join(home_dir, "IPSA", "stock_model.h5")
        # self.scaler_path = os.path.join(home_dir, "IPSA", "scaler.save")

    def load_data(self):
        data = pd.read_csv(self.csv_file)
        return data

    def prepare_data(self, data):
        # Используем все необходимые параметры
        features = data[["Open", "Close", "High", "Low", "Adj Close", "Volume"]].values

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(features)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60 : i])  # 60 предыдущих дней
            y.append(
                scaled_data[i, 0]
            )  # Предсказываем "Open" (или любой другой параметр)

        X, y = np.array(X), np.array(y)
        # Убедитесь, что X имеет правильную форму
        X = np.reshape(
            X, (X.shape[0], X.shape[1], X.shape[2])
        )  # Убедитесь, что у нас 2D массив

        return X, y

    def create_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(units=50, activation="relu"))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self):
        try:
            data = self.load_data()
            X, y = self.prepare_data(data)

            self.create_model((X.shape[1], X.shape[2]))

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    print("Using multiple GPUs...")
                    for gpu in gpus:
                        tf.config.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            history = self.model.fit(
                X,
                y,
                epochs=1000,
                batch_size=32,
                validation_split=0.2,
            )

            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False


# USAGE:

if __name__ == "__main__":
    # csv_file_path = os.path.join(home_dir, "IPSA", "combined_stock_data.csv")
    csv_file_path = "combined_stock_data.csv"
    model = StockModel(csv_file_path)
    model.train_model()
