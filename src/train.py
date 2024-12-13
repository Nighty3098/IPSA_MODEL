import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


class StockModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.model = None
        self.scaler = None

        # Настройка использования GPU
        self.setup_gpu()

    def setup_gpu(self):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    print(f"GPU {gpu} found and used.")
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")

    def load_data(self):
        # Проверка существования файла перед загрузкой данных
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The file {self.csv_file} does not exist.")

        data = pd.read_csv(self.csv_file)

        # Проверка наличия столбца "Close"
        if "Close" not in data.columns:
            raise ValueError("The CSV file must contain a 'Close' column.")

        # Удаление строк с NaN значениями в столбце "Close"
        data = data.dropna(subset=["Close"])

        return data["Close"].values.reshape(-1, 1)

    def prepare_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60 : i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y

    def create_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        # Используем Adam с уменьшенной скоростью обучения
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001
        )  # Уменьшена скорость обучения
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def train_model(self):
        try:
            data = self.load_data()
            X, y = self.prepare_data(data)

            # Проверка на наличие NaN в подготовленных данных перед обучением
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Input data contains NaN values after preparation.")

            # Создание модели с заданной формой входных данных
            self.create_model((X.shape[1], 1))
            self.model.fit(X, y, epochs=100, batch_size=32)

            model_save_path = os.path.join("stock_model.keras")
            scaler_save_path = os.path.join("scaler.save")
            self.model.save(model_save_path)
            joblib.dump(self.scaler, scaler_save_path)
            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False


if __name__ == "__main__":
    csv_file_path = os.path.join("combined_stock_data.csv")
    stock_model = StockModel(csv_file_path)
    success = stock_model.train_model()
    if success:
        print("Model trained and saved successfully.")
    else:
        print("Model training failed.")
