import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
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
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")

    def load_data(self):
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The file {self.csv_file} does not exist.")

        data = pd.read_csv(self.csv_file)

        if "Close" not in data.columns:
            raise ValueError("The CSV file must contain a 'Close' column.")

        data = data.dropna(subset=["Close"])

        return data["Close"].values.reshape(-1, 1)

    def prepare_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60 : i, 0])
            y.append(1 if scaled_data[i, 0] > scaled_data[i - 1, 0] else 0)

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y

    def create_model(self, input_shape):
        self.model = Sequential()

        # Увеличение количества нейронов и слоев с L2 регуляризацией
        self.model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=128, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=64))
        self.model.add(Dropout(0.2))

        self.model.add(
            Dense(
                units=32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            )
        )
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1, activation="sigmoid"))

        # Использование смешанной точности для ускорения обучения
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # Установка политики смешанной точности
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

        self.model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train_model(self):
        try:
            data = self.load_data()
            X, y = self.prepare_data(data)

            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Input data contains NaN values after preparation.")

            self.create_model((X.shape[1], 1))

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            split_index = int(len(X) * 0.8)
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            # Увеличение размера пакета для более эффективного использования GPU
            batch_size = 64

            # Обучение модели с валидацией и ранней остановкой
            self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=1000,
                batch_size=batch_size,
                callbacks=[early_stopping],
            )

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
