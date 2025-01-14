import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping
from keras.layers import GRU, LSTM, Dense, Dropout, Input
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2


class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(tf.timestamp() - self.epoch_start_time)


class StockModel:
    def __init__(self, csv_file, model_type="LSTM"):
        self.csv_file = csv_file
        self.model = None
        self.scaler = None
        self.model_type = model_type  # Добавляем параметр для выбора типа модели

        self.setup_gpu()
        self.setup_threading()

    def setup_gpu(self):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                print(f"Using GPU: {gpus}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("No GPU found. Using CPU.")

    def setup_threading(self):
        tf.config.threading.set_inter_op_parallelism_threads(19)
        tf.config.threading.set_intra_op_parallelism_threads(19)

    def load_data(self):
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The file {self.csv_file} does not exist.")

        data = pd.read_csv(self.csv_file)

        required_columns = ["Open", "Close", "High", "Low", "Adj Close", "Volume"]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"The CSV file must contain a '{column}' column.")

        data = data.dropna(subset=["Close"])

        features = data[required_columns].values

        return features

    def prepare_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        time_steps = 10

        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i - time_steps : i])
            y.append(scaled_data[i, 1])

        X, y = np.array(X), np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))

        return X, y

    def create_model(self, input_shape):
        self.model = Sequential()

        if self.model_type == "LSTM":
            self.model.add(
                LSTM(
                    units=1024,
                    return_sequences=True,
                    input_shape=input_shape,
                    kernel_regularizer=l2(0.01),
                )
            )
        elif self.model_type == "GRU":
            self.model.add(
                GRU(
                    units=1024,
                    return_sequences=True,
                    input_shape=input_shape,
                    kernel_regularizer=l2(0.01),
                )
            )
        else:
            raise ValueError("Unsupported model type. Choose 'LSTM' or 'GRU'.")

        self.model.add(Dropout(0.4))
        
        if self.model_type == "LSTM":
            self.model.add(LSTM(units=512, return_sequences=True, kernel_regularizer=l2(0.01)))
        else:
            self.model.add(GRU(units=512, return_sequences=True, kernel_regularizer=l2(0.01)))

        self.model.add(Dropout(0.3))

        if self.model_type == "LSTM":
            self.model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=l2(0.01)))
        else:
            self.model.add(GRU(units=256, return_sequences=True, kernel_regularizer=l2(0.01)))

        self.model.add(Dropout(0.2))

        if self.model_type == "LSTM":
            self.model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01)))
        else:
            self.model.add(GRU(units=128, return_sequences=True, kernel_regularizer=l2(0.01)))

        self.model.add(Dropout(0.2))

        if self.model_type == "LSTM":
            self.model.add(LSTM(units=64))
        else:
            self.model.add(GRU(units=64))

        self.model.add(Dropout(0.1))

        self.model.add(Dense(units=32, activation="relu"))
        self.model.add(Dense(units=1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )

    def train_model(self):
        try:
            data = self.load_data()
            X, y = self.prepare_data(data)

            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Input data contains NaN values after preparation.")

            self.create_model((X.shape[1], X.shape[2]))

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            split_index = int(len(X) * 0.8)
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            batch_size = 5

            time_history = TimeHistory()

            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=40,
                batch_size=batch_size,
                callbacks=[early_stopping, time_history],
            )

            self.plot_training_history(history)
            self.plot_training_time(time_history.times)

            model_save_path = os.path.join("stock_model.keras")
            scaler_save_path = os.path.join("scaler.save")
            self.model.save(model_save_path)
            joblib.dump(self.scaler, scaler_save_path)
            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False

    def plot_training_history(self, history):
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mean_absolute_error"], label="Training MAE")
        plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")

        plt.title("Training and Validation Mean Absolute Error")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")

        plt.legend()
        plt.savefig("training_history.png")

    def plot_training_time(self, times):
        epochs = range(1, len(times) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, times)
        plt.title("Training Time per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Time (seconds)")
        plt.grid()
        plt.savefig("training_time.png")


if __name__ == "__main__":
    csv_file_path = os.path.join("combined_stock_data.csv")
    model_type = "LSTM"  # Выбор типа модели: 'LSTM' или 'GRU'
    stock_model = StockModel(csv_file_path, model_type)
    success = stock_model.train_model()
    if success:
        print("Model trained and saved successfully.")
    else:
        print("Model training failed.")
