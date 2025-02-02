import os
import time
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)

class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_start_time)

class StockModel:
    def __init__(self, csv_file, target_column='Close', seq_length=60, train_size=0.8):
        self.csv_file = csv_file
        self.target_column = target_column
        self.seq_length = seq_length
        self.train_size = train_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.setup_gpu()
        self.setup_threading()

    def setup_gpu(self):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Using GPU: {gpus}")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU: {e}")
        else:
            logging.info("No GPU found. Using CPU.")

    def setup_threading(self):
        tf.config.threading.set_inter_op_parallelism_threads(19)
        tf.config.threading.set_intra_op_parallelism_threads(19)

    def load_data(self):
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The file {self.csv_file} does not exist.")

        data = pd.read_csv(self.csv_file, parse_dates=['Date'], index_col='Date')
        if data.isnull().values.any():
            logging.warning("Data contains missing values. Consider handling them before proceeding.")
        return data

    def prepare_data(self, data):
        feature_columns = [col for col in data.columns if col != 'Ticker']
        features = data[feature_columns].values
        target = data[[self.target_column]].values

        # Normalize data
        scaled_features = self.scaler.fit_transform(features)
        scaled_target = self.scaler.fit_transform(target)

        def create_sequences(features, target, seq_length):
            X, y = [], []
            for i in range(seq_length, len(features)):
                X.append(features[i-seq_length:i])
                y.append(target[i, 0])
            return np.array(X), np.array(y)

        train_size = int(len(scaled_features) * self.train_size)
        train_features = scaled_features[:train_size]
        test_features = scaled_features[train_size - self.seq_length:]
        train_target = scaled_target[:train_size]
        test_target = scaled_target[train_size - self.seq_length:]

        X_train, y_train = create_sequences(train_features, train_target, self.seq_length)
        X_test, y_test = create_sequences(test_features, test_target, self.seq_length)

        return X_train, y_train, X_test, y_test

    def create_model(self, input_shape):
        self.model = Sequential()

        # Optimized architecture
        self.model.add(LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        
        self.model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        
        self.model.add(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Dense( 32, activation='relu', kernel_regularizer=l2(0.01)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])

        # Visualize model
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def train_model(self):
        try:
            data = self.load_data()
            X_train, y_train, X_test, y_test = self.prepare_data(data)

            if np.isnan(X_train).any() or np.isnan(y_train).any():
                raise ValueError("Input data contains NaN values after preparation.")

            self.create_model((X_train.shape[1], X_train.shape[2]))

            early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

            time_history = TimeHistory()

            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=1000,
                batch_size=32,
                callbacks=[early_stopping, time_history, lr_scheduler],
            )

            self.plot_training_history(history)
            self.plot_training_time(time_history.times)

            model_save_path = os.path.join("stock_model.keras")
            scaler_save_path = os.path.join("scaler.save")
            self.model.save(model_save_path)
            joblib.dump(self.scaler, scaler_save_path)
            return True
        except Exception as e:
            logging.error(f"Error during training: {e}")
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
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mean_absolute_error"], label="Training MAE")
        plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
        plt.title("Training and Validation Mean Absolute Error")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid()

        plt.savefig("training_history.png", dpi=300)

    def plot_training_time(self, times):
        epochs = range(1, len(times) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, times)
        plt.title("Training Time per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Time (seconds)")
        plt.grid()
        plt.savefig("training_time.png", dpi=300)

    def evaluate_model(self, predictions, y_test):
        """Evaluate the model using various metrics."""
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)

        # Percentage error (mean error in percentage)
        mean_actual = np.mean(y_test_actual)
        percentage_error = (rmse / mean_actual) * 100

        # Output metrics
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R² (Coefficient of determination): {r2}")
        print(f"Average percentage error: {percentage_error:.2f}%")

        return mse, mae, rmse, r2, percentage_error

if __name__ == "__main__":
    csv_file_path = os.path.join("cleaned_stock_data.csv")  # Ensure the file path is correct
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The file {csv_file_path} does not exist.")

    stock_model = StockModel(csv_file_path)
    
    success = stock_model.train_model()
    if success:
        logging.info("Model trained and saved successfully.")
        
        # Get test data y_test for evaluation
        test_data = stock_model.load_data()
        _, _, X_test, y_test = stock_model.prepare_data(test_data)
        
        if X_test is not None and y_test is not None:
            predictions = stock_model.model.predict(X_test)
            stock_model.evaluate_model(predictions, y_test)
        else:
            logging.error("Model training failed.")
