import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

WINDOW_SIZE = 60
EPOCHS = 1000
BATCH_SIZE = 64
TICKER_COLUMN = "Ticker"


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df


def preprocess_data(df):
    # Define the columns to keep
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    
    # Drop irrelevant columns and select only the numeric features
    df = df.drop(columns=["Date", "Ticker"])
    df = df[numeric_cols]  # Explicitly select desired columns

    # Handle missing values
    df = df.ffill().bfill()

    # Data normalization
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Data type validation and conversion
    if df.select_dtypes(exclude="number").any().any():
        raise ValueError("Non-numeric data detected after processing")
    df = df.astype("float32")

    return df, scaler


def create_sequences(data, target_col, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, target_col])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential(
        [
            Conv1D(
                256, 5, activation="relu", padding="causal", input_shape=input_shape
            ),
            BatchNormalization(),
            Conv1D(512, 5, activation="relu", padding="causal"),
            MaxPooling1D(2),
            BatchNormalization(),
            Conv1D(1024, 3, activation="relu", padding="causal"),
            Dropout(0.4),
            # Bidirectional(LSTM(2048, return_sequences=True)),
            # Bidirectional(LSTM(2048, return_sequences=True)),
            Bidirectional(LSTM(1024, return_sequences=True)),
            Bidirectional(LSTM(1024, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(
                2048,
                activation="selu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            BatchNormalization(),
            Dropout(0.6),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


def plot_history(history):
    metrics = ["loss", "mae", "mse"]
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)
        plt.plot(history.history[metric], label="Train")
        plt.plot(history.history[f"val_{metric}"], label="Validation")
        plt.title(f"Model {metric.upper()}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()


def main(filepath):
    device = input("Choose device for training (cpu/gpu): ").strip().lower()
    if device == "gpu" and tf.config.list_physical_devices("GPU"):
        print("Using GPU for training.")
        device_name = "/GPU:0"
    else:
        print("Using CPU for training.")
        device_name = "/CPU:0"

    df = load_data(filepath)
    processed_df, scaler = preprocess_data(df)

    joblib.dump(scaler, "stock_scaler.save")

    target_col = processed_df.columns.get_loc("Close")
    data = processed_df.values

    X, y = create_sequences(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        ModelCheckpoint("best_model.keras", save_best_only=True),
        EarlyStopping(patience=15, restore_best_weights=True),
        TensorBoard(log_dir="./logs"),
        CSVLogger("training_log.csv"),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    model.save("stock_model.keras")
    plot_history(history)

    print("\nModel evaluation on test data:")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_metrics[0]:.4f}")
    print(f"Test MAE: {test_metrics[1]:.4f}")
    print(f"Test MSE: {test_metrics[2]:.4f}")


if __name__ == "__main__":
    main("cleaned_stock_data.csv")
