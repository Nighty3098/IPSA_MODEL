import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

WINDOW_SIZE = 60
EPOCHS = 1000
BATCH_SIZE = 128


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def preprocess_data(df):
    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]

    scalers = {}
    processed_dfs = []

    for ticker in df["Ticker"].unique():
        try:
            company_df = df[df["Ticker"] == ticker].copy()
            company_df = company_df.drop(columns=["Date", "Ticker"])
            company_df = company_df[numeric_cols]

            company_df = company_df.ffill().bfill()

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(company_df[numeric_cols])
            company_df = pd.DataFrame(scaled_data, columns=numeric_cols)

            scalers[ticker] = scaler
            processed_dfs.append(company_df)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    if not processed_dfs:
        raise ValueError("No data was successfully processed")

    if any(df.select_dtypes(exclude="number").any().any() for df in processed_dfs):
        raise ValueError("Non-numeric data detected after processing")

    return processed_dfs, scalers


def create_sequences(data_list, target_col, window_size=WINDOW_SIZE):
    X, y = [], []

    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]

    for company_data in data_list:
        company_values = company_data.values
        if len(company_values) < window_size + 1:
            continue

        for i in range(len(company_values) - window_size - 1):
            X.append(company_values[i : i + window_size, :])
            y.append(company_values[i + window_size, target_col])

    if not X:
        raise ValueError(
            f"Could not create sequences. Input data length must be greater than window_size ({window_size})"
        )

    return np.array(X), np.array(y)


def prepare_single_sequence(data, window_size=WINDOW_SIZE):
    """Prepare a single sequence for prediction"""
    if len(data) < window_size:
        raise ValueError(
            f"Input data length ({len(data)}) must be at least equal to window_size ({window_size})"
        )

    sequence = data[-window_size:].values
    return np.array([sequence])  # Shape: (1, window_size, features)


def build_model(input_shape, num_heads=4, ff_dim=128):
    inputs = layers.Input(shape=input_shape)

    x = layers.GaussianNoise(0.01)(inputs)

    conv1 = layers.Conv1D(
        filters=64, kernel_size=7, padding="causal", activation="relu"
    )(x)
    norm1 = layers.LayerNormalization()(conv1)
    x = layers.Dropout(0.2)(norm1)

    conv2 = layers.Conv1D(
        filters=128, kernel_size=5, padding="causal", activation="relu"
    )(x)
    norm2 = layers.LayerNormalization()(conv2)
    if x.shape[-1] != 128:
        x_proj = layers.Conv1D(128, kernel_size=1, padding="same")(x)
    else:
        x_proj = x
    x = layers.Add()([x_proj, norm2])
    x = layers.Dropout(0.2)(x)

    conv3 = layers.Conv1D(
        filters=256, kernel_size=5, padding="causal", activation="relu"
    )(x)
    norm3 = layers.LayerNormalization()(conv3)
    if x.shape[-1] != 256:
        x_proj = layers.Conv1D(256, kernel_size=1, padding="same")(x)
    else:
        x_proj = x
    x = layers.Add()([x_proj, norm3])
    x = layers.Dropout(0.2)(x)

    x = layers.MaxPooling1D(pool_size=2)(x)

    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = layers.Dropout(0.3)(attn_output)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(ff_dim * 2, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(0.3)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="mse",
        metrics=["mae", "mse", "mape"],
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

    print(df)

    processed_dfs, scalers = preprocess_data(df)

    joblib.dump(scalers, "stock_scaler.save")

    target_col = processed_dfs[0].columns.get_loc("Close")
    X, y = create_sequences(processed_dfs, target_col)
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

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest Accuracy (R² score): {r2 * 100:.2f}%")


if __name__ == "__main__":
    main("combined_stock_data.csv")
