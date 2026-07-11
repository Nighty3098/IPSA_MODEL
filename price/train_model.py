import os
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import *

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

WINDOW_SIZE = 60
FORECAST_HORIZON = 60
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def add_technical_indicators(df):
    df = df.copy()

    for period in [5, 10, 20, 50]:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
        df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

    df["Close_Dist_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["SMA_Golden_Cross"] = (df["SMA_50"] > df["SMA_20"]).astype(float)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    df["Williams_R"] = -100 * (high_14 - df["Close"]) / (high_14 - low_14 + 1e-10)
    df["ROC_10"] = df["Close"].pct_change(periods=10) * 100

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["Close"].shift())
    tr3 = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()
    df["ATR_Ratio"] = df["ATR_14"] / df["Close"]

    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-10)

    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_EMA"] = df["OBV"].ewm(span=20, adjust=False).mean()

    df["Returns_1d"] = df["Close"].pct_change()
    df["Returns_5d"] = df["Close"].pct_change(5)
    df["Returns_20d"] = df["Close"].pct_change(20)
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_20d"] = df["Log_Returns"].rolling(window=20).std() * np.sqrt(252)
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Body_Ratio"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-10)

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        df["DayOfWeek"] = dates.dt.dayofweek / 6.0
        df["DayOfMonth"] = dates.dt.day / 31.0
        df["Month"] = dates.dt.month / 12.0
        df["Quarter"] = dates.dt.quarter / 4.0
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
        df["Month_Sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * dates.dt.month / 12)

    return df.dropna()


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def preprocess_data(df):
    scalers = {}
    processed_dfs = []

    for ticker in df["Ticker"].unique():
        try:
            company_df = df[df["Ticker"] == ticker].copy()
            company_df = add_technical_indicators(company_df)
            company_df = company_df.drop(columns=["Date", "Ticker"], errors="ignore")

            numeric_cols = company_df.select_dtypes(include=[np.number]).columns.tolist()
            company_df = company_df[numeric_cols]
            company_df = company_df.ffill().bfill()
            company_df = company_df.loc[:, company_df.std() > 0]

            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(company_df)
            company_df = pd.DataFrame(scaled_data, columns=company_df.columns)

            scalers[ticker] = {"scaler": scaler, "columns": company_df.columns.tolist()}
            processed_dfs.append(company_df)
            print(f"  {ticker}: {len(company_df)} rows, {len(company_df.columns)} features")
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    if not processed_dfs:
        raise ValueError("No data was successfully processed")

    return processed_dfs, scalers


def create_sequences(data_list, target_col, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON):
    X, y = [], []

    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]

    for company_data in data_list:
        company_values = company_data.values
        total_needed = window_size + forecast_horizon

        if len(company_values) < total_needed:
            continue

        for i in range(len(company_values) - total_needed):
            X.append(company_values[i : i + window_size, :])
            y.append(company_values[i + window_size : i + total_needed, target_col])

    if not X:
        raise ValueError(f"Could not create sequences. Need at least {window_size + forecast_horizon} rows")

    return np.array(X), np.array(y)


def temporal_split(X, y, val_ratio=0.15, test_ratio=0.15):
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    return (
        X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:],
        y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
    )


def build_tcn_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.2):
    out = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate,
                        kernel_regularizer=regularizers.l2(1e-4))(x)
    out = layers.LayerNormalization()(out)
    out = layers.Activation("relu")(out)
    out = layers.SpatialDropout1D(dropout_rate)(out)

    out = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate,
                        kernel_regularizer=regularizers.l2(1e-4))(out)
    out = layers.LayerNormalization()(out)
    out = layers.Activation("relu")(out)
    out = layers.SpatialDropout1D(dropout_rate)(out)

    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)

    return layers.Add()([x, out])


def build_encoder(inputs, num_tcn_blocks=4, num_heads=8, ff_dim=128):
    x = inputs

    filters = [64, 128, 128, 256]
    for i in range(num_tcn_blocks):
        x = build_tcn_block(x, filters=filters[i], kernel_size=3, dilation_rate=2 ** i)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn = layers.Dropout(0.2)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(ff_dim * 2, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(0.2)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    return x


def build_decoder(encoder_output, forecast_horizon, ff_dim=128):
    context = layers.GlobalAveragePooling1D()(encoder_output)
    x = layers.RepeatVector(forecast_horizon)(context)

    x = layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.TimeDistributed(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)
    outputs = layers.TimeDistributed(layers.Dense(1))(x)
    outputs = layers.Reshape((forecast_horizon,))(outputs)

    return outputs


def build_model(input_shape, forecast_horizon=FORECAST_HORIZON):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.02)(inputs)
    x = layers.Conv1D(64, 1, padding="same")(x)

    encoded = build_encoder(x, num_tcn_blocks=4, num_heads=8, ff_dim=128)
    outputs = build_decoder(encoded, forecast_horizon)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        clipnorm=1.0, beta_1=0.9, beta_2=0.98,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


class WarmUpCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, total_epochs=EPOCHS, target_lr=LEARNING_RATE, min_lr=1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


class DirectionalAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.y_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        y_pred = self.model.predict(self.X_val, verbose=0)
        pred_diff = np.diff(y_pred[:, -1])
        true_diff = np.diff(self.y_val[:, -1])
        da = np.mean(np.sign(pred_diff) == np.sign(true_diff)) * 100
        logs = logs or {}
        logs["val_directional_accuracy"] = da
        print(f"  Directional Accuracy: {da:.2f}%")


def plot_history(history, save_path="training_metrics_enhanced.png"):
    metrics = ["loss", "mae", "rmse"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history.history[metric], label="Train", linewidth=1.5)
        ax.plot(history.history[f"val_{metric}"], label="Validation", linewidth=1.5)
        ax.set_title(f"Model {metric.upper()}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[3]
    lr_key = "lr" if "lr" in history.history else "learning_rate"
    if lr_key in history.history:
        ax.plot(history.history[lr_key], color="green")
        ax.set_title("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_model(model, X_test, y_test, forecast_horizon=FORECAST_HORIZON):
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test, verbose=1)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test.flatten(), y_pred.flatten())

    print(f"\nOverall: MAE={mae:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}, R2={r2:.4f}")

    print(f"\nPer-Day MAE (first 10 days):")
    for day in range(min(forecast_horizon, 10)):
        day_mae = mean_absolute_error(y_test[:, day], y_pred[:, day])
        print(f"  Day {day+1}: {day_mae:.6f}")

    pred_diff = np.diff(y_pred[:, -1])
    true_diff = np.diff(y_test[:, -1])
    da = np.mean(np.sign(pred_diff) == np.sign(true_diff)) * 100
    print(f"\nDirectional Accuracy (Day {forecast_horizon}): {da:.2f}%")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    sample_indices = np.random.choice(len(y_test), min(6, len(y_test)), replace=False)

    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]
        days = range(1, forecast_horizon + 1)
        ax.plot(days, y_test[sample_idx], "b-", label="Actual", linewidth=2)
        ax.plot(days, y_pred[sample_idx], "r--", label="Predicted", linewidth=2)
        ax.fill_between(days, y_test[sample_idx], y_pred[sample_idx], alpha=0.2, color="gray")
        ax.set_title(f"Sample {sample_idx}")
        ax.set_xlabel("Days Ahead")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions_comparison.png", dpi=150)
    plt.close()
    print("Saved: predictions_comparison.png")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "da": da}


def main(filepath):
    print("=" * 60)
    print("ENHANCED STOCK PRICE PREDICTION")
    print(f"Window: {WINDOW_SIZE} days | Horizon: {FORECAST_HORIZON} days")
    print("=" * 60)

    device = input("Choose device (cpu/gpu): ").strip().lower()
    if device == "gpu" and tf.config.list_physical_devices("GPU"):
        print("Using GPU + mixed precision")
        device_name = "/GPU:0"
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print("Using CPU")
        device_name = "/CPU:0"

    with tf.device(device_name):
        print("\n[1/6] Loading data...")
        df = load_data(filepath)
        print(f"      {len(df)} rows, {df['Ticker'].nunique()} tickers")

        print("\n[2/6] Preprocessing with technical indicators...")
        processed_dfs, scalers = preprocess_data(df)
        joblib.dump(scalers, "stock_scalers_enhanced.save")

        print("\n[3/6] Creating sequences...")
        target_col = processed_dfs[0].columns.get_loc("Close")
        print(f"      Features: {len(processed_dfs[0].columns)} -> {list(processed_dfs[0].columns[:8])}...")

        X, y = create_sequences(processed_dfs, target_col)
        print(f"      Sequences: X={X.shape}, y={y.shape}")

        X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)
        print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        print("\n[4/6] Building model...")
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.summary()

        print("\n[5/6] Training...")
        callbacks = [
            ModelCheckpoint("best_model_enhanced.keras", monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-7, verbose=1),
            WarmUpCosineDecay(warmup_epochs=10, total_epochs=EPOCHS),
            TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            CSVLogger("training_log_enhanced.csv"),
            DirectionalAccuracy(val_data=(X_val, y_val)),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=1,
        )

        model.save("stock_model_enhanced.keras")
        plot_history(history)

        print("\n[6/6] Evaluating...")
        metrics = evaluate_model(model, X_test, y_test)

        print("\n" + "=" * 60)
        print(f"Best R2: {metrics['r2']*100:.2f}% | Directional Accuracy: {metrics['da']:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main("combined_stock_data.csv")
