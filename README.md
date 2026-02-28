# Stock Price Prediction Model Documentation 📈

**Version 2.0** | **Last Updated: March 01, 2026**

---

## 📖 Overview

This project implements an advanced deep learning model for predicting stock prices using historical market data. The model leverages a hybrid architecture that combines **causal convolutional layers**, **multi‑head self‑attention**, **residual connections**, and **layer normalization** to effectively capture both short‑term patterns and long‑term dependencies in financial time series. The model is trained on multiple stock tickers and predicts the future closing price based on a window of past observations.

The codebase is written in Python, using **TensorFlow/Keras** for model construction, **scikit‑learn** for preprocessing, and **pandas** for data manipulation. The pipeline includes robust data scaling, sequence generation, training with advanced callbacks, and thorough evaluation (MAE, MSE, R², and optionally MAPE).

---

## 🚀 Features

- **Data Preprocessing**: Loads and cleans stock data for multiple tickers; applies per‑ticker MinMax scaling; handles missing values via forward/backward fill.
- **Sequence Creation**: Builds time‑series sequences with a configurable window size (default 60 days).
- **State‑of‑the‑Art Architecture**:
  - Causal Conv1D layers with residual connections and layer normalization.
  - Multi‑Head Self‑Attention for capturing global dependencies.
  - Feed‑forward networks with dropout and L2 regularization.
  - Global average pooling followed by dense heads.
- **Training Pipeline**: Automatic device selection (CPU/GPU); callbacks for early stopping, model checkpointing, learning rate reduction, and TensorBoard logging.
- **Evaluation**: Computes MAE, MSE, and R² on the test set; plots training curves.
- **Model Persistence**: Saves the final model and per‑ticker scalers for later inference.
- **Visualization**: Generates training/validation loss and metric plots.

---

## 🛠️ Requirements

Install the required packages:

```bash
pip install pandas numpy tensorflow scikit-learn joblib matplotlib
```

Or use a `requirements.txt` in InvestingAssistant repo:

```text
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.12.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.7.0
```

---

## 📂 Project Structure

```
price/
├── combined_stock_data.csv      # Input dataset (user‑provided)
├── stock_model.keras             # Final trained model
├── best_model.keras              # Best checkpoint (by val_loss)
├── stock_scaler.save              # Saved MinMaxScaler per ticker
├── training_log.csv               # Epoch‑wise training metrics
├── training_metrics.png           # Plot of loss & metrics
├── logs/                          # TensorBoard logs
└── train.py                        # Main training script
```

---

## 📊 Data Format

The input CSV must contain the following columns:

| Column        | Description                          | Type       |
|---------------|--------------------------------------|------------|
| Date          | Date of the observation              | datetime   |
| Ticker        | Stock ticker symbol                  | string     |
| Open          | Opening price                        | float      |
| High          | Highest price of the day             | float      |
| Low           | Lowest price of the day              | float      |
| Close         | Closing price (prediction target)    | float      |
| Volume        | Trading volume                       | float      |
| Dividends     | Dividends paid                       | float      |
| Stock Splits  | Stock split ratio                    | float      |

**Example:**

```csv
Date,Ticker,Open,High,Low,Close,Volume,Dividends,Stock Splits
2023-01-01,AAPL,130.28,132.67,129.61,131.86,123456789,0.0,0.0
2023-01-01,MSFT,240.22,243.15,238.75,241.01,987654321,0.0,0.0
...
```

---

## ⚙️ Configuration

The main script defines several constants at the top of `main.py`:

| Parameter       | Description                              | Default Value |
|-----------------|------------------------------------------|---------------|
| `WINDOW_SIZE`   | Number of past days used for prediction  | 60            |
| `EPOCHS`        | Maximum number of training epochs        | 1000          |
| `BATCH_SIZE`    | Batch size for training                  | 128           |

These can be adjusted directly in the source file.

---

## 🏃‍♂️ Running the Project

1. **Place your dataset** as `combined_stock_data.csv` in the project directory.
2. **Run the script**:

   ```bash
   python train.py
   ```

   You will be prompted to choose the device:
   ```
   Choose device for training (cpu/gpu):
   ```

3. **Outputs**:
   - `stock_model.keras` – the final trained model.
   - `best_model.keras` – the best model based on validation loss.
   - `stock_scaler.save` – a dictionary of `MinMaxScaler` objects for each ticker.
   - `training_log.csv` – CSV with per‑epoch metrics.
   - `training_metrics.png` – plot of loss, MAE, and MSE.
   - `logs/` – TensorBoard logs.

4. **Monitor with TensorBoard**:

   ```bash
   tensorboard --logdir logs/
   ```

   Then open `http://localhost:6006` in your browser.

---

## 🧠 Model Architecture (Improved Version 2.0)

The model is a custom deep architecture designed for time‑series forecasting. Below is a layer‑by‑layer description:

### 1. Input and Noise Regularisation
- **Input shape**: `(WINDOW_SIZE, num_features)` (e.g., `(60, 7)`).
- **GaussianNoise(0.01)** – adds small noise to inputs for better generalisation.

### 2. Convolutional Blocks with Residual Connections
Three convolutional blocks, each consisting of:
- **Causal Conv1D** (filters: 64, 128, 256; kernel sizes: 7, 5, 5; padding='causal').
- **LayerNormalization** – normalises across the feature dimension (preferred for sequences).
- **Dropout(0.2)** for regularisation.
- **Residual addition**: if the number of filters changes, a 1x1 convolution projects the skip connection.
- **Activation**: ReLU.

### 3. MaxPooling
- **MaxPooling1D(pool_size=2)** – reduces temporal dimension after convolutions.

### 4. Multi‑Head Self‑Attention Block
- **MultiHeadAttention(num_heads=4, key_dim=128)** – attends to the sequence to capture global dependencies.
- **Residual connection** around the attention layer.
- **LayerNormalisation** after addition.
- **Feed‑forward network**: Dense(ff_dim*2) → Dense(original_dim) with ReLU and Dropout(0.3).
- Another residual connection + layer norm.

### 5. Global Pooling
- **GlobalAveragePooling1D** – aggregates the sequence into a fixed‑length vector.

### 6. Dense Head
- **Dense(256, activation='relu', L2=0.001)** → BatchNormalisation → Dropout(0.4)
- **Dense(128, activation='relu', L2=0.001)** → BatchNormalisation → Dropout(0.3)
- **Output Dense(1)** – predicts the scaled closing price.

### 7. Compilation
- **Optimizer**: AdamW (learning rate = 1e-3, weight decay = 1e-4)
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MAE, MSE, and MAPE (note: MAPE can be extremely high on scaled data; interpret with caution).

---

## 📈 Training and Evaluation

### Callbacks
- `ModelCheckpoint` – saves the best model (`best_model.keras`) based on `val_loss`.
- `EarlyStopping` – stops after 15 epochs without improvement, restores best weights.
- `ReduceLROnPlateau` – reduces learning rate by factor 0.5 if `val_loss` plateaus for 5 epochs.
- `TensorBoard` – logs to `./logs/`.
- `CSVLogger` – writes epoch metrics to `training_log.csv`.

### Evaluation Metrics on Test Set
After training, the script reports:
- **Test Loss** (MSE)
- **Test MAE**
- **Test MSE** (redundant, kept for clarity)
- **R² Score** (coefficient of determination)

**Typical results** (example from a recent run):
```
Test Loss: 0.0023
Test MAE: 0.0394
Test MSE: 0.0020
Test Accuracy (R² score): 96.74%
```

*Note:* The R² score of 96.74% indicates excellent fit on the test data.

---

## 🔮 Making Predictions with the Trained Model

After training, you can load the model and scalers to predict future prices for a specific ticker:

```python
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scalers
model = load_model("stock_model.keras")
scalers = joblib.load("stock_scaler.save")

# Prepare data for a single ticker (e.g., "AAPL")
ticker = "AAPL"
df = pd.read_csv("combined_stock_data.csv")
company_df = df[df["Ticker"] == ticker].copy()
company_df = company_df.drop(columns=["Date", "Ticker"])

numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
company_df = company_df[numeric_cols].ffill().bfill()

# Scale the data using the ticker's scaler
scaler = scalers[ticker]
scaled_data = scaler.transform(company_df[numeric_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

# Create the last window of length WINDOW_SIZE
if len(scaled_df) < WINDOW_SIZE:
    raise ValueError("Not enough data to form a sequence")
sequence = scaled_df.iloc[-WINDOW_SIZE:].values  # shape: (WINDOW_SIZE, num_features)

# Add batch dimension
sequence = np.expand_dims(sequence, axis=0)  # shape: (1, WINDOW_SIZE, num_features)

# Predict
pred_scaled = model.predict(sequence)[0, 0]

# Inverse transform to get actual price
# Create a dummy row to invert only the 'Close' column
dummy = np.zeros((1, len(numeric_cols)))
dummy[0, numeric_cols.index("Close")] = pred_scaled
pred_actual = scaler.inverse_transform(dummy)[0, numeric_cols.index("Close")]

print(f"Predicted closing price for {ticker}: ${pred_actual:.2f}")
```

---

## 📝 Notes and Caveats

- **MAPE on scaled data**: The Mean Absolute Percentage Error can become astronomically large when the true value is close to zero (because of the division by a small number). **Ignore MAPE** during training if you normalise your targets to [0,1]. For business interpretation, compute MAPE after inverse‑transforming predictions.
- **Data quality**: The model assumes clean, complete data. Forward/backward fill is used, but extreme outliers may still affect performance.
- **Window size**: The default 60 days is a reasonable starting point; you may experiment with 30, 90, or 120 days.
- **Feature engineering**: Consider adding technical indicators (moving averages, RSI, MACD) to improve predictive power.
- **Overfitting**: Despite heavy regularisation, financial time series are notoriously noisy. Always validate on out‑of‑time data.
- **GPU memory**: If you encounter out‑of‑memory errors, reduce `BATCH_SIZE` or the number of filters in the convolutional layers.

---

## 📚 References

- TensorFlow/Keras: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- scikit‑learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- Pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- Matplotlib: [https://matplotlib.org/](https://matplotlib.org/)

For questions or contributions, please open an issue in the project repository.

---
