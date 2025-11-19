import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def detect_columns(df):
    date_col = None
    sales_col = None
    for col in df.columns:
        cl = col.lower()
        if 'date' in cl or 'time' in cl or 'order' in cl:
            date_col = col
        if any(x in cl for x in ['sale', 'revenue', 'amount', 'price', 'total']):
            sales_col = col
    return date_col, sales_col

# Load CSV
df = pd.read_csv('data/sales_data_sample.csv', encoding='latin1')

# Detect columns
date_col, sales_col = detect_columns(df)
if not date_col or not sales_col:
    raise ValueError(f"Could not detect date/sales columns. Found: {list(df.columns)}")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
df = df.dropna(subset=[date_col, sales_col])
df = df.sort_values(date_col)
df = df[[date_col, sales_col]]
df.columns = ['date', 'sales']
df = df[df['sales'] > 0]

if len(df) < 20:
    raise ValueError("Dataset too small. Add more sales rows (need at least 20).")

# ------ DAILY MODE (NO WEEKLY RESAMPLE) ------

values = df['sales'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Time step
time_step = min(30, max(5, len(df) // 4))

# Sequence builder
def create_sequences(values, ts):
    X, y = [], []
    for i in range(ts, len(values)):
        X.append(values[i-ts:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.1),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse')

batch_size = min(16, len(X_train))

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save
model_path = 'models/trained_lstm_model.keras'
model.save(model_path)
joblib.dump(scaler, 'models/scaler.pkl')

meta = {"time_step": time_step, "model_path": model_path}
with open("models/meta.json", "w") as f:
    json.dump(meta, f)

# Evaluate
pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = math.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)

print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
print("Training complete.")
