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

df = pd.read_csv('data/sales_data_sample.csv', encoding='latin1')

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

n = len(df)
if n < 6:
    raise ValueError(f"Not enough valid rows ({n}) to train. Need >= 6 rows.")

time_step = min(30, max(5, n // 5))

def create_sequences(values, ts):
    X, y = [], []
    for i in range(ts, len(values)):
        X.append(values[i-ts:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

scaled = MinMaxScaler(feature_range=(0,1)).fit_transform(df['sales'].values.reshape(-1,1))

X, y = create_sequences(scaled, time_step)
while X.size == 0 and time_step > 2:
    time_step = max(2, time_step // 2)
    X, y = create_sequences(scaled, time_step)

if X.size == 0:
    raise ValueError("Could not create sequences for training. Data too short.")

X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.15))
model.add(LSTM(64))
model.add(Dropout(0.15))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=max(4, min(32, len(X_train)//2)), validation_data=(X_test, y_test), verbose=1)

model_path = 'models/trained_lstm_model.keras'
model.save(model_path)
joblib.dump(MinMaxScaler(feature_range=(0,1)).fit(df['sales'].values.reshape(-1,1)), 'models/scaler.pkl')
meta = {'time_step': int(time_step), 'model_path': model_path}
with open('models/meta.json', 'w') as f:
    json.dump(meta, f)

pred = model.predict(X_test)
scaler_for_eval = joblib.load('models/scaler.pkl')
pred_inv = scaler_for_eval.inverse_transform(pred)
y_test_inv = scaler_for_eval.inverse_transform(y_test.reshape(-1,1))

rmse = math.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='Actual')
plt.plot(pred_inv, label='Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('results/predictions.png')
print("Model, scaler and metadata saved to models/ and results/predictions.png")
