import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask


app = Flask(__name__, static_folder='static')

meta_path = 'models/meta.json'
model_path_default = 'models/trained_lstm_model.keras'
scaler_path = 'models/scaler.pkl'

if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    model = load_model(meta.get('model_path', model_path_default), compile=False)
    model.compile(optimizer='adam', loss='mse')
    try:
        scaler = joblib.load(scaler_path)
    except:
        scaler = MinMaxScaler(feature_range=(0,1))
else:
    model = None
    scaler = MinMaxScaler(feature_range=(0,1))

os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

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

def prepare_series(df):
    date_col, sales_col = detect_columns(df)
    if not date_col or not sales_col:
        return None, f"Could not detect date/sales columns. Found: {list(df.columns)}"
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])
    df = df.sort_values(date_col)
    df = df[[date_col, sales_col]]
    df.columns = ['date', 'sales']
    df = df[df['sales'] > 0]
    if len(df) < 3:
        return None, "Not enough valid rows (need >= 3)."
    return df, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Please upload a CSV file.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")
    try:
        df_raw = pd.read_csv(file, encoding='latin1')
    except Exception as e:
        return render_template('index.html', error=f"Error reading CSV: {e}")

    df, err = prepare_series(df_raw)
    if err:
        return render_template('index.html', error=err)

    n = len(df)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        ts = meta.get('time_step', min(30, max(5, n//5)))
    else:
        ts = min(30, max(5, n//5))

    ts = min(ts, max(2, n-1))

    scaled = scaler.fit_transform(df['sales'].values.reshape(-1,1)) if not isinstance(scaler, MinMaxScaler) else scaler.transform(df['sales'].values.reshape(-1,1))
    if isinstance(scaled, np.ndarray) and scaled.ndim == 1:
        scaled = scaled.reshape(-1,1)

    if n < ts + 1:
        ts = max(2, n//2)
    seqs = []
    for i in range(ts, len(scaled)):
        seqs.append(scaled[i-ts:i,0])
    if len(seqs) == 0:
        last_seq = scaled[-ts:].reshape(1, ts, 1)
        seqs = [last_seq[0].flatten()]
    X = np.array(seqs).reshape(len(seqs), ts, 1)

    if model is None:
        return render_template('index.html', error="No trained model found. Run training first (train.py).")

    preds = model.predict(X)
    try:
        inv = scaler.inverse_transform(preds)
    except:
        inv = preds

    predicted_next = float(inv[-1][0]) if inv.size else float(preds[-1][0])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,4))
    recent_dates = df['date'].astype(str).tolist()[-len(inv):]
    actuals = df['sales'].tolist()[-len(inv):]
    plt.plot(recent_dates, actuals, label='Actual', marker='o')
    plt.plot(recent_dates, inv.flatten().tolist(), label='Predicted', marker='o', linestyle='--')
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()

    return render_template('results.html',
                           prediction=round(predicted_next, 2),
                           chart_path=chart_path,
                           dates=recent_dates,
                           actuals=actuals,
                           predicted=inv.flatten().tolist())

@app.route('/download_csv')
def download_csv():
    if not os.path.exists(os.path.join('static','chart.png')):
        return "No results available", 404
    df = pd.DataFrame({'date': request.args.getlist('date') or [], 'actual': request.args.getlist('actual') or [], 'predicted': request.args.getlist('predicted') or []})
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='forecast.csv')

if __name__ == '__main__':
    app.run(debug=True)
