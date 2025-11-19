import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import threading
import webbrowser

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')

meta_path = 'models/meta.json'
scaler_path = 'models/scaler.pkl'

# ------------------ LOAD MODEL ------------------
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    model_path = meta.get("model_path", "models/trained_lstm_model.keras")
    time_step_saved = meta.get("time_step", 5)

    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')

    try:
        scaler = joblib.load(scaler_path)
    except:
        scaler = MinMaxScaler()

else:
    model = None
    scaler = MinMaxScaler()


# ------------------ DETECT COLUMNS ------------------
def detect_columns(df):
    date_col = None
    sales_col = None
    for col in df.columns:
        cl = col.lower()
        if "date" in cl or "order" in cl or "time" in cl:
            date_col = col
        if any(x in cl for x in ["sale", "revenue", "amount", "price", "total"]):
            sales_col = col
    return date_col, sales_col


# ------------------ PREPARE DAILY SERIES ------------------
def prepare_series(df):
    date_col, sales_col = detect_columns(df)

    if not date_col or not sales_col:
        return None, f"Could not detect date/sales columns. Found: {list(df.columns)}"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])

    df = df.sort_values(date_col)
    df = df[[date_col, sales_col]]
    df.columns = ["date", "sales"]

    df = df[df["sales"] > 0]

    if len(df) < 10:
        return None, "Not enough data rows. Need at least 10 rows."

    return df, None


@app.route("/")
def index():
    return render_template("index.html")


# ------------------ PREDICTION ------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="Please upload a CSV file.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    try:
        df_raw = pd.read_csv(file, encoding="latin1")
    except Exception as e:
        return render_template("index.html", error=f"Error reading CSV: {e}")

    df, err = prepare_series(df_raw)
    if err:
        return render_template("index.html", error=err)

    n = len(df)

    # Load time-step from training
    ts = min(time_step_saved, max(2, n - 1))

    # Scale values
    scaled = scaler.transform(df["sales"].values.reshape(-1, 1))

    # Build sequences
    seqs = []
    for i in range(ts, len(scaled)):
        seqs.append(scaled[i - ts:i, 0])

    if len(seqs) == 0:
        return render_template("index.html", error="Not enough data for prediction.")

    X = np.array(seqs).reshape(len(seqs), ts, 1)

    preds = model.predict(X)

    try:
        inv = scaler.inverse_transform(preds)
    except:
        inv = preds

    predicted_next = float(inv[-1][0])

    # Limit graph to last 100 days
    graph_limit = 100
    plot_dates = df["date"].astype(str).tolist()[-graph_limit:]
    actuals = df["sales"].tolist()[-graph_limit:]
    predicted_vals = inv.flatten().tolist()[-graph_limit:]

    # PLOT CLEAN CHART
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))

    plt.plot(plot_dates, actuals, label="Actual", linewidth=1.5)
    plt.plot(plot_dates, predicted_vals, label="Predicted", linewidth=1.5, linestyle="--")

    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join("static", "chart.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "results.html",
        prediction=round(predicted_next, 2),
        chart_path=chart_path,
        dates=plot_dates,
        actuals=actuals,
        predicted=predicted_vals
    )


@app.route("/download_csv")
def download_csv():
    df = pd.DataFrame({
        "date": request.args.getlist("date") or [],
        "actual": request.args.getlist("actual") or [],
        "predicted": request.args.getlist("predicted") or []
    })

    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="forecast.csv")


# ------------------ AUTO OPEN BROWSER ------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
