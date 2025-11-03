# 🧠 Sales Forecasting using LSTM

### 📌 Overview
This project predicts future sales using **Deep Learning (LSTM - Long Short-Term Memory)** networks.  
It demonstrates how time-series data can be used to forecast future demand for stores or products, combining **data analytics** and **AI modeling**.

---

### 🎯 Objective
- Analyze and preprocess sales time-series data.
- Train an LSTM model to learn temporal sales patterns.
- Forecast future sales and visualize predictions.
- Compare LSTM performance against traditional models.

---

### 📊 Dataset
**Dataset:** [Store Item Demand Forecasting Challenge – Kaggle](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)  
**Columns:**
- `date`: The date of sale.  
- `store`: Store ID.  
- `item`: Item ID.  
- `sales`: Number of items sold.

---

### 🧩 Model Architecture
- Input Layer: Sequences of past sales (timesteps)
- LSTM Layer 1: 128 units, ReLU activation
- Dropout: 0.2  
- LSTM Layer 2: 64 units, ReLU activation  
- Dense Output Layer: 1 unit (predicted sales)
- Optimizer: Adam  
- Loss: Mean Squared Error (MSE)

---

### ⚙️ Requirements
