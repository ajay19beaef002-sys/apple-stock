import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# Load pre-trained model & scaler
model = load_model("stock_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Feature engineering functions here...
# compute_RSI(), create_dataset(), inverse_close_transform()

st.title("📈 Stock Price Prediction (Pretrained Model)")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
forecast_horizon = st.sidebar.slider("Forecast Days", 1, 30, 7)

if st.sidebar.button("Predict"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = compute_RSI(data['Close'])
    data = data.dropna()

    scaled_data = scaler.transform(data)
    time_step = 60
    X, y = create_dataset(scaled_data, time_step, forecast_horizon)

    # Prediction
    test_predict = model.predict(X)
    y_test_inv = inverse_close_transform(y[:, -1], scaler, scaled_data.shape[1])
    test_predict_inv = inverse_close_transform(test_predict[:, -1], scaler, scaled_data.shape[1])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-len(y_test_inv):], y=y_test_inv, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=data.index[-len(y_test_inv):], y=test_predict_inv, mode='lines', name='Predicted Price'))
    fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig)
 