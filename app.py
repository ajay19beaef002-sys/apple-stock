import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Helper functions (from notebook)
# -----------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])
        y.append(dataset[i, 3])  # Close index (Open,High,Low,Close,Volume, ...)
    return np.array(X), np.array(y)

def inverse_close_transform(scaled_close, scaler, feat_count, close_index=3):
    """
    scaled_close: 1D or 2D array of scaled close values
    scaler: fitted MinMaxScaler
    feat_count: number of features scaler was fitted on
    """
    sc = np.array(scaled_close)
    sc = sc.reshape(-1)  # flatten
    dummy = np.zeros((len(sc), feat_count))
    dummy[:, close_index] = sc
    inv = scaler.inverse_transform(dummy)[:, close_index]
    return inv

def build_lstm_model(time_step, feature_count):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, feature_count)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="wide")
st.title("📈 Stock Price Prediction — LSTM (Notebook → Streamlit)")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
time_step = st.sidebar.number_input("Time-step (history days)", min_value=10, max_value=365, value=60)
train_test_split_ratio = st.sidebar.slider("Train split (fraction)", 0.5, 0.95, 0.8)
epochs = st.sidebar.number_input("Epochs (train)", min_value=1, max_value=200, value=20)
batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=512, value=32)

col1, col2 = st.columns([1, 2])

model_path = "stock_model.h5"
scaler_path = "scaler.pkl"

with col1:
    st.subheader("Actions")
    train_btn = st.button("Train Model (on selected ticker & range)")
    load_btn = st.button("Load Saved Model (if exists)")
    predict_btn = st.button("Predict (use model & show chart)")

with col2:
    st.subheader("Status/Logs")
    status = st.empty()

# -----------------------------
# Download & prepare data
# -----------------------------
@st.cache_data(ttl=3600)
def download_and_preprocess(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df = df.dropna()
    return df

df = download_and_preprocess(ticker, start_date, end_date)
if df is None or df.shape[0] < time_step + 10:
    st.error("Not enough data for the selected date range and time-step. Try expanding the date range or decreasing time-step.")
    st.stop()

st.write(f"Data loaded: {df.shape[0]} rows")
st.dataframe(df.tail(5))

# -----------------------------
# Train model (if requested)
# -----------------------------
if train_btn:
    status.info("Starting training...")
    with st.spinner("Scaling, creating dataset and training model..."):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(df.values)  # all features
        X, y = create_dataset(scaled, time_step=time_step)
        feature_count = scaled.shape[1]

        # reshape X already shaped (samples, time_step, features)
        # split
        train_size = int(len(X) * train_test_split_ratio)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        status.write(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")

        model = build_lstm_model(time_step, feature_count)
        history = model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size),
                            validation_data=(X_test, y_test), verbose=0)

        # Save model & scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        y_train_inv = inverse_close_transform(y_train, scaler, feature_count)
        train_pred_inv = inverse_close_transform(train_pred[:,0], scaler, feature_count)

        y_test_inv = inverse_close_transform(y_test, scaler, feature_count)
        test_pred_inv = inverse_close_transform(test_pred[:,0], scaler, feature_count)

        status.success("Training finished and model saved.")
        status.write(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)):.4f}")
        status.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_inv, test_pred_inv)):.4f}")
        status.write(f"Test MAE: {mean_absolute_error(y_test_inv, test_pred_inv):.4f}")

        # Show loss plot
        st.subheader("Training Loss")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='train loss'))
        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='val loss'))
        fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig_loss, use_container_width=True)

# -----------------------------
# Load existing model (if requested)
# -----------------------------
if load_btn:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            status.success(f"Loaded model from {model_path} and scaler from {scaler_path}")
        except Exception as e:
            status.error(f"Failed to load model directly: {e}")
            status.info("Attempting to rebuild architecture and load weights (fallback).")
            try:
                # Attempt fallback: build same LSTM architecture and load weights
                feature_count = df.shape[1]
                model = build_lstm_model(time_step, feature_count)
                model.load_weights(model_path)  # works if saved weights
                scaler = joblib.load(scaler_path)
                status.success("Loaded weights into rebuilt model (fallback).")
            except Exception as e2:
                status.error(f"Fallback loading failed: {e2}")
                st.stop()
    else:
        status.warning("No saved model/scaler found. Train a model first.")

# -----------------------------
# Predict & Plot (if requested)
# -----------------------------
if predict_btn:
    # Ensure we have model and scaler
    if 'model' not in locals():
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = load_model(model_path, compile=False)
                scaler = joblib.load(scaler_path)
                status.success("Loaded saved model & scaler for prediction.")
            except Exception as e:
                status.error(f"Could not load saved model: {e}")
                st.stop()
        else:
            st.error("No model available. Train the model or load a saved model first.")
            st.stop()

    with st.spinner("Creating input sequences and predicting..."):
        scaled = scaler.transform(df.values)
        X_all, y_all = create_dataset(scaled, time_step=time_step)
        feature_count = scaled.shape[1]

        preds = model.predict(X_all)
        y_actual_inv = inverse_close_transform(y_all, scaler, feature_count)
        preds_inv = inverse_close_transform(preds[:,0], scaler, feature_count)

        # Choose last n points to plot (align with original data index)
        plot_len = min(len(y_actual_inv), 200)  # limit to recent 200 points for clarity
        plot_idx = df.index[-plot_len:]

        st.subheader(f"Predicted vs Actual — last {plot_len} days")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_idx, y=y_actual_inv[-plot_len:], mode='lines', name='Actual Close'))
        fig.add_trace(go.Scatter(x=plot_idx, y=preds_inv[-plot_len:], mode='lines', name='Predicted Close'))
        fig.update_layout(title=f"{ticker} Close Price — Actual vs Predicted", xaxis_title='Date', yaxis_title='Price ($)')
        st.plotly_chart(fig, use_container_width=True)

        st.write("Evaluation on the plotted segment:")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_actual_inv[-plot_len:], preds_inv[-plot_len:])):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_actual_inv[-plot_len:], preds_inv[-plot_len:]):.4f}")

st.markdown("---")
st.markdown("**Notes:**\n- If you trained a model with a custom Attention layer previously, loading that saved model may fail unless the exact custom layer class is present in this script and used when loading.\n- This app uses the simpler LSTM architecture (same as your original notebook core) to avoid custom-layer deserialization issues.")
