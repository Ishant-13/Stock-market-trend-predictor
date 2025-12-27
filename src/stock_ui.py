import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# -------------------------------
# Helper Functions
# -------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators used for prediction:
    - SMA 14
    - SMA 50
    - RSI 14
    """
    df["SMA_14"] = df["Close"].rolling(window=14).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# -------------------------------
# Streamlit Page Config
# -------------------------------

st.set_page_config(
    page_title="Stock Market Trend Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Stock Market Trend Predictor")

st.warning(
    "⚠️ This application is for educational purposes only and is NOT financial advice."
)

st.markdown(
    "Predict **next-day stock trend (Up / Down)** using Machine Learning "
    "and technical indicators."
)

# -------------------------------
# User Inputs
# -------------------------------

ticker = st.text_input(
    "Enter NSE stock ticker (e.g., TCS.NS, INFY.NS)",
    value="TCS.NS"
)

period = st.selectbox(
    "Select historical period",
    ["6mo", "1y", "2y", "5y"]
)

# -------------------------------
# Main Logic
# -------------------------------

if st.button("Run Analysis"):
    st.info("⏳ Downloading stock data...")

    df = yf.download(ticker, period=period, interval="1d")

    if df.empty:
        st.error("❌ No data found. Please check the ticker symbol.")
        st.stop()

    df.dropna(inplace=True)
    st.success(f"✅ Data loaded: {len(df)} trading days")

    # Feature Engineering
    df = add_technical_indicators(df)

    # Target: 1 if next day's close is higher, else 0
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df.dropna(inplace=True)

    features = [
        "Open", "High", "Low", "Close",
        "Volume", "SMA_14", "SMA_50", "RSI"
    ]

    X = df[features]
    y = df["Target"]

    # Time-series split (NO shuffle to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model Training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.metric("Accuracy", f"{accuracy:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    st.pyplot(fig)

    # -------------------------------
    # Price Chart
    # -------------------------------

    st.subheader("📈 Closing Price & Indicators")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Close"], label="Close Price", color="cyan")
    ax.plot(df["SMA_14"], label="SMA 14", linestyle="--", color="orange")
    ax.plot(df["SMA_50"], label="SMA 50", linestyle="--", color="magenta")

    ax.set_title(f"{ticker} Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # Next-Day Prediction
    # -------------------------------

    latest_data = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_data)[0]
    confidence = model.predict_proba(latest_data)[0][prediction]

    st.subheader("🔮 Next-Day Prediction")

    if prediction == 1:
        st.success(f"📈 Prediction: **UP** (Confidence: {confidence:.2f})")
    else:
        st.error(f"📉 Prediction: **DOWN** (Confidence: {confidence:.2f})")
