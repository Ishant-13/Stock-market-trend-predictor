import argparse
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def add_indicators(df):
    df["SMA_14"] = df["Close"].rolling(window=14).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).to_numpy().ravel()
    loss = (-delta.where(delta < 0, 0)).to_numpy().ravel()
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def add_target(df):
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--period", type=str, default="6mo")
    args = parser.parse_args()

    ticker = args.ticker
    period = args.period

    print(f"⏬ Downloading {ticker} ({period}) ...")
    df = yf.download(ticker, period=period, interval="1d")
    df.dropna(inplace=True)

    df = add_indicators(df)
    df = add_target(df)
    df.dropna(inplace=True)

    os.makedirs("../datasets", exist_ok=True)
    dataset_path = f"../datasets/{ticker}_dataset_used.csv"
    df.to_csv(dataset_path)
    print(f"✅ Dataset saved at {dataset_path}")

    features = ["Open", "High", "Low", "Close", "Volume", "SMA_14", "SMA_50", "RSI"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n📊 Model Performance")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    os.makedirs("../outputs", exist_ok=True)
    cm_path = f"../outputs/{ticker}_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"✅ Confusion matrix saved at {cm_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Close"], label="Closing Price")
    plt.legend()
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    price_chart_path = f"../outputs/{ticker}_closing_price.png"
    plt.savefig(price_chart_path)
    print(f"✅ Price chart saved at {price_chart_path}")

    latest = X.iloc[-1].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0][pred]

    print("\n📈 Next-Day Prediction")
    if pred == 1:
        print(f"Prediction: UP 📈 (Confidence: {proba:.2f})")
    else:
        print(f"Prediction: DOWN 📉 (Confidence: {proba:.2f})")

if __name__ == "__main__":
    main()
