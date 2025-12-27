# Stock-market-trend-predictor

# 📈 Stock Market Trend Predictor

A machine learning–based web application that predicts the **next-day stock price trend (Up/Down)** using historical market data and technical indicators.

## 🚀 Technologies Used
- Python
- Streamlit
- yFinance
- Pandas, NumPy
- Scikit-learn (Random Forest)
- Matplotlib

## 🧠 How It Works
1. Fetches historical stock data using yFinance
2. Calculates technical indicators:
   - SMA (14, 50)
   - RSI
3. Trains a Random Forest classifier
4. Predicts next-day stock movement
5. Displays results using an interactive Streamlit UI

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run src/stock_ui.py
