# 📈 Stock Trend Prediction App  

An interactive web app built with **Streamlit**, **TensorFlow/Keras**, and **yahooquery** to visualize stock price trends and predict future movements.  

## 🚀 Features  
- 🔎 Search any stock ticker (e.g., `AAPL`, `TSLA`, `MSFT`, `INFY.NS`)  
- 📊 Fetches real-time stock data using **Yahoo Query API**  
- 📉 Interactive charts with historical stock prices and moving averages  
- 🤖 Stock trend prediction using a pre-trained **LSTM deep learning model**  
- 💾 Caches data to improve performance and reduce API calls  
- ⚠️ Friendly error handling for missing or rate-limited tickers  

## 🛠️ Installation  

1. **Clone this repository**  
```bash
git clone https://github.com/omveer-07/stock-price.git
cd stock-price
````

2. **Create and activate a virtual environment** (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the app with:

```bash
streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**.

## 📂 Project Structure

```
stock-price/
│── app.py                   # Main Streamlit app
│── Latest_stock_price_model.keras   # Pre-trained LSTM model
│── requirements.txt         # Project dependencies
│── README.md                # Project documentation
```

## 👨‍💻 Author

Developed by **Omveer Singh** ✨
📧 Feel free to reach out for collaboration or improvements!