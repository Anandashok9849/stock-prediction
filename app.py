import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
import yfinance as yf
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import joblib
import datetime as dt
import os

# Plotly
import plotly.graph_objects as go
import plotly.utils
import json

# ✅ News API
import requests

app = Flask(__name__)

# Load model + scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

if not os.path.exists("static"):
    os.makedirs("static")


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        stock = request.form.get('stock', 'AAPL')

        # =========================
        # 📌 Stock Name
        # =========================
        ticker = yf.Ticker(stock)

        stock_name = stock
        try:
            info = ticker.info
            stock_name = info.get('longName') or info.get('shortName') or stock
        except:
            pass

        # =========================
        # 📰 NewsAPI Integration
        # =========================
        news = []
        try:
            API_KEY = "25b14e81477d4f0ab0fb3272b0e2817b"   # 🔥 PUT YOUR KEY HERE
            url = f"https://newsapi.org/v2/everything?q={stock_name} stock&language=en&sortBy=publishedAt&apiKey={API_KEY}"
            response = requests.get(url)
            data = response.json()

            if data.get("articles"):
                for article in data["articles"][:5]:
                    news.append({
                        "title": article["title"],
                        "link": article["url"]
                    })

        except:
            news = []

        # =========================
        # 📥 Fetch Data
        # =========================
        df = yf.download(stock, start="2018-01-01", end=dt.datetime.now())

        if df.empty:
            return render_template("index.html", error="Invalid stock symbol")

        close_data = df[['Close']]

        # =========================
        # 🔄 Scaling
        # =========================
        scaled_data = scaler.transform(close_data)

        x_test, y_test = [], []

        for i in range(100, len(scaled_data)):
            x_test.append(scaled_data[i-100:i])
            y_test.append(scaled_data[i])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # =========================
        # 🔮 Prediction
        # =========================
        y_pred = model.predict(x_test)

        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # =========================
        # 📉 Prediction Chart
        # =========================
        plt.figure(figsize=(10,5))
        plt.plot(y_test, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='orange')
        plt.title(f"{stock} Prediction vs Actual")
        plt.legend()
        plt.grid()

        plot_path = "static/result.png"
        plt.savefig(plot_path)
        plt.close()

        # =========================
        # 📈 Market Chart
        # =========================
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market Price'))
        fig.update_layout(title=f"{stock} Market Chart", template="plotly_dark")

        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # =========================
        # 🔮 Next Day Prediction
        # =========================
        last_100 = scaled_data[-100:]
        last_100 = np.reshape(last_100, (1, 100, 1))

        next_pred = model.predict(last_100)
        next_price = scaler.inverse_transform(next_pred)[0][0]

        latest_price = df['Close'].iloc[-1].item()

        return render_template(
            "index.html",
            stock=stock,
            stock_name=stock_name,
            news=news,
            mse=round(mse, 2),
            rmse=round(rmse, 2),
            plot_path=plot_path,
            graph_json=graph_json,
            next_price=round(next_price, 2),
            latest_price=round(latest_price, 2)
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
