from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from iexfinance.stocks import Stock
from config.config import API_KEY
import logging

import pandas as pd

from nlp.news_sentiment_predictor import NewsSentimentPredictor
from timeseries_models.prediction_model import StockPredictor

app = Flask(__name__)
CORS(app)

@app.route('/api/news/<ticker>', methods=['GET'])
def get_news(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    try:
        news_df = stock.get_news(last=30)
        print(news_df.columns)  # Verify column names

        # Assuming the timestamp is the index or under a different column name
        if 'datetime' not in news_df.columns:
            news_df.reset_index(inplace=True)  # Reset index in case the timestamp is in the index
            news_df.rename(columns={'index': 'datetime'}, inplace=True)  # Rename if necessary

        news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='ms', utc=True)

        today = datetime.now(timezone.utc).date()
        start_of_today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        end_of_today = start_of_today + timedelta(days=1)

        # Filter news from today
        news_filtered = news_df[(news_df['datetime'] >= start_of_today) & (news_df['datetime'] < end_of_today)]
        result = news_filtered.to_dict(orient='records')

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

@app.route('/api/chart/<ticker>', methods=['GET'])
def get_chart(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    historical_data = stock.get_historical_prices()

    closes = historical_data['close'].tolist()
    labels = historical_data['label'].tolist()

    chart_data = {
        "labels": labels,
        "datasets": [{
            "label": "Close Price",
            "data": closes
        }]
    }
    return jsonify(chart_data)

@app.route('/api/price/<ticker>', methods=['GET'])
def get_price(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    df = stock.get_quote().loc[:, 'iexRealtimePrice']
    price = df.iloc[0]  # Extract the first value
    return jsonify(price)

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict_close(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    historical_data = stock.get_historical_prices()

    predictor = StockPredictor()
    predictor.train(historical_data)
    predicted_close = predictor.predict()

    return jsonify({"predicted_close": predicted_close})

# Initialize and train model
predictor = NewsSentimentPredictor()
# Normally here you would load your training data and train the model
# For example:
# predictor.train(["some news text"], [1])  # Assuming '1' is a label for 'rise'

def fetch_news_for_prediction(ticker):
    stock = Stock(ticker, token=API_KEY)
    news_df = stock.get_news(last=30)
    if 'datetime' not in news_df.columns:
        news_df.reset_index(inplace=True)
        news_df.rename(columns={'index': 'datetime'}, inplace=True)
    news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='ms', utc=True)
    return news_df

@app.route('/api/predict/news/<ticker>', methods=['GET'])
def predict_news_impact(ticker):
    try:
        news_df = fetch_news_for_prediction(ticker)
        today = datetime.now(timezone.utc).date()
        start_of_today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        news_filtered = news_df[(news_df['datetime'] >= start_of_today)]

        if news_filtered.empty:
            return jsonify({"error": "No relevant news found for today."}), 404

        sentiment = predictor.preprocess(news_filtered)
        prediction = predictor.predict(news_filtered['headline'].iloc[0])  # Predicting based on the first headline
        return jsonify({"sentiment": sentiment, "prediction": prediction})
    except Exception as e:
        print(f"Error in predict_news_impact: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
