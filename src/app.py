from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
from iexfinance.stocks import Stock
from config.config import API_KEY

import pandas as pd
from textblob import TextBlob

from nlp_models.sentiment_analysis_model import SentimentAnalysisModel

from timeseries_models.prediction_model import StockPredictor
# from timeseries_models.lstm_stock_predictor import LSTMStockPredictor

app = Flask(__name__)
CORS(app)

@app.route('/api/news/<ticker>', methods=['GET'])
def get_news(ticker):
    token = API_KEY
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 5, 4)
    stock = Stock(ticker, token=token)
    news = stock.get_news(range="1y")
    news.index = pd.to_datetime(news.index, unit='ms')
    news_filtered = news[(news.index >= start_date) & (news.index <= end_date)]
    result = news_filtered.to_dict(orient="records")
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

@app.route('/api/predict_sentiment/<ticker>', methods=['GET'])
def predict_sentiment_close(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    historical_data = stock.get_historical_prices()
    news_data = stock.get_news(range="1y").to_dict(orient="records")

    predictor = SentimentAnalysisModel()
    features, target = predictor.preprocess(historical_data, news_data)
    predictor.train(features, target)

    # Ensure there are headlines to analyze
    if len(news_data) > 0:
        latest_headline = news_data[0]['headline']
        latest_sentiment = TextBlob(latest_headline).sentiment.polarity
        prediction = predictor.predict(latest_sentiment)
        return jsonify({"sentiment_prediction": prediction})
    else:
        return jsonify({"error": "No news headlines available for analysis"})



if __name__ == '__main__':
    app.run(debug=True)
