from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
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
