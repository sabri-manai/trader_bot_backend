from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
from iexfinance.stocks import Stock
from sklearn.linear_model import LinearRegression
from config.config import API_KEY

import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.X = None

    def train(self, historical_data):
        # Prepare the data
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['label'], format='%b %d, %y')
        df = df.sort_values(by='date')
        df['days'] = (df['date'] - df['date'].min()).dt.days
        self.X = df['days'].values.reshape(-1, 1)
        y = df['close'].values

        # Train the model
        self.model.fit(self.X, y)

    def predict(self):
        # Predict the next closing price
        if self.X is not None:
            next_day = max(self.X)[0] + 1  # Extract the last day index and increment
            future_day = np.array([[next_day]])
            return self.model.predict(future_day)[0]
        else:
            raise ValueError("Model has not been trained yet.")

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict_close(ticker):
    token = API_KEY
    stock = Stock(ticker, token=token)
    historical_data = stock.get_historical_prices()

    predictor = StockPredictor()
    predictor.train(historical_data)
    predicted_close = predictor.predict()

    return jsonify({"predicted_close": predicted_close})


if __name__ == '__main__':
    app.run(debug=True)
