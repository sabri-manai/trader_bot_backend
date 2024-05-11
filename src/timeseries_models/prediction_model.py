from flask import Flask, jsonify
from flask_cors import CORS
from iexfinance.stocks import Stock
from sklearn.ensemble import RandomForestRegressor
from config.config import API_KEY

import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

class StockPredictor:
    def __init__(self):
        # Initialize with RandomForestRegressor instead of LinearRegression
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # More trees can be better
        self.X = None
        self.y = None

    def train(self, historical_data):
        # Prepare the data
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['label'], format='%b %d, %y')
        df = df.sort_values(by='date')
        df['days'] = (df['date'] - df['date'].min()).dt.days
        self.X = df[['days']]  # Keep as DataFrame for consistency with scikit-learn
        self.y = df['close'].values

        # Train the model
        self.model.fit(self.X, self.y)

    def predict(self):
        # Predict the next closing price
        if self.X is not None:
            next_day = self.X.iloc[-1]['days'] + 1  # Simpler access
            future_day = pd.DataFrame([[next_day]], columns=['days'])
            return self.model.predict(future_day)[0]
        else:
            raise ValueError("Model has not been trained yet.")

