import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tf.keras.models import Sequential
from tf.keras.layers import Dense, LSTM
from tf.keras.optimizers import Adam

class LSTMStockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def preprocess(self, historical_data):
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['label'], format='%b %d, %y')
        df = df.sort_values(by='date')
        df['days'] = (df['date'] - df['date'].min()).dt.days
        closes = df['close'].values
        scaled_closes = self.scaler.fit_transform(closes.reshape(-1, 1))
        X, y = [], []
        for i in range(30, len(scaled_closes)):
            X.append(scaled_closes[i-30:i, 0])
            y.append(scaled_closes[i, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        self.model = model

    def train(self, X, y, epochs=20, batch_size=32):
        if self.model is None:
            self.build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        last_window = X[-1].reshape(1, X.shape[1], 1)
        prediction = self.model.predict(last_window)
        return self.scaler.inverse_transform(prediction)[0, 0]
