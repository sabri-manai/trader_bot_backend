import pandas as pd
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import numpy as np

class SentimentAnalysisModel:
    def __init__(self):
        self.model = LogisticRegression()

    def preprocess(self, historical_data, news_data):
        historical_df = pd.DataFrame(historical_data)
        news_df = pd.DataFrame(news_data)

        news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        sentiment_by_date = news_df.groupby('date')['sentiment'].mean().reset_index()

        historical_df['date'] = pd.to_datetime(historical_df['label'], format='%b %d, %y')
        df = pd.merge(historical_df, sentiment_by_date, on='date', how='left')

        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        features = df[['sentiment']].fillna(0).values
        target = df['target'].fillna(0).values

        return features, target

    def train(self, features, target):
        self.model.fit(features, target)

    def predict(self, sentiment_today):
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model is not trained yet. Call train() before predict().")
        return self.model.predict(np.array([[sentiment_today]]))[0]

