import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify
from iexfinance.stocks import Stock
from datetime import datetime, timezone
from config.config import API_KEY

class NewsSentimentPredictor:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def train(self, X, y):
        X_train_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_train_vect, y)
        print("Model trained")

    def preprocess(self, news_data):
        sentiments = [self.sentiment_analyzer.polarity_scores(news)['compound'] for news in news_data['headline']]
        return sentiments

    def predict(self, new_news):
        new_news_vect = self.vectorizer.transform([new_news])
        prediction = self.model.predict(new_news_vect)
        return prediction[0]
