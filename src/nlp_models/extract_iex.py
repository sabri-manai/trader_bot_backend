import pandas as pd
from iexfinance.stocks import Stock
from datetime import datetime
from iexfinance.utils.exceptions import IEXQueryError
from config.config import (
    API_KEY,
)


# def get_stock_news(stock_ticker, start_date, end_date, token):
#     try:
#         stock = Stock(stock_ticker, token=token)
#         news = stock.get_news(range="1y")

#         # Convert the index to datetime objects
#         news.index = pd.to_datetime(news.index, unit='ms')

#         # Filter news based on the date range
#         filtered_news = news[(news.index >= start_date) & (news.index <= end_date)]

#         return filtered_news

#     except IEXQueryError as e:
#         print(f"Error fetching news for {stock_ticker}: {e}")
#         return pd.DataFrame()

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