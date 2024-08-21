import yfinance as yf
import datetime
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob


def get_historical_prices(tickerSymbol,start_date=None,end_date=None):
    # if start date or endate are not provide default the values 
    stock_symbol = tickerSymbol
    if end_date is None: 
        end_date = datetime.datetime.today()

    if start_date is None:
        start_date = end_date - datetime.timedelta(days=5*365)  # 5 years

    # Fetch data from yf api 
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data


def get_news_data(api_key, query, start_date, end_date):
    """
    Fetch news articles for a given query.
    
    Parameters:
    api_key (str): API key for the news service.
    query (str): Search query (e.g., stock ticker).
    start_date (str): Start date (format: 'YYYY-MM-DD').
    end_date (str): End date (format: 'YYYY-MM-DD').

    Returns:
    pandas.DataFrame: DataFrame containing news headlines and publication dates.
    """
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, from_param=start_date, to=end_date, language='en')
    news_df = pd.DataFrame(articles['articles'])
    news_df = news_df[['publishedAt', 'title']]
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    return news_df


def analyze_sentiment(news_df):
    """
    Perform sentiment analysis on news headlines.

    Parameters:
    news_df (pandas.DataFrame): DataFrame containing news headlines.

    Returns:
    pandas.DataFrame: DataFrame with sentiment scores.
    """
    news_df['sentiment'] = news_df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    news_df=news_df.drop(columns=['title'])
    return news_df


def merge_data(price_df, news_df):
    """
    Merge stock price and news sentiment data on date.

    Parameters:
    price_df (pandas.DataFrame): DataFrame with stock prices.
    news_df (pandas.DataFrame): DataFrame with news sentiment.

    Returns:
    pandas.DataFrame: Merged DataFrame with stock prices and sentiment.
    """
    # Convert dates to datetime
    price_df.index = pd.to_datetime(price_df.index).date
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    
    # Resample news sentiment to daily and aggregate
    daily_sentiment = news_df.groupby('publishedAt').mean().reset_index()
    # print(daily_sentiment.head)

    # print(price_df.head)
    # Merge on date
    merged_df = price_df.reset_index().merge(daily_sentiment, left_on='Date', right_on='publishedAt', how='inner')
    merged_df.set_index('Date', inplace=True)
    return merged_df

def calculate_correlation(merged_df):
    """
    Calculate the correlation between stock prices and news sentiment.

    Parameters:
    merged_df (pandas.DataFrame): DataFrame with stock prices and sentiment.

    Returns:
    float: Correlation coefficient between stock prices and sentiment.
    """
    correlation = merged_df[['Adj Close', 'sentiment']].corr().iloc[0, 1]
    return correlation

def get_price_sentiment_correlation(ticker,start_date, end_date):
    stock_prices = get_historical_prices(ticker, start_date, end_date)
    stock_prices['Date']=stock_prices.index
    stock_prices = stock_prices.reset_index(drop=True)

    # Get news data
    api_key = '75280214a5304e8782d916c7e69d9ab5' 
    news_data = get_news_data(api_key, ticker, start_date, end_date)

    # # Analyze sentiment
    sentiment_data = analyze_sentiment(news_data)

    # Merge data
    merged_data = merge_data(stock_prices, sentiment_data)
    merged_data_cl = merged_data.dropna()

    return merged_data[merged_data['sentiment'] != 0]
