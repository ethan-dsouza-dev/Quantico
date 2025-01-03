import yfinance as yf
from decimal import Decimal, InvalidOperation
import numpy as np
import pandas as pd
from stock_indicators import Quote
from stock_indicators import indicators
from datetime import datetime

# List of stocks to use (actual stocks and test stocks)
Stocks = ['AAPL', 'NVDA', 'GOOG', 'SHOP', 'TSLA', 'META', 'AMZN', 'NFLX', 'PANW', 'SNOW']
tStocks = ['AAPL', 'NVDA']

# Function to calculate RSI
def calculate_rsi(series, length):
    delta = series.diff()  # Calculate the difference between consecutive values
    gain = np.where(delta > 0, delta, 0).flatten()  # Ensure 1D array
    loss = np.where(delta < 0, -delta, 0).flatten()  # Ensure 1D array

    # Convert to pandas Series and align with the original index
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    # Compute rolling averages
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()

    # Compute RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate EMA
def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# Function to calculate OBV
def calculate_obv(close, volume):
    direction = np.sign(close.diff())
    obv = (volume * direction).fillna(0).cumsum()
    return obv

# Function to calculate MFI
def calculate_mfi(high, low, close, volume, length):
    # Calculate Typical Price
    typical_price = (high + low + close) / 3

    # Calculate Money Flow
    money_flow = typical_price * volume

    # Separate Positive and Negative Money Flows
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0).flatten()
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0).flatten()

    # Convert to pandas Series and align with index
    positive_flow = pd.Series(positive_flow, index=close.index)
    negative_flow = pd.Series(negative_flow, index=close.index)

    # Calculate rolling sums of positive and negative money flows
    positive_mf = positive_flow.rolling(window=length, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=length, min_periods=1).sum()

    # Calculate Money Flow Index
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

# Function to calculate Accumulation/Distribution Line
def calculate_ad(high, low, close, volume):
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    ad_line = money_flow_volume.cumsum()
    return ad_line

# Return dataframe of given ticker symbol with relevant indicators
def get_indicator_data(ticker):
    df = yf.download(ticker, '2005-01-01', '2024-10-31', interval='1d')

    # Adding technical indicators to the data frame
    df['RSI'] = calculate_rsi(df['Close'], length=18)
    df['EMA18'] = calculate_ema(df['Close'], length=18)
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    df['MFI'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['Accum/Dist'] = calculate_ad(df['High'], df['Low'], df['Close'], df['Volume'])

    df = df.dropna()  # Drop rows with missing values
    return df

# Given list of stocks, convert it to dictionary of quotes where keys represent the stock names
def dictQuotes(stocks):
    dictOfQuotes = {}
    
    # DIDNT USE CHATGPT, I ADDED COMMENTS TO INCREASE QUALITY OF CODE FOR SELF SATISFACTION
    for stock in stocks:
        # Downloading the dataframe
        df = yf.download(stock, '2005-01-01', '2024-10-31', interval='1d')
        df = df.dropna()
        
        # Extrating the dates
        dates = df.index.get_level_values(0).tolist()
        df_dates = [pd.Timestamp(ts).to_pydatetime() for ts in dates]
        
        # Extrating the opens, highs, lows, closes, volumes
        opens = df['Open']
        df_opens = opens[stock].tolist()
        highs = df['High']
        df_highs = highs[stock].tolist()
        lows = df['Low']
        df_lows = lows[stock].tolist()
        closes = df['Close']
        df_closes = closes[stock].tolist()
        volumes = df['Volume']
        df_volumes = volumes[stock].tolist()
        
        # Converting it into list of Quotes
        quotes_list = [
            Quote(d,o,h,l,c,v)
            for d,o,h,l,c,v 
            in zip(df_dates, df_opens, df_highs, df_lows, df_closes, df_volumes)
        ]
        
        # Putting the quote into the dictionary with corresponding name of stock as the key
        dictOfQuotes[stock] = quotes_list
    
    return dictOfQuotes

# Creating a test dictionary for testing purposes
testDict = dictQuotes(tStocks)

# Printing out the OBV indicator to see if it matches our function
obv_test = indicators.get_mfi(testDict['AAPL'])
for obv in obv_test:
    print(obv.mfi)
print(get_indicator_data('AAPL'))
