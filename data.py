
import numpy as np
from numpy import NaN as npNaN
import pandas as pd
import yfinance as yf

import pandas_ta as ta # Technical Indicators

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# return dataframe of given ticker symbol with relevant indicators
def get_indicator_data(ticker):
    df = yf.download(ticker,'2005-01-01', '2024-10-31', interval='5d') # downloading data on 5 day time-frame

    # Adding our indicators to the data frame
    # Tells you how overbought or oversold the stock is
    df['RSI'] = ta.rsi(df['Close'], length=18) 

    # Estimated Moving Average on 18 Day timeframe
    df['EMA18'] = ta.ema(df['Close'], length=18) 

    # Shows general direction of stock based on conviction behind each days move
    # If prices are rising and the OBV is making higher peaks and higher troughs, the upward trend is likely to continue:
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    # Identifies Overbought/Oversold conditions like RSI but is volume weighted
    # df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)

    # detects whether more money is flowing in or out of a stock
    # df['Accum/Dist'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    df = df.dropna()
    return df

# print(get_indicator_data('AAPL'))