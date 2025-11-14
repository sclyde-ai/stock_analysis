# scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# standard

# my library
from .data_analysis.dataclass import ListAndStr
from . import VAR

# others
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf

from .ticker import Tickers

class Stock(Tickers):
    def __init__(self, tickers: ListAndStr=None, interval: ListAndStr='1d', value_type: str='Close', series_type: str=''):
        if tickers:
            super().__init__(tickers=tickers)
        else:
            super().__init__()
        self.interval = interval
        self.data = self.get_all_stock_data()
        # print('data')
        # print(self.data)
        self.prices = self.Value(value_type)
        self.value_type = value_type
        self.series_type = series_type
        self._set_series()
        self.prices['market'] = self.prices.sum(axis=1)
        # self.VAR = VAR.VAR(self.prices)
    
    def get_stock_data(
            self,
            ticker: str,
        ) -> pd.DataFrame:
        try:
            if self.interval in self.date_interval_list:
                data = self.get_data(f"{ticker}_{self.interval}", index_col='Date')
            else:
                data = self.get_data(f"{ticker}_{self.interval}", index_col='Datetime')
            return data
        except Exception as e:
            print(f"Could not load data : {e}")
            return None

    def get_all_stock_data(self) -> dict:
        ticker_dict = {}
        for ticker in self.tickers:
            ticker_dict[ticker] = self.get_stock_data(ticker)
            # print(ticker_dict[ticker])
        return ticker_dict

    def dropna(self, how='any', axis=0):
        self.prices = self.prices.dropna(how=how, axis=axis)
        return self
    def fillna(self, value=0):
        self.prices = self.prices.fillna(value)
        return self
    def cleanna(self):
        self.prices = self.prices.dropna(how='all', axis=1).dropna()
        return self
    def standardize(self):        
        # 各数値列を標準化
        prices_copy = self.prices.copy()
        for column in self.prices.columns:
            mean = self.prices[column].mean()
            std = self.prices[column].std()
            # 標準偏差が0の場合は除算をスキップ（すべての値が同じ場合）
            if std != 0:
                prices_copy[column] = (prices_copy[column] - mean) / std
            else:
                prices_copy[column] = 0  # すべての値を0に設定（平均が0になるため）
        self.prices = prices_copy
        return self
    
    # set to prices
    def log(self):
        index = self.prices.index
        self.prices = self.prices.apply(np.log)
        self.prices.index = index
        self.series_type += '_log'
        return self
    def diff(self):
        self.prices = self.prices.diff().dropna()
        self.series_type += '_diff'
        return self
    def _set_series(self):
        series_type_list = self.series_type.split('_')
        for series_type in series_type_list:
            if series_type == 'log':
                self.log()
            elif series_type == 'diff':
                self.diff()
            else:
                pass
        return self
    
    # return value
    def Value(self, value_type: str):
        df = pd.DataFrame()
        for ticker in self.tickers:
            print(ticker)
            series = self.data[ticker][value_type]
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    @ property
    def Close(self):
        return self.Value('Close')
    @ property
    def Open(self):
        return self.Value('Open')
    @ property
    def High(self):
        return self.Value('High')
    @ property
    def Low(self):
        return self.Value('Low')
    
    # set to prices
    def close(self):
        self.prices = self.Close
        self.value_type = 'Close'
        self._set_series()
        return self
    def open(self):
        self.prices = self.Open
        self.value_type = 'Open'
        self._set_series()
        return self
    def high(self):
        self.prices = self.High
        self.vale_type = 'High'
        self._set_series()
        return self
    def low(self):
        self.prices = self.Low
        self.value_type = 'Low'
        self._set_series()
        return self
            
    def avg(self) -> float:
        return self.prices.mean().item()
    def std(self) -> float:
        return self.prices.std().item()
    def maxlen(self) -> int:
        maxlen = 0
        for ticker in self.tickers:
            if maxlen < len(self.prices[ticker]):
                maxlen = len(self.prices[ticker])
        return maxlen

    
    def plot(self, tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue

            plt.figure(figsize=(12, 6))
            plt.plot(self.prices[ticker], label='Price')
            plt.title(f'{ticker} Stock Price {self.value_type} {self.series_type}')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.close()

    def AuotCorrelation(self, nlags: int=None, tickers: list=None) -> pd.DataFrame:
        if tickers == None:
            tickers = self.tickers
        maxlen = self.maxlen()
        if nlags == None or maxlen < nlags:
            nlags = maxlen-1
        
        autocorr_df = pd.DataFrame()
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue
            autocorr = self.prices[ticker].pct_change().dropna()
            autocorr.name = ticker
            acf_values = acf(autocorr, nlags=nlags, fft=False)
            series = pd.Series(acf_values, name=ticker)
            # autocorr_df = autocorr_df.join(series, how='outer')
            autocorr_df = pd.concat([autocorr_df, series], axis=1)
            # print(autocorr_df)
        return autocorr_df