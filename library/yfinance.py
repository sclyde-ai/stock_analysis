# scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# standard
import os
import json

# my library
from . import ListAndStr
from . import VAR

# others
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from sqlalchemy import create_engine
from dotenv import load_dotenv

# get parameter.json
with open('../parameter.json', 'r', encoding='utf-8') as f:
    json_string = f.read()
    parameter = json.loads(json_string)
    lower_tickers = [ticker.lower() for ticker in parameter['tickers']]

# date or datetime
date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']
datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']

load_dotenv(dotenv_path='../.env')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
DB_NAME = os.getenv('NEWS_DB')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/yfinance"

engine = create_engine(DATABASE_URL) 

class Tickers():
    def __init__(self, tickers: ListAndStr = parameter["tickers"], intervals: ListAndStr = parameter["intervals"]):
        lower_tickers = [ticker.lower() for ticker in tickers]
        self.tickers = lower_tickers
        self.intervals = intervals

    def get_data(
            self,
            ticker: str,
            interval: str
        ) -> pd.DataFrame:

        try:
            if interval in date_interval_list:
                data = pd.read_sql(f"{ticker}_{interval}", engine, index_col='Date')
            else:
                data = pd.read_sql(f"{ticker}_{interval}", engine, index_col='Datetime')
            print(f"Successfully loaded data for {ticker}_{interval}")
            return data
        except Exception as e:
            print(f"Could not load data for {ticker}_{interval}: {e}")
            return None

    def get_all_data(self) -> dict:
        ticker_dict = {}
        print(self.tickers)
        for ticker in self.tickers:
            interval_dict = {}
            for interval in self.intervals:
                interval_dict[interval] = self.get_data(ticker, interval)
            ticker_dict[ticker] = interval_dict
        return ticker_dict

class Stock(Tickers):
    def __init__(self, tickers: ListAndStr=lower_tickers, interval: str='1m'):
        super().__init__(tickers, intervals=[interval])
        self.tickers = tickers
        self.interval = interval
        self.data = self.get_all_data()

        self.close = self.close()
        self.open = self.open()
        self.high = self.high()
        self.low = self.low()

        self.prices = self.close
        self.prices = self.prices.dropna(how='all', axis=1)
        self.value_type = 'Close'
        self.series_type = ''

        self.VAR = VAR.VAR(self.prices)

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
    def log_diff(self):
        index = self.prices.index
        self.prices = self.prices.apply(np.log)
        self.prices.index = index
        self.prices = self.prices.diff()
        self.series_type += '_log_diff'
        return self
    
    def set_close(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker][self.interval]['Close']
        self.value_type = 'Close'
        return self
    def set_open(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker][self.interval]['Open']
        self.value_type = 'Open'
        return self
    def set_high(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker][self.interval]['High']
        self.vale_type = 'High'
        return self
    def set_low(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker][self.interval]['Low']
        self.value_type = 'Low'
        return self
    
    def close(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker][self.interval]['Close']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def open(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker][self.interval]['Open']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def high(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker][self.interval]['High']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def low(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker][self.interval]['Low']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
            
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
            # plt.plot(self.close[ticker], label='Close Price', color='blue', linewidth=2)
            # plt.plot(self.high[ticker], label='High Price', color='red', linestyle='--')
            # plt.plot(self.low[ticker], label='Low Price', color='green', linestyle='--')
            # plt.plot(self.open[ticker], label='Open Price', color='orange', linestyle=':')
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

    def CandleStick(self, day=200, moving_average: ListAndStr=[5, 10, 20 ,50, 75, 100], tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue
            # prepare history
            candlestick_history = self.history[ticker].copy()
            candlestick_history = candlestick_history.reset_index()
            candlestick_history['Date'] = candlestick_history['Date'].map(mdates.date2num)
            candlestick_history = candlestick_history[-day:]

            fig, ax = plt.subplots(figsize=(12, 6))
            # illustrate a candlestick
            candlestick_ohlc(ax, candlestick_history[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                            width=1, colorup='g', colordown='r')
            # add a moving average
            for ma in moving_average:
                candlestick_history[f'MA{ma}'] = candlestick_history['Close'].rolling(ma).mean()
                ax.plot(candlestick_history['Date'], candlestick_history[f'MA{ma}'], label=f'{ma} day moving average')

            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.title(f'{ticker} candlestick')
            plt.xlabel('Date')
            plt.ylabel('price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.close()

class Option(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.options = self.get_attribute('options')

class Holder(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.major_holders = self.get_attribute('major_holders')
        self.institutional_holders = self.get_attribute('institutional_holders')
        self.mutualfund_holders = self.get_attribute('mutualfund_holders')

class Currency(Stock):
    def __init__(self, tickers: ListAndStr, period: str='max', interval: str='1d'):
        super().__init__(tickers, period, interval)

class Finance(Tickers):
    def __init__(self, tickers, quarterly=False):
        super().__init__(tickers)
        self.balance_sheet = self.get_attribute('balance_sheet')    
        self.income_stmt = self.get_attribute('income_stmt')    
        self.cashflow = self.get_attribute('cashflow')
        if quarterly:
            self.balance_sheet = self.get_attribute('quarterly_balance_sheet')    
            self.income_stmt = self.get_attribute('quarterly_income_stmt')    
            self.cashflow = self.get_attribute('quarterly_cashflow')
        for ticker in self.tickers:
            self.balance_sheet[ticker] = self.balance_sheet[ticker].T
            self.income_stmt[ticker] = self.income_stmt[ticker].T
            self.cashflow[ticker] = self.cashflow[ticker].T

    def get_indices(self, ticker):
        # 空のDataFrame作成
        indeces = pd.DataFrame()
        
        try:
            BS = self.balance_sheet[ticker]
            PL = self.income_stmt[ticker]
            
            # 安全にデータを取得するヘルパー関数
            def get_safe(df, col):
                return df[col] if col in df.columns else pd.Series(dtype='float64')
            
            # 必要なデータを取得
            total_revenue = get_safe(PL, 'Total Revenue')
            cost_of_revenue = get_safe(PL, 'Cost Of Revenue')
            net_income = get_safe(PL, 'Net Income')
            inventory = get_safe(BS, 'Inventory')
            stockholders_equity = get_safe(BS, 'Stockholders Equity')
            treasury_shares = get_safe(BS, 'Treasury Shares Number')
            shares_issued = get_safe(BS, 'Share Issued')
            
            # 財務指標計算 (ゼロ除算を避けるためnp.where使用)
            indeces['Gross Profit'] = total_revenue - cost_of_revenue
            
            with np.errstate(divide='ignore', invalid='ignore'):
                indeces['Cost Of Revenue Ratio'] = np.where(total_revenue != 0, cost_of_revenue / total_revenue, np.nan)
                indeces['Inventory Turnover'] = np.where(inventory != 0, cost_of_revenue / inventory, np.nan)
                indeces['ROE'] = np.where(stockholders_equity != 0, net_income / stockholders_equity, np.nan)
                indeces['Treasury Stock Ratio'] = np.where(shares_issued != 0, treasury_shares / shares_issued, np.nan)
            
            # インデックスを設定
            if not indeces.empty:
                indeces.index = PL.index if not PL.empty else BS.index
                
        except KeyError as e:
            print(f"Error: {e} not found for ticker {ticker}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
        
        return indeces

class Insider(Tickers):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.insider_purchases = self.get_attribute('insider_purchases')    
        self.insider_roster_holders = self.get_attribute('insider_roster_holders')    
        self.insider_transactions = self.get_attribute('insider_transactions')