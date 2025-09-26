# scipy
import pandas as pd

# standard
import os
import json

# my library
from ... import ListAndStr

# others
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

    def get_stock_data(
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

    def get_all_stock_data(self) -> dict:
        ticker_dict = {}
        print(self.tickers)
        for ticker in self.tickers:
            interval_dict = {}
            for interval in self.intervals:
                interval_dict[interval] = self.get_stock_data(ticker, interval)
            ticker_dict[ticker] = interval_dict
        return ticker_dict
    
    def get_balance_sheet(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                BS = pd.read_sql(f"{ticker}_balance_sheet_annual", engine)
            else:
                BS = pd.read_sql(f"{ticker}_balance_sheet_quarterly", engine)
            return BS
        except Exception as e:
            print(f"Could not load data : {e}")
            return None
    
    def get_financials(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                PL = pd.read_sql(f"{ticker}_financials_annual", engine)
            else:
                PL = pd.read_sql(f"{ticker}_financials_quarterly", engine)
            return PL
        except Exception as e:
            print(f"Could not load data : {e}")
            return None

    def get_cashflow(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                CF = pd.read_sql(f"{ticker}_cashflow_annual", engine)
            else:
                CF = pd.read_sql(f"{ticker}_cashflow_quarterly", engine)
            return CF
        except Exception as e:
            print(f"Could not load data : {e}")
            return None
        
    def get_all_data(self, get_func, param=None) -> dict:
        ticker_dict = {}
        print(self.tickers)
        for ticker in self.tickers:
            ticker_param = param
            ticker_param['ticker'] = ticker
            ticker_dict[ticker] = get_func(ticker_param)
        return ticker_dict