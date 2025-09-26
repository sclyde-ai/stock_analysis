import pandas as pd
import yfinance as yf
from typing import List, Union, Dict

# scipy
import pandas as pd

# standard
import os
import json

# my library
from .dataclass import ListAndStr

# others
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/yfinance"

engine = create_engine(DATABASE_URL) 

# get parameter.json
with open('../parameter.json', 'r', encoding='utf-8') as f:
    json_string = f.read()
    parameter = json.loads(json_string)
    lower_tickers = [ticker.lower() for ticker in parameter['tickers']]

class FinancialMetrics:
    def __init__(self, fundamentals):
        self.fundamentals = fundamentals

    def _calculate_metric(self, ticker: str, calculation):
        try:
            pl_df = self.fundamentals.PL.get(ticker)
            bs_df = self.fundamentals.BS.get(ticker)

            if pl_df is None or bs_df is None:
                return pd.Series(dtype=float, name=ticker)

            # Align dataframes on their common date index
            common_index = pl_df.index.intersection(bs_df.index)
            if common_index.empty:
                return pd.Series(dtype=float, name=ticker)
            
            pl_df = pl_df.loc[common_index]
            bs_df = bs_df.loc[common_index]
            
            return calculation(pl_df, bs_df)
        except Exception as e:
            # print(f"Could not calculate metric for {ticker}: {e}")
            return pd.Series(dtype=float, name=ticker)

    def get_all_metrics(self, ticker: str) -> Dict[str, pd.Series]:
        metrics = {}
        for metric_name in [
            'gross_profit_margin', 'operating_margin', 'net_profit_margin', 'roa', 'roe', 'roic',
            'current_ratio', 'quick_ratio', 'debt_to_equity_ratio', 'equity_ratio',
            'total_asset_turnover', 'inventory_turnover', 'receivables_turnover',
            'eps', 'bps'
        ]:
            metric_func = getattr(self, metric_name)
            metrics[metric_name] = metric_func(ticker)
        return metrics

    def gross_profit_margin(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Gross Profit'] / pl['Total Revenue'])

    def operating_margin(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Operating Income'] / pl['Total Revenue'])

    def net_profit_margin(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Net Income'] / pl['Total Revenue'])

    def roa(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Net Income'] / bs['Total Assets'])

    def roe(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Net Income'] / bs['Stockholders Equity'])

    def roic(self, ticker: str) -> pd.Series:
        def calculation(pl, bs):
            tax_rate = (pl['Tax Provision'] / pl['Pretax Income']).fillna(0)
            nopat = pl['EBIT'] * (1 - tax_rate)
            return nopat / bs['Invested Capital']
        return self._calculate_metric(ticker, calculation)

    def current_ratio(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: bs['Current Assets'] / bs['Current Liabilities'])

    def quick_ratio(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: (bs['Current Assets'] - bs['Inventory']) / bs['Current Liabilities'])

    def debt_to_equity_ratio(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: bs['Total Liabilities Net Minority Interest'] / bs['Stockholders Equity'])

    def equity_ratio(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: bs['Stockholders Equity'] / bs['Total Assets'])

    def total_asset_turnover(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Total Revenue'] / bs['Total Assets'])

    def inventory_turnover(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Cost Of Revenue'] / bs['Inventory'])

    def receivables_turnover(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Total Revenue'] / bs['Accounts Receivable'])

    def eps(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: pl['Net Income Common Stockholders'] / pl['Basic Average Shares'])

    def bps(self, ticker: str) -> pd.Series:
        return self._calculate_metric(ticker, lambda pl, bs: bs['Stockholders Equity'] / bs['Share Issued'])


class Fundamentals():
    def __init__(self, tickers: ListAndStr=parameter["tickers"]):
        self.tickers = yf.Tickers(tickers)
        self.BS = self.get_all_data("balance_sheet")
        self.PL = self.get_all_data("financials")
        self.CF = self.get_all_data("cashflow")
        self._metrics = None

    def get_all_data(self, property_name: str) -> dict:
        """
        property_name: "balance_sheet", "financials", "cashflow"
        """
        data = {}
        for ticker_symbol, ticker_object in self.tickers.tickers.items():
            try:
                data[ticker_symbol] = getattr(ticker_object, property_name)
            except Exception as e:
                print(f"Could not get {property_name} for {ticker_symbol}: {e}")
        return data

    @property
    def metrics(self) -> FinancialMetrics:
        if self._metrics is None:
            self._metrics = FinancialMetrics(self)
        return self._metrics
        
