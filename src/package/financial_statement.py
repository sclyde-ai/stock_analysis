import pandas as pd
import yfinance as yf
import datetime
import time
from typing import Union

# scipy
import pandas as pd

# standard

# my library
from .data_analysis.dataclass import ListAndStr
from .ticker import Tickers

class FinancialStatement(Tickers):
    def __init__(self, tickers: ListAndStr=None):
        if tickers:
            super().__init__(tickers=tickers)
        else:
            super().__init__()
        self.BS, self.PL = self.get_all_year()

    def get_balance_sheet(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                BS = self.get_data(f"{ticker}_balance_sheet_quarterly")
            else:
                BS = self.get_data(f"{ticker}_balance_sheet_annual")
            return BS
        except Exception as e:
            print(f"Could not load data : {e}")
            return None

    def get_income_stmt(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                PL = self.get_data(f"{ticker}_income_stmt_quarterly")
            else:
                PL = self.get_data(f"{ticker}_income_stmt_annual")
            return PL
        except Exception as e:
            print(f"Could not load data : {e}")
            return None

    def get_all_data(self) -> dict:
        BS_dict = {}
        PL_dict = {}
        try:
            for ticker in self.tickers:
                BS_dict[ticker] = self.get_balance_sheet(ticker)
                PL_dict[ticker] = self.get_income_stmt(ticker)
            return BS_dict, PL_dict
        except Exception as e:
            print(f"Could not get all data: {e}")
            return None, None

    def get_by_year(self, year):
        year = int(year)
        all_bs = []
        all_pl = []

        for ticker in self.tickers:
            # 年次データを取得
            bs = self.get_balance_sheet(ticker, quarterly=False)
            pl = self.get_income_stmt(ticker, quarterly=False)

            # 'date' 列をdatetimeに変換
            if bs is not None and 'date' in bs.columns and not bs.empty:
                bs['date'] = pd.to_datetime(bs['date'])
                bs_for_year = bs[bs['date'].dt.year == year]
                if not bs_for_year.empty:
                    # 最初の行を使用し、'date'列を除外
                    bs_series = bs_for_year.iloc[0].drop('date')
                    bs_series.name = ticker
                    all_bs.append(bs_series)

            if pl is not None and 'date' in pl.columns and not pl.empty:
                pl['date'] = pd.to_datetime(pl['date'])
                pl_for_year = pl[pl['date'].dt.year == year]
                if not pl_for_year.empty:
                    # 最初の行を使用し、'date'列を除外
                    pl_series = pl_for_year.iloc[0].drop('date')
                    pl_series.name = ticker
                    all_pl.append(pl_series)

        # DataFrameを結合
        bs_df = pd.concat(all_bs, axis=1) if all_bs else pd.DataFrame()
        pl_df = pd.concat(all_pl, axis=1) if all_pl else pd.DataFrame()
        
        return bs_df.T, pl_df.T
    
    def get_all_year(self) -> dict:
        BS_dict = {}
        PL_dict = {}
        current_year = datetime.datetime.now().year
        for year in range(2000, current_year + 1):
            bs, pl = self.get_by_year(year)
            if not bs.empty:
                BS_dict[year] = bs
            if not pl.empty:
                PL_dict[year] = pl
        return BS_dict, PL_dict
    
    # profitability
    @property
    def GrossProfitMargin(self):
        GrossProfitMargin = self.PL['Gross Profit']/self.PL['Total Revenue']
        return GrossProfitMargin

    @property
    def OperatingMargin(self):
        OperatingMargin = self.PL['Operating Income']/self.PL['Total Revenue']
        return OperatingMargin
    
    @property
    def NetProfitMargin(self):
        NetProfitMargin = self.PL['Net Income']/self.PL['Total Revenue']
        return NetProfitMargin
    
    @property
    def EBITDAMargin(self):
        EBITDAMargin = self.PL['EBITDA']/self.PL['Total Revenue']
        return EBITDAMargin
    
    @property
    def ROA(self):
        ROA = self.PL['Net Income']/self.BS['Total Assets']
        return ROA
    
    @property
    def ROE(self):
        ROE = self.PL['Net Income']/self.BS['Stockholders Equity']
        return ROE
    
    @property
    def ROIC(self):
        tax_rate = self.PL['Tax Provision']/self.PL['Pretax Income']
        NOPAT = self.PL['EBIT'] * (1 - tax_rate)
        ROIC = NOPAT/self.BS['Invested Capital']
        return ROIC
    
    # solvency & liquidity
    @property
    def CurrentRatio(self):
        CurrentRatio = self.BS['Current Assets']/self.BS['Current Liabilities']
        return CurrentRatio
    
    @property
    def QuickRatio(self):
        QuickRatio = (self.BS['Current Assets'] - self.BS['Inventory'])/self.BS['Current Liabilities']
        return QuickRatio
    
    @property
    def DebtEquityRatio(self):
        DebtEquityRatio = self.BS['Total Liabilities Net Minority Interest']/self.BS['Stockholders Equity']
        return DebtEquityRatio
    
    @property
    def EquityRatio(self):
        EquityRatio = self.BS['Stockholders Equity']/self.BS['Total Assets']
        return EquityRatio

    # efficiecy 
    @property
    def TotalAssetTurnover(self):
        TotalAssetTurnover = self.PL['Total Revenue']/self.BS['Total Assets']
        return TotalAssetTurnover
    
    @property
    def InventoryTurnover(self):
        InventoryTurnover = self.PL['Cost Of Revenue']/self.BS['Inventory']
        return InventoryTurnover
    
    @property
    def ReceivablesTurnover(self):
        ReceivablesTurnover = self.PL['Total Revenue']/self.BS['Accounts Receivable']
        return ReceivablesTurnover