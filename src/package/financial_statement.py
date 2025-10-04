import pandas as pd
import yfinance as yf

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
        self.BS = self.get_all_balance_sheet()
        self.PL = self.get_all_income_stmt()

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

    def get_all_balance_sheet(self) -> dict:
        ticker_dict = {}
        for ticker in self.tickers:
            ticker_dict[ticker] = self.get_balance_sheet(ticker)
        return ticker_dict
    
    def get_all_income_stmt(self) -> dict:
        ticker_dict = {}
        for ticker in self.tickers:
            ticker_dict[ticker] = self.get_income_stmt(ticker)
        return ticker_dict
    
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