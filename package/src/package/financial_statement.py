import pandas as pd
import yfinance as yf

# scipy
import pandas as pd

# standard

# my library
from .data_analysis.dataclass import ListAndStr

class financial_statement():
    def __init__(self, tickers: ListAndStr=None):
        if tickers:
            super().__init__(tickers=tickers)
        else:
            super().__init__()
        # self.BS = self.get_all_data("balance_sheet")
        # self.PL = self.get_all_data("financials")

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

    def get_financials(
            self,
            ticker: str,
            quarterly = False
        ):
        try:
            if quarterly:
                PL = self.get_data(f"{ticker}_financials_quarterly")
            else:
                PL = self.get_data(f"{ticker}_financials_annual")
            return PL
        except Exception as e:
            print(f"Could not load data : {e}")
            return None
    
    # profitability
    def GrossProfitMargin(self):
        GrossProfitMargin = self.PL['Gross Profit']/self.PL['Total Revenue']
        return GrossProfitMargin
    
    def OperatingMargin(self):
        OperatingMargin = self.PL['Operating Income']/self.PL['Total Revenue']
        return OperatingMargin
    
    def NetProfitMargin(self):
        NetProfitMargin = self.PL['Net Income']/self.PL['Total Revenue']
        return NetProfitMargin
    
    def EBITDAMargin(self):
        EBITDAMargin = self.PL['EBITDA']/self.PL['Total Revenue']
        return EBITDAMargin
    
    def ROA(self):
        ROA = self.PL['Net Income']/self.BS['Total Assets']
        return ROA
    
    def ROE(self):
        ROE = self.PL['Net Income']/self.BS['Stockholders Equity']
        return ROE
    
    def ROIC(self):
        tax_rate = self.PL['Tax Provision']/self.PL['Pretax Income']
        NOPAT = self.PL['EBIT'] * (1 - tax_rate)
        ROIC = NOPAT/self.BS['Invested Capital']
        return ROIC
    
    # solvency & liquidity
    def CurrentRatio(self):
        CurrentRatio = self.BS['Current Assets']/self.BS['Current Liabilities']
        return CurrentRatio
    
    def QuickRatio(self):
        QuickRatio = (self.BS['Current Assets'] - self.BS['Inventory'])/self.BS['Current Liabilities']
        return QuickRatio
    
    def DebtEquityRatio(self):
        DebtEquityRatio = self.BS['Total Liabilities Net Minority Interest']/self.BS['Stockholders Equity']
        return DebtEquityRatio
    
    def EquityRatio(self):
        EquityRatio = self.BS['Stockholders Equity']/self.BS['Total Assets']
        return EquityRatio

    # efficiecy 
    def TotalAssetTurnover(self):
        TotalAssetTurnover = self.PL['Total Revenue']/self.BS['Total Assets']
        return TotalAssetTurnover
    
    def InventoryTurnover(self):
        InventoryTurnover = self.PL['Cost Of Revenue']/self.BS['Inventory']
        return InventoryTurnover
    
    def ReceivablesTurnover(self):
        ReceivablesTurnover = self.PL['Total Revenue']/self.BS['Accounts Receivable']
        return ReceivablesTurnover