import pandas as pd
import yfinance as yf

# scipy
import pandas as pd

# standard
import os
import json

# my library
from .data_analysis.dataclass import ListAndStr

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

class Fundamentals():
    def __init__(self):
        self.tickers = yf.Tickers()
        self.BS = self.get_all_data("balance_sheet")
        self.PL = self.get_all_data("financials")
        self._metrics = None

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