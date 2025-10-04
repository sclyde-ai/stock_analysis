# scipy
import pandas as pd

# standard
import os
import json
import sys

# my library
from .data_analysis.dataclass import ListAndStr

# others
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

# 1. このスクリプトファイル自身の絶対パスを取得
# .resolve() はシンボリックリンクなどを解決して完全なパスにします
script_path = Path(__file__).resolve()

# 2. 親の親のディレクトリパスを取得
# .parent で1つ上の親ディレクトリを取得できます
grandparent_dir = script_path.parent.parent.parent

# get parameter.json
with open('../parameter.json', 'r', encoding='utf-8') as f:
    json_string = f.read()
    parameter = json.loads(json_string)
    lower_tickers = [ticker.lower() for ticker in parameter['tickers']]

load_dotenv(dotenv_path='../.env')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/yfinance"
try :
    engine = create_engine(DATABASE_URL)
    print("Database connected successfully.")
except Exception as e:
    print(f"Could not connect to the database : {e}")

class Tickers:
    """ 
    """
    all_tickers = lower_tickers
    all_intervals = parameter["intervals"]
    datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
    date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']

    def __init__(self, tickers: ListAndStr=all_tickers):
        self.tickers = tickers

    def get_data(self, table_name: str, index_col: str=None) -> pd.DataFrame:
        try: 
            data = pd.read_sql(table_name, engine, index_col=index_col)
            return data
        except Exception as e:
            print(f"Could not load data : {e}")
            return None

    def get_data_dict(self) -> dict:
        data_dict = {}
        for ticker in self.tickers:
            try:
                data = self.get_data(ticker)
                print(data)
                data_dict[ticker] = data
            except Exception as e:
                print(f"Could not load data : {e}")
                continue
        return data_dict