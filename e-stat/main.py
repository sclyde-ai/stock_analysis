# 環境変数での設定（推奨）
import os
from dotenv import load_dotenv
import jpy_datareader.data as web
from jpy_datareader.data import get_data_estat_statslist

load_dotenv()
api_key = os.getenv("ESTAT_APP_ID")

def get_data_estat_statslist_(api_key=api_key, limit=10):
    statslist = web.get_data_estat_statslist(api_key=api_key, limit=10)
    return statslist

if __name__ == "__main__":
    statslist = get_data_estat_statslist_(api_key=api_key, limit=10)
    print(statslist)