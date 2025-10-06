# directoryの説明
    - yfinance_db : yfinanceからdataを取得してSQLに保存しています
    - library : 自身が学んだデータ分析手法や計算方法などを再利用できるようにpackageにしています
    - bash : 作業中に便利だと思ったcommandを保存しています

# Valid period values
    - period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
# Valid interval values
    - datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
    - date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']

# 今後の方針
    - directory全体のpackage化
    - cloud serviceを用いたserverの構築
    - APIからdataを取得
    - libraryの拡充
    - 資産運用のsimulation
    - notion dashboardの作成

# Fianancial data API
## stock data
    - yfinance
    - Alpha Vantage
    - JPX

## official data
    - e-stat

# template
    %pip install -r requirements.txt
    # !docker compose -f ../docker-compose.yml up

    # scipy
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # standard
    import os
    import sys
    import json

    # others
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    from pathlib import Path

    sys.path.append(str(Path.cwd().parent))

    # my lib
    from library import yfinance as libyf

    load_dotenv(dotenv_path='../.env')
    print(os.environ)

    # get parameter.json
    with open('../parameter.json', 'r', encoding='utf-8') as f:
        json_string = f.read()
        parameter = json.loads(json_string)
        lower_tickers = [ticker.lower() for ticker in parameter['tickers']]

    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    DB_NAME = os.getenv('NEWS_DB')
    DB_USER = os.getenv('POSTGRES_USER')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')  # docker-compose.ymlで定義したサービス名
    DB_PORT = os.getenv('DB_PORT')

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/yfinance"

    engine = create_engine(DATABASE_URL) 
    print("Successfully connected to the database.")