import os
import time
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, inspect
import json
from dotenv import load_dotenv
from datetime import timedelta

# .envファイルから環境変数を読み込む
load_dotenv()

# --- 環境変数から設定を読み込み ---
DB_NAME = os.getenv('YFINANCE_DB')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  # docker-compose.ymlで定義したサービス名
DB_PORT = os.getenv('DB_PORT')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_stock_data(ticker, interval, period="max", start=None):
    stock = yf.Ticker(ticker)
    time.sleep(1)
    data = stock.history(interval=interval, period=period, start=start)
    return data

def save_to_db(data, table_name, engine):
    if not data.empty:
        data.to_sql(table_name, engine, if_exists='append')

if __name__ == "__main__":
    try:
        with open('parameter.json', 'r', encoding='utf-8') as f:
            parameter = json.load(f)
        # Valid interval values
        datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']
        
        engine = create_engine(DATABASE_URL)
        try:
            with engine.connect() as connection:
                print(f"Successfully connected to the database.")
        except Exception as e:
            print("Failed to connect to the database after multiple retries. Exiting.")
            exit(1)
        
        inspector = inspect(engine)
        
        while True:
            for interval in parameter["intervals"]:
                for ticker in parameter["tickers"]:
                    table_name = f"{ticker.lower()}_{interval}"
                    start_date = None
                    period = parameter.get("period")

                    if inspector.has_table(table_name):
                        try:
                            if interval in datetime_interval_list:
                                index_col = "Datetime"
                            else :
                                index_col = "Date"
                            with engine.connect() as connection:
                                result = connection.execute(f'SELECT MAX("{index_col}") FROM "{table_name}"')
                                last_date = result.scalar()
                            if last_date:
                                if interval in datetime_interval_list:
                                    start_date = last_date + timedelta(minutes=1)
                                else:
                                    start_date = last_date + timedelta(days=1)
                                period = None
                        except Exception as e:
                            print(f"Could not get last date for {table_name}, using period. Error: {e}")

                    print(f"Fetching data for {ticker} with interval {interval}...")
                    stock_data = get_stock_data(ticker, interval=interval, period=period, start=start_date)

                    if not stock_data.empty:
                        # To prevent duplicates, filter out data that is already in the database
                        if start_date:
                            stock_data = stock_data[stock_data.index >= start_date]

                    if not stock_data.empty:
                        print(f"Saving data for {ticker} with interval {interval} to table {table_name}...")
                        save_to_db(stock_data, table_name, engine)
                        print(f"Successfully saved {len(stock_data)} new data points for {ticker} with interval {interval}.")
                    else:
                        print(f"No new data found for {ticker} with interval {interval}.")

            sleep_duration = parameter.get("sleep_duration_seconds", 3600)
            print(f"Finished fetching and saving stock data. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)

    except Exception as e:
        print(f"An error occurred: {e}")