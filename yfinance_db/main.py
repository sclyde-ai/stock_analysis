import os
import time
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import json

def get_stock_data(ticker, interval='1m', period='max'):
    stock = yf.Ticker(ticker)
    time.sleep(1)
    data = stock.history(interval=interval, period=period)
    return data

def save_to_db(data, table_name, engine):
    data.to_sql(table_name, engine, if_exists='replace')

if __name__ == "__main__":
    try:
        with open('parameter.json', 'r', encoding='utf-8') as f:
            json_string = f.read()
            parameter = json.loads(json_string)
        
        DATABASE_URL = os.environ.get("DATABASE_URL")
        if not DATABASE_URL:
            print("DATABASE_URL environment variable not set. Exiting.")
            exit(1)

        engine = create_engine(DATABASE_URL)
        max_retries = 10
        retry_delay = 5  # seconds
        for i in range(max_retries):
            try:
                connection = engine.connect()
                connection.close()
                print(f"Successfully connected to the database.")
                break  # 接続に成功したらループを抜ける
            except Exception as e:
                print(f"Could not connect to database (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Failed to connect to the database after multiple retries. Exiting.")
                    exit(1)

        for interval in parameter["intervals"]:
            for ticker in parameter["tickers"]:
                print(f"Fetching data for {ticker} with interval {interval}...")
                stock_data = get_stock_data(ticker, interval=interval, period=parameter["period"])
                if not stock_data.empty:
                    table_name = f"{ticker.lower()}_{interval}"
                    print(f"Saving data for {ticker} with interval {interval} to table {table_name}...")
                    save_to_db(stock_data, table_name, engine)
                    print(f"Successfully saved data for {ticker} with interval {interval}.")
                else:
                    print(f"No data found for {ticker} with interval {interval}.")

        print("Finished fetching and saving stock data.")
    except Exception as e:
        print(f"An error occurred: {e}")
