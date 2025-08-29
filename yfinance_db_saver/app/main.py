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
        print(DATABASE_URL)
        
        # Retry connecting to the database
        for _ in range(5):
            try:
                engine = create_engine(DATABASE_URL)
                connection = engine.connect()
                connection.close()
                print("Successfully connected to the database.")
                break
            except Exception as e:
                print(f"Could not connect to database: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
        else:
            print("Could not connect to the database after several retries. Exiting.")
            exit(1)

        for ticker in parameter["tickers"]:
            print(f"Fetching data for {ticker}...")
            stock_data = get_stock_data(ticker, interval=parameter["interval"], period=parameter["period"])
            if not stock_data.empty:
                print(f"Saving data for {ticker} to the database...")
                save_to_db(stock_data, f"{ticker.lower()}", engine)
                print(f"Successfully saved data for {ticker}.")
            else:
                print(f"No data found for {ticker}.")

        print("Finished fetching and saving stock data.")
    except Exception as e:
        print(f"An error occurred: {e}")
