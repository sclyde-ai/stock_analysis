
import os
import time
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    time.sleep(1)
    data = stock.history(period="1d")
    return data

def save_to_db(data, table_name, engine):
    data.to_sql(table_name, engine, if_exists='replace')

if __name__ == "__main__":
    DATABASE_URL = os.environ.get("DATABASE_URL")
    
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


    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock_data = get_stock_data(ticker)
        if not stock_data.empty:
            print(f"Saving data for {ticker} to the database...")
            save_to_db(stock_data, f"{ticker.lower()}", engine)
            print(f"Successfully saved data for {ticker}.")
        else:
            print(f"No data found for {ticker}.")

    print("Finished fetching and saving stock data.")
