import os
import time
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, inspect, text
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
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
# print(DB_PORT)

def get_stock_data(ticker_obj, interval, period="max", start=None):
    time.sleep(1)
    data = ticker_obj.history(interval=interval, period=period, start=start)
    return data

def save_to_db(data, table_name, engine, index_label):
    """
    Performs an "UPSERT" operation for PostgreSQL.
    If the table does not exist, it creates it, sets the primary key, and inserts the data.
    If the table exists, it inserts new rows or updates existing ones based on the primary key.
    """
    if data.empty:
        return

    inspector = inspect(engine)

    with engine.connect() as connection:
        if not inspector.has_table(table_name):
            # Table doesn't exist: create it, set PK, and insert the data.
            print(f"Table '{table_name}' not found. Creating it and inserting initial data.")
            try:
                # Use pandas to create the table and insert the first batch of data.
                data.to_sql(table_name, connection, if_exists='fail', index=True, index_label=index_label)
                # Set the primary key, which is crucial for future upserts.
                with connection.begin() as transaction:
                    connection.execute(text(f'ALTER TABLE "{table_name}" ADD PRIMARY KEY ("{index_label}");'))
                    transaction.commit()
                print(f"Successfully created table '{table_name}' with '{index_label}' as primary key.")
            except Exception as e:
                print(f"Error creating table '{table_name}': {e}")
        else:
            # Table exists: perform the upsert logic.
            temp_table_name = f"temp_{table_name}_{int(time.time())}"
            with connection.begin() as transaction:
                try:
                    # Step 1: Write the DataFrame to a temporary table.
                    data.to_sql(temp_table_name, connection, if_exists='replace', index=True, index_label=index_label)

                    # Step 2: Prepare column names for the SQL query.
                    df_cols = [f'"{c}"' for c in data.columns]
                    all_cols = [f'"{index_label}"'] + df_cols
                    update_stmt = ", ".join([f'{col} = EXCLUDED.{col}' for col in df_cols])

                    # Step 3: Execute the UPSERT from the temporary table.
                    upsert_sql = text(f'''
                        INSERT INTO "{table_name}" ({", ".join(all_cols)})
                        SELECT {", ".join(all_cols)} FROM "{temp_table_name}"
                        ON CONFLICT ("{index_label}") DO UPDATE SET {update_stmt};
                    ''')
                    connection.execute(upsert_sql)

                    # Step 4: Drop the temporary table.
                    connection.execute(text(f'DROP TABLE "{temp_table_name}"'))

                    transaction.commit()
                except Exception as e:
                    print(f"An error occurred during the upsert to {table_name}: {e}")
                    transaction.rollback()

def get_and_save_income_stmt(ticker_obj, ticker_str, engine):
    income_stmt_map = {
        'income_stmt_annual': ticker_obj.income_stmt,
        'balance_sheet_annual': ticker_obj.balance_sheet,
        'cashflow_annual': ticker_obj.cashflow,
        'income_stmt_quarterly': ticker_obj.quarterly_income_stmt,
        'balance_sheet_quarterly': ticker_obj.quarterly_balance_sheet,
        'cashflow_quarterly': ticker_obj.quarterly_cashflow
    }

    for name, data in income_stmt_map.items():
        if not data.empty:
            table_name = f"{ticker_str.lower()}_{name}"
            # Transpose and format data
            data = data.transpose()
            data.index.name = 'date'
            print(f"Saving {name} data for {ticker_str} to table {table_name}...")
            save_to_db(data, table_name, engine, index_label='date')
            print(f"Successfully saved {name} data for {ticker_str}.")


def get_and_save_news(ticker_obj, ticker_str, engine):
    """
    Fetches news for a given ticker and saves it to the database.
    """
    news = ticker_obj.news
    if news:
        news_list = []
        for article in news:
            # The news data can have different structures, so we check for keys gracefully.
            # The main content is often in a nested dictionary.
            content = article.get('content', article)

            news_item = {
                'uuid': article.get('id') or article.get('uuid'),
                'title': content.get('title'),
                'publisher': content.get('publisher') or content.get('provider', {}).get('displayName'),
                'provider_publish_time': content.get('provider_publish_time') or content.get('pubDate'),
                'type': content.get('type') or content.get('contentType'),
                'link': content.get('link') or content.get('canonicalUrl', {}).get('url'),
                'summary': content.get('summary'),
                'thumbnail': None
            }

            # Safely extract thumbnail
            thumbnail_data = content.get('thumbnail')
            if thumbnail_data and thumbnail_data.get('resolutions'):
                news_item['thumbnail'] = thumbnail_data['resolutions'][0].get('url')

            news_list.append(news_item)

        df = pd.DataFrame(news_list)

        # Convert provider_publish_time to datetime. It can be a string or a unix timestamp.
        def to_datetime_flexible(ts):
            if pd.isna(ts):
                return None
            if isinstance(ts, str):
                return pd.to_datetime(ts)
            # Assuming it's a number (like a unix timestamp)
            return pd.to_datetime(ts, unit='s')

        df['provider_publish_time'] = df['provider_publish_time'].apply(to_datetime_flexible)

        # Drop rows where uuid is missing, as it's our primary key
        df.dropna(subset=['uuid'], inplace=True)
        df.set_index('uuid', inplace=True)

        table_name = f"{ticker_str.lower()}_news"
        print(f"Saving news for {ticker_str} to table {table_name}...")
        save_to_db(df, table_name, engine, index_label='uuid')
        print(f"Successfully saved {len(df)} news articles for {ticker_str}.")


if __name__ == "__main__":
    try:
        with open('parameter.json', 'r', encoding='utf-8') as f:
            parameter = json.load(f)
        # Valid interval values
        datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']
        
        engine = create_engine(DATABASE_URL)
        print(engine)
        print(DATABASE_URL)
        try:
            with engine.connect() as connection:
                print(f"Successfully connected to the database.")
        except Exception as e:
            print("Failed to connect to the database after multiple retries. Exiting.")
            print(e)
            exit(1)
        
        inspector = inspect(engine)
        
        while True:
            for ticker in parameter["tickers"]:
                try:
                    print(f"--- Processing ticker: {ticker} ---")
                    ticker_obj = yf.Ticker(ticker)

                    for interval in parameter["intervals"]:
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
                        stock_data = get_stock_data(ticker_obj, interval=interval, period=period, start=start_date)

                        if not stock_data.empty:
                            # To prevent duplicates, filter out data that is already in the database
                            if start_date:
                                stock_data = stock_data[stock_data.index >= start_date]

                        if not stock_data.empty:
                            print(f"Saving data for {ticker} with interval {interval} to table {table_name}...")
                            if interval in datetime_interval_list:
                                index_name = "Datetime"
                            else:
                                index_name = "Date"
                            save_to_db(stock_data, table_name, engine, index_label=index_name)
                            print(f"Successfully saved {len(stock_data)} new data points for {ticker} with interval {interval}.")
                        else:
                            print(f"No new data found for {ticker} with interval {interval}.")

                    # Fetch and save income_stmt and news data once per ticker
                    get_and_save_income_stmt(ticker_obj, ticker, engine)
                    get_and_save_news(ticker_obj, ticker, engine)
                    print(f"--- Finished processing ticker: {ticker} ---")
                    time.sleep(1) # Add a small delay between tickers

                except Exception as e:
                    print(f"An error occurred while processing ticker {ticker}: {e}")
                    print("Continuing with the next ticker.")


            sleep_duration = parameter.get("sleep_duration_seconds", 3600)
            print(f"Finished fetching and saving data for all tickers. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)

    except Exception as e:
        print(f"An error occurred: {e}")