import os
import psycopg2
import pandas as pd

# 参照したいテーブル名（例: 'aapl_stock_data'）
TABLE_NAME = 'aapl_stock_data'

try:
    # Connect to the database
    conn = psycopg2.connect(
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        host="db",
        port=os.environ.get("DB_PORT")
    )

    # Query the data
    print(f"Attempting to read from table: {TABLE_NAME}")
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    # Close the connection
    conn.close()

    # Print the first 5 rows
    print(f"Successfully read data from {TABLE_NAME}:")
    print(df.head())

except psycopg2.errors.UndefinedTable:
    print(f"Error: Table '{TABLE_NAME}' does not exist.")
    print("Please check if the yfinance_db_saver service has run correctly and created the tables.")
except Exception as e:
    print(f"An error occurred: {e}")