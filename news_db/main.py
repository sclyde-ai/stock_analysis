import os
import time
import requests
import psycopg2
import schedule
from psycopg2.extras import execute_values
from dotenv import load_dotenv

print("--- Python script has started successfully! ---")

# .envファイルから環境変数を読み込む
load_dotenv()

# --- 環境変数から設定を読み込み ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
DB_NAME = os.getenv('NEWS_DB')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  # docker-compose.ymlで定義したサービス名
DB_PORT = os.getenv('DB_PORT')

def get_db_connection():
    """DBへの接続を試み、失敗した場合はリトライする"""
    conn = None
    while conn is None:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
        except psycopg2.OperationalError as e:
            print(f"データベース接続に失敗しました: {e}")
            print("5秒後に再試行します...")
            time.sleep(5)
    return conn

def fetch_and_save_news():
    """News APIからデータを取得し、DBに保存する関数"""
    print("ニュースの取得を開始します...")
    
    # News APIのエンドポイント (日本のビジネスカテゴリ)
    url = f"https://newsapi.org/v2/top-headlines"

    
    try:
        categories = [
            "business",
            "entertainment",
            "general",
            "health",
            "science",
            "sports",
            "technology",
        ]

        for category in categories:
            # パラメータ設定
            params = {
                "category": category,  # カテゴリ
                # "country": "jp",         # 国
                "pageSize": 20,          # 取得件数
                "apiKey": NEWS_API_KEY
            }
            response = requests.get(url, params=params)
            response.raise_for_status()  # HTTPエラーがあれば例外を発生
            data = response.json()
            articles = data.get("articles", [])

            if not articles:
                print("新しい記事は見つかりませんでした。")
                return

            # DBに保存するデータ形式に整形
            articles_to_save = [
                (
                    category,
                    article.get('source', {}).get('name'),
                    article.get('author'),
                    article.get('title'),
                    article.get('description'),
                    article.get('url'),
                    article.get('publishedAt'),
                    article.get('content'),
                )
                for article in articles
            ]

            # DBに接続して保存
            conn = get_db_connection()
            cur = conn.cursor()
            
            # ON CONFLICT句でURLの重複を無視する
            insert_query = """
                INSERT INTO articles (category, source, author, title, description, url, published_at, content)
                VALUES %s
                ON CONFLICT (url) DO NOTHING;
            """
            
            execute_values(cur, insert_query, articles_to_save)
            
            conn.commit()
            print(f"{cur.rowcount} 件の新しい記事をデータベースに保存しました。")
            
            cur.close()
            conn.close()

    except requests.exceptions.RequestException as e:
        print(f"APIリクエスト中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    print("サービスを開始します。1時間ごとにニュースを取得します。")
    
    # 最初に一度実行
    fetch_and_save_news()

    # 1時間ごとにfetch_and_save_news関数を実行するようにスケジュール
    schedule.every().hour.do(fetch_and_save_news)

    while True:
        schedule.run_pending()
        time.sleep(1)