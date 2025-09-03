# ベースとなるPythonイメージを指定
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコピー
COPY main.py .

# コンテナ起動時に実行するコマンド
CMD ["python", "main.py"]