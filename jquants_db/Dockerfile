# ベースとなるPythonイメージを指定
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 依存ライブラリのリストをコンテナにコピー
COPY requirements.txt .

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコンテナにコピー
COPY main.py .

# コンテナが起動したときに実行するコマンド
CMD ["python", "main.py"]