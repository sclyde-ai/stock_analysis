# データサイエンス向けの公式イメージをベースにする
FROM jupyter/datascience-notebook:latest

# rootユーザーに切り替えてライブラリをインストール
USER root

# requirements.txtをコンテナ内にコピー
COPY requirements.txt /tmp/

# pipでライブラリをインストール
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 元のjovyanユーザーに戻す
USER jovyan