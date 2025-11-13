# 使い方
test directory内であれば
```
    %pip install -r requirements.txt
    # !docker compose -f ../docker-compose.yml up

    import sys

    from pathlib import Path
    sys.path.append(str(Path.cwd().parent))

    from src.package.ticker import Tickers
    from src.package.stock import Stock
```

# directoryの説明
- [x] yfinance_db: yfinanceからdataを取得してSQLに保存する
- [x] library: 自身が学んだデータ分析手法や計算方法などを再利用できるようにpackageにしている
- [ ] news_db: news_apiからdataを取得する
- [ ] jquants_db: 日本取引所グループからdataを取得する
checkは稼働中を意味する

# packageの機能
- [ ] 株価
    - [ ] 株価の可視化
    - [ ] 株価予測model
- [ ] 財務諸表
    - [ ] 財務諸表の可視化
    - [ ] 財務指標の計算
- [ ] data分析
- [ ] CAPM
- [ ] C言語による計算機能
- [ ] smart contract

# data一覧
- [ ] yfinance
    - [x] 株価
    - [x] 財務諸表
    - [ ] cashflow
    - [ ] news
- [ ] news_api
- [ ] 日本取引所
- [ ] e-stat
- [ ] Alpha Vantage

# 今後の方針
- [x] directory全体のpackage化
- [ ] cloud serviceを用いたserverの構築
- [ ] 資産運用のsimulation
- [ ] dashboardの作成

- [ ] fin-py