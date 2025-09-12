# directoryの説明
yfinance_db : yfinanceからdataを取得してSQLに保存しています
library : 自身が学んだデータ分析手法や計算方法などを再利用できるようにpackageにしています
bash : 作業中に便利だと思ったcommandを保存しています

# Valid period values
period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
# Valid interval values
datetime_interval_list = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
date_interval_list = ['1d', '5d', '1wk', '1mo', '3mo']

# 今後の方針
1. cloud serviceを用いたserverの構築
2. APIからdataを取得
3. libraryの拡充
4. 資産運用のsimulation
5. notion dashboardの作成

# Fianancial data API
## stock data
yfinance
Alpha Vantage
JPX

## official data
e-stat

# port allocation
5433 : yfinance
5434 : news