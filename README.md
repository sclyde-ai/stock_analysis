# 今後の方針
1. 分析環境の構築 
2. 自作libraryの適応
3. 他のapiからのdata取得 JPX, e-stat, edinetなど
4. 一般的に使える分析環境のdocker directoryを作成
5. 自身のportfolio作成

# libraryの課題
1. stock classをticker一つに限定し、複数株を用いた分析は他のclassを作成してそれを用いる

# Valid period values
period_values = [
    '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
]

# Valid interval values
interval_values = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m', 
    '1h', '1d', '5d', '1wk', '1mo', '3mo'
]