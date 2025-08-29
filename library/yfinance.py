import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timezone, timedelta
from typing import List
import shutil
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import importlib.resources
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
from statsmodels.tsa.api import VAR
import time
from statsmodels.tsa.stattools import adfuller
from . import ListAndStr, CashTime
from statsmodels.stats.stattools import durbin_watson
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

def get_values_ranking_df(matrix, ascending=False):
    np_matrix = np.array(matrix)
    rows, cols = np_matrix.shape
    
    data = []
    for i in range(rows):
        for j in range(cols):
            data.append({
                'value': np_matrix[i, j],
                'row': i,
                'column': j,
                'index': i*cols + j
            })
    
    df = pd.DataFrame(data)
    df = df.set_index('index')
    df_sorted = df.sort_values(by='value', ascending=ascending).reset_index(drop=True)
    
    return df_sorted

class Tickers():
    def __init__(self, tickers: ListAndStr):
        self.tickers = tickers

    def get_data(
            self,
            ticker: str,
        ) -> pd.DataFrame:
        load_dotenv(dotenv_path='../.env')
        DATABASE_URL = os.environ.get("DATABASE_URL").replace("db:5432", "localhost:5433")
        engine = create_engine(DATABASE_URL)
        print("Successfully connected to the database.")
        try:
            data = pd.read_sql(ticker, engine, index_col='Date')
            print(f"Successfully loaded data for {ticker}")
        except Exception as e:
            print(f"Could not load data for {ticker}: {e}")
        return data

    def get_all_data(self) -> dict:
        ticker_dict = {}
        for ticker in self.tickers:
            ticker_dict[ticker] = self.get_data(ticker)
        return ticker_dict

class Stock(Tickers):
    def __init__(self, tickers: ListAndStr, interval: str='1m', save_path: str=None):
        super().__init__(tickers)
        self.tickers = tickers
        self.data = self.get_all_data()
        self.close = self.close()
        self.open = self.open()
        self.high = self.high()
        self.low = self.low()
        self.prices = self.close
        self.prices = self.prices.dropna(how='all', axis=1)
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            print("Index is not datetime - converting...")
            try:
                self.prices.index = pd.to_datetime(self.prices.index, utc=True)
                print("Successfully converted index to DatetimeIndex")
            except Exception as e:
                print(f"Could not convert index to datetime: {e}")
        else:
            print("Index is already datetime")

        if 'mo' in interval:
            self.prices.index = self.prices.index.strftime('%Y-%m')
        elif 'wk' in interval:
            self.prices.index = self.prices.index.strftime('%Y-%m-%d')
        elif 'd' in interval:
            self.prices.index = self.prices.index.strftime('%Y-%m-%d')
        elif 'h' in interval:
            self.prices.index = self.prices.index.strftime('%Y-%m-%d %H')
        elif 'm' in interval:
            self.prices.index = self.prices.index.strftime('%Y-%m-%d %H:%M')
        
        self.save_path = save_path
        self.series_type = ''
        if not save_path:
            self.save_path = save_path
        elif save_path.endswith('/'):
            self.save_path = save_path
        else:
            self.save_path = save_path + '/'

    def dropna(self, how='any', axis=0):
        self.prices = self.prices.dropna(how=how, axis=axis)
        return self
    def fillna(self, value=0):
        self.prices = self.prices.fillna(value)
        return self
    def cleanna(self):
        self.prices = self.prices.dropna(how='all', axis=1).dropna()
        return self
    def standardize(self):        
        # 各数値列を標準化
        prices_copy = self.prices.copy()
        for column in self.prices.columns:
            mean = self.prices[column].mean()
            std = self.prices[column].std()
            # 標準偏差が0の場合は除算をスキップ（すべての値が同じ場合）
            if std != 0:
                prices_copy[column] = (prices_copy[column] - mean) / std
            else:
                prices_copy[column] = 0  # すべての値を0に設定（平均が0になるため）
        self.prices = prices_copy
        return self

    def log(self):
        index = self.prices.index
        self.prices = self.prices.apply(np.log)
        self.prices.index = index
        self.series_type += '_log'
        return self
    def diff(self):
        self.prices = self.prices.diff().dropna()
        self.series_type += '_diff'
        return self
    def log_diff(self):
        index = self.prices.index
        self.prices = self.prices.apply(np.log)
        self.prices.index = index
        self.prices = self.prices.diff()
        self.series_type += '_log_diff'
        return self
    
    def set_close(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker]['Close']
        return self
    def set_open(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker]['Open']
        return self
    def set_high(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker]['High']
        return self
    def set_low(self):
        for ticker in self.tickers:
            self.prices[ticker] = self.data[ticker]['Low']
        return self
    
    def close(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Close']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def open(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Open']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def high(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['High']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
    def low(self):
        df = pd.DataFrame()
        for ticker in self.tickers:
            series = self.data[ticker]['Low']
            series.name = ticker 
            df = df.join(series, how='outer')
        df = df.dropna(how='all', axis=1)
        self.tickers = df.columns
        return df
            
    def avg(self):
        return self.prices.mean().item()
    def std(self):
        return self.prices.std().item()
    def maxlen(self):
        maxlen = 0
        for ticker in self.tickers:
            if maxlen < len(self.history[ticker]):
                maxlen = len(self.history[ticker])
        return maxlen

    
    def plot(self, tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue

            plt.figure(figsize=(12, 6))
            
            plt.plot(self.close[ticker], label='Close Price', color='blue', linewidth=2)
            plt.plot(self.high[ticker], label='High Price', color='red', linestyle='--')
            plt.plot(self.low[ticker], label='Low Price', color='green', linestyle='--')
            plt.plot(self.open[ticker], label='Open Price', color='orange', linestyle=':')
            
            plt.title(f'{ticker} Stock Price (Open, High, Low, Close)')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            plt.show()
            plt.close()

    def AuotCorrelation(self, lag: int=None, tickers: list=None, save_path=None) -> pd.DataFrame:
        if tickers == None:
            tickers = self.tickers
        maxlen = self.maxlen()
        if lag == None or maxlen < lag:
            lag = maxlen-1
        autocorr_df = pd.DataFrame()
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue
            autocorr = self.prices[ticker].pct_change().dropna()
            autocorr.name = ticker
            acf_values = acf(autocorr, nlags=lag, fft=False)
            series = pd.Series(acf_values, name=ticker)
            # autocorr_df = autocorr_df.join(series, how='outer')
            autocorr_df = pd.concat([autocorr_df, series], axis=1)
            # print(autocorr_df)
        if save_path:
            os.makedirs(f"{save_path}{self.prices.index[-1]}", exist_ok=True)
            autocorr_df.to_csv(f"{save_path}{self.prices.index[-1]}/autocorrelation{self.series_type}.csv")
        elif self.save_path:
            os.makedirs(f"{self.save_path}{self.prices.index[-1]}", exist_ok=True)
            autocorr_df.to_csv(f"{self.save_path}{self.prices.index[-1]}/autocorrelation{self.series_type}.csv")
        return autocorr_df
    
    def ADF(self, save_path: str=None):
        adf_df = pd.DataFrame()
        for ticker in self.tickers:
            adf_series = pd.Series()
            adf_stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(self.prices[ticker].dropna())
            adf_series['adf_stat'] = adf_stat
            adf_series['p_value'] = p_value
            adf_series['used_lag'] = used_lag
            adf_series['n_obs'] = n_obs
            adf_series['critical_value_1%'] = critical_values['1%']
            adf_series['critical_value_5%'] = critical_values['5%']
            adf_series['critical_value_10%'] = critical_values['10%']
            adf_df[ticker] = adf_series
        if save_path:
            os.makedirs(f"{save_path}{self.prices.index[-1]}", exist_ok=True)
            adf_df.to_csv(f"{save_path}{self.prices.index[-1]}/ADF{self.series_type}.csv")
        elif self.save_path:
            os.makedirs(f"{self.save_path}{self.prices.index[-1]}", exist_ok=True)
            adf_df.to_csv(f"{self.save_path}{self.prices.index[-1]}/ADF{self.series_type}.csv")
        return adf_df
    
    def VARresults(self, maxlags: int=None, save_path: str=None) -> dict:
        maxlen = len(self.prices)
        if maxlags == None or maxlen < maxlags:
            maxlags = maxlen-1
        model = VAR(self.prices)
        results = model.fit(maxlags=maxlags)
        var_dict = {}
        var_dict['params'] = results.params       
        var_dict['tvalues'] = results.tvalues
        var_dict['pvalues'] = results.pvalues
        var_dict['resid'] = results.resid
        var_dict['sigma_u'] = results.sigma_u

        print("latest date", self.prices.index[-1])
        if save_path:
            os.makedirs(f"{save_path}{self.prices.index[-1]}", exist_ok=True)
            for key, df in var_dict.items():
                df.to_csv(f"{save_path}{self.prices.index[-1]}/VAR{self.series_type}_{key}.csv")
        elif self.save_path:
            os.makedirs(f"{self.save_path}{self.prices.index[-1]}", exist_ok=True)
            for key, df in var_dict.items():
                df.to_csv(f"{self.save_path}{self.prices.index[-1]}/VAR{self.series_type}_{key}.csv")
        return var_dict

    def VARranking(self, maxlags: int=None, save_path: str=None):
        try:
            var_results = self.VARresults(maxlags=maxlags, save_path=save_path)
            print("successfully get VAR results!")
        except Exception as e:
            print(e)
            return

        ranking_df = pd.DataFrame()
        
        params_df = get_values_ranking_df(var_results['params'].drop('const')).dropna()
        ranking_df = params_df
        ranking_df = ranking_df.rename(columns={'value': 'params'})

        tvalues_df = get_values_ranking_df(var_results['tvalues'].drop('const')).dropna()
        ranking_df = pd.merge(ranking_df, tvalues_df, on=['row', 'column'])
        ranking_df = ranking_df.rename(columns={'value': 'tvalues'})

        pvalues_df = get_values_ranking_df(var_results['pvalues'].drop('const')).dropna()
        ranking_df = pd.merge(ranking_df, pvalues_df, on=['row', 'column'])
        ranking_df = ranking_df.rename(columns={'value': 'pvalues'})

        ranking_df['lag'] = ranking_df['row']//len(ranking_df.columns)
        ranking_df['ticker'] = ranking_df['row']%len(ranking_df.columns)

        return ranking_df
            
    def VARcompare(self, maxlags: int=None, save_path: str=None):
        criteria_df = pd.DataFrame()
        if maxlags == None or maxlags > len(self.prices):
            maxlags = len(self.prices)
        for lag in range(maxlags):
            try:
                model = VAR(self.prices)
                results = model.fit(maxlags=lag+1)
                criteria_series = pd.Series()
                criteria_series.name = lag+1
                criteria_series['aic'] = results.aic
                criteria_series['bic'] = results.bic
                criteria_series['hqic'] = results.hqic
                criteria_series['fpe'] = results.fpe
                criteria_series['llf'] = results.llf
                criteria_series['detomega'] = results.detomega
                # criteria_df = criteria_df.join(criteria_series, how='outer')
                criteria_df = pd.concat([criteria_df, criteria_series.to_frame().T])
            except:
                break
        if save_path:
            os.makedirs(f"{save_path}{self.prices.index[-1]}", exist_ok=True)
            criteria_df.to_csv(f"{save_path}{self.prices.index[-1]}/VAR{self.series_type}_criteria.csv")
        elif self.save_path:
            os.makedirs(f"{self.save_path}{self.prices.index[-1]}", exist_ok=True)
            criteria_df.to_csv(f"{self.save_path}{self.prices.index[-1]}/VAR{self.series_type}_criteria.csv")
        return criteria_df

    def FEVD(self, maxlags: int=None, save_path: str=None):
        try:
            maxlen = len(self.prices)
            if maxlags == None or maxlen < maxlags:
                maxlags = maxlen-1
            model = VAR(self.prices)
            results = model.fit(maxlags=maxlags)
            fevd = results.fevd(maxlags)  # 10 is the number of steps ahead
            fevd.plot()
            return
        except Exception as e:
            print(e)
            return
    
    def VAR(self, maxlags: int=None, save_path: str=None):
        print('ADF')
        try:
            adf = self.ADF(save_path=save_path)
            print(adf)
        except Exception as e:
            print(e)
        
        try:
            var_results = self.VARresults(maxlags=maxlags, save_path=save_path)
            print("successfully get VAR results!")
        except Exception as e:
            print(e)
            return
        
        try:
            ranking_df = self.VARranking(maxlags=maxlags)
            H0 = ranking_df[ranking_df['pvalues'] > 0.05]
            H1 = ranking_df[ranking_df['pvalues'] < 0.05]
            print('H0 (coef = 0)', len(H0))
            print(len(H0))
            print('H1 (coef != 0)', len(H1))
            print(len(H1))
            print('ratio', len(H1)/(len(H0) + len(H1)))
            print(H1.sort_values('params').tail(6))
            print(H1.sort_values('params').head(6))
            print("successfully get VAR ranking!")
        except Exception as e:
            print(e)
            return

        # print('resid')
        # resid_df = get_values_ranking_df(var_results['resid'])
        # print(resid_df)
        # # print(var_results['resid'])

        dw_statistics = durbin_watson(var_results['resid'])
        print("Durbin-Watson統計量（各変数ごと）:")
        print(dw_statistics)
        print(max(abs(dw_statistics-2)))

        print('sigma_u')
        sigma_u_df = get_values_ranking_df(var_results['sigma_u'])
        print(sigma_u_df)
        print(var_results['sigma_u'])

        # print('compare')
        # var_compare = self.VARcompare(maxlags=maxlags, save_path=save_path)
        # print(var_compare)

        print('FEVD')
        try:
            plt.figure()
            self.FEVD(maxlags=maxlags, save_path=save_path)
            plt.show()
            plt.close()
        except Exception as e:
            print(e)
        return ranking_df

    def multivariate_rmse(self, actual, predicted):
        """
        多変量時系列データのRMSEを計算
        
        Parameters:
        actual (2D array): 実際の値 (時点×変数)
        predicted (2D array): 予測値 (時点×変数)
        
        Returns:
        float: 全変数・全時点を考慮したRMSE
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        
        if actual.shape != predicted.shape:
            raise ValueError("actualとpredictedの形状が一致しません")
        
        # 全要素の二乗誤差を計算
        squared_errors = (actual - predicted) ** 2
        
        # 全要素の平均を取って平方根
        rmse = np.sqrt(np.mean(squared_errors))
        
        return rmse

    def VARcv(self, window_size = 140,test_size = 7, step_size = 7, maxlags=3):
        # 結果を保存するリスト
        rmse_scores = []
        n_windows = (len(self.prices) - window_size - test_size) // step_size + 1
        for i in range(n_windows):
            start_idx = i * step_size
            train_data = self.prices.iloc[start_idx:start_idx + window_size]
            test_data = self.prices.iloc[start_idx + 1:start_idx + 1 + window_size + test_size]
            
            # VARモデルの学習
            model = VAR(train_data)
            results = model.fit(maxlags=maxlags, ic='aic')  # AICで最適なラグを選択
            
            forecast = results.forecast(test_data.values, steps=len(test_data))
            
            # 予測誤差（RMSE）を計算
            rmse = self.multivariate_rmse(test_data.values, forecast)
            rmse_scores.append(rmse)
            print(f"Window {i+1}: RMSE = {rmse:.4f}")
        
        return rmse_scores
    
    def VARauto(self, maxlags, fill=False):
        print("prices")
        os.makedirs(f"{self.save_path}{self.prices.index[-1]}", exist_ok=True)
        for ticker in self.tickers:
            plt.figure()
            self.prices[ticker].plot()
            plt.title(ticker)
            name = ticker.replace(".T", "")
            plt.savefig(f"{self.save_path}{self.prices.index[-1]}/{name}_prices.png")
            plt.show()
            plt.close()
        autcorr = self.AuotCorrelation(lag=maxlags)
        autcorr_ranking = get_values_ranking_df(autcorr.iloc[1:]).dropna()
        print("autocorr")
        print(autcorr_ranking.head(6))
        print(autcorr_ranking.tail(6))
        print(autcorr.iloc[1:].std())
        # print(autcorr)
        for ticker in self.tickers:
            plot_acf(autcorr[ticker], lags=maxlags)
            # pd.plotting.autocorrelation_plot(autcorr[ticker].dropna())
            plt.title(ticker)
            name = ticker.replace(".T", "")
            plt.savefig(f"{self.save_path}{self.prices.index[-1]}/{name}_autocorr.png")
            plt.show()
            plt.close()

        print("VAR")
        try:
            if fill:
                self.standardize().log_diff().fillna()
            else:
                self.standardize().log_diff().cleanna()
            self.VAR(maxlags=maxlags)
        except Exception as e:
            print(e)

        plt.figure()
        self.prices.std().plot(kind='bar')
        # plt.title(ticker)
        # name = ticker.replace(".T", "")
        # plt.savefig(f"{self.save_path}{self.prices.index[-1]}/{name}_prices_log_diff.png")
        plt.show()
        plt.close()

        for ticker in self.tickers:
            plt.figure()
            self.prices[ticker].plot()
            plt.title(ticker)
            name = ticker.replace(".T", "")
            plt.savefig(f"{self.save_path}{self.prices.index[-1]}/{name}_prices_log_diff.png")
            plt.show()
            plt.close()
        print('mse')
        try:
            mse = self.VARcv(7)
            print(mse)
        except Exception as e:
            print(e)
        return

    def CandleStick(self, day=200, moving_average: ListAndStr=[5, 10, 20 ,50, 75, 100], tickers: ListAndStr=None):
        if tickers == None:
            tickers = self.tickers
        for ticker in tickers:
            if not ticker in self.tickers:
                print(f"{ticker} does not exist")
                continue
            # prepare history
            candlestick_history = self.history[ticker].copy()
            candlestick_history = candlestick_history.reset_index()
            candlestick_history['Date'] = candlestick_history['Date'].map(mdates.date2num)
            candlestick_history = candlestick_history[-day:]

            fig, ax = plt.subplots(figsize=(12, 6))
            # illustrate a candlestick
            candlestick_ohlc(ax, candlestick_history[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                            width=1, colorup='g', colordown='r')
            # add a moving average
            for ma in moving_average:
                candlestick_history[f'MA{ma}'] = candlestick_history['Close'].rolling(ma).mean()
                ax.plot(candlestick_history['Date'], candlestick_history[f'MA{ma}'], label=f'{ma} day moving average')

            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.title(f'{ticker} candlestick')
            plt.xlabel('Date')
            plt.ylabel('price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.close()

class Option(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.options = self.get_attribute('options')

class Holder(Stock):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.major_holders = self.get_attribute('major_holders')
        self.institutional_holders = self.get_attribute('institutional_holders')
        self.mutualfund_holders = self.get_attribute('mutualfund_holders')

class Currency(Stock):
    def __init__(self, tickers: ListAndStr, period: str='max', interval: str='1d', save_path: str=None):
        super().__init__(tickers, period, interval, save_path)

class Finance(Tickers):
    def __init__(self, tickers, quarterly=False):
        super().__init__(tickers)
        self.balance_sheet = self.get_attribute('balance_sheet')    
        self.income_stmt = self.get_attribute('income_stmt')    
        self.cashflow = self.get_attribute('cashflow')
        if quarterly:
            self.balance_sheet = self.get_attribute('quarterly_balance_sheet')    
            self.income_stmt = self.get_attribute('quarterly_income_stmt')    
            self.cashflow = self.get_attribute('quarterly_cashflow')
        for ticker in self.tickers:
            self.balance_sheet[ticker] = self.balance_sheet[ticker].T
            self.income_stmt[ticker] = self.income_stmt[ticker].T
            self.cashflow[ticker] = self.cashflow[ticker].T

    def get_indices(self, ticker):
        # 空のDataFrame作成
        indeces = pd.DataFrame()
        
        try:
            BS = self.balance_sheet[ticker]
            PL = self.income_stmt[ticker]
            
            # 安全にデータを取得するヘルパー関数
            def get_safe(df, col):
                return df[col] if col in df.columns else pd.Series(dtype='float64')
            
            # 必要なデータを取得
            total_revenue = get_safe(PL, 'Total Revenue')
            cost_of_revenue = get_safe(PL, 'Cost Of Revenue')
            net_income = get_safe(PL, 'Net Income')
            inventory = get_safe(BS, 'Inventory')
            stockholders_equity = get_safe(BS, 'Stockholders Equity')
            treasury_shares = get_safe(BS, 'Treasury Shares Number')
            shares_issued = get_safe(BS, 'Share Issued')
            
            # 財務指標計算 (ゼロ除算を避けるためnp.where使用)
            indeces['Gross Profit'] = total_revenue - cost_of_revenue
            
            with np.errstate(divide='ignore', invalid='ignore'):
                indeces['Cost Of Revenue Ratio'] = np.where(total_revenue != 0, cost_of_revenue / total_revenue, np.nan)
                indeces['Inventory Turnover'] = np.where(inventory != 0, cost_of_revenue / inventory, np.nan)
                indeces['ROE'] = np.where(stockholders_equity != 0, net_income / stockholders_equity, np.nan)
                indeces['Treasury Stock Ratio'] = np.where(shares_issued != 0, treasury_shares / shares_issued, np.nan)
            
            # インデックスを設定
            if not indeces.empty:
                indeces.index = PL.index if not PL.empty else BS.index
                
        except KeyError as e:
            print(f"Error: {e} not found for ticker {ticker}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
        
        return indeces

class Insider(Tickers):
    def __init__(self, tickers):
        super().__init__(tickers)
        self.insider_purchases = self.get_attribute('insider_purchases')    
        self.insider_roster_holders = self.get_attribute('insider_roster_holders')    
        self.insider_transactions = self.get_attribute('insider_transactions')