# scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my library
from ... import ListAndStr
from . import matrix

# statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson


class VAR():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns

    def ADF(self) -> pd.DataFrame:
        adf_df = pd.DataFrame()
        for column in self.columns:
            adf_series = pd.Series()
            adf_stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(self.df[column].dropna())
            adf_series['adf_stat'] = adf_stat
            adf_series['p_value'] = p_value
            adf_series['used_lag'] = used_lag
            adf_series['n_obs'] = n_obs
            adf_series['critical_value_1%'] = critical_values['1%']
            adf_series['critical_value_5%'] = critical_values['5%']
            adf_series['critical_value_10%'] = critical_values['10%']
            adf_df[column] = adf_series
        return adf_df
    
    def results(self, maxlags: int=None) -> dict:
        maxlen = len(self.df)
        if maxlags == None or maxlen < maxlags:
            maxlags = maxlen-1
        model = VAR(self.df)
        results = model.fit(maxlags=maxlags)
        var_dict = {}
        var_dict['params'] = results.params       
        var_dict['tvalues'] = results.tvalues
        var_dict['pvalues'] = results.pvalues
        var_dict['resid'] = results.resid
        var_dict['sigma_u'] = results.sigma_u

        # print("latest date", self.df.index[-1])
        return var_dict

    def ranking(self, maxlags: int=None) -> pd.DataFrame:
        try:
            var_results = self.results(maxlags=maxlags)
            print("successfully get VAR results!")
        except Exception as e:
            print(e)
            return

        ranking_df = pd.DataFrame()
        
        params_df = matrix.get_values_ranking_df(var_results['params'].drop('const')).dropna()
        ranking_df = params_df
        ranking_df = ranking_df.rename(columns={'value': 'params'})

        tvalues_df = matrix.get_values_ranking_df(var_results['tvalues'].drop('const')).dropna()
        ranking_df = pd.merge(ranking_df, tvalues_df, on=['row', 'column'])
        ranking_df = ranking_df.rename(columns={'value': 'tvalues'})

        pvalues_df = matrix.get_values_ranking_df(var_results['pvalues'].drop('const')).dropna()
        ranking_df = pd.merge(ranking_df, pvalues_df, on=['row', 'column'])
        ranking_df = ranking_df.rename(columns={'value': 'pvalues'})

        ranking_df['lag'] = ranking_df['row']//len(ranking_df.columns)
        ranking_df['column'] = ranking_df['row']%len(ranking_df.columns)

        return ranking_df
            
    def compare(self, maxlags: int=None):
        criteria_df = pd.DataFrame()
        if maxlags == None or maxlags > len(self.df):
            maxlags = len(self.df)
        for lag in range(maxlags):
            try:
                model = VAR(self.df)
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
        return criteria_df

    def FEVD(self, maxlags: int=None):
        try:
            maxlen = len(self.df)
            if maxlags == None or maxlen < maxlags:
                maxlags = maxlen-1
            model = VAR(self.df)
            results = model.fit(maxlags=maxlags)
            fevd = results.fevd(maxlags)  # 10 is the number of steps ahead
            fevd.plot()
            return
        except Exception as e:
            print(e)
            return
    
    def VAR(self, maxlags: int):
        print('ADF')
        try:
            adf = self.ADF()
            print(adf)
        except Exception as e:
            print(e)
        
        try:
            var_results = self.results(maxlags=maxlags)
            print("successfully get VAR results!")
        except Exception as e:
            print(e)
            return

        # dw_statistics = durbin_watson(var_results['resid'])
        # print("Durbin-Watson統計量（各変数ごと）:")
        # print(dw_statistics)
        # print(max(abs(dw_statistics-2)))

        # print('sigma_u')
        # sigma_u_df = get_values_ranking_df(var_results['sigma_u'])
        # print(sigma_u_df)
        # print(var_results['sigma_u'])

        print('compare')
        var_compare = self.compare(maxlags=maxlags)
        print(var_compare)

        print('FEVD')
        try:
            plt.figure()
            self.FEVD(maxlags=maxlags)
            plt.show()
            plt.close()
        except Exception as e:
            print(e)
        return

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

    def CrossValidation(self, window_size = 140, test_size = 7, step_size = 7, maxlags=3):
        # 結果を保存するリスト
        rmse_scores = []
        n_windows = (len(self.df) - window_size - test_size) // step_size + 1
        for i in range(n_windows):
            start_idx = i * step_size
            train_data = self.df.iloc[start_idx:start_idx + window_size]
            test_data = self.df.iloc[start_idx + 1:start_idx + 1 + window_size + test_size]
            
            # VARモデルの学習
            model = VAR(train_data)
            results = model.fit(maxlags=maxlags, ic='aic')  # AICで最適なラグを選択
            
            forecast = results.forecast(test_data.values, steps=len(test_data))
            
            # 予測誤差（RMSE）を計算
            rmse = self.multivariate_rmse(test_data.values, forecast)
            rmse_scores.append(rmse)
            print(f"Window {i+1}: RMSE = {rmse:.4f}")
        
        return rmse_scores