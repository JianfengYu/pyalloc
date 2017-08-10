""""""""""
" 用做简单线性预测
"""""""""""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt



def seasonal_adj(s: pd.Series, plot=False):
    res = sm.tsa.x13_arima_analysis(s)
    if plot:
        res.plot()

    return res.seasadj, res.trend, res.irregular


def rolling_ols(formula: str, data: pd.DataFrame, window: int, r2_adj=False, expanding=False, robust=False, M=sm.robust.norms.AndrewWave()):

    para_res = {}
    r_2_res = {}
    model_sig = {}
    forcast_res = pd.Series([])

    for i in range(len(data) - window + 1):

        if expanding:
            start_index = 0
        else:
            start_index = i

        tmp_df = data.iloc[start_index: i + window]
        forcast_x = data.iloc[i + window: i + window + 1]

        if robust:
            rlm_model = smf.rlm(formula, data=tmp_df, M=M)
            ols_result = smf.wls(formula, data=tmp_df,
                                 weights=rlm_model.fit().weights).fit()
            # ols_result = sm.WLS(rlm_model.endog, rlm_model.exog,
            #                     weights=rlm_model.fit().weights).fit()
        else:
            ols_result = smf.ols(formula, data=tmp_df).fit()

        para_res[data.index[i + window - 1]] = ols_result.params
        model_sig[data.index[i + window - 1]] = ols_result.f_pvalue

        if r2_adj:
            r_2_res[data.index[i + window - 1]] = ols_result.rsquared_adj
        else:
            r_2_res[data.index[i + window - 1]] = ols_result.rsquared

        # 一步预测
        forcast_res = forcast_res.append(ols_result.predict(forcast_x))

    para_res = pd.DataFrame(para_res).T
    r_2_res = pd.Series(r_2_res)
    model_sig = pd.Series(model_sig)

    return para_res, r_2_res.mean(), model_sig, forcast_res


def rolling_pca_reg(endog, exog, select_eig_num:int, window: int, expanding = False, r2_adj=False, robust=False, M=sm.robust.norms.AndrewWave()):

    assert exog.shape[1] >= select_eig_num, 'The dim of vars are less than the selecet eig number!'

    para_res = {}
    r_2_res = {}
    model_sig = {}
    forcast_res = pd.Series([])
    pca_r2 = []

    for i in range(len(endog) - window + 1):

        if expanding:
            start_index = 0
        else:
            start_index  = i


        tmp_endog = endog.iloc[start_index: i + window]
        tmp_exog = exog.iloc[start_index: i + window]
        forcast_x = exog.iloc[i + window: i + window + 1].values

        tmp_pca = PCA(tmp_exog.values, select_num=select_eig_num)
        tmp_pca.fit()
        pca_r2.append(tmp_pca.r_2)

        if robust:
            rlm_model = sm.RLM(endog=tmp_endog, exog=sm.add_constant(tmp_pca.lowDData), M=M)
            ols_result = sm.WLS(endog=rlm_model.endog, exog=rlm_model.exog,
                                 weights=rlm_model.fit().weights).fit()
            # ols_result = sm.WLS(rlm_model.endog, rlm_model.exog,
            #                     weights=rlm_model.fit().weights).fit()
        else:
            ols_result = smf.OLS(endog=tmp_endog, exog=sm.add_constant(tmp_pca.lowDData)).fit()

        para_res[endog.index[i + window - 1]] = ols_result.params
        model_sig[endog.index[i + window - 1]] = ols_result.f_pvalue

        if r2_adj:
            r_2_res[endog.index[i + window - 1]] = ols_result.rsquared_adj
        else:
            r_2_res[endog.index[i + window - 1]] = ols_result.rsquared

        # 一步预测
        if len(forcast_x) < 1:
            continue

        new_obs_projection = tmp_pca.projection(forcast_x).tolist()

        new_obs_projection = new_obs_projection[0][::-1]
        new_obs_projection.append(1)
        new_obs_projection = new_obs_projection[::-1]

        forcast_res[endog.index[i + window - 1]] = np.sum(np.array(ols_result.params) * np.array(new_obs_projection))

    para_res = pd.DataFrame(para_res).T
    r_2_res = pd.Series(r_2_res)
    model_sig = pd.Series(model_sig)

    return para_res, r_2_res.mean(), model_sig, forcast_res, pca_r2


class PCA(object):
    def __init__(self, data: pd.DataFrame, select_num: int):
        self.data = data
        self.sel_n = select_num

    def fit(self):
        newData, mean_v = zero_mean(self.data)
        cov = np.cov(newData, rowvar=False)

        eigVals, eigVects = np.linalg.eig(np.mat(cov))


        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
        n_eigValIndice = eigValIndice[-1:-(self.sel_n+ 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量

        n_eigValues = [eigVals[a] for a in n_eigValIndice]
        r_2 = np.sum(n_eigValues) / np.sum(eigVals) # 前n个特征值的解释力度

        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据

        # reconMat = (lowDDataMat * n_eigVect.T) + mean_v  # 重构数据

        self.cov = cov
        self.n_eigValues = n_eigValues
        self.n_eigVactors = n_eigVect
        self.r_2 = r_2
        self.lowDData = np.real(lowDDataMat) # 计算特征值可能出现复数，主要是因为高维稀疏矩阵
        self.mean_v = mean_v

    def projection(self, sample:np.array):
        return (sample-self.mean_v) * self.n_eigVactors


def cal_pca(data: pd.DataFrame, select_num: int):
    """
    主成分分析

    Parameters
    ----------
    data
    select_num

    Returns
    -------
    tuble
        lowDDataMat, reconMat
        低维特征空间的数据, 重构数据

    """
    newData, mean_v = zero_mean(data)
    cov = np.cov(newData, rowvar=False)

    eigVals, eigVects = np.linalg.eig(np.mat(cov))
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(select_num + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + mean_v  # 重构数据
    return lowDDataMat, reconMat


def zero_mean(data):
    mean_v = np.mean(data, axis=0)
    new_data = data - mean_v
    return new_data, mean_v




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # db = pd.HDFStore('data.h5')
    # asset_ind = db['asset_ind']
    # macro_ind = db['macro_ind']
    #
    # pmi = macro_ind['财新中国PMI'].dropna()
    # cpi = macro_ind['CPI:当月同比'].dropna()
    # ppi = macro_ind['PPI:全部工业品:当月同比'].dropna()
    #
    # hs300 = asset_ind['沪深300指数'].dropna()
    #
    # # hs月度的开高低收
    # hs300_c = hs300.resample('m').last()
    # hs300_o = hs300.resample('m').first()
    # hs300_h = hs300.resample('m').max()
    # hs300_l = hs300.resample('m').min()
    #
    # hs300_ret = hs300_c.pct_change()
    #
    # df = pd.DataFrame([])
    # df['pmi_p'] = pmi.pct_change()
    # df['hs300_ret'] = hs300_ret
    # df = df.dropna()
    #
    # para, r_2, model_sig, forcast = rolling_ols('hs300_ret~pmi_p', df, window=24, robust=True)
    #
    # print(para)
    # # df.hs300_ret.plot(label='true',legend=True)
    # # forcast.plot(label='forcast', legend=True)
    # (forcast - df.hs300_ret).plot()
    # plt.show()


    # # pca test
    # data = pd.DataFrame(np.random.randn(100,12))
    # pca_test = PCA(data.values, select_num=4)
    # pca_test.fit()
    # print( pca_test.projection(np.random.randn(1, 12)))
    # print(pca_test.lowDData)
    # aaa = sm.OLS(endog=np.random.randn(100, 1), exog=pca_test.lowDData).fit()
    # print(aaa.summary())

    # seasonal adjuest test
    series = pd.Series(np.random.randn(10000), index=pd.date_range(start='1990-01-01', periods=10000))
    print(series)
    series = abs(series.resample('m').last())
    print(series)
    seasonal_adj(series, plot=True)
    plt.show()
