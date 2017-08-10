"""统计模块"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import as_strided

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as col

import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体
mpl.rcParams['axes.unicode_minus'] =False # 解决保存图像是负号'-'显示为方块的问题
#sns.axes_style()，可以看到是否成功设定字体为微软雅黑。

sns.set_context("talk")


# def cross_corr(y: pd.Series, x: pd.Series, max_lag=12):
#     """计算时差相关系数"""
#     assert y.count() == len(y) and x.count() == len(x), "There are Nall values in y or x!"
#     cross_corr = []
#     corr_p_value = []
#     for lag in range(max_lag+1):
#         tmp_df = pd.DataFrame([])
#         tmp_df['x'] = x.shift(lag)
#         tmp_df['y'] = y
#         tmp_df = tmp_df.dropna()
#         # tmp_corr = np.corrcoef(tmp_df.x, tmp_df.y)[0][1]
#         tmp_corr, tmp_p = stats.pearsonr(tmp_df.x, tmp_df.y)
#         cross_corr.append(tmp_corr)
#         corr_p_value.append(tmp_p)
#
#     cross_corr = pd.Series(cross_corr, index=range(max_lag+1))
#     corr_p_value = pd.Series(corr_p_value, index=range(max_lag + 1))
#
#     return cross_corr, corr_p_value


def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x


def autocorrelation(x, maxlag):
    """
    Autocorrelation with a maximum number of lags.

    `x` must be a one-dimensional numpy array.

    This computes the same result as
        numpy.correlate(x, x, mode='full')[len(x)-1:len(x)+maxlag]

    The return value has length maxlag + 1.
    """
    x = _check_arg(x, 'x')
    p = np.pad(x.conj(), maxlag, mode='constant')
    T = as_strided(p[maxlag:], shape=(maxlag+1, len(x) + maxlag),
                   strides=(-p.strides[0], p.strides[0]))


    return T.dot(p[maxlag:].conj())


def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2 * maxlag, mode='constant')
    T = as_strided(py[2 * maxlag:], shape=(2 * maxlag + 1, len(y) + 2 * maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def plot_cross_corr(y: pd.Series, x: pd.Series, max_lag=12):


    # 相关系数的统计检验、画图
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.xcorr(x, y, usevlines=True, maxlags=max_lag, normed=True)
    # ax1.grid(True)
    # ax1.axhline(0, color='black')
    lags, c, _, _ = plt.xcorr(x=x, y=y, usevlines=True, maxlags=max_lag, normed=True)
    plt.scatter(lags, c, marker='o')
    # plt.xlabel('Lags of {0} corresponding to {1}'.format(x.name, y.name))
    # plt.title('Cross Correlation Coefficients')
    plt.xlabel('{0} 相对于 {1} 的滞后期'.format(x.name, y.name))
    plt.ylim(-1,1)
    plt.title('时差相关系数')
    # plt.show()

    return lags, c


def cal_spe(predict_value: pd.Series, true_value: pd.Series):
    """计算预测误差"""
    assert predict_value.index == true_value.index, "预测值和真实值的index不相同！"
    assert predict_value.isnull.sum() == 0, "预测值有缺失"
    assert true_value.isnull.sum() == 0, "真实值有缺失"

    return ((predict_value - true_value) ** 2).sum()


def cal_cspe(predict_value: pd.Series, true_value: pd.Series, pre_mean:pd.Series,
             window_size, cspe_plot=False, rolling=False, title=None):
    """cspe画图"""

    assert len(predict_value) == len(true_value), "The index of predict value and true value are not match!"
    assert len(pre_mean) == len(true_value), "The index of pre mean value and true value are not match!"

    # 1. cspe画图
    daily_se_predict = (predict_value - true_value)**2

    # 滚动计算均值
    #pre_mean = true_value.rolling(window=window_size).mean().shift(1)
    daily_se_mean = (pre_mean - true_value)**2

    # 从每日的预测平方差计算累计的预测平方差(移动平均的预测均误超过预测值的累计值)
    daily_se_vs =(daily_se_mean - daily_se_predict).dropna()
    if rolling == True:
        cspe = daily_se_vs.rolling(window=window_size, center=False).sum()
    else:
        cspe = daily_se_vs.cumsum()

    # 画图
    if cspe_plot is True:
        # cspe.plot(title='Cumulative Squared Prediciton Error')
        if title is None:
            tt = '相对预测均方误差累计'
        else:
            tt = title

        cspe.plot(title=tt)
        plt.axhline(0, color='black')
        plt.show()

    return cspe, daily_se_vs


def oos_test(predict_value: pd.Series, true_value: pd.Series, pre_mean:pd.Series,
             window_size, sig_level=0.1):
    """样本外test"""

    assert len(predict_value) == len(true_value), "The index of predict value and true value are not match!"
    assert len(pre_mean) == len(true_value), "The index of pre mean value and true value are not match!"

    # 1. cspe画图

    daily_se_predict = (predict_value - true_value)**2

    # 滚动计算均值
    #pre_mean = true_value.rolling(window=window_size).mean().shift(1)
    daily_se_mean = (pre_mean - true_value)**2

    # # 从每日的预测平方差计算累计的预测平方差(移动平均的预测均误超过预测值的累计值)
    # daily_se_vs =(daily_se_mean - daily_se_predict).dropna()
    # if rolling == True:
    #     cspe = daily_se_vs.rolling(window=window_size, center=False).sum()
    # else:
    #     cspe = daily_se_vs.cumsum()
    #
    # # 画图
    # if cspe_plot is True:
    #     # cspe.plot(title='Cumulative Squared Prediciton Error')
    #     if title is None:
    #         tt = '相对预测均方误差累计'
    #     else:
    #         tt = title
    #     cspe.plot(title=tt)
    #     plt.axhline(0, color='black')
    #     plt.show()

    # 2. 计算oos_r2
    r_os = 1 - (daily_se_predict.sum()) / (daily_se_mean.sum())

    if r_os < 0:
        return r_os, np.NaN, False

    # McCracken 2004 F test
    f_statistic = (len(true_value) - window_size + 1)*(daily_se_mean.sum() - daily_se_predict.sum()) / (daily_se_mean.sum())
    f_critical_value = stats.f.cdf(1-sig_level, 1, len(true_value) - window_size + 1)
    if f_statistic>f_critical_value:
        sig = True
    else:
        sig = False

    return r_os, f_statistic, sig



def plot_corr(corr_df: pd.DataFrame, add_text=True, fig_size=(12, 15)):
    data = corr_df.values
    label_ticks = corr_df.columns
    picture = DrawPcolor()
    picture.Pcolor(data, AddText=add_text, fig_size=fig_size)
    picture.Set_labelticks(label_ticks, label_ticks)




class DrawPcolor(object):
    """
    相关系数画图

    Example

def Main():
    data = np.random.rand(5, 4)
    Corr = np.corrcoef(data)
    xlabel_ticks=["a",'b','c','d','e']#range(Corr.shape[0]);#
    ylabel_ticks=["A","B","C",'D','E']#range(Corr.shape[1]);#
    MyPict=DrawPcolor()
    MyPict.Pcolor(Corr,3,4,4,AddText=True,b=1) # params 3,4,4,b has no effect on the exec
    MyPict.Set_labelticks(xlabel_ticks,ylabel_ticks)

if __name__=='__main__':
    Main()
    """
    def __init__(self):
        ## self define the colorbar
        startcolor = '#006400'  # a dark green
        midcolor = '#ffffff'    # a bright white
        endcolor = '#ee0000'    # a dark red
        self.Mycmap = col.LinearSegmentedColormap.from_list('MyColorbar', [startcolor, midcolor,
                                                                           endcolor])  # use the "fromList() method

    def Pcolor(self, data, AddText=True, size=(12,10),*args, **kwargs):
        # *args is a tuple,**kwargs is a dict;
        # Here args means the Matrix Corr,kwargs includes the key of the " AddText function"
        self.fig = plt.figure(figsize=size)
        self.ax = self.fig.add_subplot(111)
        self.Data = data
        heatmap = self.ax.pcolor(self.Data, cmap=self.Mycmap, alpha=0.8, vmin=-1, vmax=1)  # cmap=plt.cm.Reds)
        self.fig.colorbar(heatmap)
        # want a more natural, table-like display
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.set_xticks(np.arange(self.Data.shape[0]) + 0.5, minor=False)
        self.ax.set_yticks(np.arange(self.Data.shape[1]) + 0.5, minor=False)

        if AddText == True:
            for y in range(self.Data.shape[1]):
                for x in range(self.Data.shape[0]):
                    self.ax.text(x + 0.5, y + 0.5, '%.2f' % self.Data[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 )
        self.fig.show()

    def Set_labelticks(self, *tick_labels):
        # put the major ticks at the middle of each cell
        self.ax.set_xticklabels(tick_labels[0], rotation=90, minor=False, ha='left')
        self.ax.set_yticklabels(tick_labels[1], rotation=0, minor=False)
        self.fig.show()



if __name__ == "__main__":


    db = pd.HDFStore('data.h5')
    asset_ind = db['asset_ind']
    macro_ind = db['macro_ind']

    pmi = macro_ind['财新中国PMI'].dropna()
    hs300 = asset_ind['沪深300指数'].dropna()

    # hs月度的开高低收
    hs300_c = hs300.resample('m').last()
    hs300_o = hs300.resample('m').first()
    hs300_h = hs300.resample('m').max()
    hs300_l = hs300.resample('m').min()

    hs300_ret = hs300_c.pct_change()
    # hs300_curet_max = (hs300_h - hs300_o) / hs300_o
    # hs300_curet_min = (hs300_l - hs300_o) / hs300_o

    # print(hs300_ret.head())
    # print(hs300_curet_max.head())
    # print(hs300_curet_min.head())

    # hs300_ret.plot()
    # hs300_curet_max.plot()
    # hs300_curet_min.plot()
    # plt.show()

    df = pd.DataFrame([])
    df['pmi_p'] = pmi.pct_change()
    df['hs300_ret'] = hs300_ret
    df = df.dropna()

    # # 自相关系数
    # cro_corr, cro_p = cross_corr(df.hs300_ret, df.pmi_p, plot=True)
    # print(cro_corr, cro_p)
    # df.index.name = 'Time'
    # print(df.head())


    # ols = pd.ols(y=df.hs300_ret, x=df.pmi_p, nw_lags=1, window_type='rolling', window=24)
    # print(ols)

    pre_mean = df.hs300_ret.rolling(window=12).mean().shift(1) # 前12期的移动平均
    # pre_mean = df.hs300_ret.rolling(window=2).mean() # 用未来数据做测试

    df['pre_mean'] = pre_mean
    df['predict_mean'] = df.hs300_ret.ewm(halflife=12).mean().shift(1)
    df = df.dropna()
    print(df)

    a= oos_test(pre_mean=df.pre_mean, predict_value=df.predict_mean, true_value=df.hs300_ret, window_size=12, sig_level=0.1)
    cal_cspe(pre_mean=df.pre_mean, predict_value=df.predict_mean, true_value=df.hs300_ret, window_size=12,cspe_plot=True)
    print(a)

    # # 相关系数画图
    # def Main():
    #     data = np.random.rand(5, 4)
    #     Corr = np.corrcoef(data)
    #     xlabel_ticks = ["a", 'b', 'c', 'd', 'e']  # range(Corr.shape[0]);#
    #     ylabel_ticks = ["A", "B", "C", 'D', 'E']  # range(Corr.shape[1]);#
    #     MyPict = DrawPcolor()
    #     MyPict.Pcolor(Corr, AddText=True)
    #     MyPict.Set_labelticks(xlabel_ticks, ylabel_ticks)
    #     plt.show()
    #
    #
    # Main()