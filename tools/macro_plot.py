import pandas as pd
import numpy as np
import scipy as spy

import matplotlib.colors as col

from numpy.lib.stride_tricks import as_strided as strided

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style("white")

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 设定颜色
startcolor = '#006400'              # a dark green
midcolor = '#ffffff'                # a bright white
endcolor = '#ee0000'                # a dark red
grenn_to_red_cmap = col.LinearSegmentedColormap.from_list(
    'MyColorbar', [startcolor, midcolor, endcolor])


def cal_percentileofscore(s: pd.Series) -> pd.Series:
    """
    计算序列中各元素的分位数值

    Parameters
    ----------
    s: pd.Series
        原始序列

    Returns
    -------
    pd.Series
        序列中各元素的分位数值
    """
    tmp_s = s.copy()
    s_pct_score = tmp_s.apply(lambda x: spy.stats.percentileofscore(tmp_s.sort_values(), x))
    return s_pct_score


def cal_rolling_correlation(a, w):
    """cal_correlation_df"""
    n, m = a.shape[0], 2
    s1, s2 = a.strides
    b = strided(a, (m, w, n - w + 1), (s2, s1, s1))
    b_mb = b - b.mean(1, keepdims=True)
    b_ss = (b_mb ** 2).sum(1) ** .5
    return (b_mb[0] * b_mb[1]).sum(0) / (b_ss[0] * b_ss[1])


def rolling_correlation(df: pd.DataFrame, w: int) -> pd.Series:
    """
    计算滚动的相关系数

    Parameters
    ----------
    df: pd.DataFrame
        数据DataFrame，两列
    w: int
        滚动的区间长度

    Returns
    -------
    pd.Series
        滚动相关系数序列

    """
    assert len(df.dropna()) == len(df), "数据中有NaN值"
    a = df.values
    return pd.Series(cal_rolling_correlation(a, w), df.index[w - 1:])


def plot_mothly_ret_heatmap(s: pd.Series, annot=False, cmap=grenn_to_red_cmap) -> None:
    """
    画不同年月的收益热图

    Parameters
    ----------
    s: pd.Series
        收益序列
    annot: bool
        是否在图中标出数值
    cmap:
        颜色图，用matplotlib.color.LinearSegmentedColormap.from_list方法设定

    Returns
    -------
    None
    """

    # change to log return
    tmp_df = np.log(1 + s).to_frame()

    tmp_df['Year'] = tmp_df.index.year
    tmp_df['Month'] = tmp_df.index.month

    res = tmp_df.groupby(['Year', 'Month']).sum()
    res = np.exp(res) - 1
    res = res.unstack()
    res.columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    sns.heatmap(res, cmap=cmap, annot=annot)
    sns.plt.title(s.name + '收益率')
    plt.show()


def plot_corr_heatmap(df: pd.DataFrame, annot=True, cmap=grenn_to_red_cmap) -> None:
    """
    画相关系数热图

    Parameters
    ----------
    df: pd.DataFrame
        数据DataFrame，每一行为一个样本
    annot: bool
        是否需要标注数值
    cmap:
        颜色图，用matplotlib.color.LinearSegmentedColormap.from_list方法设定

    Returns
    -------

    """
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap=cmap, annot=annot, vmin=-1, vmax=1)
    plt.show()


def plot_rolling_sharpe(ret: pd.Series, window: int, min_periods: int) -> None:
    """
    画滚动的sharpe图

    Parameters
    ----------
    ret: pd.Series
        收益率序列
    window: int
        滚动区间长度
    min_periods: int
        最小数据个数

    Returns
    -------

    """
    rolling_sharpe = ret.rolling(window=window, min_periods=min_periods).mean() / \
                     ret.rolling(window=window, min_periods=min_periods).std()
    rolling_sharpe = rolling_sharpe.dropna()

    rolling_sharpe.plot()

    plt.axhline(rolling_sharpe.iloc[-1], linestyle='--', label='now')
    plt.axhline(rolling_sharpe.median(), c='r', label='Median')
    plt.axhline(rolling_sharpe.mean(), c='y', label='Average')
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    print('Hello World')