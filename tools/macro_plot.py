import pandas as pd
import numpy as np
import scipy as spy

from typing import Union
import matplotlib.colors as col

from numpy.lib.stride_tricks import as_strided as strided

import matplotlib.pyplot as plt
import seaborn as sns

from pyalloc.config import TRADING_DAYS_A_YEAR
from pyalloc.tools.analyze import freq_periods

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

"""计算"""


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

def cal_rolling_sharpe(ret: pd.Series, rf: Union[pd.Series, None], window: int, min_periods: int, freq='D') -> pd.Series:
    """
    计算rolling sharpe

    Parameters
    ----------
    ret: pd.Series
        收益率序列
    freq: str
        频率, 'M' 月度，'W' 周度，'D' 日度， 默认为'D'
    window:
        窗口
    min_periods:
        最小区间

    Returns
    -------

    """
    period = freq_periods(freq)

    if rf is None:
        ex_mean = ret.rolling(window=window, min_periods=min_periods).mean()
        std = ret.rolling(window=window, min_periods=min_periods).std()
        return (ex_mean / std).dropna() * np.sqrt(period)

    index = ret.dropna().index.intersection(rf.dropna().index)
    ex_mean = (ret - rf).reindex(index).rolling(window=window, min_periods=min_periods).mean()
    std = ret.reindex(index).rolling(window=window, min_periods=min_periods).std()
    rolling_sharpe = (ex_mean / std).dropna() * np.sqrt(period)
    return rolling_sharpe


"""画图"""

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


def plot_hist_level(s: pd.Series, label:str) -> None:
    """
    画历史水平图

    Parameters
    ----------
    s: pd.Series
        序列
    Returns
    -------

    """
    s.plot(label=label)
    plt.axhline(s.iloc[-1], linestyle='--', label='now: {0:.2f}'.format(s.iloc[-1]))
    plt.axhline(s.median(), c='r', label='Median: {0:.2f}'.format(s.median()))
    plt.axhline(s.mean(), c='y', label='Average: {0:.2f}'.format(s.mean()))
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
    plt.show()


def cal_duration(s: pd.Series, level: float, up=False) -> pd.Series:
    """
    计算累计持续时间

    Parameters
    ----------
    s: pd.Series
        时间序列
    level: float
        比较水平
    up: bool, default False
        比较的方向，True为高于level

    Returns
    -------

    """

    if up:
        dummy = s > level
    else:
        dummy = s < level

    cum_days = dummy.cumsum()  # 累计的符合天数

    _, stop_duration_day = cal_duration_periods(s, level, up)

    pre_cum_days = cum_days[stop_duration_day].copy()
    pre_cum_days = pre_cum_days.reindex(dummy.index).fillna(method='ffill').fillna(0)  # 之前累计的天数

    return cum_days - pre_cum_days

def cal_duration_periods(s: pd.Series, level: float, up=False) -> tuple:
    """符合条件的起止日期"""
    if up:
        dummy = s > level
    else:
        dummy = s < level

    cum_days = dummy.cumsum()  # 累计的符合天数
    duration_days = cum_days != cum_days.shift(1)  # 持续增加的天为true

    stop_duration_dummy = (duration_days - duration_days.shift(-1)) == 1
    start_duration_dummy = (duration_days - duration_days.shift(-1)) == -1

    stop_duration_day = stop_duration_dummy[stop_duration_dummy].index  # 停止持续的日期
    start_duration_day = start_duration_dummy[start_duration_dummy].index  # 开始持续的日期

    return start_duration_day, stop_duration_day


if __name__ == '__main__':
    ret = pd.DataFrame(np.random.randn(50, 2), index=pd.date_range(start='2010-01-01', periods=50), columns=['A', 'B'])
    dummy = ret < 0
