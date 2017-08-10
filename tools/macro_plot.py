import pandas as pd
import numpy as np
import scipy as spy

import matplotlib.colors as col

from numpy.lib.stride_tricks import as_strided as strided

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style("white")

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体
mpl.rcParams['axes.unicode_minus'] =False # 解决保存图像是负号'-'显示为方块的问题

# 设定颜色
startcolor = '#006400'  # a dark green
midcolor = '#ffffff'  # a bright white
endcolor = '#ee0000'  # a dark red
Mycmap = col.LinearSegmentedColormap.from_list(
    'MyColorbar', [startcolor, midcolor, endcolor])


def cal_percentileofscore(s: pd.Series) -> None:
    """calculate the percentile of score of a series"""
    s_pct_score = s.apply(lambda x: spy.stats.percentileofscore(s.sort_values(), x))
    return s_pct_score

# cal_correlation_df
def rolling_correlation(a, w):
    # from numpy.lib.stride_tricks import as_strided as strided
    n, m = a.shape[0], 2
    s1, s2 = a.strides
    b = strided(a, (m, w, n - w + 1), (s2, s1, s1))
    b_mb = b - b.mean(1, keepdims=True)
    b_ss = (b_mb ** 2).sum(1) ** .5
    return (b_mb[0] * b_mb[1]).sum(0) / (b_ss[0] * b_ss[1])

def rolling_correlation_df(df, w):
    a = df.values
    return pd.Series(rolling_correlation(a, w), df.index[w-1:])


def plot_mothly_ret(s: pd.Series, annot=False):

    tmp_df = s.copy()
    # change to log return
    tmp_df = np.log(1 + s).to_frame()

    tmp_df['Year'] = tmp_df.index.year
    tmp_df['Month'] = tmp_df.index.month

    res = tmp_df.groupby(['Year', 'Month']).sum()
    res = np.exp(res) - 1
    res = res.unstack()
    res.columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    sns.heatmap(res, cmap=Mycmap, annot=annot)
    sns.plt.title(s.name + '收益率')
    plt.show()


def plot_corr_hearmap(df: pd.DataFrame, annot=True):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap=Mycmap, annot=annot, vmin=-1, vmax=1)
    plt.show()

if __name__ == '__main__':
    print('Hello World')