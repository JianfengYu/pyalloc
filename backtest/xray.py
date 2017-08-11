"""
收益和风险的分解，收益特征的分析
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

# 解决中文显示问题
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import pyalloc.tools.analyze as als
from pyalloc.backtest.strategy import Strategy
from pyalloc.data.api_config import wind_edb_dict


def dict_value_to_key(dic: dict) -> dict:
    res = { dic[a]: a for a in dic.keys()}
    return res
code_map = dict_value_to_key(wind_edb_dict)

class Xray:

    def __init__(self, strategy: Strategy):

        self._s_name = strategy._name

        self._sdt = strategy._nv.index[0]
        self._edt = strategy._nv.index[-1]

        # 合成行情df
        self._quote = pd.DataFrame([])
        for asset in strategy._datasource.data.keys():
            self._quote[asset] = strategy._datasource.data[asset]['pct_change']
        self._quote['cash'] = .0

        self._daily_weight = strategy._weight
        self._rebalance_weight = strategy._rebalance_weight

        self._nv = strategy._nv
        self._ret = strategy._s_ret
        self._turnover = strategy.turnover
        self._cost = strategy.cost
        self._backtest_report = strategy._report

        # self._strategy = strategy

    def run(self):
        """进行所有的分析"""
        self.plot_rebalance_weight()
        self.plot_daily_weight()
        self.return_analyser(plot=True)
        self.plot_return_distribution()
        self.plot_drawdown()

    def plot_return_distribution(self):
        """每日收益分布作图"""
        daily_ret = self._ret.dropna()
        sns.distplot(daily_ret, kde=False, fit=stats.norm)

        loc, _ = plt.yticks()
        plt.yticks(loc, ['{0:.2f}%'.format(a * 100) for a in np.array(loc) / len(daily_ret)])
        plt.title(self._s_name + ' Daily Return Distribution')
        plt.show()

    def return_analyser(self, start_date=None, end_date=None, plot=False,
                        code_dict=code_map):
        """收益分解"""
        if start_date is None and end_date is None:
            ret_daily = self._ret.dropna()
        else:
            ret_daily = self._ret.dropna()[start_date:end_date]

        logret_daily = np.log(ret_daily +1)
        asset_ret = self._quote.reindex(ret_daily.index)
        asset_ret.rename(columns=code_dict, inplace=True)  # 资产收益

        # 归因
        # 计算每日开盘权重，需要在调仓日进行修正
        asset_weight = self._daily_weight.shift(1).reindex(ret_daily.index)  # 前一日收盘权重
        asset_weight.rename(columns=code_dict, inplace=True)

        # 每日的log return分解
        asset_contribution = ((asset_ret * asset_weight).T / ret_daily * logret_daily.T).T

        asset_contribution_total = asset_contribution.sum() / logret_daily.sum()

        asset_contribution_total['Transaction Cost'] = -(self._cost / ret_daily * logret_daily).sum()
        asset_contribution_total['Trading PL'] = 1 - asset_contribution_total.sum()

        self.return_contribution = asset_contribution_total

        # 业绩归因柱状图
        if plot is True:
            aaa = pd.DataFrame({'Attribution': asset_contribution_total}).reset_index()
            aaa.rename_axis({'index': 'Asset'}, inplace=True, axis=1)
            # sns.distplot(b4.res_nv.pct_change().dropna(), kde=False, bins=50)
            g = sns.barplot(x='Asset', y='Attribution', data=aaa,
                            palette=sns.color_palette("Paired"))

            sns.plt.title(self._s_name + ' Attribution')

            yloc, _ = sns.plt.yticks()
            sns.plt.yticks(yloc, ['{0:.1f}%'.format(a * 100) for a in yloc])
            sns.plt.xticks(rotation=30)

            # 增加annotate
            def annotateBars(row, ax=g):
                for p in ax.patches:
                    sign = 1 if p.get_y() >= 0 else -1
                    large0 = 1 if p.get_y() >=0 else 0
                    ax.annotate('{0:.1f}%'.format(100 * sign * p.get_height()),
                                (p.get_x() + p.get_width() / 2., large0 * p.get_height()),
                                ha='center', va='center', fontsize=11, color='gray', rotation=30, xytext=(0, 20),
                                textcoords='offset points')
            aaa.apply(annotateBars, ax=g, axis=1)

            sns.plt.ylabel('% of Total Return')
            # sns.plt.rcParams['image.cmap'] = 'Paired'
            sns.plt.show()

    def plot_drawdown(self, legend=True, figure_size=(12, 6)):
        """绘制回撤分析图"""

        daily_dd = self._backtest_report._max_drawdown_info[-1]
        # 回撤分析图
        daily_dd.plot(kind='area', title=self._s_name + ' Drawback', label=self._s_name, legend=legend,
                      alpha=0.6, color='Gray', ylim=(daily_dd.min() - 0.1, 0), figsize=figure_size)

        mdd_sdt = self._backtest_report._max_drawdown_info[1]
        mdd_edt = self._backtest_report._max_drawdown_info[2]
        mdd_range = self._backtest_report._max_drawdown_info[3]

        # 最大回撤区间
        daily_dd[mdd_sdt:mdd_edt].plot(kind='area', label='Max Drawback Periods({0}days)'.format(mdd_range),
                                       legend=legend, alpha=0.8, color='Green')

        locs, labels = plt.yticks()
        plt.yticks(locs, ['{0:.0f}%'.format(a * 100) for a in locs])

        plt.legend(loc=3)
        plt.show()

    def plot_daily_weight(self, figure_size=(12, 6)):
        """绘制每日的权重图"""

        weight = self._daily_weight.copy().rename(columns=code_map)
        weight.index = [pd.datetime.strftime(a, '%Y-%m') for a in weight.index]

        weight.plot(title=self._s_name + ' Daily Weight', kind='area', stacked=True, figsize=figure_size, alpha=0.9,
                    color=sns.color_palette("Paired"))

        step = len(weight.index) // 8
        plt.xticks(range(0, len(weight.index), step), [weight.index[i] for i in range(0, len(weight.index), step)])

        loc, _ = plt.yticks()
        plt.yticks(loc, ['{0:.0f}%'.format(a * 100) for a in loc])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)  # put legend on the right
        plt.show()

    def plot_rebalance_weight(self, figure_size=(12, 6)):
        """绘制调仓日的权重图"""
        w = self._rebalance_weight.rename(columns=code_map)
        w.index = [pd.datetime.strftime(a, '%Y-%m') for a in w.index]
        w.plot(title=self._s_name + '  Weight', kind='bar', stacked=True, figsize=figure_size, alpha=0.9,
                    color=sns.color_palette("Paired"))

        step = len(w.index) // 8
        plt.xticks(range(0, len(w.index), step), [w.index[i] for i in range(0, len(w.index), step)])
        plt.yticks(np.arange(0, 1.1, 0.2), ['{0:.0f}%'.format(a * 100) for a in np.arange(0, 1.1, 0.2)])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        plt.show()


    @staticmethod
    def draw_weight(weight: pd.Series, title: str, figure_size=(12, 6)):
        """绘制给定权重的图"""

        w = weight.copy()
        w.index = [pd.datetime.strftime(a, '%Y-%m') for a in w.index]
        w.plot(title=title, kind='bar', stacked=True, figsize=figure_size, alpha=0.9,
                    color=sns.color_palette("Paired"))

        step = len(w.index) // 8
        plt.xticks(range(0, len(w.index), step), [w.index[i] for i in range(0, len(w.index), step)])
        plt.yticks(np.arange(0, 1.1, 0.2), ['{0:.0f}%'.format(a * 100) for a in np.arange(0, 1.1, 0.2)])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        plt.show()

if __name__ == '__main__':
    # 获取收益率数据
    pass
