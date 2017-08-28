"""
收益和风险的分解，收益特征的分析，策略之间的比较
"""
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from old.api_config import wind_edb_dict
from pyalloc.backtest.strategy import Strategy

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_style("whitegrid")


def dict_value_to_key(dic: dict) -> dict:
    res = {dic[a]: a for a in dic.keys()}
    return res
code_map = dict_value_to_key(wind_edb_dict)


class Xray:
    """对单一策略回测结果的分析"""

    def __init__(self, strategy: Strategy):

        self._s = strategy
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
        self._average_rebalance_interval = 0

       # self._strategy = strategy

    def run(self):
        """进行常规的分析"""
        self.plot_rebalance_weight()            # 调仓权重图
        self.plot_daily_weight()                # 每日权重图
        self.return_attribution(plot=True)      # 收益归因图
        self.plot_return_distribution()         # 收益分布图
        self.plot_drawdown()                    # 回撤分析图
        self.plot_turnvoer()                    # 换手率分析图

    def return_analyze(self):
        #TODO 收益的进一步分析，包括按月的柱状图，按年的柱状图，滚动sharpe ratio
        pass

    def risk_analyze(self):
        #TODO 风险的进一步分析，包括滚动sharpe ratio，VaR值 vs 累计损失
        pass

    def plot_return_distribution(self):
        """每日收益分布作图"""
        daily_ret = self._ret.dropna()
        sns.distplot(daily_ret, kde=False, fit=stats.norm)

        loc, _ = plt.yticks()
        plt.yticks(loc, ['{0:.2f}%'.format(a * 100) for a in np.array(loc) / len(daily_ret)])
        plt.title(self._s_name + ' Daily Return Distribution')
        plt.show()

    def return_attribution(self, start_date=None, end_date=None, plot=False,
                           code_dict=code_map):
        """收益分解"""
        if start_date is None and end_date is None:
            ret_daily = self._ret.dropna()
        else:
            ret_daily = self._ret.dropna()[start_date:end_date]

        logret_daily = np.log(ret_daily +1)
        asset_ret = self._quote.reindex(ret_daily.index)[self._s._env.datasource.sids]
        asset_ret.rename(columns=code_dict, inplace=True)  # 资产收益

        # 归因
        # 计算每日开盘权重，需要在调仓日进行修正
        asset_weight = self._daily_weight.shift(1).reindex(ret_daily.index)  # 前一日收盘权重
        asset_weight.rename(columns=code_dict, inplace=True)

        # 每日的log return分解
        asset_contribution = ((asset_ret * asset_weight).T / ret_daily * logret_daily.T).T

        asset_contribution_total = asset_contribution.sum() / logret_daily.sum()

        # asset_contribution_total['Transaction Cost'] = -(self._cost / ret_daily * logret_daily).sum()
        asset_contribution_total['Trading PL'] = 1 - asset_contribution_total.sum()  # 残留为交易当天PL

        self.return_contribution = asset_contribution_total

        # 业绩归因柱状图
        if plot is True:
            aaa = pd.DataFrame({'Attribution': asset_contribution_total}).reset_index()
            aaa.rename_axis({'index': 'Asset'}, inplace=True, axis=1)
            # sns.distplot(b4.res_nv.pct_change().dropna(), kde=False, bins=50)
            plt.figure(figsize=(8, 6))

            g = sns.barplot(x='Asset', y='Attribution', data=aaa,
                            palette=sns.color_palette("Paired"))

            sns.plt.title(self._s_name + ' Attribution')

            yloc, _ = sns.plt.yticks()
            sns.plt.yticks(yloc, ['{0:.1f}%'.format(a * 100) for a in yloc])
            sns.plt.xticks(rotation=30)


            # 增加annotate
            def annotateBars(row, ax=g):
                for p in ax.patches:
                    # print(p.get_x(), p.get_y(), p.get_width(), p.get_height())
                    sign = 1 if p.get_y() >= 0 else -1
                    large0 = p.get_height() if p.get_height() >=0 else 0
                    # print(p.get_height(), large0)
                    ax.annotate('{0:.1f}%'.format(100 * sign * p.get_height()),
                                (p.get_x() + p.get_width() / 2., large0),
                                ha='center', va='center', fontsize=11, color='gray', rotation=30, xytext=(0, 15),
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

        s = weight.index
        if len(s) < 8:
            step = 1
        elif len(s) < 16:
            step = len(s) // 2
        elif len(s) < 32:
            step = len(s) // 4
        else:
            step = len(s) // 8

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

        if len(w) < 8:
            step = 1
        elif len(w) < 16:
            step = len(w) // 2
        elif len(w) < 32:
            step = len(w) // 4
        else:
            step = len(w) // 8

        plt.xticks(range(0, len(w.index), step), [w.index[i] for i in range(0, len(w.index), step)])
        plt.yticks(np.arange(0, 1.1, 0.2), ['{0:.0f}%'.format(a * 100) for a in np.arange(0, 1.1, 0.2)])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        plt.show()

    def plot_turnvoer(self, figure_size=(12, 6)):
        """换手率柱状图"""
        s = self._turnover.copy()
        s = s[s != 0]

        s.plot.barh(color='c', label='Turnover')
        loc, _ = plt.xticks()
        plt.xticks(loc, ['{0:.0f}%'.format(a * 100) for a in loc])

        if len(s) < 8:
            step = 1
        elif len(s) < 16:
            step = len(s) // 2
        elif len(s) < 32:
            step = len(s) // 4
        else:
            step = len(s) // 8
        plt.yticks(range(0, len(s.index), step), [s.index[i].strftime('%Y-%m-%d')
                                                  for i in range(0, len(s.index), step)])

        y_min, y_max = plt.ylim()
        plt.vlines(x=s.mean(), ymin=y_min, ymax=y_max, label='Mean: {0:.1f}%'.format(s.mean()*100),
                   linestyles='--', colors='orange')
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)

        # 计算平均调仓间隔天数
        index_list = list(self._daily_weight.index)
        rebalance_point = [index_list.index(a) for a in self._rebalance_weight.index]
        average_rebalance_day = np.mean([rebalance_point[i] - rebalance_point[i-1]
                                         if i>0 else rebalance_point[i]
                                         for i in range(len(rebalance_point))])
        self._average_rebalance_interval = average_rebalance_day

        plt.title('Average Interval Between Rebalance: {0} Days'.format(int(average_rebalance_day)))
        plt.show()



class Comparator:
    """对多个策略进行比较"""

    def __init__(self, strategies: List[Strategy]):
        self._strategies = strategies
        self._nvs = {s.name: s.net_value for s in strategies}

        #TODO



if __name__ == '__main__':

    pass
