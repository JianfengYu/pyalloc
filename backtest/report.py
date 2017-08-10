import abc
import pandas as pd
import numpy as np
from tabulate import tabulate

import pyalloc.tools.analyze as anz


class Report(metaclass=abc.ABCMeta):

    def __init__(self, s_nv: pd.Series):
        self._total_return = .0
        self._annal_return = .0
        self._annal_volatility = .0
        self._max_drawdown_info = None
        self._max_wait_days = None
        self._s_nv = s_nv

    @abc.abstractmethod
    def analyze(self):
        raise NotImplementedError('.analyze() not defined!')

    def __repr__(self):
        return "Report({0})".format(self.__dict__)


class BacktestReport(Report):
    """回测结果report"""

    def __init__(self, s_nv: pd.Series, bench_nv=None, rf=None, true_beta=False):
        super().__init__(s_nv)

        self._bench_nv = bench_nv
        self._rf = rf

        self._true_beta = true_beta

        self._beta = .0
        self._alpha = .0
        self._sharpe = .0
        self._information_ratio = .0
        self._tracking_error = .0
        self._Sortino = .0
        self._Treynor = .0
        self._calmar = .0
        self._downside_risk = .0

        self._bench_total_return = .0
        self._bench_annal_return = .0

        self._timing_indicator = None

        self._bench_flag = True if bench_nv is not None else False
        self._rf_flag = True if rf is not None else False

    def analyze(self):

        self._total_return = self._s_nv[-1] / self._s_nv[0] - 1
        self._annal_return = anz.cal_annal_return(self._s_nv, freq='D')
        self._annal_volatility = anz.cal_annal_volatility(self._s_nv, freq='D')

        self._max_wait_days = anz.cal_max_wait_periods(self._s_nv)
        self._max_drawdown_info = anz.cal_max_drawdown_info(self._s_nv)
        # max_dd, start_date, end_date, dt_range, drawdown_here = self._max_drawdown_info

        self._calmar = anz.cal_calmar(self._s_nv, freq='D')
        self._sharpe = anz.cal_sharpe(self._s_nv, rf=self._rf, freq='D')
        self._downside_risk = anz.cal_downside_risk(self._s_nv, self._bench_nv, freq='D')

        self._timing_indicator = anz.cal_timing_indicator(self._s_nv, self._bench_nv, self._rf)

        if self._bench_flag:
            if anz.check(self._s_nv, self._bench_nv):
                self._bench_total_return = self._bench_nv[-1] / self._bench_nv[0] - 1
                self._bench_annal_return = anz.cal_annal_return(self._bench_nv, freq='D')
                self._information_ratio = anz.cal_information_ratio(self._s_nv, self._bench_nv, freq='D')
                self._tracking_error = anz.cal_tracking_error(self._s_nv, self._bench_nv, freq='D')

        elif self._rf_flag:
            if anz.check(self._s_nv, self._bench_nv, self._rf_flag):
                self._beta = anz.cal_beta(self._s_nv, self._bench_nv, self._rf)

            if self._true_beta:  # 用真实的策略beta计算alpha
                self._alpha = anz.cal_alpha(self._s_nv, self._bench_nv, self._rf, beta=self._beta)
                self._Treynor = anz.cal_treynor(self._s_nv, self._bench_nv, self._rf, freq='D')
            else:  # 直接计算超额收益，即beta=1
                self._alpha = anz.cal_alpha(self._s_nv, self._bench_nv, self._rf, beta=1)

        if self._rf_flag:
            self._Sortino = anz.cal_sortino(self._s_nv, self._rf, freq='D')

    def output(self):

        # 先分析，才有输出
        self.analyze()

        basic_risk_indicators = []
        basic_risk_indicators.append(['Total Returns', "{0:.3%}".format(self._total_return)])
        basic_risk_indicators.append(['Annual Returns', "{0:.3%}".format(self._annal_return)])
        basic_risk_indicators.append(['Bench Total Returns', "{0:.3%}".format(self._bench_total_return)])
        basic_risk_indicators.append(['Bench Annual Returns', "{0:.3%}".format(self._bench_annal_return)])
        basic_risk_indicators.append(['Alpha', "{0:.4}".format(self._alpha)])
        basic_risk_indicators.append(['Beta', "{0:.4}".format(self._beta)])
        basic_risk_indicators.append(['Annual Volatility', "{0:.4}".format(self._annal_volatility)])
        basic_risk_indicators.append(['MaxDrawdown', "{0:.3%}".format(self._max_drawdown_info[0])])
        basic_risk_indicators.append(['MaxDrawdown Period', "{0} to {1}, {2} days".format(
            self._max_drawdown_info[1], self._max_drawdown_info[2], self._max_drawdown_info[3])]
                                     )
        basic_risk_indicators.append(['Max Waite Period', "{0} days".format(self._max_wait_days)])
        basic_risk_indicators.append(['Sharpe', "{0:.4}".format(self._sharpe)])
        basic_risk_indicators.append(['Information Ratio', "{0:.4}".format(self._information_ratio)])

        basic_risk_indicators.append(['Calmar', "{0:.4}".format(self._calmar)])

        basic_risk_indicators.append(['Sortino', "{0:.4}".format(self._Sortino)])

        basic_risk_indicators.append(['Tracking Error', "{0:.4}".format(self._tracking_error)])
        basic_risk_indicators.append(['Downside Risk', "{0:.4}".format(self._downside_risk)])

        basic_risk_indicators.append(['Win Rate', "{0:.3%}".format(self._timing_indicator[0])])
        basic_risk_indicators.append(['Loss Rate', "{0:.3%}".format(self._timing_indicator[1])])
        basic_risk_indicators.append(['Win Odds', "{0:.5}".format(self._timing_indicator[2])])
        basic_risk_indicators.append(['PL Ratio', "{0:.5}".format(self._timing_indicator[3])])

        headers = ["Indicator", "Value"]
        print(tabulate(basic_risk_indicators, headers))