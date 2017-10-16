"""
本模块用于计算分析类指标
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyalloc.config import TRADING_WEEKS_A_YEAR, TRADING_MONTHS_A_YEAR, TRADING_DAYS_A_YEAR


# 辅助函数
def match_data(bench: pd.Series, *args):
    """
    将不同的pandas.Series按照时间index对齐
    Parameters
    ----------
    bench
        对齐所要参照的基准

    args
        需要对对齐的序列

    Returns
    -------
    pandas.DataFrame
        对齐后的DataFrame

    """

    assert isinstance(bench.index, pd.DatetimeIndex), "Index should be pandas.DatetimeIndex!"

    res = bench.to_frame()
    for i in range(len(args)):
        res[args[i].name] = args[i]

    return res


def check(*args) -> bool:
    """
    对pandas.Series进行合规检查，包括index是否为pandas.DatetimeIndex，是否有空值，长度是否相等且大于2

    Parameters
    ----------
    args
        待判断的序列

    Returns
    -------
    bool
        通过则返回 True，否则报错

    """
    length = len(args[0])
    assert length > 2, "The length of time series should be longer than 2!"

    for i in range(len(args)):
        assert isinstance(args[i].index, pd.DatetimeIndex), "The index should be pandas.DatetimeIndex!"
        assert args[i].isnull().sum() == 0, "There are NaN values!"
        assert len(args[i]) == length, "The length of data are not same!"

    return True


def freq_periods(freq: str) -> float:
    """
    根据频率返回1年中的时期数目

    Parameters
    ----------
    freq: str
        'M', 月度
        'W', 周度
        'D', 日度

    Returns
    -------
    float
        一年中的时期数，如一年有12个月

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')

    return periods


def de_annualize(ret_annualized: pd.Series, freq='D'):
    """
    计算年化收益对应频率的收益

    Parameters
    ----------
    ret_annualized
        pd.Series 年化的收益率序列
    freq
        需要转换成的频率
    Returns
    -------
    pd.Series
        一定时期内的收益

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')
    s = np.exp(np.log(ret_annualized + 1) / periods) - 1
    return s


"""
计算策略整体表现类指标（日频）
"""


def cal_annal_return(net_value: pd.Series, freq='D') -> float:
    """
    计算日频净值的年化收益率

    Parameters
    ----------
    net_value
        策略日净值的时间序列，支持原始账户总资产的时间序列
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        年化收益率

    """
    # dt_range = pd.period_range(net_value.index[0], net_value.index[-1])
    # 用交易日计算年化收益
    periods = freq_periods(freq)

    annal_re = pow(net_value.values[-1] / net_value.values[0],
                   periods / (len(net_value) - 1)) - 1

    return annal_re


def cal_annal_volatility(net_value: pd.Series, freq='D') -> float:
    """
    计算日频净值的年化波动率

    Parameters
    ----------
    net_value
        策略日净值的时间序列，支持原始账户总资产的时间序列
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        年化波动率

    """
    periods = freq_periods(freq)

    rtns = net_value.pct_change().dropna()

    if len(rtns) <= 1:
        return .0
    vlt = np.sqrt(periods) * rtns.std(ddof=1)
    return vlt


def cal_max_drawdown(net_value: pd.Series) -> float:
    """
    计算最大回撤

    Parameters
    ----------
    net_value
        净值序列

    Returns
    -------
    float
        最大回撤

    """

    # 计算当日之最大的净值
    max_here = net_value.expanding(min_periods=1).max()
    drawdown_here = net_value / max_here - 1

    # 计算最大回撤开始和结束时间
    tmp = drawdown_here.sort_values().head(1)
    max_dd = float(tmp.values)

    return max_dd


def cal_max_drawdown_info(net_value: pd.Series) -> tuple:
    """
    计算日频净值的最大回撤相关信息

    Parameters
    ----------
    net_value
        策略日净值的时间序列，支持原始账户总资产的时间序列

    Returns
    -------
    tuple
        (最大回撤，最大回撤开始日期，最大回撤结束日期，最大回撤持续天数, 每日的回撤)

    """
    max_here = net_value.expanding(min_periods=1).max()  # 计算当日之前的最大净值
    drawdown_here = net_value / max_here - 1  # 计算当日的回撤

    # 计算最大回撤和结束时间
    tmp = drawdown_here.sort_values().head(1)
    max_dd = float(tmp.values)
    end_date = tmp.index.strftime('%Y-%m-%d')[0]

    # 计算开始时间
    tmp = net_value[:end_date]
    tmp = tmp.sort_values(ascending=False).head(1)
    start_date = tmp.index.strftime('%Y-%m-%d')[0]

    # 计算回撤持续天数
    dt_range = len(pd.period_range(start_date, end_date))

    return max_dd, start_date, end_date, dt_range, drawdown_here


def cal_tracking_error(net_value: pd.Series, bench_nv: pd.Series, freq='D') -> float:
    """
    计算年化跟踪误差
        跟踪误差是指组合收益率与基准收益率(大盘指数收益率)之间的差异的收益率标准差，反映了基金管理的风险。

    Parameters
    ----------
    net_value
        策略日净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准日净值的时间序列，支持原始基准指数的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        年化跟踪误差

    References
    _______
        Wiki： https://en.wikipedia.org/wiki/Tracking_error
    """
    periods = freq_periods(freq)

    diff = (net_value.pct_change() - bench_nv.pct_change()).dropna()
    te = np.sqrt((diff * diff).sum() / len(diff) * periods)

    return te


def cal_information_ratio(net_value: pd.Series, bench_nv: pd.Series, freq='D') -> float:
    """
    计算年化信息比率
        表示单位主动风险所带来的超额收益
        IR = E[R_p-R_b]/sqrt(Var(R_p-R_b))
        若Var(R_p-R_b)为零，返回带符号的np.inf

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        年化信息比率

    References
    _______
        Wiki： https://en.wikipedia.org/wiki/Information_ratio
    """
    periods = freq_periods(freq)

    r_p = net_value.pct_change().dropna()
    r_b = bench_nv.pct_change().dropna()
    diff = r_p - r_b
    if diff.std() != 0:
        ir = (diff.mean() / diff.std()) * np.sqrt(periods)
    else:
        ir = np.sign(diff.mean()) * np.inf

    return ir


def cal_beta(net_value: pd.Series, bench_nv: pd.Series, rf: pd.Series) -> float:
    """
    计算策略的历史beta
        资本资产定价模型（CAPM）的Beta
        若bench_nv的方差为0，返回np.NaN

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益率的时间序列

    Returns
    -------
    float
        策略的beta

    References
    _______
        Wiki： https://en.wikipedia.org/wiki/Capital_asset_pricing_model
    """
    re_p = (net_value.pct_change() - rf).dropna()  # 策略超额收益序列
    re_b = (bench_nv.pct_change() - rf).dropna()

    if re_b.var() == 0:
        return np.NaN
    else:
        beta = re_p.cov(re_b) / re_b.var()
        return beta


def cal_alpha(net_value: pd.Series, bench_nv: pd.Series, rf: pd.Series, beta=1, freq='D') -> float:
    """
    计算策略的alpha
        默认计算投资组合相对基准的超额收益，即beta=1。
        若要用资本资产定价模型（CAPM）的Alpha，可指定参数beta为CAPM模型的beta。

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益率的时间序列

    beta
        策略的beta值，默认为1。

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略的alpha

    """

    annal_rp = cal_annal_return(net_value, freq)
    annal_rb = cal_annal_return(bench_nv, freq)
    annal_rf = cal_annal_return((rf + 1).cumprod(), freq)

    alpha = annal_rp - annal_rf - beta * (annal_rb - annal_rf)
    return alpha


def cal_sharpe(net_value: pd.Series, rf=None, freq='D') -> float:
    """
    计算年化Sharpe比
        SharpeRatio = E[R_p-R_f] / sqrt(Var(R_p-R_f))
        若Var(R_p-R_f)为0，返回带符号的np.inf

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略的年化Sharpe比

    References
    _______
    https://en.wikipedia.org/wiki/Sharpe_ratio

    """
    periods = freq_periods(freq)

    if rf is None:
        re_p = net_value.pct_change().dropna()
    else:
        re_p = (net_value.pct_change() - rf).dropna()

    if re_p.std() != 0:
        sharpe = re_p.mean() / re_p.std() * np.sqrt(periods)
    else:
        sharpe = np.sign(re_p.mean()) * np.inf

    return sharpe


def cal_m2_measure(net_value: pd.Series, bench_nv: pd.Series, rf=None, freq='D') -> float:
    """
    Modigliani risk-adjusted performance
        M2 = (Rp-Rf).mean() * (sigma(Rb-Rf)/sigma(Rp-Rf)) + Rf.mean()

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        风险收益的时间序列

    freq
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略的M2值

    References
    _______
    https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance#cite_note-3
    Modigliani, Franco (1997). "Risk-Adjusted Performance". Journal of Portfolio Management. 1997 (Winter): 45–54.

    """
    periods = freq_periods(freq)

    if rf is None:
        re_p = net_value.pct_change().dropna()
        rb = bench_nv.pct_change().dropna()
    else:
        re_p = (net_value.pct_change() - rf).dropna()
        rb = (bench_nv.pct_change() - rf).dropna()

    sigma_ratio = rb.std() / re_p.std()

    m2 = (re_p.mean() * sigma_ratio + rf.mean()) * periods if rf is not None else re_p.mean() * sigma_ratio * periods

    return m2


def cal_m2_alpha(net_value: pd.Series, bench_nv: pd.Series, rf=None, freq='D') -> float:
    """
    risk-adjusted performance alpha of M2 measure
        M2 = (Rp-Rf).mean() * (sigma(Rb-Rf)/sigma(Rp-Rf)) + Rf.mean()
        M2_alpha = M2 - Rf.mean()

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        风险收益的时间序列

    freq
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略M2测度下的

    References
    _______
    https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance#cite_note-3
    Modigliani, Franco (1997). "Risk-Adjusted Performance". Journal of Portfolio Management. 1997 (Winter): 45–54.
    """
    periods = freq_periods(freq)
    m2 = cal_m2_measure(net_value, bench_nv, rf, freq)
    m2_alpha = m2 - rf.mean() * periods
    return  m2_alpha


def cal_downside_risk(net_value: pd.Series, bench_nv=None, r_min=.0, freq='D') -> float:
    """
    计算年化的下行波动率
        真实收益率低于给定收益率的平方的期望，再取sqrt。若不指定给定的收益率，则默认为0.
        Wiki中的target semi deviation (TSV)。

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列
        默认为None，即仅考虑收益率为负的情况，不与benchmark进行比较

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    r_min:float， default .0
        最小收益水准，默认为0

    Returns
    -------
    float
        策略的年化下行波动率

    References
    _______
    https://en.wikipedia.org/wiki/Downside_risk

    """
    periods = freq_periods(freq)

    if bench_nv is not None and r_min != 0:
        raise ValueError('You can only assign one between bench_nv and r_min!')

    r_p = net_value.pct_change().dropna()

    if bench_nv is not None:
        r_b = bench_nv.pct_change().dropna()
        dummy = r_p < r_b
        diff = r_p[dummy] - r_b[dummy]
    else:
        diff = r_p[r_p < r_min]

    if len(diff) <= 1:
        return 0.
    else:
        return np.sqrt((diff*diff).sum()/len(r_p) * periods)


def cal_sortino(net_value: pd.Series, rf: pd.Series, freq='D') -> float:
    """
    计算年化sortino比率
        年化的超额收益除以年化的下行波动率。这里的下行波动率为target semi deviation (TSV)，详见Wiki。
        若下行波动率为0，则返回带符号的np.inf

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略的年化Sortino比率

    References
    _______
    Sortino Ratio
        https://en.wikipedia.org/wiki/Sortino_ratio

    Downside Risk
        https://en.wikipedia.org/wiki/Downside_risk

    """

    annal_rp = cal_annal_return(net_value, freq=freq)
    annal_rf = cal_annal_return((1 + rf).cumprod(), freq=freq)
    downside_risk = cal_downside_risk(net_value, bench_nv=(1 + rf).cumprod(), freq=freq)

    if downside_risk != 0:
        return (annal_rp - annal_rf) / downside_risk
    else:
        return np.sign(annal_rp - annal_rf) * np.inf


def cal_calmar(net_value: pd.Series, freq='D') -> float:
    """
    计算年化的Calmar比率
        组合的年化收益率与历史最大回撤之间的比率

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略年化Calmar比率

    References
    _______
    [1] https://en.wikipedia.org/wiki/Calmar_ratio
    [2] Young, Terry W. (1 October 1991), "Calmar Ratio: A Smoother Tool", Futures (magazine)

    """
    maxdd = cal_max_drawdown(net_value)
    annal_ret = cal_annal_return(net_value, freq=freq)

    if maxdd != 0:
        calmar = annal_ret / abs(maxdd)
    else:
        calmar = np.sign(annal_ret) * np.inf

    return calmar


def cal_treynor(net_value: pd.Series, bench_nv: pd.Series, rf: pd.Series, freq: str) -> float:
    """
    计算年化Treynor比率
        组合的超额收益除以组合的beta值。

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益的时间序列

    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为D。

    Returns
    -------
    float
        策略年化Treynor比率

    References
    _______
    https://en.wikipedia.org/wiki/Treynor_ratio
    """
    beta = cal_beta(net_value, bench_nv, rf)
    annal_rp = cal_annal_return(net_value, freq=freq)
    annal_rf = cal_annal_return((1 + rf).cumprod(), freq=freq)

    if beta == 0:
        return np.sign(annal_rp - annal_rf) * np.inf
    elif beta == np.NaN:
        return np.NaN
    else:
        return (annal_rp - annal_rf) / beta


def cal_max_wait_periods(net_value: pd.Series) -> int:
    """
    计算再创新高最长等待天数

    Parameters
    ----------
    net_value: pd.Series
        净值序列

    Returns
    -------
    int
        再创新高最长等待天数

    """

    max_here = net_value.expanding(min_periods=1).max()  # 计算当日之前的账户最大值
    new_high_dummy = net_value == max_here  # 得到是否达到最大值的日期dummy
    max_wait_days = np.max(np.bincount(new_high_dummy.cumsum()))

    return int(max_wait_days)


def portfolio_value_at_risk(re: pd.Series, cov: pd.DataFrame, weight: pd.Series, alpha=0.01) -> float:
    """
    计算组合的预期VaR，可认为是极端情况下的最大回撤

    Parameters
    ----------
    re: pd.Series
        收益序列
    cov: pd.DataFrame
        方差协方差矩阵
    weight: pd.Series
        权重序列
    alpha: float
        正态分布假定下的损失发生概率

    Returns
    -------
    float
        组合的预期Value at Risk

    """
    re_p = re.dot(weight)
    std_p = np.sqrt(weight.dot(cov).dot(weight))
    dist = norm.ppf(alpha)

    if re_p + dist * std_p > 0:
        return 0
    else:
        return re_p + dist * std_p


"""
计算风险类指标（日内）
"""

"""
计算择时类策略指标
"""


def cal_timing_indicator(net_value: pd.Series, bench_nv=None, rf=None) -> tuple:
    """
    计算择时策略的相关指标，包括
        win_rate，获胜概率，即获胜的天数除以总天数。
        lose_rate，失败概率，即失败的天数除以总天数。
        win_odds，胜算，即获胜概率除以失败概率，越大越好。
        PL_ratio, 盈亏比，获胜天数的收益除以失败天数的损失。

    Parameters
    ----------
    net_value
        策略净值的时间序列，支持原始账户总资产的时间序列

    bench_nv
        基准净值的时间序列，支持原始账户总资产的时间序列

    rf
        无风险收益的时间序列

    Returns
    -------
    tuple
        win_rate, loss_rate, win_odds, PL_ratio
    """

    if bench_nv is None:
        if rf is None:
            check(net_value)
            diff_nv = net_value.pct_change()
        else:
            check(net_value, rf)
            diff_nv = net_value.pct_change() - rf
    else:
        check(net_value, bench_nv)
        rtns_b = bench_nv.pct_change()
        diff_nv = net_value.pct_change() - rtns_b

    diff_nv = diff_nv.dropna()

    win_rate = sum(diff_nv > 0) / len(diff_nv)
    loss_rate = sum(diff_nv < 0) / len(diff_nv)
    win_odds = win_rate / (1 - win_rate) if win_rate is not 1 else np.inf

    loss_ret_mean = np.mean(diff_nv[diff_nv < 0])  # 失败日的每日平均损失
    win_ret_mean = np.mean(diff_nv[diff_nv > 0])  # 获胜日的每日平均收益

    if loss_ret_mean != 0:
        pl_ratio = abs(win_ret_mean / loss_ret_mean)
    else:
        pl_ratio = np.inf

    return win_rate, loss_rate, win_odds, pl_ratio
