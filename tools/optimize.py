import pandas as pd
import numpy as np
import warnings
# import cvxopt as opt
# from cvxopt import matrix, blas, solvers, log, div, spdiag

# turn off progress printing
# solvers.options['show_progress'] = False

import scipy.optimize as sopt  # 解凸优化


def risk_parity_solver(cov: pd.DataFrame, tol=1e-6, max_loop=100000):
    """
    求解 risk parity

    Parameters
    ----------
    cov: np.matrix
        证券的协方差矩阵，要求正定，如果不正定会抛出异常
    tol: float
        误差容忍，默认为 1e-6
    max_loop: int
        最大迭代步数，默认为 100000

    Returns
    -------
    np.array
        列向量表示的权重

    Examples
    --------
    >>> import numpy as np
    >>> C = np.matrix([[10, 20], [20, 100]])
    >>> x = risk_parity_solver(C)
    >>> print(x)
    [[ 0.75974693]
     [ 0.24025307]]

    >>> np.array(C * x) * x
    array([[ 9.42278461],
       [ 9.42278461]])

    References
    ---------
    FLORIN SPINU (2013), AN ALGORITHM FOR COMPUTING RISK PARITY WEIGHTS
    """
    C = np.matrix(cov.values)

    # 证券数量
    n = C.shape[0]

    assert n > 1, "至少有两个证券"

    # 检查 C 是否正定
    try:
        np.linalg.cholesky(C)
    except Exception as e:
        print(e, "协方差矩阵不正定，无法求解")
        raise e

    # 等风险占比
    b = np.ones((n, 1)) * (1/n)

    # 初始 u，表示 Cx 与 bx^{-1} 的差
    u = np.ones((n, 1))

    # 根据 u 计算初始权重
    x = (np.sqrt(np.sum(b)) / np.sqrt(u.T @ C @ u))[0, 0] * u

    # see reference formula （15）
    lbd_star = 0.95 * (3 - np.sqrt(5)) / 2

    # 循环求解
    # 注意：矩阵乘法使用 * 或 @ 都一样，但 @ 意义更明确
    for i in range(max_loop):
        u = C @ x - b * (1 / x)
        H = C + np.diag((b*1/(x**2)).T[0])
        dx = H**(-1) @ u
        delta = np.abs(np.max(dx / x))
        lbd = np.sqrt(u.T @ dx)[0, 0]

        if lbd > lbd_star:
            x = np.array(x - dx / (1 + delta))
            continue

        if lbd > tol:
            x = np.array(x - dx)

        else:
            break

    res = x/x.sum()

    return [a[0] for a in res]


def risk_budget_solver(cov: pd.DataFrame, risk_budget: list, tol=1e-10):
    """
    风险预算优化器

    Parameters
    ----------
    cov: pd.DataFrame
        方差协方差矩阵

    risk_budget: list
        风险预算

    tol: float
        迭代优化的精度

    Returns
    -------

    """
    assert len(risk_budget) == cov.shape[0], "The number of risk budget is not match!"

    cov = np.mat(cov)

    x0 = np.ones(cov.shape[0]) / cov.shape[0]  # 优化的初始值X0，次数默认将其设定为总体平均值

    # 定义目标函数及其导数为： 其中sign>0表示求解最小值,sign<0表示求解最大值
    def func_risk_budget(x, sign=1.0, cov_day=cov, b=risk_budget):
        n = cov_day.shape[0]
        temp = 0
        """ Objective function """
        for stepi in range(n):
            # print(x[stepi])
            temp = temp + (x[stepi] * (cov_day * np.mat(x).T)[stepi, 0] / ((np.mat(x) * cov_day * np.mat(x).T)[0, 0]) -
                           b[stepi]) ** 2
        return sign * temp

    # 约束条件，注意ineq此处含义是>=，等式及不等式右侧默认为0，如sum(x)-1=0

    cons_risk_budget = ({'type': 'eq', 'fun': lambda x: np.array([sum(x) - 1])},
                        {'type': 'ineq', 'fun': lambda x: np.array(x)},
                        {'type': 'ineq', 'fun': lambda x: 1 - np.array(x)})

    res_risk_budget = sopt.minimize(func_risk_budget, x0, constraints=cons_risk_budget, method='SLSQP',
                                   options={'disp': False}, tol=tol)
    # print(type(res_risk_budget))
    return res_risk_budget['x']


def Markovitz_solver(r, C, tau=None, bound=None, target_vol=None, x0=None, tol=1e-10,
                     constrains=({'type': 'eq',  'fun': lambda w: sum(w) - 1.0})
                     ):
    """
    Markovitz优化器

    Parameters
    ----------
    r
        预期收益
    C
        方差协方差矩阵
    tau
        风险厌恶系数，和目标波动率必须制定一个
    bound
        边界条件
    constrains
        限制条件，默认权重之和为1
    target_vol
        目标波动率
    x0
        迭代初始值，程序中默认为均配
    tol
        优化容忍最小误差

    Returns
    -------

    """
    assert tau is not None , "Tau should be specified!"

    # 检查 C 是否正定
    try:
        np.linalg.cholesky(C)
    except Exception as e:
        print(e, "协方差矩阵不正定，无法求解")
        raise e

    numAsset = len(r)

    # 初始权重
    if x0 is None:
        w0 = 1.0 * np.ones(numAsset) / numAsset
    else:
        w0 = x0

    # # 边界条件，默认不能卖空
    # if bound is None:
    #     bounds = [(0, 1) for x in range(numAsset)]
    # else:
    #     bounds = bound

    # 风险厌恶系数设置
    t = tau

    if target_vol is not None:
        cons = (
            {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: target_vol - np.dot(np.dot(w, C), w)}
        )
    else:
        cons = constrains


    def objFunc(w, r, C, tau): # tau为风险厌恶系数
        val = t/2 * np.dot(np.dot(w, C), w) - sum(w * r)
        return val

    result = sopt.minimize(objFunc, w0, (r, C, t), method='SLSQP', constraints=cons, bounds=bound, tol=tol)
    w_opt = result.x

    return w_opt


def max_sharpe_solver(r, C, bound=None,
                      constrains = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0}),
                      min_weight=0.05, Num=500, tol= 1e-10):
    """
    最大化shape ratio，迭代方法，保证了资产的最小权重

    Parameters
    ----------
    r
        期望收益

    C
        期望方差协方差矩阵

    bound
        边界条件，[(w_low, w_up) for i in range(len(r))]的形式给入

    min_weight
        资产的权重不能小于min_weight，否则为0。

    Num
        迭代次数

    Returns
    -------
    w_s
        最优权重

    """
    numAsset = len(r)

    # 初始权重
    w = 1.0 * np.ones(numAsset) / numAsset

    # # 边界条件，默认不能卖空
    # if bound is None:
    #     bounds = [(0, 1)] * numAsset
    # else:
    #     bounds = bound

    # # 权重之和为1
    # constrain = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0})

    N = Num

    # 记录结果
    s_max = -100.0              # 最优sharpe ratio
    w_s = np.zeros(numAsset)    # 组合的最优配置比例
    r_s = 0.0                   # 组合最优时期望收益
    C_s = 0.0                   # 组合最优时的方差

    # def objFunc(w, r, C, tau): # tau为风险厌恶系数
    #     val = tau * np.dot(np.dot(w, C), w) - sum(w * r)
    #     return val

    for tau in [10 ** (5.0 * t / N - 1.0) for t in range(N)]:
        # result = opt.minimize(objFunc, w, (r, C, tau), method='SLSQP', constraints=constrain, bounds=bound)
        # w_opt = result.x

        w_opt = Markovitz_solver(r=r, C=C, tau=tau, bound=bound, constrains=constrains, tol=tol)

        for i in range(numAsset):
            if w_opt[i] < min_weight:
                w_opt[i] = 0.0
        w_opt = w_opt / sum(w_opt)

        r_opt = sum(w_opt * r)
        C_opt = np.dot(np.dot(w_opt, C), w_opt)
        s = r_opt / C_opt

        if s_max < s:
            s_max = s
            w_s = w_opt
            r_s = r_opt
            C_s = C_opt

    return w_s #, s_max, r_s, C_s


def Markovitz_l2_penalty(r, C, xb, A, tau=None, bound=None, target_vol=0.1,
                         x0=None, tol=1e-10, lmbd=0.5,
                         constraints=({'type': 'eq',  'fun': lambda w: sum(w) - 1.0})
                         ):
    assert tau is not None or target_vol is not None, "One of the tau and target_vol should be specified!"

    numAsset = len(r)

    # 初始权重
    if x0 is None:
        w0 = 1.0 * np.ones(numAsset) / numAsset
    else:
        w0 = x0

    # 风险厌恶系数设置 可以由目标波动率反推出
    if tau is None:
        t = np.sqrt(np.dot(np.dot(r, np.linalg.inv(C)), r)) / target_vol
    else:
        t = tau

    lmbd_scaled = lmbd * t

    def objFunc(w, r, C, tau):  # tau为风险厌恶系数
        val = t / 2 * np.dot(np.dot(w, C), w) - sum(w * r) + lmbd_scaled / 2 * np.dot(np.dot(w-xb, A), w-xb)
        return val

    result = sopt.minimize(objFunc, w0, (r, C, t), method='SLSQP', constraints=constraints, bounds=bound, tol=tol)
    w_opt = result.x

    return w_opt


""""""""""""""
" 以下暂时不用 "
""""""""""""""


def max_sharpe_solver1(r, C, bound=None, tol= 1e-10):
    """
    最大化sharpe ratio（理论求解器）

    Parameters
    ----------
    r
        期望收益

    C
        期望方差协方差矩阵

    bound
        边界条件，[(w_low, w_up) for i in range(len(r))]的形式给入

    min_weight
        资产的权重不能小于min_weight，否则为0。

    Num
        迭代次数

    Returns
    -------
    w_s
        最优权重

    """

    numAsset = len(r)

    # 初始权重
    w = 1.0 * np.ones(numAsset) / numAsset

    # 权重之和为1
    constrain = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0})


    def negative_sharpe(w):
        w = np.array(w)
        ret = np.dot(r,w)
        var = np.dot(np.dot(w, C), w)
        return -ret/var

    result = sopt.minimize(negative_sharpe, w, method='SLSQP', bounds=bound, constraints=constrain, tol=tol)
    w_s = result.x

    return w_s


def Markovitz_mu_solver(r, C, target_vol=0.4, bound=None, constrains=None, tol= 1e-10):
    """
    Markovitz 优化器

    Parameters
    ----------
    r
        期望收益
    C
        期望方差协方差举证
    tau
        风险厌恶系数，越大越风险厌恶
    bound
        边界条件
    Returns
    -------

    """
    numAsset = len(r)

    # 判断可行域
    if np.min(np.var(C)) > target_vol * target_vol:
        warnings.warn("The target vol is less than the smallest vol!")
        return np.array((np.var(C) <= np.min(np.var(C))) * 1)

    else:
        vol = target_vol

    # 初始权重
    w0 = 1.0 * np.ones(numAsset) / numAsset

    # # 边界条件，默认不能卖空
    # if bound is None:
    #     bounds = [(0, 1) for x in range(numAsset)]
    # else:
    #     bounds = bound
    #
    # 权重之和为1
    if constrains is None:
        cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
                     {'type': 'ineq', 'fun': lambda w: vol * vol - np.dot(np.dot(w, C), w)})
    else:
        cons = constrains

    def objFunc(w, r, C):  # tau为风险厌恶系数
        val = - sum(w * r)
        return val

    result = sopt.minimize(objFunc, w0, (r, C), method='SLSQP', constraints=cons, bounds=bound, tol=tol)
    w_opt = result.x

    return w_opt


""""""""""
"" 以下为原始code
"""""""""""

# def markovitz(re_day: pd.Series, cov_day: pd.DataFrame, target_vol: float, mu=None):
#     n = len(re_day)
#     S = matrix(np.asmatrix(cov_day))
#     pbar = matrix(np.array(re_day))
#
#     if mu is None:
#         m = float(np.sqrt(np.dot(np.dot(re_day, np.linalg.inv(cov_day)), re_day)) / target_vol)
#     else:
#         m = mu
#
#     G = - matrix(np.eye(n))
#     h = matrix(0.0, (n ,1))
#     A = matrix(1.0, (1, n))
#     b = matrix(1.0)
#
#     sol = solvers.qp(m*S, -pbar, G, h, A, b)
#
#     return list(sol['x'])
#
#
# def markovitz2(re_day: pd.Series, cov_day: pd.DataFrame, target_vol: float, mu=None, max_weight=0.8):
#     n = len(re_day)
#     S = matrix(np.asmatrix(cov_day))
#     pbar = matrix(np.array(re_day))
#
#     if mu is None:
#         m = float(np.sqrt(np.dot(np.dot(re_day, np.linalg.inv(cov_day)), re_day)) / target_vol)
#     else:
#         m = mu
#
#     G1 = - np.eye(n)
#     G2 = np.eye(n)
#     G = matrix(np.vstack((G1, G2)))
#
#     h1 = np.zeros(n)
#     h2 = np.ones(n) * max_weight
#     h = matrix(np.hstack((h1,h2)))
#
#     A = matrix(1.0, (1, n))
#     b = matrix(1.0)
#
#     sol = solvers.qp(m*S, -pbar, G, h, A, b)
#
#     return list(sol['x'])
#
#
# def markovitz_c_l(re_day: pd.Series, cov_day: pd.DataFrame, target_vol: float):
#     """
#     带杠杆
#     """
#     n = len(re_day)
#     S = matrix(np.asmatrix(cov_day))
#     pbar = matrix(np.array(re_day))
#
#     mu = np.sqrt(np.dot(np.dot(re_day, np.linalg.inv(cov_day)), re_day)) / target_vol
#
#     G1 = - np.eye(n)
#     G2 = np.eye(n)
#     G = matrix(np.vstack((G1, G2)))
#
#     h1 = np.zeros(n)
#     h1[-3] = 0.35
#     h2 = np.ones(n)
#     h2[-2] = 1.35
#     h2[-1] = 0.3
#     # h2[-3] = 0.2
#     h = matrix(np.hstack((h1, h2)))
#
#     A = matrix(1.0, (1, n))
#     b = matrix(1.0)
#
#     sol = solvers.qp(mu*S, -pbar, G, h, A, b)
#
#     return list(sol['x'])

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # 获取收益率数据
    db = pd.HDFStore('..\data\DB.h5')
    ret = db['ret_index']
    ret = ret[['Bond', 'Stock']]
    ret = ret.dropna()
    db.close()

    bound = [(0, 1)]*2

    # res_1 = max_sharpe_solver(ret.mean(), ret.cov(), bound=bound)
    #
    # print(res_1)

    res_2 = Markovitz_solver(ret.mean(), ret.cov(), target_vol=0.1, bound=bound)
    print(res_2)

    res_rp = risk_parity_solver(ret.cov())
    print(res_rp)

    res_3 = Markovitz_l2_penalty(ret.mean(), ret.cov(), xb=res_rp, A=np.diag(np.diag(ret.cov())), target_vol=0.1,
                                 bound=bound)
    print(res_3)
