import numpy as np
import pandas as pd
import scipy as spy

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Dict, Union, List
from pyalloc.config import TRADING_DAYS_A_YEAR

sns.set_style("whitegrid")

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class EventStudy:

    def __init__(self, ret_df: pd.DataFrame, event_date: Dict[str, list],
                 pre_min=60, after_min=120, pre_window=60, after_window=120):

        self._ret_df = ret_df
        self._event_date = deepcopy(event_date)
        self._valid_ob = event_date.keys()
        self._pre_min = pre_min
        self._after_min = after_min
        self._pre_window = pre_window
        self._after_window = after_window
        self._abnormal_ret = None
        self._ret_window = None

        self.check()

    def check(self):
        """
        检查有效性, 剔除不符合计算长度的event date

        Returns
        -------

        """
        assert isinstance(self._ret_df.index, pd.DatetimeIndex), "The index of ret_df has to be pd.DatetimeIndex!"
        assert len(set(self._valid_ob) - set(self._ret_df.columns)) == 0, "The keys of event_date are out of ret_df.colums!"
        assert self._pre_min >= self._pre_window and self._after_min >= self._after_window, "Window range are not match!"

    def cal_abnormal_ret(self, pre_window: int, after_window: int) -> pd.DataFrame:

        day_count = [ a - pre_window for a in range(pre_window)] + [ b for b in range(after_window+1)]
        result = pd.DataFrame([], index=day_count)

        for ob in self._event_date.keys():
            tmp_ret = self._ret_df[ob].copy()
            tmp_event_date = self._event_date[ob]

            total_index = tmp_ret.index.union(pd.DatetimeIndex(tmp_event_date)).sort_values()
            tmp_ret = tmp_ret.reindex(total_index).fillna(0)  # 包含eventday的收益序列

            for date in tmp_event_date:

                event_point = list(tmp_ret.index).index(date)

                if event_point > self._pre_min and len(total_index)-2 - event_point > self._after_min:

                    event_slice = [a + event_point for a in day_count]

                    # print(event_slice)
                    # print(len(tmp_ret))
                    try:
                        tmp_ret_slice = tmp_ret[event_slice].values
                        result[(ob, date)] = pd.Series(tmp_ret_slice, index=day_count)
                    except Exception as e:
                        print(day_count)
                        print(date, event_point, self._pre_min, self._after_min)
                        print(len(tmp_ret), len(total_index))
                        print(event_slice)
                        raise e

        return result

    def plot(self, title='Event Study', fill_between=True, q_l=0.25, q_u=0.75):

        # 事件发生前后的收益
        self._ret_window = self.cal_abnormal_ret(pre_window=self._pre_window, after_window=self._after_window
                                                   )

        cumsum = (self._ret_window+1).cumprod() - 1

        self._abnormal_ret = self._ret_window.mean(axis=1)
        car = (self._abnormal_ret + 1).cumprod() - 1

        # self._abnormal_ret.plot(kind='bar', color='#836FFF', alpha=0.8)
        plt.vlines(self._abnormal_ret.index, 0, self._abnormal_ret.values,
                   colors='#836FFF', alpha=0.9, label='AR')
        car.plot(kind='line', color='r', label='CAR')

        if fill_between:
            plt.fill_between(cumsum.index,
                         cumsum.quantile(q_l, axis=1).values,
                         cumsum.quantile(q_u, axis=1).values, alpha=0.5)
        plt.title(title)
        plt.legend(loc=2)
        plt.show()



if __name__ == "__main__":
    ret = pd.DataFrame(np.random.randn(50, 2), index=pd.date_range(start='2010-01-01', periods=50), columns=['我', 'B'])
    event_day = {
        '我': [pd.to_datetime('2010-02-01')],
        'B': [pd.to_datetime('2010-01-13')]
    }

    event_study = EventStudy(ret_df=ret, event_date=event_day, pre_min=5, after_min=5, pre_window=5, after_window=5)
    print(event_study.cal_abnormal_ret(pre_window=5, after_window=5))
    event_study.plot('哈哈')