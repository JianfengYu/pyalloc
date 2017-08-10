import abc
import logging
from typing import Union, Dict

import pandas as pd
import numpy as np

from WindPy import w
from datetime import *

from pyalloc.backtest.enums import Frequency

logger = logging.getLogger(__name__)

class Loader(metaclass=abc.ABCMeta):
    """数据加载基类"""

    @abc.abstractmethod
    def read(self, sids:(str, list), start: Union[pd.Timestamp, str],
             end: Union[pd.Timestamp, str], frequency=None) -> pd.DataFrame:
        """读取资产数据"""

class WindEDBReader(Loader):

    def __init__(self):
        if not w.isconnected():
            w.start()

    def read(self, sids:(str, list), start: Union[pd.Timestamp, str],
             end: Union[pd.Timestamp, str], frequency=None) -> pd.DataFrame:

        start_date = pd.to_datetime(start) - pd.to_timedelta('1D') # 计算收益向前多取一天
        end_date = pd.to_datetime(end)

        wind_edb_res = w.edb(sids, start_date, end_date)
        df = pd.DataFrame(wind_edb_res.Data, index=wind_edb_res.Codes, columns=wind_edb_res.Times).T
        # df = df.fillna(method='ffill').pct_change().dropna()
        return df

class HDFLoader(Loader):

    def __init__(self, path):
        self._path = path

    def read(self, sids:(str, list), start: Union[pd.Timestamp, str],
             end: Union[pd.Timestamp, str], frequency=None) -> Dict[str, pd.DataFrame]:

        start_date = pd.to_datetime(start).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end).strftime('%Y-%m-%d')

        db = pd.HDFStore(self._path)
        result = {}
        for sid in sids:
            try:
                result[sid] = db[sid][start_date:end_date]
            except Exception as e:
                print(e)

        db.close()
        return result

if __name__ == '__main__':
    # # TEST EDB
    from pyalloc.data.api_config import wind_edb_dict

    ind_cn_simple = [
        '上证综合指数',
        '中债综合指数',
        '南华综合指数',
        'SHIBOR_3m'
    ]

    ind_cn_simple = '上证综合指数'

    # codes = [wind_edb_dict[a] for a in ind_cn_simple]
    codes = wind_edb_dict[ind_cn_simple]

    start = '2010-06-02'
    end = '2011-08-01'

    print(codes)
    DATA_LOADER = WindEDBReader()
    df = DATA_LOADER.read(codes, start, end)

    print(start, df.head(1), end, df.tail(1))
