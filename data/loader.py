import abc
import logging
from typing import Union, Dict
import os

import pandas as pd
import numpy as np

from WindPy import w
from datetime import *

from pyalloc.backtest.enums import Frequency
from pyalloc.config import TRADING_DAYS_A_YEAR

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

        # reading config file
        # print(os.getcwd())
        edb_config = pd.read_excel('D:\PersonalProjects\pyalloc\pyalloc\EDB_config.xlsx')
        self._config = {}
        for row in edb_config.itertuples(index=False, name='EDB'):
            self._config[row.code] = row

    def get_raw_data(self, sids:(str, list), start: Union[pd.Timestamp, str],
             end: Union[pd.Timestamp, str], frequency=None) -> pd.DataFrame:

        start_date = pd.to_datetime(start) - pd.to_timedelta('1D') # 计算收益向前多取一天
        end_date = pd.to_datetime(end)

        wind_edb_res = w.edb(sids, start_date, end_date)
        df = pd.DataFrame(wind_edb_res.Data, index=wind_edb_res.Codes, columns=wind_edb_res.Times).T
        # df = df.fillna(method='ffill').pct_change().dropna()
        return df

    def read(self, sids:(str, list), start: Union[pd.Timestamp, str],
             end: Union[pd.Timestamp, str], frequency=None) -> Dict[str, pd.DataFrame]:
        # 读取原始数据
        df = self.get_raw_data(sids, start, end, frequency)
        result = {}
        # 不同数据采用不同的处理方法
        for column in df.columns:
            print(column)
            edb = self._config[column]
            # 指数数据计算每日变动
            if edb.type == 0:
                result[column] = pd.DataFrame(
                    {'pct_change': df[column].fillna(method='ffill').pct_change().dropna()}
                )
            # 百分比数据
            elif edb.type == 1:
                # result[column] = df[column]/100
                print('{0} 非行情数据，不能被用于回测！'.format(edb.cn_name))
                pass #TODO
            # 年化收益数据
            elif edb.type == 2:
                result[column] = pd.DataFrame(
                    {'pct_change': (np.exp(np.log(df[column]/100 + 1)/ TRADING_DAYS_A_YEAR) - 1).dropna()}
                )
            # 宏观存量
            elif edb.type == 3:
                print('{0} 非行情数据，不能被用于回测！'.format(edb.cn_name))
                pass #TODO
            # 宏观增量
            elif edb.type ==4:
                print('{0} 非行情数据，不能被用于回测！'.format(edb.cn_name))
                pass #TODO
            else:
                print("{0} 指标类型不能被识别!".format(edb.cn_name))

        return result


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
    # from pyalloc.data.api_config import wind_edb_dict
    #
    # ind_cn_simple = [
    #     '上证综合指数',
    #     '中债综合指数',
    #     '南华综合指数',
    #     'SHIBOR_3m'
    # ]
    #
    # ind_cn_simple = '上证综合指数'
    #
    # # codes = [wind_edb_dict[a] for a in ind_cn_simple]
    #
    # codes = list(wind_edb_dict.values())
    # reverse_dict = dict([(wind_edb_dict[key], key) for key in wind_edb_dict.keys()])
    #
    # start = '1990-01-01'
    # end = pd.datetime.now().strftime('%Y-%m-%d')
    #
    # # print(codes)
    # DATA_LOADER = WindEDBReader()
    # df = DATA_LOADER.read(codes, start, end)
    # df.rename(columns=reverse_dict, inplace=True)
    #
    # db = pd.HDFStore('pyalloc.h5')
    # db['edb'] = df
    # db.close()
    #
    # df.to_csv('edb_df.csv')

    # 测试EDB数据读取
    config = pd.read_excel('EDB_config.xlsx')
    for row in config.itertuples(index=False, name='EDB'):
        print(row)