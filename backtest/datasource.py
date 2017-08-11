import abc
import pandas as pd
from typing import Union

from pyalloc.backtest.enums import Frequency
from pyalloc.data.loader import WindEDBReader, HDFLoader


class DataSource(metaclass=abc.ABCMeta):
    """
    数据源基类
    DataSource 是一个由 {str:dataframe} 组成的数据集，每个 key 对应于一个证券，每个 dataframe 都具有相同的 pd.DatatimeIndex
    """

    def __init__(self, sids: list, frequency: Frequency):
        self._sids = list(set(sids))        # 去掉重复
        self._frequency = frequency         # 数据频率
        self._data = {}                     # dict of pd.DataFrame 每一个键表示一个证券，DataFrame的columns表示证券信息，Index为时间
        self._columns = None                # list of str pd.DataFrame的列名，默认所有DataFrame的列都一样
        self._iter = {}                     # dict of iter 对应于 _data, key为sid，value为DataFrame.itertuples(),由于迭代
        self._index = []                    # 对齐之后共有的时间索引转换成的 list 初始化为空
        self._size = 0

        self._nrow = 0                      # 迭代器位置标记

        self._external_data = None          # 外部数据,如因子数据和宏观数据

    def _check_valid(self):
        """
        检查数据是否符合规范，尤其是是否 DataFrame 是否具有必要的字段
        """
        assert len(self._sids) > 0, "证券代码列表不能为空"

        # 必须字段
        assert 'pct_change' in self._columns, "数据必须包含 pct_change 字段"

    @property
    def sids(self) -> list:
        return self._sids

    @property
    def columns(self) -> list:
        return self._columns

    @property
    def index(self) -> list:
        return self._index

    @property
    def frequency(self):
        return self._frequency

    @property
    def data(self):
        return self._data

    @property
    def external_data(self):
        return self._external_data

    def __getitem__(self, sid) -> pd.DataFrame:
        """
        获取数据源底层的数据

        Parameters
        ----------
        sid: str
            证券代码：形如 600000.SH，交易代码+交易所形成的该证券的 **唯一** 标识

        Returns
        -------
        pd.DataFrame
            sid 对应的 DataFrame
        """
        return self._data[sid]

    def __len__(self):
        """
        获取 DataSource 的长度，即 共有的时间索引 的长度

        Returns
        -------
        int
            所有 DataFrame 共有的 DatetimeIndex 的长度
        """
        return self._size

    def __iter__(self):
        # 初始化迭代器位置
        self._nrow = 0
        # 生成 _iter 字典
        for sid, df in self._data.items():
            self._iter[sid] = df.itertuples(index=True, name='Quote')

        return self

    def __next__(self) -> dict:
        """
        获取下一个时刻点的行情字典

        Returns
        -------
        dict
            其中一个 key 为 time，value为当前时间
            其他 key 为证券代码，value为当前的行情(namedtuple)，如
                Quote(Index="2016-01-01", price=15.16, quantity=50200)
        """
        bars = {}

        # 对每个证券
        for sid, it in self._iter.items():
            bars[sid] = next(it)
        bars['time'] = self._index[self._nrow]
        self._nrow += 1  # 移动 + 1

        return bars

    def get_hist(self, until: int, length: int) -> dict:
        """
        获取指定段的数据

        Parameters
        ----------
        until: int
            截止的位置
        length: int
            往回取的长度

        Returns
        -------
        dict of pd.DataFrame
            所有标的指定段的历史数据
        """
        assert length > 0, "所取数据长度必须大于 0, length = {0}".format(length)

        hists = {}
        if length > until - 1:
            return hists

        for sid, df in self._data.items():
            hists[sid] = df[until - length - 1: until - 1]

        return hists

    def to_hdf(self, path: str):
        """
        将数据保存为hdf文件

        Parameters
        ----------
        path: str
            hdf文件保存路径

        Returns
        -------

        """
        db = pd.HDFStore(path)
        for sid in self._sids:
            db[sid] = self._data[sid]
        db.close()


class WindDataSource(DataSource):
    """主要用于从Wind读取和存储"""

    def __init__(self, sids: list, start: Union[pd.Timestamp, str], end: Union[pd.Timestamp, str],
                 benchmark: Union[str, None], riskfree: Union[str, None], frequency=None):
        super().__init__(sids, frequency)
        self._benchmark = benchmark

        print('loading data from wind edb')

        loader = WindEDBReader()
        self._data = loader.read(self._sids, start, end)

        for sid in self._sids:
            if self._columns is None:
                self._columns = self._data[sid].columns
            if len(self._index) == 0:
                self._index = self._data[sid].index
            else:
                # 获得所有index的交集
                self._index = self._index.intersection(self._data[sid].index)

        if benchmark is not None and benchmark not in self._sids:
            if isinstance(benchmark, str):
                df_bench = loader.read(benchmark, start, end)[benchmark]

                if not isinstance(df_bench.index, pd.DatetimeIndex):
                    raise Exception("DataFrame 的 index 必须是 pd.DatetimeIndex 类型")

                self._data[benchmark] = pd.DataFrame({'pct_change': df_bench[benchmark]})
            else:
                raise Exception("benchmark must be a sid")

        if riskfree is not None and riskfree not in self._sids:
            if isinstance(riskfree, str):
                df_riskfree = loader.read(riskfree, start, end)[riskfree]

                if not isinstance(df_riskfree.index, pd.DatetimeIndex):
                    raise Exception("DataFrame 的 index 必须是 pd.DatetimeIndex 类型")

                self._data[riskfree] = pd.DataFrame({'pct_change': df_riskfree[riskfree]})
            else:
                raise Exception("riskfree must be a sid")

        self._index = self._index.tolist()

        self._size = len(self._index)

        self._check_valid()


class HDFDataSource(DataSource):

    def __init__(self, hdfpath, sids: list, start: Union[pd.Timestamp, str], end: Union[pd.Timestamp, str],
                 benchmark: str, riskfree: str, frequency=None):
        super().__init__(sids, frequency)
        self._benchmark = benchmark
        self._path = hdfpath

        loader = HDFLoader(self._path)
        self._data = loader.read(self._sids, start, end)

        for sid in self._sids:
            if self._columns is None:
                self._columns = self._data[sid].columns
            if len(self._index) == 0:
                self._index = self._data[sid].index
            else:
                self._index = self._index.intersection(self._data[sid].index)

        if benchmark is not None and benchmark not in self._sids:
            if isinstance(benchmark, str):
                df_bench = loader.read(benchmark, start, end)[benchmark]

                if not isinstance(df_bench.index, pd.DatetimeIndex):
                    raise Exception("DataFrame 的 index 必须是 pd.DatetimeIndex 类型")

                self._data[benchmark] = pd.DataFrame({'pct_change': df_bench[benchmark]})
            else:
                raise Exception("benchmark must be a sid")

        if riskfree is not None and riskfree not in self._sids:
            if isinstance(riskfree, str):
                df_riskfree = loader.read(riskfree, start, end)[riskfree]

                if not isinstance(df_riskfree.index, pd.DatetimeIndex):
                    raise Exception("DataFrame 的 index 必须是 pd.DatetimeIndex 类型")

                self._data[riskfree] = pd.DataFrame({'pct_change': df_riskfree[riskfree]})
            else:
                raise Exception("riskfree must be a sid")

        self._index = self._index.tolist()

        self._size = len(self._index)

        self._check_valid()


if __name__ == "__main__":
    pass
    # # 从wind更新所有回测行情数据
    #
    # from pyalloc.data.api_config import wind_edb_dict
    # edb_config = pd.read_excel('D:\PersonalProjects\pyalloc\pyalloc\EDB_config.xlsx')
    #
    # sids = []
    # for iter in edb_config.itertuples():
    #     if iter.type == 0 or iter.type ==2:
    #         sids.append(iter.code)
    #
    # benchmark_name = '上证综合指数'
    # riskfree_name = 'SHIBOR_3m'
    #
    # benchmark = wind_edb_dict[benchmark_name]
    # riskfree = wind_edb_dict[riskfree_name]
    #
    # start = '1990-01-01'
    # end = pd.datetime.now().strftime('%Y-%m-%d')
    #
    # data_source = WindDataSource(sids, start, end, benchmark, riskfree)
    # data_source.to_hdf('test_db.h5')
    # print(data_source.data)
    #
    # data_source2 = HDFDataSource('test_db.h5', sids, '2007-01-01', end, benchmark, riskfree)
    # print('reading data from HDF...')
    # print(data_source2.data)
