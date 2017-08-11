import pandas as pd
from sqlalchemy import create_engine

# sql engine
zyyx_conn = create_engine(
    "mysql+pymysql://simu_sywg:fG37g9KfceV0YEsS@106.75.45.237:15077/CUS_FUND_DB?charset=gbk"
)

sws_conn = create_engine(
    "mssql+pymssql://pu_zhulan:pu_zhulan@192.30.1.40:1433/StructuredFund?charset=utf8"
)


# Wind quant api

wind_edb_dict = {
    # 国内宏观指标
    'M1同比':                             'M0001383',
    'M2同比':                             'M0001385',
    '金融机构新增人民币贷款':                'M0009973',
    '社会融资规模':                         'M5206730',
    '工业增加值当月同比':                    'M0000545',
    'PMI':                                'M0017126',
    '财新中国PMI':                         'M0000138',
    'CPI当月同比':                         'M0000612',
    'PPI全部工业品当月同比':                 'M0001227',
    '进出口金额当月同比':                    'M0000605',
    '70个大中城市新建住宅价格指数当月同比':     'S2707403',

    # 国内股票指数
    '上证综合指数':                         'M0020188',
    '上证50指数':                          'M0020223',
    '沪深300指数':                         'M0020209',
    '创业板指数':                          'M0062531',
    '中证500指数':                         'M0062541',

    # 国内债券指数
    '中债综合指数':                         'M0051552',
    '中债国债总指数':                       'M0051559',
    '中债企业债总指数':                     'M0051566',
    '中债信用债总指数':                     'M0265754',

    # 国内商品期货指数
    '南华综合指数':                         'S0105896',
    '南华工业品指数':                       'S0105897',
    '南华农产品指数':                       'S0105898',
    '南华金属指数':                         'S0105899',
    '南华能化指数':                         'S0105900',
    '南华贵金属指数':                       'S0200883',

    '上海金交所黄金现货Au9999':              'S0035819',

    # 美国股市
    'S&P500':                             'G0001672',

    # 香港股市
    '恒生指数':                            'G0001693',

    # 常用利率
    'SHIBOR_3m':                          'M0017142'
}


wind_wsd_dict = {
    # 基金指数
    '中证股票基金':                        'H11021.CSI',
    '中证债券基金':                        'H11023.CSI',
    '中证货币基金':                        'H11025.CSI'
}


if __name__ == '__main__':
    # sql_zzyx = """
    # select * from v_fund_asset_data limit 5
    # """

    # ## 获取照样永续数据库中的所有view
    # sql = """
    # show table status where comment='view'
    # """

    # sql = """
    # SELECT DISTINCT index_name FROM t_fund_index
    # """
    #
    # df = pd.read_sql(sql, zyyx_conn)
    # print(df)
    df = pd.DataFrame(wind_edb_dict, index=['code']).T
    df.to_csv('EDB_config.csv')