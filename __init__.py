import os
import logging
from . import backtest, data

def init_logging(filename="./strategy.log"):
    """
    日志初始化

    Parameters
    ----------
    filename: str
        日志输出文件名，默认为 ./strategy.log
    """

    # 文件夹路径检查
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)

    #初始化日志
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename, 'w', 'utf-8')
    handler.setFormatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    root_logger.addHandler(handler)