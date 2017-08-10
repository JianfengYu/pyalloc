
class Context:
    """全局运行对象， 用于记录当前时间和行情"""

    cur_time = None
    cur_quotes = None # 当前的行情
    pre_quotes = None