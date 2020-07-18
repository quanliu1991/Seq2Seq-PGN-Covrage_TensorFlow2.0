#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


from multiprocessing import Pool

import numpy as np
import pandas as pd

from utils.common_utils import cpu_cores


def multiproc_df(df, func, cores=cpu_cores):
    # 按照进程数切分数据
    data_split = np.array_split(df, cores)
    # 进程池
    pool = Pool(cores)
    # 数据分发与合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭进程池
    pool.close()
    # 执行完close后不会有新的进程加入到pool, join函数等待所有子进程结束
    pool.join()
    return data
