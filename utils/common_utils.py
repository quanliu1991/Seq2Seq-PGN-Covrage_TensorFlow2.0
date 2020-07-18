#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import logging
from multiprocessing import cpu_count
import jieba


# 配置日志
def init_logger():
    logging.basicConfig(level=logging.INFO, format='[ %(asctime)s ] - [ %(levelname)s ]: %(message)s')
    logger = logging.getLogger()
    return logger


# 多进程核心数
# cpu_cores = cpu_count()
cpu_cores=1

# 配置分词器
def init_jieba(path_userdict, cores=1):
    if path_userdict:
        jieba.load_userdict(path_userdict)
    if cores > 0:
        jieba.enable_parallel(cores)


# 停用词工具
class SWTool(object):
    def __init__(self, path_stopwords=None):
        super(SWTool, self).__init__()
        self.stopwords = self.load(path_stopwords)
        self.logger = init_logger()

    def load(self, path_stopwords):
        self.stopwords = []
        if not path_stopwords:
            return self.stopwords
        try:
            with open(path_stopwords, 'r', encoding='utf-8') as f:
                self.stopwords = f.readlines()
                self.stopwords = [sw.strip() for sw in self.stopwords]
        except RuntimeError as e:
            self.logger.error(e)
        return self.stopwords

    def __call__(self, words):
        return [word for word in words if word not in self.stopwords]
