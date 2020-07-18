#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/3/11 16:46
from utils.common_utils import cpu_cores
from utils.data_utils import AutoMasterDataHandler

# 语料数据前处理
def preprocess():
    # 加载数据处理器
    data_handler = AutoMasterDataHandler(cores=cpu_cores)
    # 建立数据集、词向量、词表
    data_handler.build_dataset()


if __name__ == "__main__":
    preprocess()