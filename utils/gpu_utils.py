#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import os

import tensorflow as tf

from utils.common_utils import init_logger

# 日志配置
logger = init_logger()


def config_gpu(use_gpu=True):
    # 如果指定使用CPU
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return None, None
    # 如果指定使用GPU
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    if not physical_gpus:
        return None, None
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info('{} physical GPU(s), {} logical GPU(s)'.format(len(physical_gpus), len(logical_gpus)))
            return physical_gpus, logical_gpus
    except RuntimeError as e:
        logger.error(e)
        return None, None
