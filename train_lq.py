#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import os
import tensorflow as tf
from seq2seq.model_seq2seq_lq import Seq2Seq
from seq2seq.train_helper_lq import train_model
import time

from test import decoder
from utils import config
from utils.batcher import Batcher
from utils.common_utils import init_logger
from utils.data import Vocab
from utils.gpu_utils import config_gpu
from utils.wv_utils import WVTool
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 日志配置
logger = init_logger()


def train():
    # 配置GPU
    use_gpu = True
    config_gpu(use_gpu=use_gpu)

    # 读取字典 和 词向量矩阵
    vocab = Vocab(config.path_vocab, config.vocab_size)

    wvtool = WVTool(ndim=config.emb_dim)
    embedding_matrix = wvtool.load_embedding_matrix(path_embedding_matrix=config.path_embedding_matrixt)

    # 构建模型
    logger.info('构建Seq2Seq模型 ...')
    model=Seq2Seq(config.batch_size,embedding_matrix=embedding_matrix)


    # 存档点管理
    ckpt = tf.train.Checkpoint(Seq2Seq=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=config.dir_ckpt, max_to_keep=10)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logger.info('最新存档点加载自: {}'.format(ckpt_manager.latest_checkpoint))
    else:
        logger.info('无可加载的存档点: 重新训练 ...')

    # 获取训练数据
    batcher = Batcher(config.path_seg_train, vocab, mode='train',
                      batch_size=config.batch_size, single_pass=False)

    time.sleep(10)
    # 训练模型
    # 输入：训练数据barcher，模型，词表，存档点，词向量矩阵
    train_model(batcher, model=model,  ckpt_manager=ckpt_manager)



if __name__ == '__main__':
    # 训练模型
    train()
    decoder()


