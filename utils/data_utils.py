#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15

import os
import pathlib
import re

import jieba
import numpy as np
import pandas as pd

from utils.common_utils import cpu_cores, init_logger, init_jieba, SWTool
from utils.multiproc_utils import multiproc_df
from utils.wv_utils import WVTool
from utils import config


class AutoMasterDataHandler(object):

    def __init__(self, cores=cpu_cores):
        super(AutoMasterDataHandler, self).__init__()
        self.cores = cores
        self.wvtool = WVTool(ndim=config.emb_dim, cores=self.cores, epochs=config.wv_epochs)
        self.logger = init_logger()
        self.cols = {'X': ['权利要求'], 'Y': ['正例文本']}
        self.cols['X+Y'] = self.cols['X'] + self.cols['Y']
        self.max_len = {'X': None, 'Y': None}
        self.df_all, self.df_train, self.df_eval, self.df_test, self.df_merged = pd.DataFrame(
            columns=['专利序号', '权利要求', '正例文本']), pd.DataFrame(columns=['专利序号', '权利要求', '正例文本']), pd.DataFrame(
            columns=['专利序号', '权利要求', '正例文本']), pd.DataFrame(columns=['专利序号', '权利要求', '正例文本']), pd.DataFrame(
            columns=['专利序号', '权利要求', '正例文本'])
        self.path_userdict = None
        self.path_stopwords = None
        self.swtool = None
        self.path_data_train, self.path_data_eval, self.path_data_test = None, None, None
        self.path_seg_train, self.path_seg_test, self.path_seg_eval, self.path_seg_merged = None, None, None, None
        self.path_pad_train_X, self.path_pad_test_X, self.path_pad_train_Y = None, None, None
        self.path_wv_model, self.path_embedding_matrix = None, None
        self.path_vocab, self.path_vocab_w2i, self.path_vocab_i2w = None, None, None
        self.initialize_params()

    def initialize_params(self):
        # 自定义词典
        self.path_userdict = config.path_userdict
        init_jieba(path_userdict=self.path_userdict, cores=self.cores)
        # 停用词
        self.path_stopwords = config.path_stopwords
        self.swtool = SWTool(self.path_stopwords)
        # 训练、验证、测试数据
        self.path_data_all = config.path_data_all
        self.path_data_train = config.path_data_train
        self.path_data_eval = config.path_data_eval
        self.path_data_test = config.path_data_test
        # 分词结果
        self.path_seg_train = config.path_seg_train
        self.path_seg_eval = config.path_seg_eval
        self.path_seg_test = config.path_seg_test
        self.path_seg_merged = config.path_seg_merged
        # PADDING结果
        self.path_pad_train_X =config.path_pad_train_X
        self.path_pad_eval_X = config.path_pad_eval_X
        self.path_pad_train_Y = config.path_pad_train_Y
        # 词向量模型、词嵌入矩阵和字典
        self.path_wv_model = config.path_wv_mode
        self.path_embedding_matrix = config.path_embedding_matrixt
        self.path_vocab = config.path_vocab
        self.path_vocab_w2i = config.path_vocab_w2i
        self.path_vocab_i2w = config.path_vocab_i2w

    def build_dataset(self):
        # 载入原始数据
        self.load_raw_data(path_data_all=self.path_data_all)
        # 分词
        self.tokenize()
        # 并划分训练、验证、测试集
        self.divide_dataset(self.df_all)
        # 首次训练词向量
        self.wvtool.build_model(self.path_seg_merged, first=True)
        # PADDING并增量训练词向量（UNK、BOS、EOS、PAD）
        self.pad(path_pad_train_X=self.path_pad_train_X,
                 path_pad_eval_X=self.path_pad_eval_X,
                 path_pad_train_Y=self.path_pad_train_Y)
        # 保存词向量模型, 字典, 词向量矩阵
        self.wvtool.save_model(self.path_wv_model)
        self.wvtool.save_embedding_matrix(self.path_embedding_matrix)
        self.wvtool.save_vocab(path_vocab=self.path_vocab,
                               path_vocab_w2i=self.path_vocab_w2i,
                               path_vocab_i2w=self.path_vocab_i2w)
        return self.df_train, self.df_eval, self.df_test, self.df_all

    def load_raw_data(self, path_data_all):
        if path_data_all:
            self.path_data_all = path_data_all
            self.df_all = pd.read_csv(self.path_data_all)
        return self.df_train, self.df_test

    def tokenize(self):
        # 去掉空值
        self.df_all.dropna(subset=self.cols['X+Y'], how='any', inplace=True)
        # 文本预处理
        self.df_all=multiproc_df(self.df_all, self.proc_df)
        # self.proc_df(self.df_all)
        return self.df_all


    def load_seg_data(self, path_seg_train, path_seg_test, path_seg_merged):
        if path_seg_train:
            self.path_seg_train = path_seg_train
            self.df_train = pd.read_csv(path_seg_train, header=0, index_col=None)
        if path_seg_test:
            self.path_seg_test = path_seg_test
            self.df_test = pd.read_csv(path_seg_test, header=0, index_col=None)
        if path_seg_merged:
            self.path_seg_merged = path_seg_merged
            self.df_merged = pd.read_csv(path_seg_merged, header=0, index_col=None)
        return path_seg_train, path_seg_test, path_seg_merged

    def divide_dataset(self, df_all, train_ratio=8, eval_ratio=1):
        # 划分训练 验证、测试集
        # 目前复杂度为数据行数，可以进一步优化。
        _index_div = 0
        for _,row in df_all.iterrows():
            # 划分训练集
            if _index_div < train_ratio:
                self.df_train= self.df_train.append(row, ignore_index=True)
            # 划分验证集
            if train_ratio <= _index_div and _index_div < (train_ratio + eval_ratio):
                self.df_eval= self.df_eval.append(row, ignore_index=True)
            # 划分测试集
            if _index_div == (train_ratio + eval_ratio):
                self.df_test=self.df_test.append(row, ignore_index=True)
                _index_div = -1
            _index_div += 1
        print("train_num,eval_num.test_num")
        print(len(self.df_train),len(self.df_eval),len(self.df_test))
        # 保存全部数据、训练、验证、测试集
        self.df_all.to_csv(self.path_seg_merged, index=None, header=True)
        self.df_train.to_csv(self.path_seg_train, index=None, header=True)
        self.df_eval.to_csv(self.path_seg_eval, index=None, header=True)
        self.df_test.to_csv(self.path_seg_test, index=None, header=True)

        return self.df_train, self.df_eval, self.df_test, self.df_all

    def pad(self, path_pad_train_X, path_pad_eval_X, path_pad_train_Y):
        # 训练集的'X'和'Y'，测试集的'X'
        self.df_train['X'] = self.df_train[self.cols['X']].apply(lambda x: ' '.join(x), axis=1)
        self.df_eval['X'] = self.df_eval[self.cols['X']].apply(lambda x: ' '.join(x), axis=1)
        self.df_train['Y'] = self.df_train[self.cols['Y']].apply(lambda x: ' '.join(x), axis=1)
        # 获取句子序列长度
        if config.max_enc_steps is None:
            max_len_X_train = self.wvtool.max_len(self.df_train['X'])
            max_len_X_test = self.wvtool.max_len(self.df_eval['X'])
            self.max_len['X'] = max(max_len_X_train, max_len_X_test)
        else:
            self.max_len['X']=config.max_enc_steps
        if config.max_dec_steps is None:
            self.max_len['Y'] = self.wvtool.max_len(self.df_train['Y'])
        else:
            self.max_len['Y']=config.max_dec_steps
        # PADDING
        self.df_train['X'] = self.wvtool.pad_df(self.df_train['X'], self.max_len['X'], self.wvtool.vocab)
        self.df_eval['X'] = self.wvtool.pad_df(self.df_eval['X'], self.max_len['X'], self.wvtool.vocab)
        self.df_train['Y'] = self.wvtool.pad_df(self.df_train['Y'], self.max_len['Y'], self.wvtool.vocab)
        # 保存PAD结果
        if path_pad_train_X:
            self.path_pad_train_X = path_pad_train_X
            self.df_train[['X']].to_csv(path_pad_train_X, header=True, index=None)
            self.wvtool.build_model(self.path_pad_train_X, first=False)
        if path_pad_eval_X:
            self.path_pad_test_X = path_pad_eval_X
            self.df_eval[['X']].to_csv(path_pad_eval_X, header=True, index=None)
            self.wvtool.build_model(self.path_pad_test_X, first=False)
        if path_pad_train_Y:
            self.path_pad_train_Y = path_pad_train_Y
            self.df_train[['Y']].to_csv(path_pad_train_Y, header=True, index=None)
            self.wvtool.build_model(self.path_pad_train_Y, first=False)

        return self.df_train['X'], self.df_eval['X'], self.df_train['Y']

    def load_pad_data(self, path_pad_train_X, path_pad_test_X, path_pad_train_Y):
        if path_pad_train_X:
            self.path_pad_train_X = path_pad_train_X
            self.df_train[['X']] = pd.read_csv(path_pad_train_X, header=True, index_col=None)
        if path_pad_test_X:
            self.path_pad_test_X = path_pad_test_X
            self.df_test[['X']] = pd.read_csv(path_pad_test_X, header=True, index_col=None)
        if path_pad_train_Y:
            self.path_pad_train_Y = path_pad_train_Y
            self.df_train[['Y']] = pd.read_csv(path_pad_train_Y, header=True, index_col=None)
        return self.df_train['X'], self.df_test['X'], self.df_train['Y']

    def proc_df(self, df):
        for col in df.columns:
            if col not in self.cols['X+Y']:
                continue
            df[col] = df[col].apply(self.proc_sentence)
        return df

    def proc_sentence(self, sentence):
        sentence = AutoMasterDataHandler.clean_sentence(sentence)
        words = jieba.lcut(sentence)
        # words = self.swtool(words)
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return ' '.join(words)

    @classmethod
    def clean_sentence(cls, sentence):
        if not isinstance(sentence, str):
            return ''
        sentence = re.sub(r'[\s+\-\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:；;+——()?【】“”！？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                          ' ', sentence)
        sentence = re.sub(r"[a-zA-Z]*", "", sentence)
        sentence = re.sub(r"[0-9]*", "", sentence)

        return sentence

    def preprocess_sentence(self, sentence):
        path_root = pathlib.Path(os.path.abspath(__file__)).parent.parent
        wv1 = WVTool()
        wv1.load_vocab(os.path.join(path_root, 'data', 'wv', 'vocab.txt'))
        wv1.load_vocab_dict(os.path.join(path_root, 'data', 'wv', 'vocab.w2i.txt'))
        sentence = self.proc_sentence(self.clean_sentence(sentence))
        sentence = wv1.pad_sentence(sentence, len(sentence), wv1.vocab)
        sentence = wv1.sent2idx(sentence, wv1.vocab_w2i)
        return sentence
