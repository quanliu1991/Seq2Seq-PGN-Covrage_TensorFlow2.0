#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import argparse


def get_model_params():
    parser = argparse.ArgumentParser()
    # 带默认值的参数
    parser.add_argument('--max_enc_len', default=10, type=int, help='编码器输入最大长度')
    parser.add_argument('--max_dec_len', default=10, type=int, help='解码器输入最大长度')
    parser.add_argument('--batch_size', default=256, type=int, help='batch大小(训练)')
    parser.add_argument('--batch_test', default=64, type=int, help='batch大小(测试)')
    parser.add_argument('--epochs', default=200, type=int, help='训练epoch总数')
    parser.add_argument('--embedding_dim', default=512, type=int, help='词向量维度')
    parser.add_argument('--enc_units', default=512, type=int, help='编码器隐层单元数')
    parser.add_argument('--dec_units', default=512, type=int, help='解码器隐层单元数')
    parser.add_argument('--att_units', default=32, type=int, help='注意力中间层单元数')
    parser.add_argument('--learning_rate', default=1.E-4, type=float, help='学习率')
    parser.add_argument('--min_dec_steps', default=1, type=int, help='beam search最小长度')
    parser.add_argument('--max_dec_steps', default=32, type=int, help='beam search最大长度')
    parser.add_argument('--rnn_type', default='gru', type=str, help='RNN单元类型')
    parser.add_argument('--train_mode', default='cv_no_hidden', type=str, help='训练模式')
    parser.add_argument('--train_size', default=-1, type=int, help='训练样本大小')
    parser.add_argument('--test_mode', default='greedy', type=str, help='测试模式')
    parser.add_argument('--beam_size', default=3, type=int, help='beam search宽度')
    parser.add_argument('--test_size', default=-1, type=int, help='训练样本大小')
    # 布尔型参数，特殊处理
    use_gpu_parser = parser.add_mutually_exclusive_group(required=False)
    use_gpu_parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='使用GPU')
    use_gpu_parser.add_argument('--use_cpu', dest='use_gpu', action='store_false', help='使用GPU')
    parser.set_defaults(use_gpu=True)
    # 将参数存在字典中
    args = parser.parse_args()
    params = vars(args)
    return params
