#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/6/7 22:24
import tensorflow as tf


class ELMo_layer(tf.keras.layers):
    def __init__(self, hiddendim):
        self.hidden_dim = hiddendim
        # self.lstm_fw_cell=tf.keras.layers.LSTMCell()
        # self.lstm_bw_cell=tf.keras.layers.LSTMCell()
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim), merge_mode='concat')

    def __call__(self, *args, **kwargs):
        output, h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM())
