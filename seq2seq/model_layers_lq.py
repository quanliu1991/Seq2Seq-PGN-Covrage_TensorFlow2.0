#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/3/25 9:52
import os

import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras

from utils import config

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix, **kwarg):
        super(Encoder, self).__init__(**kwarg)
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'
                                       , dropout=config.drop_rate)
        self.W_h = tf.keras.layers.Dense(enc_units, use_bias=False)
        self.dropout=tf.keras.layers.Dropout(config.drop_rate)

    def __call__(self, x, hidden):
        x = self.embedding(x)
        x=self.dropout(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.dropout(output)
        encoder_feature = tf.reshape(output, [-1, self.enc_units])
        encoder_feature = self.W_h(encoder_feature)
        # encoder_feature = tf.expand_dims(tf.reshape(encoder_feature, [self.batch_sz, -1, self.enc_units]), axis=2)

        return output, encoder_feature, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Attention(keras.layers.Layer):
    def __init__(self, hidden_dim, is_coverage, **kwargs):
        # self.linear = LinearLayer()
        self.hidden_dim = hidden_dim
        self.is_coverage = is_coverage
        super(Attention, self).__init__(**kwargs)

        # attention
        if self.is_coverage:
            self.W_c = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.decode_proj = tf.keras.layers.Dense(self.hidden_dim)
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def __call__(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        batch_size = encoder_outputs.get_shape()[0]  # if this line fails, it's because the batch size isn't defined
        h_dim = encoder_outputs.get_shape()[2]  # if this line fails, it's because the batch size isn't defined
        t_k = encoder_outputs.get_shape()[1]  # if this line fails, it's because the batch size isn't defined
        # attn_size = encoder_outputs.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*dec_units
        # dec_fea_expanded = tf.expand_dims(tf.expand_dims(dec_fea, 1), 1)
        # dec_fea_expanded = tf.expand_dims(tf.expand_dims(dec_fea, 1), 1)
        # 增加一个维度，并将维度改为t_k
        dec_fea_expanded = tf.tile(tf.reshape(dec_fea, (batch_size, 1, h_dim)), multiples=(1, t_k, 1))
        dec_fea_expanded=tf.reshape(dec_fea_expanded,[-1,h_dim])

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*dec_units

        if self.is_coverage:
            coverage_input = tf.reshape(coverage, [-1, 1])  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*dec_units
            att_features = att_features + coverage_feature

        e = tf.nn.tanh(att_features)  # B * t_k x 2*dec_units
        scores = self.v(e)  # B * t_k x 1

        scores = tf.reshape(scores, [batch_size, -1])  # B x t_k

        attn_dist_ = tf.nn.softmax(scores, axis=1) * enc_padding_mask  # B x t_k
        normalization_factor = tf.reduce_sum(attn_dist_, 1, keepdims=True)  # [[3], [3]]
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = tf.expand_dims(attn_dist, 1)  # B x 1 x t_k
        c_t = tf.matmul(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = tf.reshape(c_t, [batch_size, -1])  # B x 2*dec_units

        attn_dist = tf.reshape(attn_dist, [batch_size, -1])  # B x t_k

        if self.is_coverage:
            coverage = tf.reshape(coverage, [batch_size, -1])
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, vocab_size, embedding_dim, embedding_matrix, pointer_gen=True, **kwarg):

        self.pointer_gen = True
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pointer_gen = pointer_gen
        self.attention = Attention(self.dec_units,True)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=config.drop_rate)
        self.dropout=tf.keras.layers.Dropout(config.drop_rate)

        if self.pointer_gen:
            self.p_gen_linear = tf.keras.layers.Dense(1)

            # p_vocab
        self.out1 = tf.keras.layers.Dense(self.dec_units)
        self.out2 = tf.keras.layers.Dense(self.vocab_size)
        super(Decoder, self).__init__(**kwarg)

    def __call__(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                 extra_zeros, enc_batch_extend_vocab, coverage, step):

        if step == 0:
            c_t, _, coverage_next = self.attention(s_t_1, encoder_outputs, encoder_feature,
                                                   enc_padding_mask, coverage)

            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)  # batch_size x 1 x embedding_dim
        y_t_1_embd=self.dropout(y_t_1_embd)
        y_t_1_embd=tf.expand_dims(y_t_1_embd,1)
        gru_out, s_t = self.gru(y_t_1_embd, s_t_1)  # batch_size x 1 x embedding_dim   batch_size x embedding_dim
        gru_out=self.dropout(gru_out)

        s_t_hat = s_t  # batch_size x embedding_dim
        c_t, attn_dist, coverage_next = self.attention(s_t_hat, encoder_outputs, encoder_feature,
                                                       enc_padding_mask, coverage)

        if step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            y_t_1_embd = tf.reshape(y_t_1_embd, [-1, self.embedding_dim])
            p_gen_input = tf.concat((c_t, s_t_hat, y_t_1_embd), 1)  # B x (2*2*dec_units + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = tf.sigmoid(p_gen)

        output = tf.concat((tf.reshape(gru_out, [-1, self.dec_units]), c_t), 1)  # B x dec_units * 3
        output = self.out1(output)  # B x dec_units

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = tf.nn.softmax(output, axis=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            if extra_zeros is not None:
                vocab_dist_ = tf.concat([vocab_dist_, extra_zeros], 1)
            shape_ = vocab_dist_.shape[1]
            enc_batch_extend_vocab=tf.expand_dims(enc_batch_extend_vocab,2)
            attn_vocab_dist_ = tf.convert_to_tensor([tf.scatter_nd(indices, updates, [shape_]) for (indices, updates) in
                                                     zip(enc_batch_extend_vocab, attn_dist_)])
            final_dist = vocab_dist_ + attn_vocab_dist_
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


'''
class LinearLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

    def __call__(self, args, output_size, bias, bias_start=10.0):
        if args is None or (isinstance(args, (list, tuple)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        matrix = self.add_weight(name="Matrix", shape=(total_arg_size, output_size),
                                 initializer='uniform',
                                 trainable=True)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = self.add_weight(name="Bias", shape=(output_size,), initializer=tf.constant_initializer(bias_start))
        return res + bias_term


class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建所需要的参数"""
        # x * w + b.  input_shape:[None , a ] w:[a,b] output_shape:[None b]
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)


class PointerGeneratorLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        pass

    def build(self, input_shape):
        self.wh = self.add_weight()
        self.ws = self.add_weight()
        self.wx = self.add_weight()
        self.bias = self.add_weight()

    def call(self, ht, st, xt):
        """完成正向计算"""
        self.pgen = self.activation(ht @ self.wh + st @ self.ws + xt @ self.wx + self.bias)
        return self.pgen

'''
if __name__ == "__main__":
    # input=[[1., 2., 3.], [2., 3., 4.], [3., 5., 7.]]
    # input = tf.convert_to_tensor(input)
    # attention=Attention(4,True)
    # print(attention.W_c(input))
    # x = [[1., 2., 3.], [2., 3., 4.], [3., 5., 7.]]
    # y = [[2., 2., 3., 6], [3., 4., 6., 8], [4., 6., 7., 8.]]
    # x = tf.convert_to_tensor(x)
    # y = tf.convert_to_tensor(y)
    # linear = LinearLayer()
    # result = linear([x, y], 128, True)
    # print(result)

    input_encoder = [[2, 2, 3], [2, 3, 4], [3, 5, 7], [4, 5, 6]]
    input_encoder = tf.convert_to_tensor(input_encoder)
    embedding_matrix = np.zeros([100, 4])
    encoder = Encoder(100, 4, 8, 4, embedding_matrix)
    hid = encoder.initialize_hidden_state()
    encoder_outputs, encoder_out_fea, encoder_state = encoder(input_encoder, hid)
    print("encoder_outputs=", encoder_outputs)
    print("encoder_out_fea=", encoder_out_fea)
    print("encoder_state=", encoder_state)
    # attention = Attention(8, False)
    s_t_hat = tf.convert_to_tensor(
        [[3., 4., 3., 4., 3., 4., 3., 4.], [3., 4., 3., 4., 3., 4., 3., 4.], [3., 4., 3., 4., 3., 4., 3., 4.],
         [3., 4., 3., 4., 3., 4., 3., 4.]])
    y_t_1 = tf.convert_to_tensor([[3], [4], [3], [5]])
    enc_padding_mask = [[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]]
    extra_zeros = tf.zeros([4, 3], tf.float32)
    # enc_batch_extend_vocab=tf.convert_to_tensor([[100,101,102],[100,101,102],[100,101,102],[100,101,102]])
    enc_batch_extend_vocab = tf.convert_to_tensor(
        [[[50], [101], [102]], [[100], [101], [102]], [[100], [101], [102]], [[100], [101], [102]]])

    coverage=tf.zeros_like(input_encoder,dtype=float)
    decoder = Decoder(8, 100, 4, embedding_matrix, True)
    sorces = decoder(y_t_1, s_t_hat, encoder_outputs, encoder_out_fea, enc_padding_mask, extra_zeros=extra_zeros,
                     enc_batch_extend_vocab=enc_batch_extend_vocab, coverage=coverage, step=0)
    # sorces = attention(s_t_hat, encoder_outputs, encoder_out_fea,
    #                    enc_padding_mask=enc_padding_mask, coverage=False)

    print("sdfg:::::::::::::::::", sorces)
    # indices = tf.constant([[[0], [1], [3], [2]],[[0], [1], [100], [2]]])
    # updates = tf.constant([[5, 5, 5, 5],[5, 5, 3, 4]])
    # shape = [103]
    # for (indices, updates, shape) in zip(indices, updates, shape):
    #     print(indices)
    #     print(indices)
    #     print(indices)
    # a =tf.convert_to_tensor([[1,2,3],[2,3,4]])
    # print(a.shape[1])
    #
    # scatter =tf.convert_to_tensor([tf.scatter_nd(indices, updates, shape) for (indices, updates) in zip(indices, updates)])
    #
    # print(scatter)

"""
(4, 2, 1, 3)= (4, 2, 1, 3)+(4, 1, 1, 3)
    a=tf.convert_to_tensor([[[[2, 2, 3]],[[2, 3, 4]]], [[[2, 3, 4]],[[2, 3, 4]]], [[[3, 5, 7]],[[2, 3, 4]]], [[[4, 5, 6]],[[2, 3, 4]]]])
    b=tf.convert_to_tensor([[[[2, 2, 3]]], [[[2, 3, 4]]], [[[3, 5, 7]]], [[[4, 5, 6]]]])
    c=a+b
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(c)"""
