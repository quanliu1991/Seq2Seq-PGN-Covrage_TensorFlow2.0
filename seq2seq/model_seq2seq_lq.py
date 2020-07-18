#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/3/25 10:01


from typing import Tuple
import numpy as np
import tensorflow as tf
from seq2seq.model_layers_lq import Encoder, Decoder
from utils import config



class Seq2Seq(tf.keras.Model):
    def __init__(self, batch_size,embedding_matrix: np.ndarray) -> None:
        super(Seq2Seq, self).__init__()

        self.vocab_size=config.vocab_size
        self.embedding_dim=config.emb_dim
        self.enc_units=config.hidden_dim
        self.dec_units=config.hidden_dim
        self.batch_size=batch_size
        self.embedding_matrix = embedding_matrix
        self.encoder = Encoder(vocab_size= self.vocab_size,
                               embedding_dim= self.embedding_dim,
                               enc_units= self.enc_units,
                               batch_sz=self.batch_size,
                               embedding_matrix=self.embedding_matrix)
        self.decoder = Decoder( dec_units=self.dec_units,
                                vocab_size= self.vocab_size,
                               embedding_dim= self.embedding_dim,
                               embedding_matrix=self.embedding_matrix)

    def encode_all(self, enc_input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        hidden = self.encoder.initialize_hidden_state()
        enc_output, encoder_feature, enc_hidden = self.encoder(x=enc_input, hidden=hidden)
        return enc_output, encoder_feature, enc_hidden

    def decode_one(self, dec_input: tf.Tensor, dec_hidden: tf.Tensor,
                   enc_output: tf.Tensor, encoder_feature: tf.Tensor, enc_padding_mask,
                   extra_zeros, enc_batch_extend_vocab, coverage, step):

        assert (isinstance(enc_output, tf.Tensor))
        # context_vector, attention_weights = self.attention(dec_hidden, values=enc_output)
        # prediction, dec_hidden = self.decoder(inputs=dec_input, hidden=context_vector, mode=mode)
        final_dist, s_t, c_t, attn_dist, p_gen, coverage = self.decoder(dec_input, dec_hidden, enc_output,
                                                                        encoder_feature, enc_padding_mask,
                                                                        extra_zeros, enc_batch_extend_vocab, coverage,
                                                                        step)
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


    def teacher_forcing_decode(self, enc_input, dec_input, enc_padding_mask,dec_padding_mask ,extra_zeros, enc_batch_extend_vocab,
                               coverage):
        # 最终输出结果列表
        step_loss_list, coverage_list = [], []
        # 编码器输出
        enc_output, encoder_feature, enc_hidden = self.encode_all(enc_input)
        # 复制隐层状态
        dec_hidden = enc_hidden
        # dec_target是以'句子开始标识符'开头，因此移动一位
        dec_target = dec_input[:, 1:]
        # 最大解码步数
        max_steps = dec_target.shape[1]
        # 计算输出
        for t in range(max_steps):
            y_t_1 = dec_input[:, t]
            # 单步解码
            final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = self.decode_one(y_t_1, dec_hidden,
                                                                                    enc_output, encoder_feature,
                                                                                    enc_padding_mask, extra_zeros,
                                                                                    enc_batch_extend_vocab, coverage, t)

            # 下次计算的输入: 基于Teacher Forcing
            output = dec_target[:, t]
            # dec_input = tf.expand_dims(output, axis=1)
            dec_hidden = s_t
            coverage = next_coverage
            # 根据output从final_dist取出p(w*)

            # 构建一个numpy的arange列表，其长度为tensor的行数
            line = np.arange(len(output)).reshape(-1, 1)
            output=np.expand_dims(output,1)
            index = np.hstack((line, output))
            p_w = tf.gather_nd(final_dist, index)

            # 计算step_loss
            step_loss = -tf.math.log(p_w + config.eps)
            if config.is_coverage:
                step_coverage_loss = tf.reduce_sum(tf.minimum(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, t]
            step_loss = step_loss * step_mask
            step_loss_list.append(step_loss)

            # 存储结果
            # step_loss_list.append(p_w)
            coverage_list.append(coverage)
        # 返回结果
        return step_loss_list, coverage_list

    @property
    def trainable_variables(self):
        return self.encoder.trainable_variables + \
               self.decoder.trainable_variables




