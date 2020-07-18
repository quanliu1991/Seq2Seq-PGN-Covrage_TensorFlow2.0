#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/5/21 16:31
import numpy as np

from utils import config
import tensorflow as tf


def get_input_from_batch(batch):
    batch_size = len(batch.enc_lens)

    # enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_batch = batch.enc_batch

    # enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_padding_mask = batch.enc_padding_mask
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        # enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        enc_batch_extend_vocab = batch.enc_batch_extend_vocab
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            # extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
            extra_zeros = tf.zeros([batch_size, batch.max_art_oovs])

    # c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
    c_t_1 = tf.zeros([batch_size, 2 * config.hidden_dim])

    coverage = None
    if config.is_coverage:
        coverage = tf.zeros_like(enc_batch,dtype=float)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


def get_output_from_batch(batch):
    dec_batch = batch.dec_batch
    dec_padding_mask = batch.dec_padding_mask
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = dec_lens

    target_batch = batch.target_batch

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch