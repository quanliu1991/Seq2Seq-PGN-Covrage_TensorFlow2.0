#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15

import tensorflow as tf

from seq2seq.model_utils import get_input_from_batch
from utils import data, config


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)





def sort_beams(beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)



def beam_search(batch,model,vocab):
    #batch should have only one example
    enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
        get_input_from_batch(batch)

    encoder_outputs, encoder_feature, encoder_hidden = model.encode_all(enc_batch)
    # s_t_0 = model.reduce_state(encoder_hidden)
    s_t_0 = encoder_hidden

    # dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
    # dec_h = dec_h.squeeze()
    # dec_c = dec_c.squeeze()

    #decoder batch preparation, it has beam_size example initially everything is repeated
    beams = [Beam(tokens=[vocab.word2id(data.START_DECODING)],
                  log_probs=[0.0],
                  state=(s_t_0[0]),
                  context = c_t_0[0],
                  coverage=(coverage_t_0[0] if config.is_coverage else None))
             for _ in range(config.beam_size)]
    results = []
    steps = 0
    while steps < config.max_dec_steps and len(results) < config.beam_size:
        latest_tokens = [h.latest_token for h in beams]
        latest_tokens = [t if t < vocab.size() else vocab.word2id(data.UNKNOWN_TOKEN) \
                         for t in latest_tokens]
        y_t_1 = tf.convert_to_tensor(latest_tokens)

        all_state =[]


        all_context = []

        for h in beams:
            # state_h, state_c = h.state
            # all_state_h.append(state_h)
            # all_state_c.append(state_c)
            all_state.append(h.state)

            all_context.append(h.context)

        # s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))

        # s_t_1 = tf.expand_dims(tf.reshape(tf.concat(all_state, 0),[-1,config.hidden_dim]),0)
        s_t_1 = tf.reshape(tf.concat(all_state, 0),[-1,config.hidden_dim])
        # c_t_1 = torch.stack(all_context, 0)

        coverage_t_1 = None
        if config.is_coverage:
            all_coverage = []
            for h in beams:
                all_coverage.append(h.coverage)
            coverage_t_1 = tf.concat(all_coverage, 0)

        final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = model.decoder(y_t_1, s_t_1,
                                                    encoder_outputs, encoder_feature, enc_padding_mask,
                                                    extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
        log_probs = tf.math.log(final_dist)
        topk_log_probs, topk_ids = tf.nn.top_k(log_probs, config.beam_size * 2)

        # dec_h, dec_c = s_t
        # dec_h = dec_h.squeeze()
        # dec_c = dec_c.squeeze()

        all_beams = []
        num_orig_beams = 1 if steps == 0 else len(beams)
        for i in range(num_orig_beams):
            h = beams[i]
            state_i = (s_t[i])
            context_i = c_t[i]
            coverage_i = (coverage_t[i] if config.is_coverage else None)

            for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                new_beam = h.extend(token=topk_ids[i][j],
                               log_prob=topk_log_probs[i] [j],
                               state=state_i,
                               context=context_i,
                               coverage=coverage_i)
                all_beams.append(new_beam)

        beams = []
        for h in sort_beams(all_beams):
            if h.latest_token == vocab.word2id(data.STOP_DECODING):
                if steps >= config.min_dec_steps:
                    results.append(h)
            else:
                beams.append(h)
            if len(beams) == config.beam_size or len(results) == config.beam_size:
                break

        steps += 1

    if len(results) == 0:
        results = beams

    beams_sorted = sort_beams(results)

    return beams_sorted[0]














# def merge_batch_beam(t: tf.Tensor):
#     # 输入: tensor of shape [batch_size, beam_size ...]
#     # 输出: tensor of shape [batch_size * beam_size, ...]
#     batch_size, beam_size = t.shape[0], t.shape[1]
#     return tf.reshape(t, shape=[batch_size * beam_size] + list(t.shape[2:]))
#
#
# def split_batch_beam(t: tf.Tensor, beam_size: int):
#     # 输入: tensor of shape [batch_size * beam_size ...]
#     # 输出: tensor of shape [batch_size, beam_size, ...]
#     return tf.reshape(t, shape=[-1, beam_size] + list(t.shape[1:]))
#
#
# def tile_beam(t: tf.Tensor, beam_size: int):
#     # 输入: tensor of shape [batch_size, ...]
#     # 输出: tensor of shape [batch_size, beam_size, ...]
#     multipliers = [1, beam_size] + [1] * (t.shape.ndims - 1)
#     return tf.tile(tf.expand_dims(t, axis=1), multipliers)
#
#
# class Hypothesis(object):
#     def __init__(self, tokens, log_probs, hidden):
#         # all tokens from time step 0 to the current time step t
#         self.tokens = tokens
#         # log probabilities of all tokens
#         self.log_probs = log_probs
#         # decoder hidden state after the last token decoding
#         self.hidden = hidden
#
#     def extend(self, token, log_prob, hidden):
#         """
#         Method to extend the current hypothesis by adding the next decoded token and
#         all information associated with it
#         """
#         tokens = self.tokens + [token]
#         log_probs = self.log_probs + [log_prob]
#         return Hypothesis(tokens, log_probs, hidden)
#
#     @property
#     def latest_token(self):
#         return self.tokens[-1]
#
#     @property
#     def tot_log_probs(self):
#         return sum(self.log_probs)
#
#     @property
#     def avg_log_probs(self):
#         return self.tot_log_probs / len(self.log_probs)
