#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @author   : Quan Liu
# @date     : 2020/5/21 16:01

import time
import tensorflow as tf
from tqdm import tqdm
from seq2seq.model_seq2seq_lq import Seq2Seq
from seq2seq.model_utils import get_input_from_batch, get_output_from_batch
from utils import config
from utils.common_utils import init_logger
logger = init_logger()

def eval_model(batcher,
                model: Seq2Seq,
                ckpt_manager: tf.train.CheckpointManager) -> float:


    # 训练参数
    batch_size = config.eval_batch_size
    learning_rate = config.lr




    # 优化器
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=learning_rate)


    # 训练
    # @tf.function
    def train_step(batcher) -> float:
        with tf.GradientTape() as tape:
            # 获取训练输入数据
            enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = get_input_from_batch(
                batcher)
            dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch=get_output_from_batch(batcher)

            # # 解码器的首次输入 = 句子开始标签
            # dec_input = tf.expand_dims([bos_idx] * dec_target.shape[0], 1)
            # 计算输出
            step_losses, coverage = model.teacher_forcing_decode(enc_batch, dec_batch, enc_padding_mask, dec_padding_mask,extra_zeros, enc_batch_extend_vocab,
                               coverage)
            # 计算 batch loss (句子开始标识不计算在内)

            sum_losses = tf.reduce_sum(tf.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / dec_lens_var
            batch_loss = tf.reduce_mean(batch_avg_loss)
            # print(batch_loss)
            # 可训练参数
            variables = model.trainable_variables
            # 计算梯度
            gradients = tape.gradient(batch_loss, variables)
            # 更新梯度
            optimizer.apply_gradients(zip(gradients, variables))
            # 返回 batch loss
            return batch_loss



    # 进行验证
    start = time.time()
    total_loss = 0.
    # 遍历所有的Batch
    with tqdm(range(config.step_per_epoch_eval), unit='batch', desc='验证进度') as pbar:
        for step in pbar:
            batch=batcher.next_batch()
            # batch=batcher.next_batch()
            batch_loss = train_step(batch)
            total_loss += batch_loss

            pbar.set_postfix({'size': '{:d}'.format(batch_size),
                              'eval_loss': '{:.6f}'.format(batch_loss),
                              'eval_average': '{:.6f}'.format(total_loss / (step + 1))})
            pbar.update(1)

    epoch_loss = total_loss / config.step_per_epoch_eval#steps_per_epoch

    logger.info('验证集花费时间为 {:.2f} 分钟, Loss = {:.6f}'.format(
        (time.time() - start) / 60., epoch_loss))
    print('eval_loss:',epoch_loss.numpy())

    # 返回loss
    return epoch_loss
