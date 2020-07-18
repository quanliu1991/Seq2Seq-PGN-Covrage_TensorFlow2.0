#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# from seq2seq.batcher import build_train_batch
from eval import evaluate
from seq2seq.model_seq2seq_lq import Seq2Seq
from seq2seq.model_utils import get_input_from_batch, get_output_from_batch
from utils import config
from utils.common_utils import init_logger
from utils.wv_utils import WVTool

# 日志配置
logger = init_logger()



def train_model(batcher,
                model: Seq2Seq,
                ckpt_manager: tf.train.CheckpointManager) -> float:


    # 训练参数
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = config.lr
    max_enc_len = config.max_enc_steps
    max_dec_len = config.max_dec_steps



    # 优化器
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=learning_rate)

    # 损失函数
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    # @tf.function
    # def loss_func(real: tf.Tensor, pred: tf.Tensor) -> float:
    #     # 计算 损失
    #     loss = loss_object(real, pred)
    #     # 忽略 pad 和 unk
    #     pad_mask = tf.math.equal(real, pad_idx)
    #     # unk_mask = tf.math.equal(real, unk_idx)
    #     # mask = tf.math.logical_not(tf.math.logical_or(pad_mask, unk_mask))
    #     # unk mask会导致预测结果中缺少UNK项，故而去掉，只保留pad mask
    #     mask = tf.math.logical_not(pad_mask)
    #     mask = tf.cast(mask, dtype=loss.dtype)
    #     loss *= mask
    #     # 返回按 平均的 loss
    #     return tf.reduce_mean(loss)

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

    # 构建训练数据集


    # 进行训练
    epoch_loss = -1
    train_loss=[]
    eval_loss=[]
    for epoch in tqdm(range(epochs)):
        start = time.time()
        total_loss = 0.
        # 遍历所有的Batch
        with tqdm(range(config.step_per_epoch), unit='batch', desc='训练进度') as pbar:
            for step in pbar:
                batch=batcher.next_batch()
                # batch=batcher.next_batch()
                batch_loss = train_step(batch)
                total_loss += batch_loss

                pbar.set_postfix({'size': '{:d}'.format(batch_size),
                                  'loss': '{:.6f}'.format(batch_loss),
                                  'average': '{:.6f}'.format(total_loss / (step + 1))})
                pbar.update(1)
        # 定期保存模型: 每个Epoch
        path_ckpt_save = ckpt_manager.save()
        # 计算平均loss:
        epoch_loss = total_loss / config.step_per_epoch#steps_per_epoch
        logger.info('Epoch {:3d}: 新存档点保存在 {}'.format(
            epoch + 1, path_ckpt_save))
        logger.info('Epoch {:3d}: 训练花费时间为 {:.2f} 分钟, Loss = {:.6f}'.format(
            epoch + 1, (time.time() - start) / 60., epoch_loss))
        print('Epoch:',epoch + 1,'\ntrain_loss:',epoch_loss.numpy())
        train_loss.append(epoch_loss.numpy())
        eval_epoch_loss=evaluate()
        eval_loss.append(eval_epoch_loss.numpy())
    print('train_loss:',train_loss)
    print('eval_loss:',eval_loss)
    # 返回loss
    return epoch_loss



