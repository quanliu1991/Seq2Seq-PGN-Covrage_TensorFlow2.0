import math
import os
import pathlib

# 根目录位置
path_root = pathlib.Path(os.path.abspath(__file__)).parent.parent
path_data=os.path.join(path_root, 'data')

# 存档点位置
dir_ckpt = os.path.join(path_root, 'model', 'checkpoints')
# os.makedirs(dir_ckpt, 0o755, exist_ok=True)

# 其它文件位置
# 用户词典
path_userdict=os.path.join(path_data, 'userdict', 'userdict.AutoMaster.txt')
# 停用词表
path_stopwords=os.path.join(path_data,  'stopwords', 'stopwords.AutoMaster.txt')

# 原始训练、验证、测试数据
path_data_all=os.path.join(path_data, 'raw', 'all_raw_data.csv')
path_data_train=os.path.join(path_data, 'raw', 'corpus_train.csv')
path_data_eval=os.path.join(path_data, 'raw', 'corpus_eval.csv')
path_data_test=os.path.join(path_data, 'raw', 'corpus_test.csv')

# 经过分词的训练、验证、测试数据
path_seg_train=os.path.join(path_data, 'seg', 'train.seg.csv')
path_seg_eval=os.path.join(path_data, 'seg', 'eval.seg.csv')
path_seg_test=os.path.join(path_data, 'seg', 'test.seg.csv')
#经过分词的训练+验证+测试数据
path_seg_merged=os.path.join(path_data, 'seg', 'merged.seg.csv')

# padding训练X、训练Y、验证X
path_pad_train_X=os.path.join(path_data, 'pad', 'train.pad.X.csv')
path_pad_train_Y=os.path.join(path_data, 'pad', 'train.pad.Y.csv')
path_pad_eval_X=os.path.join(path_data, 'pad', 'eval.pad.X.csv')

# 词向量模型、词嵌入矩阵
path_wv_mode=os.path.join(path_data, 'wv', 'wv.model')
path_embedding_matrixt=os.path.join(path_data, 'wv', 'embedding.matrix.txt')

# 词表、词表正向词典（词-->序号）、词表逆向词典（序号-->词）
path_vocab=os.path.join(path_data, 'wv', 'vocab.txt')
path_vocab_w2i=os.path.join(path_data, 'wv', 'vocab.w2i.txt')
path_vocab_i2w=os.path.join(path_data, 'wv', 'vocab.i2w.txt')

# Hyperparameters
epochs=200
wv_epochs=10
hidden_dim= 512
emb_dim= 256
batch_size=256
eval_batch_size=256
max_enc_steps=30
max_dec_steps=30
beam_size=4
min_dec_steps=5
vocab_size=12831
example_num=8056#训练集总数
eval_num=1007#验证集总数
test_num=1007#测试集总数

cov_loss_wt = 1
eps = 1e-7
lr=1e-4
drop_rate=0.3
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
lr_scheduler_gamma=0.8
lr_scheduler_step_size=2

pointer_gen = True
is_coverage = True

max_iterations = 10000
use_gpu=True
step_per_epoch=math.floor(example_num/batch_size)#一个epoch的训练次数
step_per_epoch_eval=math.floor(eval_num/eval_batch_size)#一个epoch的训练次数