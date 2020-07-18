#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15


import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from utils.common_utils import init_logger


class CustomToken(object):
    def __init__(self, name, word, index):
        super(CustomToken, self).__init__()
        self.name = name
        self.word = word
        self.index = index


class WVTool(object):
    def __init__(self, ndim=512, cores=1, epochs=10):
        super(WVTool, self).__init__()
        self.ndim = ndim
        self.cores = cores
        self.epochs = epochs
        self.model, self.vocab = None, None
        self.vocab_w2i, self.vocab_i2w = None, None
        self.embedding_matrix = None
        self.logger = init_logger()
        self.labels = {
            'PAD': CustomToken(name='PAD', word='<PAD>', index=-1),
            'UNK': CustomToken(name='UNK', word='<UNK>', index=-1),
            'BOS': CustomToken(name='BOS', word='<BOS>', index=-1),
            'EOS': CustomToken(name='EOS', word='<EOS>', index=-1)
        }

    def build_model(self, path_corpus, first=True):
        if not path_corpus:
            return self.model, self.vocab, self.embedding_matrix
        if not first and not self.model:
            return self.model, self.vocab, self.embedding_matrix
        if first:
            self.logger.info('开始训练词向量......: {}'.format(path_corpus))
            self.model = Word2Vec(LineSentence(path_corpus),
                                  size=self.ndim,
                                  sg=1,
                                  workers=self.cores,
                                  iter=self.epochs,
                                  window=5,
                                  min_count=2)
        else:
            self.logger.info('增量训练词向量......: {}'.format(path_corpus))
            self.model.build_vocab(LineSentence(path_corpus), update=True)
            self.model.train(LineSentence(path_corpus),
                             epochs=self.epochs,
                             total_examples=self.model.corpus_count)
        self.build_vocab_from_model()
        self.build_embedding_matrix()
        return self.model, self.vocab, self.embedding_matrix

    def build_vocab_from_model(self, model=None):
        if model:
            assert (isinstance(model, Word2Vec))
            self.model = model
        else:
            assert (isinstance(self.model, Word2Vec))
        self.vocab = self.model.wv.index2word
        self.build_vocab_from_list(vocab=self.vocab)
        self.build_labels()
        return self.vocab, self.vocab_w2i, self.vocab_i2w

    def build_vocab_from_list(self, vocab):
        assert (isinstance(vocab, list))
        self.vocab = vocab
        self.vocab_w2i = {word: index for index, word in enumerate(self.vocab)}
        self.vocab_i2w = {index: word for index, word in enumerate(self.vocab)}
        self.build_labels()
        return self.vocab, self.vocab_w2i, self.vocab_i2w

    def build_vocab_from_dict(self, vocab_dict, reverse=False):
        assert (isinstance(vocab_dict, dict))
        if reverse:
            self.vocab_i2w = vocab_dict
            self.vocab_w2i = self.reverse_vocab_dict(vocab_dict)
        else:
            self.vocab_w2i = vocab_dict
            self.vocab_i2w = self.reverse_vocab_dict(vocab_dict)
        self.vocab = sorted(self.vocab_w2i.keys())
        self.build_labels()
        return self.vocab, self.vocab_w2i, self.vocab_i2w

    def build_labels(self):
        assert (isinstance(self.vocab_w2i, dict))
        for k, v in self.labels.items():
            if v.word in self.vocab_w2i:
                self.labels[k].index = self.vocab_w2i[v.word]
            else:
                self.labels[k].index = -1
        return self.labels

    def build_embedding_matrix(self, model=None):
        if model:
            assert (isinstance(model, Word2Vec))
            self.model = model
        else:
            assert (isinstance(self.model, Word2Vec))
        self.embedding_matrix = self.model.wv.vectors

    def save_model(self, path_model):
        assert (isinstance(self.model, Word2Vec))
        self.logger.info('保存词向量模型到: {}'.format(path_model))
        self.model.save(path_model)

    def load_model(self, path_model):
        self.logger.info('读取词向量模型从: {}'.format(path_model))
        self.model = Word2Vec.load(path_model)
        self.build_vocab_from_model()
        self.build_embedding_matrix()
        return self.model

    def save_vocab(self, path_vocab, path_vocab_w2i, path_vocab_i2w):
        if isinstance(path_vocab, str):
            with open(path_vocab, 'w', encoding='utf-8') as f:
                for i in self.vocab:
                    f.write('{}\n'.format(i))
        if isinstance(path_vocab_w2i, str):
            with open(path_vocab_w2i, 'w', encoding='utf-8') as f:
                for k, v in self.vocab_w2i.items():
                    f.write('{}\t{}\n'.format(k, v))
        if isinstance(path_vocab_i2w, str):
            with open(path_vocab_i2w, 'w', encoding='utf-8') as f:
                for k, v in self.vocab_i2w.items():
                    f.write('{}\t{}\n'.format(k, v))

    def load_vocab(self, path_vocab):
        with open(path_vocab, 'r', encoding='utf-8') as f:
            vocab = [w.strip() for w in f.readlines()]
        return self.build_vocab_from_list(vocab=vocab)

    def load_vocab_dict(self, path_vocab_dict, reverse=False):
        with open(path_vocab_dict, 'r', encoding='utf-8') as f:
            self.vocab_w2i, self.vocab_i2w = {}, {}
            for line in f.readlines():
                key, value = line.strip().split('\t')
                if reverse:
                    idx, word = int(key), str(value)
                else:
                    word, idx = str(key), int(value)
                self.vocab_w2i[word] = idx
                self.vocab_i2w[idx] = word
            self.vocab = sorted(self.vocab_w2i.keys())
            self.build_labels()
        return self.vocab, self.vocab_w2i, self.vocab_i2w

    def save_embedding_matrix(self, path_embedding_matrix):
        np.savetxt(path_embedding_matrix, self.embedding_matrix, fmt='%0.8f')

    def load_embedding_matrix(self, path_embedding_matrix):
        self.embedding_matrix = np.loadtxt(path_embedding_matrix)
        self.ndim = self.embedding_matrix.shape[1]
        return self.embedding_matrix

    @classmethod
    def reverse_vocab_dict(cls, vocab):
        assert (isinstance(vocab, dict))
        return {v: k for k, v in vocab.items()}

    @classmethod
    def max_len(cls, df, sep=' '):
        max_len = df.apply(lambda x: x.count(sep) + 1)
        return int(np.mean(max_len) + 2 * np.std(max_len))

    def pad_sentence(self, sentence, max_len, vocab, sep=' '):
        words = sentence.strip().split(sep)
        words = words[:max_len]
        sentence = [word if word in vocab else self.labels['UNK'].word for word in words]
        sentence = [self.labels['BOS'].word] + sentence + [self.labels['EOS'].word]
        # sentence = [self.labels['BOS'].word] + words + [self.labels['EOS'].word]
        sentence = sentence + [self.labels['PAD'].word] * (max_len - len(words))
        return sep.join(sentence)

    def pad_df(self, df_corpus, max_len, vocab, sep=' '):
        return df_corpus.apply(lambda x: self.pad_sentence(x, max_len, vocab, sep=sep))

    def sent2idx(self, sentence, vocab_w2i=None, sep=' '):
        vocab_dict = vocab_w2i if vocab_w2i else self.vocab_w2i
        words = sentence.split(sep)
        indice = [self.word2idx(word, vocab_dict) for word in words]
        return indice

    def idx2sent(self, indice, vocab_i2w=None, sep=' '):
        vocab_dict = vocab_i2w if vocab_i2w else self.vocab_i2w
        words = [self.idx2word(index, vocab_dict) for index in indice]
        return sep.join(words)

    def word2idx(self, word, vocab_w2i=None):
        vocab_dict = vocab_w2i if vocab_w2i else self.vocab_w2i
        return vocab_dict[word] if word in vocab_dict else self.labels['UNK'].index
        # return vocab_dict[word] if word in vocab_dict else self.labels['UNK'].index


    def idx2word(self, index, vocab_i2w=None):
        vocab_dict = vocab_i2w if vocab_i2w else self.vocab_i2w
        return vocab_dict[index] if index in vocab_dict else self.labels['UNK'].word
