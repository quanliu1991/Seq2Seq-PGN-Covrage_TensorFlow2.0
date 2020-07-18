"""
分词工具的抽象类、和各个分词工具子类
"""
import re

import jieba
import hanlp
import pkuseg
import thulac
import pynlpir



# import pyltp

class Tokenizer:
    def __init__(self):
        pass

    def __str__(self):
        pass

    def cut(self, sentence):
        """
        分词方法
        :return:
        """
        pass

    def load_txt(self):
        """
        加载自定义词典
        :return:
        """
        pass


class Jieba(Tokenizer):
    def __init__(self):
        super().__init__()


    def __str__(self):
        return "Jieba"

    def cut(self, sentence):
        words = jieba.lcut(sentence)
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return " ".join(words)


class HanLP(Tokenizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = hanlp.load('CTB6_CONVSEG')

    def __str__(self):
        return "HanLP"

    def cut(self, sentence):
        words = self.tokenizer(sentence)
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return " ".join(words)


class LTP(Tokenizer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "LTP"

    def cut(self):
        pass


class Pkuseg(Tokenizer):
    def __init__(self):
        super().__init__()
        self.seg = pkuseg.pkuseg()  # 以默认配置加载模型

    def cut(self, sentence):
        words = self.seg.cut(sentence)  # 进行分词
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return " ".join(words)

    def __str__(self):
        return "pkuseg"


class THULAC(Tokenizer):
    def __init__(self):
        super().__init__()
        # thu1 = thulac.thulac(user_dict=None, model_path=None, T2S=False, seg_only=True, filt=False)  # 默认模式
        # self.thu1 = thulac.thulac(model_path="/home/liuquan/project/tokenizers/model/thulac_model/models/",
                                  # seg_only=True)  # 默认模式
        self.thu1 = thulac.thulac(seg_only=True)

    def __str__(self):
        return "thulac"

    def cut(self, sentence):
        # words = thu1.cut(sentence, text=True)  # 进行一句话分词
        words = self.thu1.cut(sentence, text=True)  # 进行一句话分词
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return " ".join(words)


class Snownlp(Tokenizer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Snownlp"

    def cut(self, sentence):

        # words = [re.sub('\s', '', word) for word in words]
        # words = list(filter(None, words))
        # return " ".join(words)
        pass

class Nlpir(Tokenizer):
    def __init__(self):
        super().__init__()
        pynlpir.open()

    def __str__(self):
        return "Nlpir"

    def cut(self, sentence):
        words = pynlpir.segment(sentence, pos_tagging=False)
        words = [re.sub('\s', '', word) for word in words]
        words = list(filter(None, words))
        return " ".join(words)
