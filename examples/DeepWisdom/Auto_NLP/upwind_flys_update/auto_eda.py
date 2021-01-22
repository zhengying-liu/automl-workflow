# -*- coding: utf-8 -*-
# @Date    : 2019/12/30 14:30
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    : Exploratory data analysis
import numpy as np
from scipy import stats
from collections import Counter

NLP_WORDS_STAT_METHODS = {"avg","max","min","median","mode","std"}

def ohe2cat(label):
    label = np.reshape(label, (1, label.shape[0]))
    return np.argmax(label, axis=1)[0]


class AutoEDA(object):
    """
    接受输入tensor，按模态进行不同数据的EDA
    """

    def __init__(self, meta, x, y):
        self.set_meta(meta)

        self.label_counter = self.label_distribution(y)
        label_index = self._get_label_index(y, self.label_counter)
        self.grouped_feats = self._get_x_by_label(x, label_index)
        self.show_basic_info()

    def set_meta(self, meta):
        self.num_samples = meta["num_samples"]
        self.label_types = meta["output_dim"]
        self.language = meta["language"]

    def show_basic_info(self):
        print("==============BASIC INFO==================")
        print("NUM OF SAMPLES: ", self.num_samples)
        print("NUM OF LABEL TYPES: ", self.label_types)
        print("LANGUAGE:", self.language)
        print("LABEL DISTRIBUTION:", self.label_counter)
        print("==============BASIC INFO==================")

    def _get_label_index(self, y, label_counter):
        label_index = {}.fromkeys(label_counter.keys())
        for key, val in label_counter.items():
            label_index[key] = np.argwhere(y == key).ravel()
        return label_index

    def _get_x_by_label(self, x, label_index):
        x_grouped_dict = {}.fromkeys(label_index.keys())
        for key, val in label_index.items():
            # todo: 超大文本，不能直接np.array
            x_texts = np.array(x)[label_index[key]]
            x_grouped_dict[key] = x_texts
        return x_grouped_dict

    def label_distribution(self, y, return_ratio=True):
        # 公共EDA, 所有模态均适合
        """
        获取训练集/测试集的样本分布，返回dict,包含类别样本数/比例
        :return:
        """
        label_counter = Counter(y)
        if return_ratio:
            label_ratio = {}.fromkeys(label_counter.keys())
            for key, val in label_counter.items():
                label_ratio[key] = round(float(val) / float(self.num_samples), 3)
            return label_ratio
        return label_counter

    def count_word_numbers(self, raw_text, labels):
        """
        获取不同label下的样本字数统计
        :param:raw_text: 官方切分，英文按space, 中文按字
        :return:
        """
        word_num = len(raw_text)
        print("count word numbers: {}!\n".format(word_num))

    def get_text_length_stat_with_label(self, method="all"):
        """
        获取不同label分布下的文本长度的统计特征：avg/max/min/median/mode
        :param method: 指定获取文本长度特征方法，默认是全部统计特征
        :return: dict
        """

        def get_statistic_val(method):
            dict = {}.fromkeys(self.label_counter.keys())
            if method == "avg":
                func = np.average
            elif method == "max":
                func = np.max
            elif method == "min":
                func = np.min
            elif method == "median":
                func = np.median
            elif method == "mode":
                func = stats.mode
            elif method == "std":
                func = np.std

            for key, val in self.grouped_feats.items():
                if method == "mode":
                    dict[key] = np.round(float(func([len(_text) for _text in val])[0][0]), 3)
                else:
                    dict[key] = np.round(func([len(_text) for _text in val]), 3)
            return dict

        if method == "all":
            full_statistic_dict = {}.fromkeys(NLP_WORDS_STAT_METHODS)
            for _method in NLP_WORDS_STAT_METHODS:
                full_statistic_dict[_method] = get_statistic_val(_method)
            # print(full_statistic_dict)
            return full_statistic_dict
        elif method in NLP_WORDS_STAT_METHODS:
            # print({method: get_statistic_val(method)})
            return {method: get_statistic_val(method)}
        else:
            raise ValueError

    def get_text_length_stat(self, method, label_stat_dict):
        if label_stat_dict:
            if method in label_stat_dict:
                stats = label_stat_dict[method]
                return np.mean(list(stats.values()))
