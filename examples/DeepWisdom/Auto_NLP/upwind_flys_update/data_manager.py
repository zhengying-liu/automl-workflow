"""
MIT License

Copyright (c) 2019 Lenovo Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import re
import time
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from keras.preprocessing import text
from keras.preprocessing import sequence

import jieba_fast as jieba
# try:
#     os.system("pip uninstall spcay")
#     os.system("pip install spacy>=2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple")
# except:
#     print("import spacy fail!")


# dir_path = os.path.abspath(os.path.dirname(__file__))
# os.system("pip install {}/en_core_web_sm-2.2.5.tar.gz".format(dir_path))
#
# import spacy
# en_nlp_processor = spacy.load("en_core_web_sm")

# import nltk.data
# os.system("wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip")
# from nltk import downloader
# downloader = downloader.Downloader(server_index_url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip')
# nltk.download("wordnet")
import time
# from Auto_NLP.upwind_flys_update.auto_augment import eda as augment_fn
# from Auto_NLP.upwind_flys_update.auto_augment import random_deletion, random_insertion, random_swap
# from Auto_NLP.upwind_flys_update.auto_eda import AutoEDA, ohe2cat
# from bpemb import BPEmb

# from Auto_NLP.upwind_flys_update.time_utils import TimerD

CHI_WORD_LENGTH = 2
MAX_CHAR_LENGTH = 96
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 601
# MAX_SEQ_LENGTH = 301
# MAX_VALID_PERCLASS_SAMPLE = 600
MAX_VALID_PERCLASS_SAMPLE = 400
MAX_SAMPLE_TRIAN = 18000
# MAX_TRAIN_PERCLASS_SAMPLE = 800
MAX_TRAIN_PERCLASS_SAMPLE = 1000
# MAX_TRAIN_PERCLASS_SAMPLE = 3500
# RNN_UNITS = [64, 128, 256]

noisy_pos_tags = ["PROP"]
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import string
# from imblearn.over_sampling import SMOTE
from Auto_NLP.upwind_flys_update.time_utils import info

punctuations = string.punctuation
min_token_length = 2
# dir = os.path.abspath(os.path.dirname(__file__))
# cache_dir = "/app/codalab/AutoDL_sample_code_submission/Auto_NLP/upwind_flys_update/"
# print(os.path.join(dir, "en"))
# info("download en bpe word vector...")
# en_bpe_file = os.path.join(dir, "en", "en.wiki.bpe.vs200000.d300.w2v.bin")
# zh_bpe_file = os.path.join(dir, "zh", "zh.wiki.bpe.vs200000.d300.w2v.bin")
# if not os.path.exists(en_bpe_file):
#     os.system(
#         "wget -P {} http://175.24.101.84/bpe_en/en.wiki.bpe.vs200000.d300.w2v.bin".format(os.path.join(dir, "en")))
# if not os.path.exists(zh_bpe_file):
#     os.system(
#         "wget -P {} http://175.24.101.84/bpe_zh/zh.wiki.bpe.vs200000.d300.w2v.bin".format(os.path.join(dir, "zh")))
# dir = os.path.abspath(os.path.dirname(__file__))

# en_bpe_encoder = BPEmb(lang="en", dim=300, vs=200000, cache_dir=dir)
# zh_bpe_encoder = BPEmb(lang="zh", dim=300, vs=200000, cache_dir=dir)

import multiprocessing
from multiprocessing import Pool

NCPU = multiprocessing.cpu_count() - 1

# from Auto_NLP.upwind_flys_update.utils import full_stop_words

MAX_EN_CHAR_LENGTH = 35


##### list 分段切分函数：接近等长划分.
def chunkIt(seq, num):
    """
    :param seq: 原始 list 数据
    :param num: 要分chunk是数量.
    :return:
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def clean_en_text_parallel(dat, worker_num=NCPU, partition_num=10):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)
    # data = p.map(clean_en_text, sub_data_list)
    data = p.map(clean_en_original, sub_data_list)
    p.close()

    # 把 list of list of str 结果 flat 回到 list of str
    flat_data = [item for sublist in data for item in sublist]
    return flat_data


def clean_en_original(dat, ratio=0.1, is_ratio=True):
    # def clean_en_text(dat, ratio=0.1, is_ratio=True):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])
        ret.append(line)
    return ret


def clean_en_text(dat, ratio=0.1, is_ratio=True, rmv_stop_words=True):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:" + '0123456789'))
    ret = []
    for line in dat:
        line = line.strip()
        line = line.translate(trantab)
        line_split = line.split()
        line_split = [word.lower() for word in line_split if (len(word) > 1)]
        # line_split = [word.lower() for word in line_split if (len(word) < MAX_EN_CHAR_LENGTH and len(word) > 1)]
        if len(line_split) == 0:
            print("Empty text!\n\n\n")
        # fixme: 是否要去除stopwords
        # line_split = [tok for tok in line_split if (tok not in stopwords and tok not in punctuations) ]
        # line_split = [tok for tok in line_split if (tok not in punctuations)]
        if rmv_stop_words:
            new_line_split = list(set(line_split).difference(full_stop_words))
            new_line_split.sort(key=line_split.index)
            if len(new_line_split) == 0:
                print("Empty text!\n\n")
                new_line_split = line_split
        else:
            new_line_split = line_split

        if is_ratio:
            NUM_WORD = max(int(len(new_line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(new_line_split) > NUM_WORD:
            line = " ".join(new_line_split[0:NUM_WORD])
        else:
            line = " ".join(new_line_split)
        ret.append(line)
    return ret


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))
    # return ''.join(jieba.cut(text, cut_all=False))


def check_x_avg_length(input_x):
    return np.mean([len(x.split(" ")) for x in input_x])

def sample_input_data(input_x, input_y, num_classes, max_num=500):
        train_label_distribution = np.sum(np.array(input_y), 0)
        all_index = []
        meta_train_index = []
        print("train_distribution: ", train_label_distribution)  # 获取对应label的分布
        for i in range(num_classes):
            all_index.append(
                list(np.where((input_y[:, i] == 1) == True)[0]))

        for i in range(num_classes):  # 按label类别抽取
            # fixme: 对0边界处理
            if len(all_index[i]) < max_num and len(all_index[i]) > 0:
                tmp = all_index[i] * int(
                    max_num / len(all_index[i]))
                tmp += random.sample(all_index[i],
                                     max_num - len(tmp))
                meta_train_index += tmp
            else:
                meta_train_index += random.sample(
                    all_index[i], max_num)

        random.shuffle(meta_train_index)

        train_sample_x = [input_x[i] for i in meta_train_index]
        train_sample_y = input_y[meta_train_index, :]
        return train_sample_x, train_sample_y


class DataGenerator(object):
    def __init__(self,
                 x_train, y_train,
                 metadata,
                 imbalance_level=-1):

        self.meta_data_x, \
        self.meta_data_y = x_train, y_train
        self.metadata = metadata

        self.num_classes = self.metadata['class_num']
        self.num_samples_train = self.metadata['train_num']
        self.language = metadata['language']

        print("num_samples_train:", self.num_samples_train)
        print("num_class_train:", self.num_classes)

        self.val_index = None
        self.tokenizer = None
        self.max_length = None
        self.sample_num_per_class = None
        self.data_feature = {}
        self.eda_feature = {}
        self.update_y = None
        self.diff_add_x = None
        self.pseudo_x_train_size = 0
        self.full_x = []
        self.full_y = np.array([])
        # if self.language == "EN":
        #     self.bpe_encoder = en_bpe_encoder
        # else:
        #     self.bpe_encoder = zh_bpe_encoder
        # cache_dir = "/app/codalab/AutoDL_sample_code_submission_orig/Auto_NLP/upwind_flys_update/"

        self.x_dict = {i: [] for i in range(self.num_classes)}
        self.imbalance_flg = False
        self.do_generate_sample = False
        self.empty_class_ = []
        self.meta_train_x = []
        self.meta_train_y = np.array([])

        self.full_index = None
        self.imbalance_level = imbalance_level
        self.MAX_TRAIN_PERCLASS_SAMPLE = MAX_TRAIN_PERCLASS_SAMPLE
        print("Init Data Manager! Imbalance_level is {}".format(self.imbalance_level))
        # if self.num_classes<=5 and self.num_samples_train<35000 and self.imbalance_level<=1:
        if self.num_classes <= 5 and self.imbalance_level <= 1:
            self.MAX_TRAIN_PERCLASS_SAMPLE = 3000
        elif self.num_classes == 2 and self.imbalance_level<=1:
            self.MAX_TRAIN_PERCLASS_SAMPLE = 3500

        print("Init Data Manager! MAX_TRAIN_PERCLASS_SAMPLE is {}".format(self.MAX_TRAIN_PERCLASS_SAMPLE))

    # @timeit
    def snoop_data(self, metadata):
        time_end_ = time.time()
        self.snoop_avg_text_length = check_x_avg_length(self.meta_data_x)
        time_end = time.time()
        print('EDA.get_text_length_stat_with_label cost {} sec'.format(time_end - time_end_))
        print("self.snoop_avg_text_length ", self.snoop_avg_text_length)

    def update_meta_data(self, x_train, y_train, use_diff=False):
        if use_diff:
            diff_add_x = list(set(x_train).difference(self.x_val_raw))  # 新增数据， 去除和原始评估集的交集
            diff_id_x = [x_train.index(_x) for _x in diff_add_x]

            self.diff_add_x = [x_train[x] for x in diff_id_x]
            self.meta_data_x.extend(self.diff_add_x)
            # self.diff_add_x = self.meta_data_x
            print("check meta_data_x", len(self.meta_data_x))
            update_y = y_train[diff_id_x]

        else:
            self.meta_data_x = self.meta_data_x + x_train
            # self.diff_add_x = self.meta_data_x
            self.diff_add_x = x_train
            update_y = y_train
        # 添加新增数据对应label
        self.update_y = update_y
        self.meta_data_y = np.concatenate([self.meta_data_y, update_y], axis=0)

        # self.update_y = self.meta_data_y

    def set_sample_num_per_class(self, sample_num_per_class):
        self.sample_num_per_class = sample_num_per_class

    # generate validation dataset index
    def sample_valid_index(self):
        all_index = []
        empty_class = []
        train_label_distribution = np.sum(np.array(self.meta_data_y), 0)
        train_label_ratio = train_label_distribution/np.sum(train_label_distribution)
        for i in range(self.num_classes):
            if list(np.where((self.meta_data_y[:, i] == 1) == True)[0]) == 0:
                # 不产生对于的评估数据
                empty_class.append(i)
                continue
            all_index.append(
                list(np.where((self.meta_data_y[:, i] == 1) == True)[0]))

        val_index = []
        for i in range(self.num_classes):
            if train_label_ratio[i]<0.01:
                ratio = 0.075
            else:
                ratio = 0.2
            # todo: 检查这里采样值应当大于0
            tmp = random.sample(all_index[i],
                                int(len(all_index[i]) * ratio))
            if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
                tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
            val_index += tmp
            all_index[i] = list(
                set(all_index[i]).difference(set(tmp)))
            # 增加全局x_dict
            self.x_dict[i] = all_index[i]

        self.all_index = all_index

        self.val_index = val_index
        # self.empty_class_ = empty_class

    def auto_augment(self, meta_train_index, num_aug=4):
        augment_index = []
        for i in range(self.num_classes):  # 按label类别抽取
            tmp = random.sample(meta_train_index[i], min(int(self.sample_num_per_class * 0.1), 100))

            augment_index += tmp
        random.shuffle(augment_index)
        augment_train_x = [
            self.meta_data_x[i]
            for i in augment_index
        ]
        augment_train_y = self.meta_data_y[augment_index, :]

        agx, agy = [], []
        for x, y in zip(augment_train_x, augment_train_y):
            _agx = augment_fn(x, num_aug=num_aug)
            _agy = [y for i in range(len(_agx))]
            agx += _agx
            agy.extend(_agy)
        return agx, np.array(agy)



    # generate training meta dataset index
    def sample_train_index(self, all_index):
        train_label_distribution = np.sum(np.array(self.meta_data_y), 0)
        self.train_label_distribution = train_label_distribution
        print("train_distribution: ", train_label_distribution)  # 获取对应label的分布
        self.max_sample_num_per_class = int(
            np.max(train_label_distribution) * 4 / 5)

        if self.sample_num_per_class is None:
            if self.num_samples_train <  MAX_SAMPLE_TRIAN:
                self.sample_num_per_class = self.max_sample_num_per_class
            else:
                self.sample_num_per_class = min(self.max_sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)
        # fixme: 如果sample_num_per_class不为空的情况
        else:
            # 避免类别数多的情况下，第一次进样少，导致后面连续采样过低
            self.sample_num_per_class = max(self.max_sample_num_per_class, int(np.mean(train_label_distribution)))
        print("check sample_train_index samplle_num_per_class:{}".format(self.sample_num_per_class))
        # self.sample_num_per_class = self.max_sample_num_per_class
        # if self.imbalance_flg:
        #     self.sample_num_per_class = min(self.sample_num_per_class, int(np.mean(train_label_distribution)))

        meta_train_index = []

        if self.imbalance_flg:
            max_sample_num = min(self.sample_num_per_class, int(np.mean(train_label_distribution)))
            # max_sample_num =min(max_sample_num, 3500)
            max_sample_num = min(self.MAX_TRAIN_PERCLASS_SAMPLE, max_sample_num)
        else:
            # max_sample_num = self.sample_num_per_class
            max_sample_num = min(self.sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)
        # max_sample_num = self.sample_num_per_class
        print("max_sample_num is {}".format(max_sample_num))

        for i in range(self.num_classes):  # 按label类别抽取
            # if len(all_index[i]) < max_sample_num and len(all_index[i])>0:
            if len(all_index[i]) == 0:
                continue
            if len(all_index[i]) < max_sample_num and len(all_index[i]) > 0:
                tmp = all_index[i] * int(
                    max_sample_num / len(all_index[i]))
                tmp += random.sample(all_index[i],
                                     max_sample_num - len(tmp))
                meta_train_index += tmp
            else:
                meta_train_index += random.sample(
                    all_index[i], max_sample_num)

        random.shuffle(meta_train_index)
        self.meta_train_index = meta_train_index
        return meta_train_index

    def sample_valid_index_add(self):
        add_index = []
        empty_class = []
        # todo: 存在某些类别评估样本数为零
        for i in range(self.num_classes):
            if list(np.where((self.update_y[:, i] == 1) == True)[0]) == 0:
                # 不产生对于的评估数据
                empty_class.append(i)
                continue
            add_index.append(
                list(np.where((self.update_y[:, i] == 1) == True)[0]))
        val_index = []
        for i in range(self.num_classes):
            tmp = random.sample(add_index[i],
                                int(len(add_index[i]) * 0.2))
            if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
                tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
            val_index += tmp
            add_index[i] = list(
                set(add_index[i]).difference(set(tmp)))
            self.x_dict[i].extend(add_index[i])
        self.add_index = add_index
        self.add_val_index = val_index
        # self.empty_class_ = empty_class

    def do_random_generate_sample(self, num=1):
        # 默认只随机产生一个样本，减少干扰
        sentence_list = []
        for i in range(num):
            rand_int = random.randint(0, len(self.meta_data_x) - 1)
            sentence_list.append(self.meta_data_x[rand_int])
        generate_samples = []
        for sentence in sentence_list:
            # print("select sentence: {}".format(sentence))
            # fixme: 针对英文
            words = sentence.split(" ")
            # print(words)
            if len(words) <= 1:
                new_words = ","
            else:
                w_i = random.randint(0, len(words) - 1)
                new_words = words[w_i]
            # words = [word for word in words if word is not '']
            # new_words = random_deletion(words, p=0.8)
            generate_samples.append(new_words)
        return generate_samples

    def check_imbalance_and_generate_samples(self, all_index):
        new_generate_index = []
        if self.update_y is None:
            train_label_distribution = np.sum(np.array(self.meta_data_y), 0)
        else:
            train_label_distribution = np.sum(np.array(self.update_y), 0)
        print("new sample train_distribution: ", train_label_distribution)  # 获取对应label的分布
        self.normal_std = np.std(train_label_distribution) / np.sum(train_label_distribution)
        print("check normalized std {}", self.normal_std)
        self.empty_class_ = [i for i in range(train_label_distribution.shape[0]) if train_label_distribution[i] == 0]
        if self.normal_std >= 0.1 or 0.0 in train_label_distribution:
            self.imbalance_flg = True
        # imbalance_flg = False

        for i in range(self.num_classes):
            new_generate_index.append([])

        for i in range(self.num_classes):  # 按label类别抽取
            if len(all_index[i]) == 0:
                self.do_generate_sample = True
                # todo: 产生随机样本
                new_samples = self.do_random_generate_sample(num=1)
                new_generate_index[i] = new_samples

            else:
                # todo: 是否要在这里做采样
                pass
        # print(new_generate_index)
        return new_generate_index

    def sample_train_index_new(self, train_y):
        all_index = []
        for i in range(self.num_classes):
            if list(np.where((train_y[:, i] == 1) == True)[0]) == 0:
                # 不产生评估数据
                continue
            all_index.append(
                list(np.where((train_y[:, i] == 1) == True)[0]))
        print("Update add_index use full_data!!!")
        self.add_index = all_index

        train_label_distribution = np.sum(np.array(train_y), 0)
        self.train_label_distribution = train_label_distribution
        print("sample_train_index_new train_distribution: ", train_label_distribution)  # 获取对应label的分布
        self.max_sample_num_per_class = int(
            np.max(train_label_distribution) * 4 / 5)

        if self.sample_num_per_class is None:
            if self.num_samples_train < MAX_SAMPLE_TRIAN:
                self.sample_num_per_class = self.max_sample_num_per_class
            else:
                self.sample_num_per_class = min(self.max_sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)
        else:
            # 避免类别数多的情况下，第一次进样少，导致后面连续采样过低
            self.sample_num_per_class = max(self.max_sample_num_per_class, int(np.mean(train_label_distribution)))
        print("check sample_train_index_new samplle_num_per_class:{}".format(self.sample_num_per_class))
        meta_train_index = []

        if self.imbalance_flg:
            max_sample_num = min(self.sample_num_per_class, int(np.mean(train_label_distribution)))
            # max_sample_num = min(max_sample_num, 3500)
            max_sample_num = min(max_sample_num, self.MAX_TRAIN_PERCLASS_SAMPLE)
        else:
            # max_sample_num = self.sample_num_per_class
            max_sample_num = min(self.sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)
        # max_sample_num = self.sample_num_per_class
        print("max_sample_num is {}".format(max_sample_num))

        for i in range(self.num_classes):  # 按label类别抽取
            # if len(all_index[i]) < max_sample_num and len(all_index[i])>0:
            if len(all_index[i])==0:
                # fixme: 考虑0的边界
                continue
            elif len(all_index[i]) < max_sample_num and len(all_index[i]) > 0:
                tmp = all_index[i] * int(
                    max_sample_num / len(all_index[i]))
                tmp += random.sample(all_index[i],
                                     max_sample_num - len(tmp))
                meta_train_index += tmp
            else:

                meta_train_index += random.sample(
                    all_index[i], max_sample_num)

        random.shuffle(meta_train_index)
        self.meta_train_index = meta_train_index
        return meta_train_index

    # 生成增量的训练数据集
    def sample_train_index_add(self, add_index):
        train_label_distribution = np.sum(np.array(self.update_y), 0)
        print("new sample train_distribution: ", train_label_distribution)  # 获取对应label的分布
        self.max_sample_num_per_class = int(
            np.max(train_label_distribution) * 4 / 5)

        if self.sample_num_per_class is None:
            if self.num_samples_train < MAX_SAMPLE_TRIAN:
                self.sample_num_per_class = self.max_sample_num_per_class
            else:
                self.sample_num_per_class = min(self.max_sample_num_per_class, self.MAX_TRAIN_PERCLASS_SAMPLE)

        info("start sample data")
        max_sample_num = min(self.sample_num_per_class, int(np.mean(train_label_distribution)))
        if self.imbalance_flg:
            max_sample_num = int(max_sample_num * self.normal_std)
        # max_sample_num = self.sample_num_per_class
        print("max_sample_num is {}".format(max_sample_num))

        meta_train_add_index = []
        for i in range(self.num_classes):  # 按label类别抽取
            if len(add_index[i]) == 0:
                continue
            elif len(add_index[i]) < self.sample_num_per_class and len(add_index[i]) > 0:

                if self.imbalance_flg:
                    if len(add_index[i]) < max_sample_num:

                        tmp = add_index[i] * int(
                            max_sample_num / len(add_index[i]))
                        tmp += random.sample(add_index[i],
                                             max_sample_num - len(tmp))

                    else:
                        tmp = random.sample(
                            add_index[i], max_sample_num)
                    # tmp = add_index[i] * int(
                    #     self.sample_num_per_class / len(add_index[i]))
                    # tmp += random.sample(add_index[i],
                    #                  self.sample_num_per_class - len(tmp))  # 再采样n个样本， n取差值
                else:
                    tmp = add_index[i] * int(
                        self.sample_num_per_class / len(add_index[i]))
                    tmp += random.sample(add_index[i],
                                         self.sample_num_per_class - len(tmp))

                meta_train_add_index += tmp
            else:  # 随机抽取
                if self.imbalance_flg:
                    meta_train_add_index += random.sample(
                        add_index[i], max_sample_num)
                else:
                    meta_train_add_index += random.sample(
                        add_index[i], self.sample_num_per_class)

        info("end sample data")
        random.shuffle(meta_train_add_index)
        self.meta_train_add_index = meta_train_add_index
        return meta_train_add_index

    def sample_dataset_from_metadatset_full_train(self):
        if self.full_index is None:
            all_index = []
            for i in range(self.num_classes):
                if list(np.where((self.meta_data_y[:, i] == 1) == True)[0]) == 0:
                    continue
                all_index.append(
                    list(np.where((self.meta_data_y[:, i] == 1) == True)[0]))
            self.full_x = all_index
        print("check size of meta_train_y:",self.meta_train_y.shape[0])
        self.new_generate_samples_idx = self.check_imbalance_and_generate_samples(self.full_x)
        self.sample_train_index_new(self.meta_train_y)
        print("length of add sample_index", len(self.meta_train_index))
        train_x = [self.meta_train_x[i] for i in self.meta_train_index]
        train_y = self.meta_train_y[self.meta_train_index, :]
        self.imbalance_flg = False
        return train_x, train_y

    def _sample_from_full(self):
        # 清空历史记录数据
        self.meta_train_y = np.array([])
        self.meta_train_x = []

        # 0219: 重复采样之前，保证meta_train_x是原始分别
        train_add_index = []
        for i in range(len(self.all_index)):
            train_add_index.extend(self.all_index[i])
        self.meta_train_x = [self.meta_data_x[i] for i in train_add_index]
        self.meta_train_y = self.meta_data_y[train_add_index, :]

        self.sample_train_index(self.all_index)  # 生成全量的meta_train_index，用于获得训练文本
        self.imbalance_flg = False
        print("length of sample_index", len(self.meta_train_index))
        print("length of val_index", len(self.val_index))

        train_x = [self.meta_data_x[i] for i in self.meta_train_index]
        train_y = self.meta_data_y[self.meta_train_index, :]

        if self.do_generate_sample:
            print("Do Radam Create Samples!")
            for i in range(self.num_classes):
                new_samples = self.new_generate_samples_idx[i]
                if len(new_samples) == 0:
                    continue
                train_x.extend(new_samples)
                new_label = np.zeros((len(new_samples), self.num_classes))
                new_label[:, i] = 1
                train_y = np.concatenate([train_y, new_label], axis=0)
            self.do_generate_sample = False

        valid_x = [self.meta_data_x[i] for i in self.val_index]
        valid_y = self.meta_data_y[self.val_index, :]

        # train_add_index = []
        # for i in range(len(self.all_index)):
        #     train_add_index.extend(self.all_index[i])
        # train_diff_x = [self.meta_data_x[i] for i in train_add_index]
        # train_diff_y = self.meta_data_y[train_add_index, :]

        # 更新全量的train 样本

        # self.meta_train_x = train_x
        # self.meta_train_y = train_y

        return train_x, train_y, valid_x, valid_y


    def sample_dataset_from_metadataset_all_data(self):
        print("Reset full val index!")
        self.sample_valid_index()
        self.new_generate_samples_idx = self.check_imbalance_and_generate_samples(self.all_index)
        print("check empty class {}".format(self.empty_class_))
        print("check input data imbalance:{}".format(self.imbalance_flg))
        print("check do generate sample flg:{}".format(self.do_generate_sample))
        return self._sample_from_full()

    def sample_dataset_from_metadataset_iter(self, use_val=False):
        if self.val_index is None:
            print("None val index, set valid_index!")
            self.sample_valid_index()  # 生成首次的all_index/val_index

        self.new_generate_samples_idx = self.check_imbalance_and_generate_samples(self.all_index)
        print("check empty class {}".format(self.empty_class_))
        print("check input data imbalance:{}".format(self.imbalance_flg))
        print("check do generate sample flg:{}".format(self.do_generate_sample))


        print("first Process Data for SVM!")

        if self.diff_add_x is None:  # 首次，没有新增样本的时候

            train_add_index = []
            for i in range(len(self.all_index)):
                train_add_index.extend(self.all_index[i])
            train_diff_x = [self.meta_data_x[i] for i in train_add_index]
            train_diff_y = self.meta_data_y[train_add_index, :]
            # 更新全量的train 样本
            if self.meta_train_y.shape[0] == 0:
                self.meta_train_x = train_diff_x
                self.meta_train_y = train_diff_y
            else:
                # self.meta_train_x.extend(train_diff_x)
                self.meta_train_x = self.meta_train_x + train_diff_x
                self.meta_train_y = np.concatenate([self.meta_train_y, train_diff_y], axis=0)

            self.sample_train_index(self.all_index)  # 生成首次的meta_train_index，用于获得训练文本
            self.imbalance_flg = False
            print("length of sample_index", len(self.meta_train_index))
            print("length of val_index", len(self.val_index))

            train_x = [self.meta_data_x[i] for i in self.meta_train_index]
            train_y = self.meta_data_y[self.meta_train_index, :]

            if self.do_generate_sample:
                print("Do Radam Create Samples!")
                for i in range(self.num_classes):
                    new_samples = self.new_generate_samples_idx[i]
                    if len(new_samples) == 0:
                        continue
                    train_x.extend(new_samples)
                    new_label = np.zeros((len(new_samples), self.num_classes))
                    new_label[:, i] = 1
                    train_y = np.concatenate([train_y, new_label], axis=0)
                self.do_generate_sample = False

            valid_x = [self.meta_data_x[i] for i in self.val_index]
            valid_y = self.meta_data_y[self.val_index, :]


            return train_x, train_y, valid_x, valid_y

        else:
            # 在新样本上重新划分训练集和评估集
            self.sample_valid_index_add()

            val_diff_x = [self.diff_add_x[i] for i in self.add_val_index]
            val_diff_y = self.update_y[self.add_val_index, :]

            train_add_index = []
            for i in range(len(self.add_index)):
                train_add_index.extend(self.add_index[i])
            train_diff_x = [self.diff_add_x[i] for i in train_add_index]
            train_diff_y = self.update_y[train_add_index, :]

            if use_val:
                # train_diff_x.extend(val_diff_x)
                train_diff_x = train_diff_x+val_diff_x
                train_diff_y = np.concatenate([train_diff_y, val_diff_y], axis=0)
            # 更新全量的train 样本
            if self.meta_train_y.shape[0] == 0:
                self.meta_train_x = train_diff_x
                self.meta_train_y = train_diff_y
            else:
                # self.meta_train_x.extend(train_diff_x)
                self.meta_train_x = self.meta_train_x+train_diff_x
                self.meta_train_y = np.concatenate([self.meta_train_y, train_diff_y], axis=0)
            print("Check meta_train_x size {}, meta_train_y size {}".format(len(self.meta_train_x), self.meta_train_y.shape[0]))
            # self.sample_train_index_add(self.add_index)
            print("meta_train_y", self.meta_train_y.shape[0])
            self.sample_train_index_new(self.meta_train_y)
            self.new_generate_samples_idx = self.check_imbalance_and_generate_samples(self.add_index)
            print("check empty class {}".format(self.empty_class_))
            print("check input data imbalance: ".format(self.imbalance_flg))
            self.imbalance_flg = False
            print("length of add sample_index", len(self.meta_train_index))
            train_x = [self.meta_train_x[i] for i in self.meta_train_index]
            train_y = self.meta_train_y[self.meta_train_index, :]
            print("Check train_x size {}, train_y size {}".format(len(train_x), train_y.shape[0]))

            if self.do_generate_sample:
                print("DO Radam Create Samples!")
                for i in range(self.num_classes):
                    new_samples = self.new_generate_samples_idx[i]
                    if len(new_samples) == 0:
                        continue
                    train_x.extend(new_samples)
                    new_label = np.zeros((len(new_samples), self.num_classes))
                    new_label[:, i] = 1
                    train_y = np.concatenate([train_y, new_label], axis=0)
                self.do_generate_sample = False

            print("check train_diff_y: ", train_diff_y.shape)

            self.val_index = self.add_val_index
            return train_x, train_y, val_diff_x, val_diff_y


    def dataset_preporocess(self, x_train, rmv_stop_words=True):
        if self.language == 'ZH':
            print("this is a ZH dataset")
            x_train = clean_zh_text(x_train)
            word_avr = np.mean([len(i) for i in x_train])
            test_num = self.metadata['test_num']
            chi_num_chars_train = int(word_avr * len(x_train) /
                                      CHI_WORD_LENGTH)
            chi_num_chars_test = int(word_avr * test_num / CHI_WORD_LENGTH)

            self.meta_data_feature = {
                'chi_num_chars_train': chi_num_chars_train,
                'chi_num_chars_test': chi_num_chars_test,
                'language': self.language
            }
            self.set_feature_mode()

            if self.feature_mode == 1:
                x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            self.meta_data_feature = {
                'language': self.language
            }
            self.set_feature_mode()
            # x_train = clean_en_text(x_train)
            start = time.time()
            x_train = clean_en_text_parallel(x_train)
            # x_train = clean_en_using_spacy(x_train)
            end = time.time()
            print("time cost {}".format(end - start))

        return x_train, self.feature_mode


    def set_feature_mode(self):
        if self.meta_data_feature['language'] == 'ZH':
            chi_num_chars_train, chi_num_chars_test = self.meta_data_feature["chi_num_chars_train"], \
                                                      self.meta_data_feature["chi_num_chars_test"]
        cond_word_1 = self.meta_data_feature['language'] == 'EN'
        cond_word_2 = self.meta_data_feature['language'] == 'ZH' \
                      and chi_num_chars_train < 2e5 and chi_num_chars_test < 4e5

        if cond_word_1 or cond_word_2:
            self.feature_mode = 1
        else:
            self.feature_mode = 0
            # self.load_pretrain_emb = False

        print("the feature mode is", self.feature_mode)


    def dataset_postprocess_bpe(self, x_train, y_train, model_name, call_num, concate_val=True):
        # 转成对应bpe id即可
        if model_name == 'svm':
            self.valid_x = x_train[len(self.meta_train_index):]
            self.valid_y = y_train[len(self.meta_train_index):, :]
            print("check valid_x", len(self.valid_x))
            print("check valid_y", self.valid_y.shape)
            self.x_train = x_train[0:len(self.meta_train_index)]
            self.y_train = y_train[0:len(self.meta_train_index), :]

            self.svm_x_train = x_train[0:len(self.meta_train_index)]
            self.svm_y_train = y_train[0:len(self.meta_train_index), :]

            self.x_train, self.svm_token = self.vectorize_data(self.x_train)
            self.x_val_raw = self.valid_x
        else:
            if call_num == 1:
                self.max_length = None
                x_train_ids = self.bpe_encoder.encode_ids(x_train + self.valid_x)

            else:
                x_train_ids = self.bpe_encoder.encode_ids(x_train)

            x_train_ids = np.asarray(x_train_ids)
            if self.max_length == None:
                self.max_length = len(max(x_train_ids, key=len))
                ave_length = np.mean([len(i) for i in x_train_ids])
                # ave_length_1 = np.mean(x_train_ids)
                std_length = np.std([len(i) for i in x_train_ids])

                print("max_length_word_training:", self.max_length)  # 获取最大文本长度
                print("ave_length_word_training:", ave_length)  # 获取平均文本长度
                print("std_length_word_training:", std_length)

            if self.max_length > MAX_SEQ_LENGTH:
                self.max_length = MAX_SEQ_LENGTH

            x_train_ids = sequence.pad_sequences(x_train_ids, maxlen=self.max_length, padding='post',
                                                 value=self.bpe_encoder.vectors.shape[0])

            # print("check pad seq ", x_train_ids[:10])
            print("check shape of x_train_ids ", x_train_ids.shape)
            if call_num > 1:
                index_size = len(self.meta_train_index)
            else:
                index_size = len(self.meta_train_index)

            print("max_length_training:", self.max_length)
            print("num_featrues_training:", self.bpe_encoder.vectors.shape[0])
            if call_num == 1:
                self.valid_x = x_train_ids[-len(self.valid_x):]
                x_train_ids = x_train_ids[:-len(self.valid_x)]
            # x_train_ids = self.bpe_encoder.encode_ids(x_train)
            _valid_x = x_train_ids[index_size:index_size + len(self.val_index)]
            _valid_y = y_train[index_size:index_size + len(self.val_index), :]
            if concate_val:
                self.valid_x = np.concatenate([_valid_x, self.valid_x], axis=0)
                self.valid_y = np.concatenate([_valid_y, self.valid_y], axis=0)

            self.x_train = np.concatenate([x_train_ids[:index_size], x_train_ids[index_size + len(self.val_index):]],
                                          axis=0)

            self.y_train = np.concatenate([y_train[:index_size, :], y_train[index_size + len(self.val_index):, :]],
                                          axis=0)
            # self.y_train = y_train[:index_size, :]
            print("check 3: x_train size is {} and y train size is {}".format(self.x_train.shape[0],
                                                                              self.y_train.shape[0]))
            self.num_features = 200000
            self.data_feature['num_features'] = self.num_features
            self.data_feature['num_class'] = self.num_classes
            self.data_feature['max_length'] = self.max_length
            self.data_feature['input_shape'] = x_train_ids.shape[1:][0]
            self.data_feature["rnn_units"] = 128
            self.data_feature["filter_num"] = 64


    def dataset_postprocess(self, x_train, y_train, model_name, extra_vocab=None, call_num=0, concate_val=True):
        if model_name == 'svm':
            self.valid_x = x_train[len(self.meta_train_index):]
            self.valid_y = y_train[len(self.meta_train_index):, :]
            print("check valid_x", len(self.valid_x))
            print("check valid_y", self.valid_y.shape)
            self.x_train = x_train[0:len(self.meta_train_index)]
            self.y_train = y_train[0:len(self.meta_train_index), :]

            self.svm_x_train = x_train[0:len(self.meta_train_index)]
            self.svm_y_train = y_train[0:len(self.meta_train_index), :]

            self.x_train, self.svm_token = self.vectorize_data(self.x_train)
            self.x_val_raw = self.valid_x
            # self.data_feature = None

        else:
            # x_train: 已经拼接

            if call_num == 1:
                # print(self.valid_x[:10])
                x_train, self.valid_x, self.word_index, self.num_features, self.tokenizer, self.max_length = self.sequentialize_data(
                    x_train, self.feature_mode, tokenizer=self.tokenizer, max_length=self.max_length,
                    val_contents=self.valid_x)
            else:
                x_train, self.word_index, self.num_features, self.tokenizer, self.max_length = self.sequentialize_data(
                    x_train, self.feature_mode, tokenizer=self.tokenizer, max_length=self.max_length)

            self.x_train = x_train
            self.y_train = y_train
            print("max_length_training:", self.max_length)
            print("num_featrues_training:", self.num_features)

            if call_num > 1:
                index_size = len(self.meta_train_index)
            else:
                index_size = len(self.meta_train_index)

                # _valid_x = self.x_train[len(self.meta_train_index)+len(self.svm_x_train)+self.pseudo_x_train_size:]
            _valid_x = self.x_train[index_size:index_size + len(self.val_index)]
            _valid_y = self.y_train[index_size:index_size + len(self.val_index), :]
            if concate_val:
                self.valid_x = np.concatenate([_valid_x, self.valid_x], axis=0)
                self.valid_y = np.concatenate([_valid_y, self.valid_y], axis=0)

            self.x_train = np.concatenate([x_train[:index_size], x_train[index_size + len(self.val_index):]], axis=0)
            self.y_train = np.concatenate([y_train[:index_size, :], y_train[index_size + len(self.val_index):, :]], axis=0)
            # self.y_train = y_train[:index_size, :]
            print("check 3: x_train size is {} and y train size is {}".format(self.x_train.shape[0], self.y_train.shape[0]))

            self.data_feature['num_features'] = self.num_features
            self.data_feature['word_index'] = self.word_index
            self.data_feature['num_class'] = self.num_classes
            self.data_feature['max_length'] = self.max_length
            self.data_feature['input_shape'] = x_train.shape[1:][0]
            self.data_feature["rnn_units"] = 128
            self.data_feature["filter_num"] = 64

        # for svm vectorize data


    def vectorize_data(self, x_train, x_val=None, analyzer='word'):
        # vectorizer = HashingVectorizer(n_features=30000, analyzer=analyzer, ngram_range=(1, 1))
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=20000, analyzer=analyzer)

        if x_val:
            full_text = x_train + x_val
        else:
            full_text = x_train
        vectorizer.fit(full_text)
        train_vectorized = vectorizer.transform(x_train)
        if x_val:
            val_vectorized = vectorizer.transform(x_val)
            return train_vectorized, val_vectorized, vectorizer
        return train_vectorized, vectorizer

    # Vectorize for cnn


    def sequentialize_data(self, train_contents, feature_mode, val_contents=None, tokenizer=None, max_length=None):
        """Vectorize data into ngram vectors.

        Args:
            train_contents: training instances
            val_contents: validation instances
            y_train: labels of train data.

        Returns:
            sparse ngram vectors of train, valid text inputs.
        """
        if tokenizer is None:
            if feature_mode == 0:
                tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE,
                                           char_level=True,
                                           oov_token="UNK")
            elif feature_mode == 1:
                tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)

            tokenizer.fit_on_texts(train_contents + val_contents)
        x_train = tokenizer.texts_to_sequences(train_contents)

        if val_contents:
            x_val = tokenizer.texts_to_sequences(val_contents)

        if max_length == None:
            max_length = len(max(x_train, key=len))
            ave_length = np.mean([len(i) for i in x_train])
            std_length = np.std([len(i) for i in x_train])

            print("max_length_word_training:", max_length)  # 获取最大文本长度
            print("ave_length_word_training:", ave_length)  # 获取平均文本长度
            print("std_length_word_training:", std_length)
            # max_length = max_length - int(std_length / 2)

        if max_length > MAX_SEQ_LENGTH:
            max_length = MAX_SEQ_LENGTH

        x_train = sequence.pad_sequences(x_train, maxlen=max_length, padding='post')
        if val_contents:
            x_val = sequence.pad_sequences(x_val, maxlen=max_length, padding='post')

        word_index = tokenizer.word_index
        num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
        print("vacab_word:", len(word_index))
        if val_contents:
            return x_train, x_val, word_index, num_features, tokenizer, max_length
        else:
            return x_train, word_index, num_features, tokenizer, max_length


    def sequentialize_data_no_padding(self, train_contents, feature_mode, val_contents=[], tokenizer=None, max_length=None,
                                      Max_Vocab_Size=None):
        if Max_Vocab_Size is None:
            Vocab_Size = MAX_VOCAB_SIZE
        else:
            Vocab_Size = Max_Vocab_Size
        print("Max Vocab Size is {}".format(Vocab_Size))
        if tokenizer is None:
            if feature_mode == 0:
                tokenizer = text.Tokenizer(num_words=Vocab_Size,
                                           char_level=True,
                                           oov_token="UNK")
            elif feature_mode == 1:
                tokenizer = text.Tokenizer(num_words=Vocab_Size)

            tokenizer.fit_on_texts(train_contents)

        _max_length = max_length
        word_index = tokenizer.word_index
        num_features = min(len(word_index) + 1, Vocab_Size)
        print("vacab_word:", len(word_index))

        if val_contents:
            return word_index, num_features, tokenizer, _max_length
        else:
            return word_index, num_features, tokenizer, _max_length
