"""MIT License

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
SOFTWARE."""
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import argparse
import time
import gzip
import gc

os.system("pip install jieba_fast -i https://pypi.tuna.tsinghua.edu.cn/simple")

# os.system("pip install fastNLP -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install pathos -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install bpemb -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install keras-radam -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.system("pip install wordninja")
os.system("apt-get install wget")

# os.system("python -m pip install scikit-learn==0.21.0  -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.system("pip install imbalanced-learn==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.system("python; import sklearn")
# os.system("pip install scikit-learn==0.21.0  -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.system("pip install keras-bert")
# os.system("pip install 'kashgari<1.0.0'")
import os
import jieba_fast as jieba
import math
from Auto_NLP.upwind_flys_update.model_manager import ModelGenerator
from Auto_NLP.upwind_flys_update.data_manager import DataGenerator
from Auto_NLP.upwind_flys_update.data_manager import sample_input_data
from Auto_NLP.DeepBlueAI import ac
# from Auto_NLP.upwind_flys_update.preprocess_utils import clean_en_with_different_cut as  clean_en_original
from Auto_NLP.upwind_flys_update.preprocess_utils import clean_en_original
# from meta_utils import feature_dict
import numpy as np
import logging
import sys, getopt
import keras
from functools import reduce

# import wordninja
from keras.preprocessing import sequence  # from tensorflow.python.keras.preprocessing import sequence
from keras import backend as K
# from keras_radam import RAdam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

# from AutoNLP.upwind_flys_update.pytf_finetune_bert import FineTuneBertModel
# from keras_bert import extract_embeddings
# from sentence_transformers import SentenceTransformer
# from kashgari.embeddings import BERTEmbedding

print(keras.__version__)

nltk_data_path = '/root/nltk_data/corpora'
wordnet_path = os.path.join(os.path.dirname(__file__), "wordnet")
print(wordnet_path)
os.system("mkdir /root/nltk_data")
os.system("mkdir {}".format(nltk_data_path))
os.system("cp -r {} {}".format(wordnet_path, nltk_data_path))

from nltk.corpus import wordnet

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
import json
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras
from keras.models import load_model

# sess = K.set_session()
MAX_SEQ_LENGTH = 301
# MAX_SEQ_LENGTH = 601
# Limit on the number of features. We use the top 20K features
MAX_VOCAB_SIZE = 20000
MAX_CHAR_LENGTH = 96  # 128
MIN_SAMPLE_NUM = 6000

SAMPLE_NUM_PER_CLASS = 800
SAMPLE_NUM_PER_CLASS_ZH = 1000
SAMPLE_NUM_PER_CLASS_EN = 5000

NUM_EPOCH = 1
VALID_RATIO = 0.1
TOTAL_CALL_NUM = 120  # 120
NUM_MIN_SAMPLES = 8000
UP_SAMPING_FACTOR = 10

NUM_UPSAMPLING_MAX = 100000
INIT_BATCH_SIZE = 32
CHI_WORD_LENGTH = 2
EMBEDDING_SIZE = 300
verbosity_level = 'INFO'
MAX_EN_CHAR_LENGTH = 35

import string

# from numba import cuda
# from imblearn.keras import BalancedBatchGenerator

# from nltk.corpus import stopwords

# english_stopwords = stopwords.words('english')


punctuations = string.punctuation
from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import EnglishStemmer, SnowballStemmer
from scipy import stats

stemmer = SnowballStemmer('english')

# stemmer = EnglishStemmer()
TFIDF_VOCAB = None
# from sklearn.svm import LinearSVC
# from pathos.multiprocessing import ProcessingPoll as PPool
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

from Auto_NLP.upwind_flys_update.time_utils import info
from Auto_NLP.deepWisdom.pytf_finetune_bert import FineTuneBertModel
from Auto_NLP.deepWisdom.pytf_finetune_bert import pretrained_models_download
from Auto_NLP.upwind_flys_update.data_generator import DataGenerator as BatchDataGenerator
from Auto_NLP.upwind_flys_update.utils import set_mp, clean_data, pad_sequence, full_stop_words, clean_en_text_single

# from Auto_NLP.DeepBlueAI.model_db import Model as DB_Model
from Auto_NLP.DeepBlueAI.model_iter_db import Model as DB_Model

pretrained_models_download()

weights_file = os.path.join(os.path.dirname(__file__), "model_cnn.h5")

global svm_tokenizer


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger(verbosity_level)

from multiprocessing import Pool
import multiprocessing

import scipy.sparse as sp

NCPU = multiprocessing.cpu_count() - 1
import torch

# num_cores = multiprocessing.cpu_count()
# num_partitions = num_cores - 2  # I like to leave some cores for other
# processes
print(NCPU)


# from sklearn.feature_extraction.text import TfidfVectorizer

def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        oldval = sa[0]
        k0 = 0
        for k in range(1, m):
            if sa[k] != oldval:
                R[k0:k] = sum(R[k0:k]) / (k - k0)
                k0 = k
                oldval = sa[k]
        R[k0:m] = sum(R[k0:m]) / (m - k0)
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S


def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0: return R
    average = lambda x: reduce(
        lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] +
                      (1. / (j[0] + 1)) * j[1]), enumerate(x))[1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))


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
        # print("add!")
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def clean_zh_text_parallel(dat, worker_num=NCPU, partition_num=10, vocab=None, fn=None):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)
    # data = p.map(clean_zh_word_text, sub_data_list)
    data = p.map(fn, sub_data_list)
    p.close()
    flat_data = [item for sublist in data for item in sublist]

    return flat_data


def clean_en_text_parallel(dat, worker_num=NCPU, partition_num=10, vocab=None, fn=None):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)
    data = p.map(fn, sub_data_list)
    # data = p.map(clean_en_original, sub_data_list)
    p.close()

    # 把 list of list of str 结果 flat 回到 list of str
    flat_data = [item for sublist in data for item in sublist]
    # flat_data = [p.get() for p in data][0]
    # print(flat_data[:3])
    return flat_data


def detect_digits(input_str):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:"))
    input_str = input_str.strip()
    clean_line = input_str.translate(trantab)
    cnt = 0
    words = clean_line.strip().split()
    for word in words:
        if word.isdigit():
            # print(word)
            cnt += 1
    return round(float(cnt) / float(len(words)), 4)


def detect_supper_and_digits(input_str_list):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:"))
    upper_cnt, digits_cnt = [], []
    for input_str in input_str_list:
        input_str = input_str.strip()
        clean_line = input_str.translate(trantab)
        cnt = 0
        digit_cnt = 0
        words = clean_line.strip().split()
        for word in words:
            if word.istitle() or word.isupper():
                # print(word)
                cnt += 1
            if word.isdigit():
                # print(word)
                digit_cnt += 1
        if len(words) > 0:
            upper_cnt.append(round(float(cnt) / float(len(words)), 5))
            digits_cnt.append(round(float(digit_cnt) / float(len(words)), 5))
    return np.average(upper_cnt), np.average(digits_cnt)


def detect_punctuation(input_str_lst):
    trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:" + '0123456789'))
    cnt = []
    for input_str in input_str_lst:
        input_str = input_str.strip()
        clean_line = input_str.translate(trantab)
        cnt_original = len(input_str.split())
        cnt_clean = len(clean_line.split())
        if cnt_original == 0:
            cnt.append(0.0)
        else:
            cnt.append(round(float(cnt_original - cnt_clean) / float(cnt_original), 5))
    return np.average(cnt)


def get_word(str):
    return str + " "


def clean_zh_word_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！? ～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub('', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        # ret.append
        # s = _tokenize_chinese_words(line)
        # line_ = list(map(get_word, line))
        ret.append(line)

    return ret


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！? ～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub('', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        # ret.append
        # s = _tokenize_chinese_words(line)
        # line_ = list(map(get_word, line))
        ret.append(line)

    return ret


def categorical_focal_loss_fixed(y_true, y_pred):
    """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

    gamma = 2.
    alpha = .25
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Sum the losses in mini_batch
    return K.sum(loss, axis=1)


def convert_data(tokenizer,
                 train_contents,
                 max_length_fixed,
                 val_contents=None):
    x_train = tokenizer.texts_to_sequences(train_contents)

    if val_contents:
        x_val = tokenizer.texts_to_sequences(val_contents)

    max_length = len(max(x_train, key=len))
    ave_length = np.mean([len(i) for i in x_train])
    info("max_length_word_training:", max_length)
    info("ave_length_word_training:", ave_length)

    x_train = sequence.pad_sequences(x_train, maxlen=max_length_fixed)
    if val_contents:
        x_val = sequence.pad_sequences(x_val, maxlen=max_length_fixed)

    if val_contents:
        return x_train, x_val
    else:
        return x_train


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))
    # return ''.join(jieba.cut(text, cut_all=False))


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):
    """ 
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path
        self.model = None
        self.call_num = 0
        self.load_pretrain_emb = True
        # self.load_pretrain_emb = False
        self.emb_size = EMBEDDING_SIZE
        self.batch_size = INIT_BATCH_SIZE
        self.total_call_num = TOTAL_CALL_NUM
        self.valid_cost_list = [[]] * 20
        self.auc = 0
        self.svm = True
        self.svm_model = None
        self.svm_token = None
        self.tokenizer = None
        self.model_weights_list = [[]] * 20
        # 0: char based   1: word based   2: doc based
        self.feature_mode = 1
        self.vocab = None
        # self.use_bpe = True
        self.use_bpe = False
        self.reduce_lr = False
        # "text_cnn" "lstm" "sep_cnn_model"
        self.model_mode = 'text_cnn'
        self.fasttext_embeddings_index = None
        self.add_pseudo_data = False
        self.avg_word_per_sample = 0.0
        self.use_pretrain_model = False
        self.use_tf_direct = True
        # self.mp_pooler = set_mp(processes=4)
        self.mp_pooler = None
        self.svm_model = None
        self.imbalance_level = -1
        # 0: binary_crossentropy
        # 1: categorical_crossentropy
        # 2: sparse_categorical_crossentropy
        self.metric = 1
        self.max_length = 0
        self.seq_len_std = 0.0
        finetune_classifer = FineTuneBertModel(metadata=self.metadata)
        self.ft_model = finetune_classifer

        # self.ft_model = None
        self.num_features = MAX_VOCAB_SIZE
        # load pretrian embeding
        if self.load_pretrain_emb:
            self._load_emb()

        self.db_model = DB_Model(self.metadata, fasttext_emb=self.fasttext_embeddings_index)
        normal_lr = LearningRateScheduler(self.lr_decay)
        self.callbacks = []
        early_stopping = EarlyStopping(monitor="loss", patience=15)
        self.callbacks.append(normal_lr)

        self.best_val_auc = 0.0
        self.best_cnn_auc = 0.0
        self.best_rcnn_auc = 0.0
        self.best_call_num = 0
        self.best_val = {0: 0.0}
        self.encode_test = False
        self.cur_lr = None
        self.tokenize_test = False
        self.clean_x = []
        self.clean_y = []
        self.index_to_token = {}
        self.clean_valid_x = []
        self.bert_check_length = 0
        self.start_ft_bert = False
        self.start_ft_bert_call_num = 0
        self.bert_auc = 0.0
        # self.best_ft_model = []
        self.update_bert = False
        self.best_bert_pred = []
        self.lrs = [0.016, 0.0035]
        self.scos = [-1]
        self.his_scos = []
        self.best_sco = -1
        self.best_res = []
        self.best_val_res = [0] * 30
        self.best_test_res = [0] * 30
        self.use_db_model = False
        self.use_multi_svm = True
        self.start_cnn_call_num = 4
        self.imbalance_flow_control = 1
        self.split_val_x = False
        self.valid_on_first_round = False  # 是否在第一轮做评估
        self.is_best = False
        self.feature_dict = {}
        self.time_record = {}
        self.start_db_model = False
        self.first_stage_done = False
        self.test_result = [0] * 30
        self.bert_result = []
        self.bert_output_patience = 3
        self.second_stage_done = False
        self.do_zh_tokenize = False
        self.cut_type = 0  # cut_type: 0: 直接截断； 1：前面+后面； 2：抽取关键词作为截断
        self.first_cnn_done = False
        self.model_id = 0
        self.select_cnn = False
        self.finish_first_cnn_call_num = 0
        self.best_scores = []
        self.x_test_clean_word = None
        self.hist_test = [[]] * 20
        self.model_weights_update_flg = [[]] * 20
        self.use_char = False
        self.seg_test_word = False


    def clean_vocab(self):
        trantab = str.maketrans(dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]?.\/+_~:" + '0123456789'))
        # new_vocab = {}
        for token, index in self.vocab.items():
            _token = token.translate(trantab)
            self.index_to_token[index] = _token

    def ensemble(self):
        feat_size = len(self.his_scos) + 1
        return np.mean(self.best_test_res[1:feat_size], axis=0)

    def define_models(self):
        models = ['CNN', 'GRU', '', '', '', '']
        methods = ['', 'char-level', 'word-level + pretrained embedding300dim', 'word-level + 64dim-embedding', '', '',
                   '']
        return models, methods

    def to_json(self, name, feature=None):
        json_file = os.path.join(os.path.dirname(__file__), name + ".json")
        # json_obj = json.dumps(self.feature_dict)
        if feature is None:
            feature = self.feature_dict
        with open(json_file, "w") as f:
            json.dump(feature, f)

    def train_svm(self, train_x, train_y):
        self.svm_model = self.model_manager.build_model(model_name='svm',
                                                        data_feature=self.data_generator.data_feature)

        # 采样svm，保证进入svm的样本不超过20000个
        if len(train_x) > 20000:
            info("Redo sample size limitation for SVM! Use up to 20000 samples")
            self.data_generator.x_train = train_x[:20000]
            self.data_generator.y_train = train_y[:20000, :]

        else:
            self.data_generator.x_train = train_x
            self.data_generator.y_train = train_y

        # 对合并后的数据进行tfidf特征制作
        if self.use_char == True:
            analyzer = 'char'
        else:
            analyzer = 'word'
        train_vectorized, self.data_generator.svm_token = self.data_generator.vectorize_data(
            self.data_generator.x_train, analyzer=analyzer)

        self.svm_model.fit(train_vectorized, ohe2cat(self.data_generator.y_train))
        self.svm_token = self.data_generator.svm_token

        # 此时的valid_x已经添加之前的验证集样本
        if self.start_cnn_call_num>self.imbalance_flow_control:
            valid_auc = self._valid_auc(self.clean_valid_x, self.data_generator.valid_y, svm=True,
                                    model=self.svm_model)
        else:
            valid_auc = 0.0
        return valid_auc

    def ft_pretrain_model(self, x_train, y_train):
        if self.bert_check_length <= 64:
            self.ft_model.finetune_config.max_seq_length = 64
        else:
            self.ft_model.finetune_config.max_seq_length = 128

        if not self.start_ft_bert:
            # del self.model
            self.ft_model.finetune_config.num_train_epochs = 2
            self.ft_model.finetune_config.per_gpu_train_batch_size = 16
            self.ft_model.finetune_config.warmup_steps = int(0.1 * (3000 // 16 * 2))

            gc.collect()
            K.clear_session()
            self.start_ft_bert = True
            self.start_ft_bert_call_num = self.call_num
            info("start_ft_bert_call_num is {}".format(self.start_ft_bert_call_num))

        if self.call_num >= self.start_ft_bert_call_num + 2:  #
            self.ft_model.finetune_config.learning_rate = max(3 * self.ft_model.finetune_config.learning_rate / 5, 1e-5)
            self.ft_model.finetune_config.num_train_epochs = 1
            self.ft_model.finetune_config.warmup_steps = 0

        if self.metadata["language"] == "ZH":
            # x_train = clean_zh_text_parallel(x_train, vocab=None)
            x_train = clean_zh_text(x_train)
            # fixme: 不需要分词
            x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            # x_train = clean_en_text_parallel(x_train, vocab=None)
            x_train = clean_en_original(x_train)

        if len(x_train) > 3000:
            max_num = int(3000.0 / float(self.metadata["class_num"]))
            _x_train, _y_train = sample_input_data(x_train, y_train, self.metadata["class_num"], max_num=max_num)
        else:
            _x_train, _y_train = x_train, y_train

        info("Current Max Length is {}".format(np.max([len(x) for x in _x_train])))
        info("Current Avg Length is {}".format(np.average([len(x) for x in _x_train])))

        ft_model = self.ft_model.train_model_process(_x_train, ohe2cat(_y_train), self.ft_model.model)
        y_eval = self.ft_model.model_eval_process(self.clean_valid_x, ohe2cat(self.data_generator.valid_y),
                                                  ft_model)
        bert_auc = self._autodl_auc(self.data_generator.valid_y, y_eval)
        info("bert_auc is {} and best bert_auc is {}".format(bert_auc, self.bert_auc))
        if bert_auc > self.bert_auc:
            info("update bert ft model!\n ")
            # 仅考虑连续auc不上升的case，当auc出现更优结果，又重新计算patience
            self.bert_output_patience = 3
            self.update_bert = True
            self.bert_auc = bert_auc
        else:
            self.bert_output_patience -= 1
            self.update_bert = False

        if self.bert_auc > self.best_val_auc:
            self.use_pretrain_model = True
            self.selcet_svm = False
            return
        else:
            info("update: model save and reload!")
            self.use_pretrain_model = False
            return

    def sample_data_from_input(self, y_train):
        if y_train.shape[0] > 0:
            # 更新新增数据的index
            info("start sample_dataset_from_metadataset_iter for call_num={}!".format(self.call_num))
            if self.call_num >= self.start_cnn_call_num:
                use_val = True
            else:
                use_val = False
            print("use_val",use_val)
            # if self.start_cnn_call_num == 1 and not self.split_val_x:  # 极不均衡数据集，从全量里采样

            if self.start_cnn_call_num == self.imbalance_flow_control and not self.split_val_x and self.call_num==self.start_cnn_call_num:
                # if not self.split_val_x:
                train_diff_x, train_diff_y, val_diff_x, val_diff_y = self.data_generator.sample_dataset_from_metadataset_all_data()
                info("finish sample_dataset_from_metadataset_iter for call_num={}!".format(self.call_num))

                return train_diff_x, train_diff_y, val_diff_x, val_diff_y

            # elif self.start_cnn_call_num > 1:
            elif self.start_cnn_call_num > 1:
                train_diff_x, train_diff_y, val_diff_x, val_diff_y = self.data_generator.sample_dataset_from_metadataset_iter(
                    use_val)
                # if self.call_num == 0 and self.imbalance_level == 2:
                if self.imbalance_level == 2:
                    self.data_generator.meta_train_x = self.data_generator.meta_data_x
                    self.data_generator.meta_train_y = self.data_generator.meta_data_y

                info("check train_diff_x size  {}  and val_diff_x size {}".format((len(train_diff_x)), len(val_diff_x)))
                info("finish sample_dataset_from_metadataset_iter for call_num={}!".format(self.call_num))
                return train_diff_x, train_diff_y, val_diff_x, val_diff_y

            # else:
            #     train_diff_x, train_diff_y = self.data_generator.sample_dataset_from_metadatset_full_train()
            #     info("Use full data random sample!")
            #     return train_diff_x, train_diff_y, None, None

        else:  # no sample input
            train_diff_x, train_diff_y = self.data_generator.sample_dataset_from_metadatset_full_train()
            info("Use full data random sample!")
            return train_diff_x, train_diff_y, None, None

    def run_first_svm(self, train_diff_x, train_diff_y, val_diff_x, val_diff_y):
        info("start clean_Data!")

        if self.metadata["language"] == "ZH":
            # train_diff_x_preprocessed = clean_zh_text_parallel(train_diff_x, vocab=None)
            start = time.time()
            # train_diff_x_preprocessed = clean_zh_text(train_diff_x)
            # train_diff_x_preprocessed =clean_zh_text_parallel(train_diff_x, fn=clean_zh_word_text)
            train_diff_x = np.array(train_diff_x, dtype='object')
            train_diff_x_preprocessed = ac.clean_text_zh_seg1(train_diff_x, MAX_SEQ_LENGTH)
            end = time.time()
            self.time_record["clean_zh_text_train"] = end - start
            # print(train_diff_x_preprocessed[:5])
            start = time.time()
            # train_diff_x_preprocessed = list(map(_tokenize_chinese_words, train_diff_x_preprocessed))
            end = time.time()
            # self.time_record["_tokenize_chinese_words_train"] = end - start
            start = time.time()
            # valid_x = clean_zh_text_parallel(val_diff_x, fn=clean_zh_word_text)
            val_diff_x = np.array(val_diff_x, dtype='object')
            valid_x = ac.clean_text_zh_seg1(val_diff_x, MAX_SEQ_LENGTH)
            # valid_x = clean_zh_text(val_diff_x)
            end = time.time()
            self.time_record["clean_zh_text_valid"] = end - start
            start = time.time()
            # valid_x = list(map(_tokenize_chinese_words, valid_x))
            end = time.time()
            # self.time_record["_tokenize_chinese_words_valid"] = end - start
        else:
            start = time.time()
            train_diff_x_preprocessed = clean_en_original(train_diff_x)
            end = time.time()
            self.time_record["clean_en_original_train"] = end - start
            start = time.time()
            valid_x = clean_en_original(val_diff_x)
            end = time.time()
            self.time_record["clean_en_original_valid"] = end - start
            # valid_x = clean_en_text_parallel(val_diff_x, vocab=None)

        info("b4: check preprocessed train_data size:{}, label size:{}".format(len(train_diff_x_preprocessed),
                                                                               train_diff_y.shape[0]))
        info("end clean_Data!")
        self.svm_x_train = train_diff_x_preprocessed
        self.svm_y_train = train_diff_y

        # gc.collect()
        self.data_generator.valid_x = val_diff_x
        self.data_generator.valid_y = val_diff_y
        self.clean_valid_x = valid_x
        self.data_generator.x_val_raw = self.data_generator.valid_x

        if len(self.svm_x_train) > 20000:
            info("Redo sample size limitation for SVM! Use up to 20000 samples")
            self.data_generator.x_train = self.svm_x_train[:20000]
            self.data_generator.y_train = self.svm_y_train[:20000, :]
        else:
            self.data_generator.x_train = self.svm_x_train
            self.data_generator.y_train = self.svm_y_train

        info("After: check preprocessed train_data size:{}, label size:{}".format(len(self.svm_x_train),
                                                                                  self.svm_y_train.shape[0]))

        if not self.valid_on_first_round:  # 如果不在第一轮评估，默认直接出点
            self.data_generator.x_train = self.data_generator.x_train + valid_x
            self.data_generator.y_train = np.concatenate([self.data_generator.y_train, val_diff_y], axis=0)

        info("start vectorize_data!")
        if self.metadata["language"] == "ZH":
            analyzer = 'char'
        else:
            analyzer = "word"

        print("check type of x_train {}".format(type(self.data_generator.x_train)))

        start = time.time()
        train_vectorized, self.data_generator.svm_token = self.data_generator.vectorize_data(
            self.data_generator.x_train, analyzer=analyzer)
        end = time.time()
        self.time_record["vectorize_data"] = end - start
        # print(self.data_generator.svm_token.vocabulary_)
        # self.data_generator.y_train = train_diff_y
        print("check train_vectorized shape{}".format(train_vectorized.shape))
        info("end vectorize_data!")
        start = time.time()
        self.model.fit(train_vectorized, ohe2cat(self.data_generator.y_train))
        end = time.time()
        self.time_record['svm fit'] = end - start
        self.svm_token = self.data_generator.svm_token

        if not self.valid_on_first_round:
            valid_auc = 0.0
        else:
            start = time.time()
            valid_auc = self._valid_auc(valid_x, self.data_generator.valid_y, svm=True)
            if self.empty_class_ and self.kurtosis < 0:
                valid_auc = valid_auc * 1 * (1 -
                                             (float(len(self.empty_class_)) / float(self.metadata["class_num"])))
            end = time.time()
            self.time_record["valid_auc"] = end - start

        info("original valid_auc_svm: {}".format(valid_auc))

        self.valid_auc_svm = valid_auc
        info("valid_auc_svm {}".format(self.valid_auc_svm))

    def set_cnn_params(self):
        ############################## 第一阶段 CNN 设置模型参数 ####################################

        self.data_generator.data_feature[
            'num_features'] = self.data_generator.num_features  # self.data_generator.bpe_encoder.vectors.shape[0]                                                                                                                  # self.data_generator.num_features
        self.data_generator.data_feature['num_class'] = self.data_generator.num_classes
        self.data_generator.data_feature['max_length'] = self.max_length
        self.data_generator.data_feature['input_shape'] = self.max_length
        self.data_generator.data_feature["rnn_units"] = 128
        self.data_generator.data_feature["filter_num"] = 64
        self.data_generator.data_feature["word_index"] = self.data_generator.word_index

    def build_tokenizer(self, preprocessed_dat):
        ############################## 构建tokenizer ####################################
        self.set_max_seq_len()
        self.data_generator.feature_mode = 1
        Max_Vocab_Size = self.set_max_vocab_size(preprocessed_dat)
        # if self.use_multi_svm:
        #     Max_Vocab_Size = self.set_max_vocab_size(preprocessed_dat)
        #
        # else:
        #     Max_Vocab_Size = self.set_max_vocab_size(preprocessed_dat)

        self.data_generator.word_index, self.data_generator.num_features, \
        self.data_generator.tokenizer, self.max_length = self.data_generator.sequentialize_data_no_padding(
            preprocessed_dat, self.data_generator.feature_mode,
            tokenizer=None,
            max_length=self.max_length,
            Max_Vocab_Size=Max_Vocab_Size)
        # for word, index in self.data_generator.word_index.items():
        #     if index<30:
        #         print("word: {}, index {}".format(word, index))

    def run_first_stage_model(self, preprocessed_dat, train_diff_y):
        bs_x_train = preprocessed_dat
        bs_y_train = train_diff_y
        num_epochs = 1

        info("Train on {} samples".format(bs_y_train.shape[0]))
        bs_training_generator = BatchDataGenerator(bs_x_train, bs_y_train, batch_size=self.batch_size,
                                                   mp_pooler=self.mp_pooler,
                                                   bpe_encoder=None,
                                                   language=self.metadata["language"],
                                                   max_length=self.max_length if self.max_length else 100,
                                                   vocab=None,
                                                   tokenizer=self.data_generator.tokenizer,
                                                   num_features=self.data_generator.num_features)

        history = self.model.fit_generator(generator=bs_training_generator, verbose=1,
                                           epochs=num_epochs,
                                           callbacks=self.callbacks,
                                           shuffle=True)

        return history

    def preprocess_data(self, x):
        if self.metadata["language"] == "ZH":
            if self.call_num >= self.start_cnn_call_num:
                info("use word-level")
                # preprocessed_dat = clean_zh_text_parallel(x, vocab=None, fn=clean_zh_text)
                x = np.array(x, dtype='object')
                preprocessed_dat = ac.clean_text_zh_seg1(x, MAX_SEQ_LENGTH)
                preprocessed_dat = list(map(_tokenize_chinese_words, preprocessed_dat))
            else:
                # fixme: 先不用，因为后面用前N次的结果 build word
                info("use char-level")
                # preprocessed_dat = clean_zh_text_parallel(x, vocab=None, fn=clean_zh_word_text)
                # preprocessed_dat = clean_zh_text_parallel(x, vocab=None, fn=clean_zh_text)
                x = np.array(x, dtype='object')
                preprocessed_dat = ac.clean_text_zh_seg1(x, MAX_SEQ_LENGTH)
                # self.use_char = True
                preprocessed_dat = list(map(_tokenize_chinese_words, preprocessed_dat))
            # print(preprocessed_dat[:3])
        else:
            # preprocessed_dat = clean_en_text_parallel(train_diff_x, vocab=None)
            preprocessed_dat = clean_en_original(x)
        return preprocessed_dat

    def set_max_vocab_size(self, input_x):
        avg_punct_cnt = detect_punctuation(input_x)
        avg_upper_cnt, avg_digit_cnt = detect_supper_and_digits(input_x)

        info("avg_punct_cnt is {} and avg_upper_cnt is {} and avg_digit_cnt is {}".format(avg_punct_cnt,
                                                                                          avg_upper_cnt,
                                                                                          avg_digit_cnt))
        if avg_punct_cnt <= 0.02:
            Max_Vocab_Size = 30000
        else:
            Max_Vocab_Size = 20000
        info("set Max_Vocab_Size:{}".format(Max_Vocab_Size))
        if "avg_punct_cnt" not in self.feature_dict:
            self.feature_dict["avg_punct_cnt"] = float(avg_punct_cnt)
            self.feature_dict["avg_upper_cnt"] = float(avg_upper_cnt)
            self.feature_dict["avg_digit_cnt"] = float(avg_digit_cnt)
            print("feature_dict:", self.feature_dict)
            self.to_json(name="new_feature")

        return Max_Vocab_Size

    def set_max_seq_len(self):
        if self.max_length > MAX_SEQ_LENGTH:
            self.max_length = MAX_SEQ_LENGTH
            info("update max_length {}".format(self.max_length))
        if self.seq_len_std > 150:
            self.max_length = 301
            info("update max_length {}".format(self.max_length))

    def train(self, x_train, y_train, remaining_time_budget=None):
        """model training on train_dataset.It can be seen as metecontroller
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
:        """
        if self.done_training:
            return

        if self.use_tf_direct:
            pass

        info("check input_y {}".format(y_train.shape))

        if self.call_num == 0:
            # if self.imbalance_level == 2:
            if self.imbalance_level == 2 or self.metadata["class_num"] >= 5:
                self.callbacks.pop(0)  # 不decay lr
            info("start preprocessing for call_num=0!")
            self.data_generator = DataGenerator(x_train, y_train, self.metadata, self.imbalance_level)

            self.data_generator.feature_mode = 1

            start = time.time()
            self.data_generator.snoop_data(metadata=self.metadata)
            end = time.time()
            self.time_record["snoop_data"] = end - start

            self.model_manager = ModelGenerator(self.data_generator.feature_mode,
                                                load_pretrain_emb=self.load_pretrain_emb,
                                                fasttext_embeddings_index=self.fasttext_embeddings_index)

        else:
            if y_train.shape[0] > 0:
                # use_diff = self.call_num<=4
                use_diff = False
                info("start update_meta_data!")
                self.data_generator.update_meta_data(x_train, y_train, use_diff)
                info("end update_meta_data!")

        info("check meta_data_y {}".format(self.data_generator.meta_data_y.shape))

        # 每次从全局采样
        info("start sample_data_from_input!")
        start = time.time()
        print(y_train)
        train_diff_x, train_diff_y, val_diff_x, val_diff_y = self.sample_data_from_input(y_train)
        end = time.time()
        if "sample_data_from_input" not in self.time_record:
            self.time_record["sample_data_from_input"] = end - start
        info("end sample_data_from_input!")

        ############################ 预训练模型 阶段 ##############################
        # 进入预训练模型部分
        if self.second_stage_done and self.avg_word_per_sample <= 12 and self.bert_check_length <= 156:  # 结束第二个阶段再进入预训练模型
            if self.start_ft_bert and not self.use_pretrain_model:
                # 不继续使用bert
                # fixme: 考虑设一个缓冲区，允许bert多训练几次
                self.use_pretrain_model = False
                return
            else:
                if self.bert_output_patience > 0:
                    return self.ft_pretrain_model(train_diff_x, train_diff_y)
                else:
                    self.use_pretrain_model = False
                    return
        ############################ DB模型训练 阶段 ##############################
        # 进入db模型部分

        elif self.first_stage_done:
            if not self.start_db_model:
                # 第一次进入db时，先清空现有的sess
                del self.model
                gc.collect()
                K.clear_session()
                self.start_db_model = True
                info("check samples {}".format(len(self.data_generator.meta_train_x)))

            if self.db_model.model_id == len(
                    self.db_model.cand_models) and self.db_model.data_id == self.db_model.max_data:
                self.second_stage_done = True
                info("finish second stage!")
                return

            self.db_model.train_iter((self.data_generator.meta_data_x, self.data_generator.meta_data_y),
                                     eval_dataset=(self.data_generator.valid_x, self.data_generator.valid_y),
                                     remaining_time_budget=remaining_time_budget)

            db_auc = self.db_model.best_sco  # 本身是一个集成结果
            if db_auc == -1:
                db_auc = 0.0
            # if db_auc >= self.best_val_auc * 0.97:
            if db_auc >= self.best_val_auc * 0.95:
                print("Use db_model when db_auc is {} and best_val_auc is {}".format(db_auc, self.best_val_auc))
                self.use_db_model = True
                if self.db_model.Xtest is None:
                    self.db_model.START = True
                return
            else:
                print("Do not Use db_model when db_auc is {} and best_val_auc is {}".format(db_auc, self.best_val_auc))
                # self.use_db_model = False
                # 这里需要保持db model内部的状态，不然会漏状态
                if self.db_model.START == False and self.db_model.best_sco == 0.02:
                    self.db_model.is_best = False
                    self.db_model.LASTROUND = False
                    # pass
                elif self.db_model.START == True:
                    self.db_model.START = False
                # sess = tf.Session(config=config)
                # K.set_session(sess)
                info("update: model save and reload!")
                # self.model = load_model(weights_file)
                self.use_db_model = False
                return

        ############################ 第一阶段 SVM/CNN/RCNN ##############################
        # 进入第一阶段训练: 选择模型：svm/cnn/rcnn
        self.model_name = self.model_manager.model_pre_select(self.call_num, self.data_generator.snoop_avg_text_length,
                                                              cnn_done_status=self.first_cnn_done)
        info("finish model_pre_select!")

        # 第一阶段先选svm，再选CNN
        if self.call_num == 0:
            info("start build svm model!")
            start = time.time()
            self.model = self.model_manager.build_model(self.model_name, self.data_generator.data_feature)
            end = time.time()
            self.time_record["build model"] = end - start
            info("finish build svm model!")

        # 第一阶段前置第一个SVM训练过程：数据处理，模型训练，模型评估（默认不评估）
        if self.call_num == 0:
            self.run_first_svm(train_diff_x, train_diff_y, val_diff_x, val_diff_y)

        # 进入第一阶段数据处理阶段，只处理增量数据
        else:  # 处理call_num>0的所有情况
            info("clean full_x start")
            info("b4: check preprocessed train_data size:{}, label size:{}".format(len(train_diff_x),
                                                                                   train_diff_y.shape[0]))
            preprocessed_dat = self.preprocess_data(train_diff_x)
            info("check preprocessed_dat size {}".format(len(preprocessed_dat)))
            # 增量 前处理后的样本
            info("b4: check preprocessed train_data size:{}, label size:{}".format(len(train_diff_x),
                                                                                   train_diff_y.shape[0]))
            if not self.data_generator.tokenizer:
                # 在构建tokenizer之前，存下前N次的预处理文本，作为tokenizer fit的样本
                if self.metadata["language"] == "ZH" and self.call_num==1:
                    self.svm_x_train = preprocessed_dat
                    self.svm_y_train = train_diff_y
                    # self.clean_valid_x = list(map(_tokenize_chinese_words, self.clean_valid_x))
                else:
                    self.svm_x_train.extend(preprocessed_dat)
                    self.svm_y_train = np.concatenate([self.svm_y_train, train_diff_y], axis=0)

            info("after:check preprocessed train_data size:{}, label size:{}".format(len(self.svm_x_train),
                                                                                     self.svm_y_train.shape[0]))
            info("clean full_x end")

            ############################ 新增dataset_read_num的评估数据处理 ##############################
            if y_train.shape[0] > 0:
                # 有新增样本才增加valid
                # if self.start_cnn_call_num > 1:  # 走N个SVM再切换CNN
                if self.start_cnn_call_num > self.imbalance_flow_control:  # 走N个SVM再切换CNN
                    info("run multi_svm!")
                    if self.call_num < self.start_cnn_call_num:  # 得到全局评估数据，后面不再增加

                        self.data_generator.valid_x = np.concatenate([self.data_generator.valid_x, val_diff_x], axis=0)
                        self.data_generator.valid_y = np.concatenate([self.data_generator.valid_y, val_diff_y], axis=0)
                        self.data_generator.x_val_raw = self.data_generator.valid_x

                        valid_x = self.preprocess_data(val_diff_x)
                        if self.metadata["language"] == "ZH" and self.call_num == 1:
                            self.clean_valid_x = valid_x
                            self.data_generator.valid_y = val_diff_y
                        else:
                            self.clean_valid_x = np.concatenate([self.clean_valid_x, valid_x], axis=0)
                        info("check preprocessed valid_data_y size:{}".format(self.data_generator.valid_y.shape[0]))
                        info("check preprocessed valid_data size:{}".format(len(self.data_generator.valid_x)))
                        info("check preprocessed valid_data size:{}".format(len(self.clean_valid_x)))
                        info("check preprocessed valid_data_raw size:{}".format(len(self.data_generator.x_val_raw)))

                else:
                    if not self.split_val_x and self.call_num==self.start_cnn_call_num:
                        self.split_val_x = True
                        info("run single_svm!")
                        self.data_generator.valid_x = val_diff_x
                        self.data_generator.valid_y = val_diff_y
                        valid_x = self.preprocess_data(val_diff_x)
                        self.clean_valid_x = valid_x
                        info("check preprocessed valid_data_y size:{}".format(self.data_generator.valid_y.shape[0]))
                        info("check preprocessed valid_data size:{}".format(len(self.data_generator.valid_x)))
                        info("check preprocessed valid_data size:{}".format(len(self.clean_valid_x)))
                        info("check preprocessed valid_data_raw size:{}".format(len(self.data_generator.x_val_raw)))

            ############################## 进入第一阶段 前N个 SVM 训练 #################################
            if self.call_num < self.start_cnn_call_num and self.call_num > 0 and self.use_multi_svm:  # (对于call_num: 1,2,3，走SVM)
                info("train svm model!")
                valid_auc = self.train_svm(preprocessed_dat, train_diff_y)
                info("original valid_auc_svm: {}".format(valid_auc))
                self.valid_auc_svm = valid_auc
                # if self.split_val_x: # 插入点不进行评估
                #     self.valid_auc_svm = 0.0
                info("valid_auc_svm: {}".format(self.valid_auc_svm))
                self.selcet_svm = True
                return
            ############################## 进入第一阶段 深度模型 训练 #################################
            else:
                train_num = self.call_num
                start_offset = self.start_cnn_call_num
                ############################## 进入第一阶段 TextCNN 训练 ###################################
                if self.call_num == self.start_cnn_call_num:  # 从第N+1个call num开始build cnn模型以及embedding encoder
                    if self.start_cnn_call_num == self.imbalance_flow_control:
                        self.build_tokenizer(preprocessed_dat)
                    else:
                        # if self.metadata["language"]=="ZH":
                        #     info("build tokenizer using word-level data!")
                        # #     # self.use_char = False
                        #     self.build_tokenizer(preprocessed_dat)
                        #
                        # else:
                            self.build_tokenizer(self.svm_x_train)
                    self.set_cnn_params()
                    self.model_weights_list[self.model_id] = []
                    self.valid_cost_list[self.model_id] = []
                    info("start build text_cnn model!")
                    self.model = self.model_manager.build_model(self.model_name, self.data_generator.data_feature)
                    info("finish build text_cnn model!")

                ############################## 进入第一阶段 TextRCNN 训练 ###################################
                '''
                elif self.first_cnn_done and not self.first_stage_done:  # CNN 训练结束，重新buid text_rcnn模型
                    start_offset = self.finish_first_cnn_call_num+1
                    train_num = self.call_num
                    if self.call_num == self.finish_first_cnn_call_num + 1:
                        self.model_id += 1

                        # 切换模型
                        self.model = None
                        gc.collect()
                        K.clear_session()

                        self.model_name = self.model_manager.model_pre_select(self.call_num,
                                                                              self.data_generator.snoop_avg_text_length,
                                                                              cnn_done_status=self.first_cnn_done)

                        info("start build text_rcnn model!")
                        self.model = self.model_manager.build_model(self.model_name, self.data_generator.data_feature)
                        self.model_weights_list[self.model_id] = []
                        self.valid_cost_list[self.model_id] = []
                        self.callbacks = []
                        # RCNN 采用大学习率，及快速decay策略
                        lrate = LearningRateScheduler(self.step_decay)
                        self.callbacks.append(lrate)
                        info("finish build text_rcnn model!")
                '''
                history = self.run_first_stage_model(preprocessed_dat, train_diff_y)
                self.feedback_simulation(history, train_num=train_num, start_offset=start_offset)

    def rebuild_predict_prob(self, prediction):
        # new_prob_arary = np.zeros((prediction.shape[0], self.metadata["class_num"]))
        new_prob_arary = prediction
        val_label_distribution = np.sum(np.array(self.data_generator.valid_y), 0)
        self.empty_class_ = [i for i in range(val_label_distribution.shape[0]) if val_label_distribution[i] == 0]
        self.kurtosis = stats.kurtosis(val_label_distribution)
        self.nomalized_std = np.std(val_label_distribution) / np.sum(val_label_distribution)
        info("check empty_class {}".format(self.empty_class_))
        info("check kurtosis is {}".format(self.kurtosis))
        if self.empty_class_:
            info("do rebuild")
            for sample_i in range(prediction.shape[0]):
                np_median_value = np.median(prediction[sample_i])
                for empty_cls in self.empty_class_:
                    new_prob_arary[sample_i][empty_cls] = np_median_value

        return new_prob_arary

    def lr_decay(self, epoch):
        if self.call_num == 1 or self.cur_lr is None:
            self.cur_lr = self.model_manager.lr
        if self.call_num % 7 == 0:
            self.cur_lr = 3 * self.cur_lr / 5
        self.cur_lr = max(self.cur_lr, 0.0001)
        info("recompile lr {}".format(self.cur_lr))
        lr = self.cur_lr
        return lr

    def step_decay(self, epoch):
        epoch = (self.call_num - self.finish_first_cnn_call_num) // 3
        initial_lrate = self.model_manager.lr  # 0.016 #0.0035 #
        drop = 0.65  # 0.65
        epochs_drop = 1.0  # 2.0
        if (self.call_num - self.finish_first_cnn_call_num) <= 5:
            lrate = initial_lrate
        else:
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        lrate = max(lrate, 0.0001)
        info("recompile lr {}".format(lrate))
        return lrate

    def _get_valid_columns(self, solution):
        """Get a list of column indices for which the column has more than one class.
        This is necessary when computing BAC or AUC which involves true positive and
        true negative in the denominator. When some class is missing, these scores
        don't make sense (or you have to add an epsilon to remedy the situation).

        Args:
          solution: array, a matrix of binary entries, of shape
            (num_examples, num_features)
        Returns:
          valid_columns: a list of indices for which the column has more than one
            class.
        """
        num_examples = solution.shape[0]
        col_sum = np.sum(solution, axis=0)
        valid_columns = np.where(1 - np.isclose(col_sum, 0) -
                                 np.isclose(col_sum, num_examples))[0]
        return valid_columns

    def _autodl_auc(self, solution, prediction, valid_columns_only=True):
        """Compute normarlized Area under ROC curve (AUC).
        Return Gini index = 2*AUC-1 for  binary classification problems.
        Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
        for the predictions. If solution and prediction are not vectors, the AUC
        of the columns of the matrices are computed and averaged (with no weight).
        The same for all classification problems (in fact it treats well only the
        binary and multilabel classification problems). When `valid_columns` is not
        `None`, only use a subset of columns for computing the score.
        """
        if valid_columns_only:
            valid_columns = self._get_valid_columns(solution)
            if len(valid_columns) < solution.shape[-1]:
                logger.warning(
                    "Some columns in solution have only one class, " +
                    "ignoring these columns for evaluation.")
            solution = solution[:, valid_columns].copy()
            prediction = prediction[:, valid_columns].copy()
        label_num = solution.shape[1]
        auc = np.empty(label_num)
        for k in range(label_num):
            r_ = tiedrank(prediction[:, k])
            s_ = solution[:, k]
            if sum(s_) == 0:
                print(
                    "WARNING: no positive class example in class {}".format(k +
                                                                            1))
            npos = sum(s_ == 1)
            nneg = sum(s_ < 1)
            auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
        return 2 * mvmean(auc) - 1

    def _valid_auc(self, x_valid, y_valid, svm=False, model=None):

        if svm:
            x_valid = self.svm_token.transform(x_valid)
            # info('y_valid.shape', y_valid.shape)
            if model is None:
                result = self.model.predict_proba(x_valid)
                # print("check result {}".format(result[:5,:]))
                result = self.rebuild_predict_prob(result)
                # print("check result {}".format(result[:5, :]))
            else:
                result = model.predict_proba(x_valid)
                # print("check result {}".format(result[:5, :]))
                result = self.rebuild_predict_prob(result)
                # print("check result {}".format(result[:5, :]))
            # info('result.shape', result.shape)
        else:
            info("Valid on {} samples".format(len(x_valid)))
            bs_eval_generator = BatchDataGenerator(x_valid, y_valid, batch_size=self.batch_size,
                                                   mp_pooler=self.mp_pooler,
                                                   bpe_encoder=None,
                                                   language=self.metadata["language"],
                                                   max_length=self.max_length if self.max_length else 100,
                                                   # vocab=self.tf_idf_vocab,
                                                   vocab=None,
                                                   # tokenizer=None,
                                                   tokenizer=self.data_generator.tokenizer,
                                                   num_features=self.data_generator.num_features,
                                                   shuffle=False)
            result = self.model.predict_generator(bs_eval_generator)
            # result = self.rebuild_predict_prob(result)
            info("show shape of y_valid {}".format(y_valid.shape))
            info("show shape of result {}".format(result.shape))
            # print("result:", result)

        return self._autodl_auc(y_valid, result)  # y_test

    def output_logic(self):
        # self.test_result[0]: CNN 最后结果
        # self.test_result[1]：DB 最后结果

        if not self.first_stage_done:
            info("Output in first stage!")
            # 第一阶段没有结束: 目前选择：svm or CNN or RCNN
            if self.selcet_svm:
                info("select svm in first stage!")
                if self.svm_model:
                    info("use new svm model!")
                    x_test = self.svm_token.transform(self.x_test_clean_word)
                    result = self.svm_model.predict_proba(x_test)
                    self.svm_result = result

                    # todo: 合并svm result
                    info("load svm again!!!")
                    return result
            else:
                info("use CNN/RCNN in first stage!")

                result = self.model.predict(self.x_test,
                                            batch_size=self.batch_size * 16)

                self.hist_test[self.model_id].append(result)
                ensemble_result = np.mean(self.hist_test[self.model_id], axis=0)

                info(
                    "model_id is {} and hist_test size is {}".format(self.model_id, len(self.hist_test[self.model_id])))
                info("model is is {} and hist val auc size is {}".format(self.model_id,
                                                                         len(self.valid_cost_list[self.model_id])))
                print("val cost list {}".format(self.valid_cost_list[self.model_id]))
                print("model weight update flg {}".format(self.model_weights_update_flg[self.model_id]))
                ############################ 单模型ensemble  ####################################

                #################################################################################
                if self.first_cnn_done:
                    if isinstance(self.test_result[0], int):
                        # self.test_result[0] = result
                        self.test_result[0] = ensemble_result
                    # self.test_result[0] = result

                    if self.select_cnn:
                        # result = self.test_result[0]
                        ensemble_result = self.test_result[0]

                    else:
                        # self.test_result[1] = result
                        self.test_result[1] = ensemble_result
                        # result = np.mean(self.test_result, axis=0)
                        ensemble_result = np.mean(self.test_result[:2], axis=0)

                return ensemble_result
                # return result

        elif self.first_stage_done and not self.second_stage_done:
            info("Output in second stage!")

            # 第二阶段没有结束：只有两个选择：db 模型 or 第一阶段最优模型
            if self.use_db_model:
                info("Use db Model!!")
                db_result = self.db_model.test(self.x_test_raw)

                # 如果db输出为空，返回第一个阶段结果
                # if db_result.shape[0] == 0:
                if len(db_result)==0:
                    info("DB result is empty!")
                    if isinstance(self.test_result[2], int):
                        # result = np.mean(self.test_result[:2], axis=0)
                        result = self.test_result[1]
                    else:
                        result = np.mean(self.test_result[1:3], axis=0)
                    return result
                else:
                    info("DB result is Not empty!")
                    self.test_result[2] = db_result
                    result = np.mean(self.test_result[1:3], axis=0)
                    return result
            else:
                if self.start_db_model:
                    info("start_db_model!")
                    # todo: 可以是N个ensemble
                    # result = self.test_result[0]
                    # result = np.mean(self.test_result, axis=0)
                    # result = self.test_result[1]
                    if isinstance(self.test_result[2], int):
                        # result = np.mean(self.test_result[:2], axis=0)
                        result = self.test_result[1]
                    else:
                        result = np.mean(self.test_result[1:3], axis=0)
                else:
                    info("Not start_db_model!")
                    # 如果当前是CNN训练的最后一次输出，保留当前输出
                    result = self.model.predict(self.x_test,
                                                batch_size=self.batch_size * 16)

                    self.hist_test[self.model_id].append(result)
                    ensemble_result = np.mean(self.hist_test[self.model_id], axis=0)

                    if self.first_stage_done:
                        # self.test_result[1] = result
                        self.test_result[1] = ensemble_result

                return result

        else:
            info("Output in third stage!")
            # 第三阶段没有结束：只有两个选择：预训练模型 or 前两阶段最优模型
            if self.use_pretrain_model:
                info("use pretrain_model {}".format(self.use_pretrain_model))
                if self.update_bert:  # 如果更新了bert模型，采用更新的参数进行预测
                    info("use update_bert {}".format(self.update_bert))
                    result = self.ft_model.model_predict_process(self.x_test_clean, self.ft_model.model)
                    self.best_bert_pred = result
                    self.bert_result.append(result)
                    if len(self.bert_result) > 0:  # ensemble前N次 bert结果
                        result = np.mean(self.bert_result, axis=0)
                    info("bert result size 1 {}".format(len(self.bert_result)))

                else:  # 否则，用历史结果出点
                    result = np.mean(self.bert_result, axis=0)
                    info("bert result size 2 {}".format(len(self.bert_result)))
                return result
            else:
                # fixme: 比前两阶段大
                if self.bert_auc * 0.98 > max(self.best_val_auc, self.db_model.best_sco):  # 表明已经存在训练过的bert模型且结果远高于前两阶段
                    result = np.mean(self.bert_result, axis=0)
                    info("use bert ensemble")
                elif self.bert_auc > 0.0:  # 已存在训练过的bert模型，但结果没有远超过前两阶段
                    # self.test_result.extend(self.bert_result)
                    if len(self.bert_result)>0:
                        info("use bert + CNN ensemble when bert result size is {}".format(self.bert_result))
                        self.test_result[3] = np.mean(self.bert_result, axis=0)
                        result = np.mean(self.test_result[1:4], axis=0)
                    else:
                        result = np.mean(self.test_result[1:3], axis=0)
                        info("use bert + CNN ensemble")
                else:  # 表面当前只有CNN模型
                    result = np.mean(self.test_result[:3], axis=0)
                    info("bert result size 2 {}".format(len(self.bert_result)))
                    info("use CNN ensemble")
                return result

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # model = models.load_model(self.test_input_path + 'model.h5')

        train_num, self.test_num = self.metadata[
                                       'train_num'], self.metadata['test_num']
        self.class_num = self.metadata['class_num']
        info("num_samples_test: {}".format(self.test_num))
        info("num_class_test: {}".format(self.class_num))

        # if self.call_num == 0 or self.call_num == 1:
        self.x_test_raw = x_test
        if self.call_num == 0:
            info("start clean x_test!")
            # tokenizing Chinese words
            if self.metadata['language'] == 'ZH':
                # x_test = clean_zh_text_parallel(x_test)
                start = time.time()
                # x_test = clean_zh_text(x_test)
                # x_test = clean_zh_text_parallel(x_test, fn=clean_zh_word_text)
                x_test = np.array(x_test, dtype='object')
                x_test = ac.clean_text_zh_seg1(x_test, MAX_SEQ_LENGTH)
                end = time.time()
                self.time_record["clean_zh_text_test"] = end - start
                start = time.time()
                # x_test = list(map(_tokenize_chinese_words, x_test))
                # x_test = ac.clean_text_zh_seg2(x_test, 0)
                # x_test = [' '.join(s) for s in x_test]
                end = time.time()
                self.time_record["_tokenize_chinese_words_test"] = end - start

            else:
                # x_test = clean_en_text_parallel(x_test, vocab=None)
                start = time.time()
                x_test = clean_en_original(x_test)
                end = time.time()
                self.time_record["clean_en_original_test"] = end - start

            self.x_test_clean = x_test
            info("finish clean x_test!")

            start = time.time()

            x_test = self.svm_token.transform(self.x_test_clean)
            # x_test = parallelize_dataframe(x_test, vectorize)
            end = time.time()
            self.time_record["svm_token_transform_test"] = end - start
            start = time.time()
            result = self.model.predict_proba(x_test)
            end = time.time()
            self.time_record["svm_predict_proba"] = end - start
            self.to_json(name="time_record", feature=self.time_record)

            self.svm_result = result
            self.call_num = self.call_num + 1
            return result  # y_test

        if self.metadata['language'] == 'ZH':
            if not self.x_test_clean_word:
                # redo clean use jieba_fast
                x_test_raw = np.array(self.x_test_raw, dtype='object')
                self.x_test_clean_word = ac.clean_text_zh_seg1(x_test_raw, MAX_SEQ_LENGTH)

            # if not self.use_char and not self.seg_test_word:
                self.x_test_clean_word = list(map(_tokenize_chinese_words, self.x_test_clean_word))
                self.seg_test_word = True
                # if not self.use_char:
                #     self.x_test_clean_word = list(map(_tokenize_chinese_words, self.x_test_clean_word))


        else:
            self.x_test_clean_word = self.x_test_clean

        if self.call_num > self.start_cnn_call_num - 1 or self.selcet_svm == False:
            self.tokenizer = self.data_generator.tokenizer

            if not self.use_pretrain_model:
                info("start encode x_text!")
                if not self.encode_test and self.use_bpe:
                    x_test_clean = self.data_generator.bpe_encoder.encode_ids(self.x_test_clean)  # 经过前处理的x_test
                    self.x_test = sequence.pad_sequences(x_test_clean,
                                                         maxlen=self.max_length,
                                                         padding='post',
                                                         value=self.data_generator.bpe_encoder.vectors.shape[0])

                    self.encode_test = True
                else:
                    if not self.tokenize_test:
                        self.tokenizer = self.data_generator.tokenizer
                        self.x_test = self.tokenizer.texts_to_sequences(self.x_test_clean_word)
                        self.x_test = sequence.pad_sequences(self.x_test,
                                                             maxlen=self.max_length,
                                                             padding='post')
                        self.tokenize_test = True

                info("finish encode x_text!")

        result = self.output_logic()
        # Cumulative training times
        self.call_num = self.call_num + 1
        if self.call_num >= self.total_call_num:
            self.done_training = True

        return result  # y_test

    def _load_glove_emb(self):
        EMB_DIR = os.path.join(os.path.dirname(__file__), 'emb')
        embedding_data = {}

        with open(os.path.join(EMB_DIR, 'glove.6B.300d.txt'), 'r', encoding="utf-8") as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]

        print('Found %s gloveText word vectors.' %
              len(embedding_data))
        self.fasttext_embeddings_index = embedding_data

    def _load_emb(self):
        # loading pretrained embedding

        FT_DIR = '/app/embedding'
        fasttext_embeddings_index = {}
        if self.metadata['language'] == 'ZH':
            f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'), 'rb')
        elif self.metadata['language'] == 'EN':
            f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'), 'rb')
        else:
            raise ValueError('Unexpected embedding path:'
                             ' {unexpected_embedding}. '.format(
                unexpected_embedding=FT_DIR))

        for line in f.readlines():
            values = line.strip().split()
            if self.metadata['language'] == 'ZH':
                word = values[0].decode('utf8')
            else:
                word = values[0].decode('utf8')
            coefs = np.asarray(values[1:], dtype='float32')
            fasttext_embeddings_index[word] = coefs

        info('Found %s fastText word vectors.' %
             len(fasttext_embeddings_index))
        self.fasttext_embeddings_index = fasttext_embeddings_index

    def check_early_stop_conditon(self, train_num, start_offset, pre_auc, valid_auc):
        # 15
        early_stop_conditon2 = (train_num - start_offset) >= 5 \
                               and (self.valid_cost_list[self.model_id][train_num - (start_offset + 1)] - valid_auc) > 0 \
                               and (self.valid_cost_list[self.model_id][train_num - (start_offset + 2)] -
                                    self.valid_cost_list[self.model_id][train_num - (start_offset + 1)]) > 0

        early_stop_conditon1 = self.auc < pre_auc and self.auc > 0.96 and (train_num - start_offset) > 20
        if early_stop_conditon1 or early_stop_conditon2:
            print("use train_num is {}，start_offset is {} ".format(train_num, start_offset))
            if early_stop_conditon2:
                self.model.set_weights(self.model_weights_list[self.model_id][train_num - (start_offset + 2)])
                info("load weight...and done_training when early_stop_conditon2")
            if (train_num - start_offset) >= 10 and early_stop_conditon1:  # 20
                self.model.set_weights(self.model_weights_list[self.model_id][train_num - (start_offset + 1)])
                info("load weight...and done_training when early_stop_conditon1")
        return (early_stop_conditon1 or early_stop_conditon2)

    def set_next_round_sample_size(self, history):
        # Dynamic sampling ,if accuracy is lower than 0.65 ,Increase sample size
        self.sample_num_per_class = self.data_generator.sample_num_per_class
        if history.history['acc'][0] < 0.65:
            self.sample_num_per_class = min(4 * self.data_generator.sample_num_per_class,
                                            self.data_generator.max_sample_num_per_class)

        # 增加下一轮进入模型的样本数量，避免因为前期样本太少，模型效果不提升
        if self.data_generator.max_sample_num_per_class > self.sample_num_per_class:
            self.sample_num_per_class = self.data_generator.max_sample_num_per_class

        info("set_sample_num_per_class: {}".format(self.sample_num_per_class))
        self.data_generator.set_sample_num_per_class(self.sample_num_per_class)

    def update_best_val(self, val_auc, best_val_auc, start_offset, best_call_num):
        if val_auc < best_val_auc:
            # 如果小于最好结果，采用最好结果
            self.is_best = False
            info("check model_weights_list size:{}".format(len(self.model_weights_list[self.model_id])))
            info("use best result when call_num is {}! and best_auc is {}!".format(self.best_call_num,
                                                                                   self.best_val_auc))
            # best_call_num = self.best_call_num
            print("use index is {}".format(self.best_call_num - start_offset))
            print("current model weights size is {}".format(len(self.model_weights_list[self.model_id])))

            ########################## 允许更多的评估权重输出 ######################################
            if np.std([val_auc, best_val_auc])<3e-3:  # 如果当前评估AUC与最佳AUC的偏差在可控范围内，允许输出
                self.model_weights_update_flg[self.model_id].append(best_call_num - start_offset)

            else: # 否则，保存最佳结果
                if self.best_call_num >= start_offset:
                    self.model.set_weights(self.model_weights_list[self.model_id][self.best_call_num - start_offset])
                    self.model_weights_update_flg[self.model_id].append(self.best_call_num - start_offset)

        else:
            self.model_weights_update_flg[self.model_id].append(best_call_num - start_offset)
            self.is_best = True
            # 否则，更新最好结果
            best_call_num = self.call_num  # 当前的call_num为全局的call_num，一直递增
            self.best_call_num = self.call_num
            self.best_val[best_call_num] = val_auc
            info("update best result when call_num is {}! and best_auc is {}!".format(self.best_call_num,
                                                                                      val_auc))




    def feedback_simulation(self, history, train_num, start_offset=0):
        # Model Selection and Sample num from Feedback Dynamic Regulation of Simulator

        self.set_next_round_sample_size(history)
        # Early stop and restore weight automatic
        # 评估当前方案
        info("check size of clean_valid_x: {}".format(len(self.clean_valid_x)))
        valid_auc = self._valid_auc(self.clean_valid_x, self.data_generator.valid_y)
        info("valid_auc: {}".format(valid_auc))

        pre_auc = self.auc
        self.auc = valid_auc
        self.valid_cost_list[self.model_id].append(valid_auc)

        ##################### 先判断当前轮模型是否达到当前模型训练最优 #########################################
        if not self.first_cnn_done:
            # 如果没有结束CNN，此时CNN 训练评估中
            print("check auc {} and best_cnn_auc {}".format(self.auc, self.best_cnn_auc))
            self.update_best_val(self.auc, self.best_cnn_auc, start_offset, train_num)
            if self.is_best:
                self.best_cnn_auc = self.auc

        else:
            # 结束CNN，进入RCNN
            print("check auc {} and best_rcnn_auc {}".format(self.auc, self.best_rcnn_auc))
            self.update_best_val(self.auc, self.best_val_auc, start_offset, train_num)
            if self.is_best:
                self.best_rcnn_auc = self.auc




        ##################### 再比较当前模型最优与其他模型效果 #################################################
        self.select_cnn = self.best_cnn_auc * 0.97 > self.best_rcnn_auc
        self.best_val_auc = max(self.best_cnn_auc, self.best_rcnn_auc)
        # select which model is activated
        self.selcet_svm = self.valid_auc_svm > self.best_val_auc

        stop_condition = self.check_early_stop_conditon(train_num, start_offset, pre_auc, valid_auc)

        if not self.first_cnn_done and stop_condition:
            # fixme 设置状态
            self.first_cnn_done = True
            self.first_stage_done = True
            self.finish_first_cnn_call_num = self.call_num

        # 没有添加rcnn的时候不进入这里
        # elif self.first_cnn_done and stop_condition:
        #     self.first_stage_done = True

        model_weights = self.model.get_weights()
        self.model_weights_list[self.model_id].append(model_weights)
