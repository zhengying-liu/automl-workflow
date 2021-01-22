# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import time
import jieba
import numpy as np
import sys, getopt
import math
import gc

import keras
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, GRU, Activation
from keras.layers import Dropout, Embedding, Dot, Concatenate, PReLU
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler  # , EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from functools import reduce
from keras.layers import CuDNNGRU

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from Auto_NLP.DeepBlueAI.tf_model import *
from Auto_NLP.DeepBlueAI import ac

# from Auto_NLP.DeepBlueAI.get_embedding import GET_EMBEDDING

import warnings

warnings.filterwarnings('ignore')

try:
    import gzip
except:
    os.system('pip3 install gzip')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # True  # to log device placement (on which device the operation ran)
config.gpu_options.per_process_gpu_memory_fraction = 0.5

MAX_SEQ_LENGTH = 50
MAX_VOCAB_SIZE = 200000  # Limit on the number of features. We use the top 20K features

from sklearn.metrics import roc_auc_score


def auc_metric(solution, prediction, task='binary.classification'):
    '''roc_auc_score() in sklearn is fast than code provided by sponsor
    '''
    if solution.sum(axis=0).min() == 0:
        return np.nan
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc * 2 - 1)


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.
    Args:
        num_classes: Number of classes.

    Returns:
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def CNN_Model(seq_len, num_classes, num_features, embedding_matrix=None):
    in_text = Input(shape=(seq_len,))
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    trainable = True
    if embedding_matrix is None:
        x = Embedding(num_features, 64, trainable=trainable)(in_text)
    else:
        x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

    x = Conv1D(128, kernel_size=5, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128)(x)  #
    x = PReLU()(x)
    x = Dropout(0.35)(x)  # 0
    x = BatchNormalization()(x)

    y = Dense(op_units, activation=op_activation)(x)

    md = keras.models.Model(inputs=[in_text], outputs=y)

    return md


def RNN_Model(seq_len, num_classes, num_features, embedding_matrix=None):
    in_text = Input(shape=(seq_len,))
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    trainable = True
    if embedding_matrix is None:
        x = Embedding(num_features, 64, trainable=trainable)(in_text)
    else:
        x = Embedding(num_features, 300, trainable=trainable, weights=[embedding_matrix])(in_text)

    x = CuDNNGRU(128, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128)(x)  #
    x = PReLU()(x)
    x = Dropout(0.35)(x)  # 0
    x = BatchNormalization()(x)

    y = Dense(op_units, activation=op_activation)(x)

    md = keras.models.Model(inputs=[in_text], outputs=y)

    return md


def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
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


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):

    def __init__(self, metadata, train_output_path="./", test_input_path="./", fasttext_emb=None):
        print('************************Init Model************************************')
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.epoch = 1
        self.max_epoch = 8

        self.model = None

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.te_y = None
        self.split_val = True

        self.word_index = None
        self.max_length = None
        self.seq_len = None
        self.num_features = None

        self.scos = [-1]
        self.his_scos = []

        self.patience = 3
        self.k = 0

        self.Xtest = None

        self.best_sco = -1
        self.best_res = []

        self.best_val_res = [0] * 30
        self.best_test_res = [0] * 30

        self.model_num = -1

        self.model_id = 0
        self.cand_models = ['CNN', 'GRU']
        self.lrs = [0.0035, 0.016]

        self.data_id = 0
        self.max_data = 3

        self.max_seq_len = 1600

        self.is_best = False
        self.new_data = False

        self.test_id = 0

        self.embedding_matrix = None
        self.fasttext_emb = fasttext_emb

        self.START = True  # 定义是否是第一个点
        self.FIRSTROUND = True  # 定义是否是第一轮
        self.LASTROUND = False
        self.FIRSTEPOCH = 6

        self.FIRST_CUT = 1200
        self.SENTENCE_LEN = 6000
        self.SAMPLENUM = 100000

        self.emb_size = 64
        self.out_size = 128
        self.sess_config = config
        self.TRAIN_STATUS = False  # 定义现在db模型是否在训练中
        self.Switch_to_New_Model = True
        self.use_ft_tf_model = False

    def step_decay(self, epoch):
        epoch = self.epoch
        initial_lrate = self.lrs[self.model_id]  # 0.016 #0.0035 #
        drop = 0.65
        epochs_drop = 1.0  # 2.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def is_done(self):
        if self.model_id == len(self.cand_models):
            if self.data_id == self.max_data:
                self.done_training = True
            else:
                self.model_id = 0

    def get_batch_size(self, data_size, batch_size):

        N = 7633305600 / (self.seq_len * self.emb_size * self.out_size)

        batch_size = min(batch_size, N)
        batch_size = max(batch_size, 4)

        return int(batch_size)

    def train_single_model(self, model_name):
        pass

    def get_embedding_new(self, num_features, word_index, fasttext_embeddings_index=None):
        EMBEDDING_DIM = 300
        embedding_matrix = np.zeros((num_features, EMBEDDING_DIM))
        cnt = 0
        # if version == 'RENMIN':
        #     fasttext_embeddings_index = GET_EMBEDDING_NEW.fasttext_embeddings_index_zh if self.metadata['language'] == 'ZH' else GET_EMBEDDING.fasttext_embeddings_index_en
        # elif version == 'official':
        #     fasttext_embeddings_index = GET_EMBEDDING.fasttext_embeddings_index_zh if self.metadata['language'] == 'ZH' else GET_EMBEDDING.fasttext_embeddings_index_en
        for word, i in word_index.items():
            if i >= num_features:  # if index of word > num_features
                continue
            embedding_vector = fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                # TODO: Words not found in embedding index should be the mean of all other word's embeddings.
                embedding_matrix[i] = np.zeros(300)
                cnt += 1
        print('fastText oov words: %s' % cnt)

        return embedding_matrix

    def _process_feature(self, x_train, y_train, x_eval, y_eval, data_lan, data_type, deal_seg, sentence_len,
                         sample_num):
        ###Sample
        x_train = np.array(x_train, dtype='object')
        y_train = np.array(y_train, dtype='object')
        x_eval = np.array(x_eval, dtype='object')
        y_eval = np.array(y_eval, dtype='object')
        len_train = len(x_train)
        index = [i for i in range(len_train)]
        np.random.shuffle(index)
        index = index[0:sample_num]
        x_train = x_train[index]
        y_train = y_train[index]
        print('SAMPLE_POS_NEG:', np.sum(y_train, axis=0))
        print('#################Sample From ', len_train, ' To ', len(x_train), ' ######################')
        x_train, word_index, num_features, max_length = self.deal_data(x_train, data_lan, data_type, deal_seg,
                                                                       sentence_len)

        num_classes = self.metadata['class_num']
        self.word_index = word_index
        self.max_length = max_length
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_train, ohe2cat(y_train), test_size=0.2,
        #                                                                       random_state=666)
        if self.split_val:
            print("split X_val use train_test_split!!!")
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_train, ohe2cat(y_train),
                                                                                  test_size=len(x_eval),
                                                                                  random_state=666)
        else:
            print("Do not split X_val, use outside X_val!!!")
            self.X_train = x_train
            self.y_train = ohe2cat(y_train)

            # data_type=1, 不需要build word index
            x_eval = self.deal_data(x_eval, data_lan, data_type=1, deal_seg=deal_seg, sentence_len=max_length)
            self.X_val = x_eval
            self.y_val = ohe2cat(y_eval)


        te_y = np.eye(num_classes)[self.y_val]
        self.te_y = te_y

        self.seq_len = len(x_train[0])
        self.num_features = num_features

    def preprocess_features(self, train_dataset, eval_dataset):
        self.embedding_matrix = None

        x_train, y_train = train_dataset
        x_eval, y_eval = eval_dataset

        start = time.time()
        data_type = 0
        if self.metadata['language'] == 'ZH':
            data_lan = 0
        else:
            data_lan = 1

        if self.START:
            print('When enter the system firstly, generate data for training:', self.epoch)
            deal_seg = 1
            sentence_len = self.FIRST_CUT
            self._process_feature(x_train, y_train, x_eval, y_eval, data_lan=data_lan, data_type=data_type,
                                  deal_seg=deal_seg, sentence_len=sentence_len, sample_num=self.SAMPLENUM)
        else:
            sentence_len = self.SENTENCE_LEN
            if self.data_id < 2:
                deal_seg = self.data_id + 1
                self._process_feature(x_train, y_train, x_eval, y_eval, data_lan=data_lan, data_type=data_type,
                                      deal_seg=deal_seg, sentence_len=sentence_len, sample_num=len(x_train))

            if self.data_id == 1:
                self.emb_size = 300
                self.embedding_matrix = self.get_embedding_new(num_features=self.num_features,
                                                               word_index=self.word_index,
                                                               fasttext_embeddings_index=self.fasttext_emb)
            elif self.data_id == 0 or self.data_id == 2:
                self.emb_size = 64
                self.embedding_matrix = None

            self.data_id += 1
            self.new_data = True

        print("###initail:", time.time() - start)

    def deal_data(self, data, data_lan, data_type, deal_seg, sentence_len):
        # fixme: call .so file as module.

        # import ac
        if data_type == 0:
            s1 = time.time()
            t1 = time.time()
            if deal_seg == 1:
                if data_lan == 0:
                    data = ac.clean_text_zh_seg1(data, sentence_len)
                else:
                    data = ac.clean_text_en_seg1(data, sentence_len)
            elif deal_seg == 2:
                if data_lan == 0:
                    data = ac.clean_text_zh_seg2(data, sentence_len)
                else:
                    data = ac.clean_text_en_seg2(data, sentence_len)

            t2 = time.time()

            num_sentence = len(data)
            t = np.array(data, dtype='object')
            MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, word2index, text_lens = ac.bulid_index(t, num_sentence)

            print('*****************************DataNum:', num_sentence)
            print('*****************************DataLen:', np.mean(text_lens))

            t3 = time.time()
            max_length = MAX_SEQ_LENGTH
            res = ac.texts_to_sequences_and_pad(t, num_sentence, word2index, max_length, text_lens, data_type)
            num_features = min(len(word2index) + 1, MAX_VOCAB_SIZE)
            #    print ('###num_features:', num_features)
            t4 = time.time()
            print('###clean ', t2 - t1, 's')
            print('###build', t3 - t2, 's')
            print('###seq', t4 - t3, 's')

            s2 = time.time()
            print('###init data tot use time ', s2 - s1, 's')
            return res, word2index, num_features, max_length
        else:
            s1 = time.time()
            if deal_seg == 1:
                if data_lan == 0:
                    data = ac.clean_text_zh_seg1(data, sentence_len)
                else:
                    data = ac.clean_text_en_seg1(data, sentence_len)
            elif deal_seg == 2:
                if data_lan == 0:
                    data = ac.clean_text_zh_seg2(data, sentence_len)
                else:
                    data = ac.clean_text_en_seg2(data, sentence_len)
            num_sentence = len(data)
            print("num_sentence: ", num_sentence)
            t = np.array(data, dtype='object')
            word2index = self.word_index
            max_length = self.max_length
            res = ac.texts_to_sequences_and_pad(t, num_sentence, word2index, max_length, None, data_type)
            return res

    def train_start_stage(self):
        pass

    def train_firstround_stage(self):
        pass

    def build_and_compile_model(self, train_dataset):
        is_balance = 0

        if (self.START) and (self.metadata['class_num'] == 2) and (is_balance) and self.use_ft_tf_model:
            print('using tf model...')
            config = {
                'sequence_length': self.seq_len,
                'embedding_size': self.emb_size,
                'vocabulary_size': self.num_features,
                'num_classes': self.metadata['class_num']
            }
            # sess = K.get_session()
            model = FT_tf_model(config)
            self.use_ft_tf_model = True


        else:
            print('###Use model:', self.cand_models[self.model_id])

            start3 = time.time()

            print('Embedding Size of This Model: {}.'.format(self.emb_size))

            if self.cand_models[self.model_id] == 'CNN':
                model = CNN_Model(self.seq_len, num_classes=self.metadata['class_num'],
                                  num_features=self.num_features, embedding_matrix=self.embedding_matrix)
            elif self.cand_models[self.model_id] == 'GRU':
                if self.seq_len > self.max_seq_len:
                    self.model_id += 1
                    print("!!! self.seq_len > self.max_seq_len, is_done !!!")
                    self.is_done()
                    return

                model = RNN_Model(self.seq_len, num_classes=self.metadata['class_num'],
                                  num_features=self.num_features, embedding_matrix=self.embedding_matrix)
            elif self.cand_models[self.model_id] == 'Att':
                if self.seq_len > self.max_seq_len:
                    self.model_id += 1
                    self.is_done()
                    return

                model = GRU_Attention_Model(self.seq_len, num_classes=self.metadata['class_num'],
                                            num_features=self.num_features, embedding_matrix=self.embedding_matrix)

            print("###Bluid model:", time.time() - start3)

            if self.metadata['class_num'] == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'

            opt = keras.optimizers.Adam(lr=0.001)
            model.compile(optimizer=opt, loss=loss, metrics=['acc'])
            self.use_ft_tf_model = False

        self.model_num += 1
        return model

    def update_train_parameters(self):
        lrate = LearningRateScheduler(self.step_decay)
        callbacks = [lrate]

        X_train = self.X_train
        y_train = self.y_train
        batch_size = 64

        if len(y_train) > 10000:
            if self.epoch < 4:
                batch_size = (6 - self.epoch) * 32 * int(len(self.X_train) / 10000)  # 256
            else:
                batch_size = 16 * int(len(self.X_train) / 6000)  # 256  10000

            batch_size = min(batch_size, 2048)
            batch_size = max(batch_size, 32)

        if self.epoch == 1 and len(y_train) > 5000:
            batch_size = max(batch_size, 128)

        batch_size = self.get_batch_size(len(self.y_train), batch_size)
        print('###train batch size:', batch_size)
        return callbacks, batch_size

    def trasf_res(self, result, test_num, class_num):
        # category class list to sparse class list of lists
        y_test = np.zeros([test_num, class_num])

        if self.metadata['class_num'] == 2:
            result = result.flatten()
            y_test[:, 0] = 1 - result
            y_test[:, 1] = result
            # print(y_test)
        else:
            y_test = result
        return y_test

    def evaluate_model(self, model):
        pred = None
        max_auc = np.max(self.scos)

        #if self.epoch == 1:
        if self.epoch == 0:
            val_auc = 0.001 * self.epoch
        else:
            batch_size = self.get_batch_size(len(self.y_val), 1024)
            print('###val batch size:', batch_size)
            print("check x_val {}".format(self.X_val))
            result = model.predict(self.X_val, batch_size=batch_size)

            pred = self.trasf_res(result, len(self.y_val), self.metadata['class_num'])
            val_auc = auc_metric(self.te_y, pred)

        if val_auc > max_auc:
            self.k = 0
            self.best_val_res[self.model_num] = pred
        else:
            self.k += 1

        self.scos.append(val_auc)
        print('val aucs:', self.scos)
        return max_auc, val_auc

    def control_logic(self):
        pass

    def ensemble(self):

        feat_size = len(self.his_scos) + 1
        print("check his_scors {} and feat_size {}".format(len(self.his_scos), feat_size))
        print("check best_test_res {}".format(self.best_test_res))
        best_val = [element for element in self.best_test_res if not isinstance(element, int)]
        if best_val:
            return np.mean(best_val, axis=0)
        else:
            return np.array([])


    def train_iter(self, train_dataset, eval_dataset, remaining_time_budget=None, do_clean=False):
        print('\n--- remaining_time_budget \n', remaining_time_budget)
        if remaining_time_budget <= self.metadata['time_budget'] * 0.125:
            self.done_training = True
            self.model = None
            return

        ##################### 外部控制早停策略 #################################
        if do_clean:
            model = self.model
            del model
            gc.collect()
            K.clear_session()

            # if self.FIRSTROUND:
            #     self.FIRSTROUND = False
            #     self.LASTROUND = True
            # else:
            self.model_id += 1

            self.is_done()

            # Reset number of epoch and patience of early-stopping
            self.epoch = 1
            self.k = 0

            # The score in Sample Stage do not append in history score
            if not self.LASTROUND:
                self.his_scos.append(self.scos)
            self.scos = [-1]
        #################### 外部控制早停策略 ##################################

        print("Current START is {}, Current FIRSTROUND is {}, Current epoch is {}".format(self.START, self.FIRSTROUND,
                                                                                          self.epoch))
        t1 = time.time()

        if self.START:
            data = train_dataset[0]
            len_sum = 0
            shape = min(len(data), 10000)
            for i in data[:shape]:
                len_sum += len(i)

            len_mean = len_sum // shape
            print('current len mean {} constraint {}'.format(len_mean, self.SENTENCE_LEN))
            cut = 0
            if len_mean > self.FIRST_CUT:
                print('len mean {} FIRST CUT {} need cut.'.format(len_mean, self.FIRST_CUT))
                cut = 1
            len_mean = min(self.FIRST_CUT, len_mean)

            len_mean_for_compute = max(100, len_mean)
            sample_row = int(-90.8 * len_mean_for_compute + 128960)
            print(
                'sample_row= int(-90.8*len_mean_for_compute + 128960),  len_mean_for_compute={}, sample_row={}'.format(
                    len_mean_for_compute, sample_row))

            MAX_SAMPLE_ROW = 100000
            MIN_SAMPLE_ROW = 16666

            sample_row = min(sample_row, MAX_SAMPLE_ROW)
            sample_row = max(sample_row, MIN_SAMPLE_ROW)
            print('len mean {}'.format(len_mean))

            sample = 1
            if sample_row >= len(data):
                sample = 0

            cut = 1
            sample = 1
            if cut == 0 and sample == 0:
                self.START = False
                self.FIRSTROUND = False
                self.LASTROUND = False
            else:
                self.FIRST_CUT = len_mean  # 平均文本长度
                self.SAMPLENUM = sample_row

            print('****************************************************************************************')
            print('Num of Data', len(data), 'Sample Num：', sample_row)
            print('Text Length：', len_sum // shape, 'Cut：', len_mean)
            print('Is Sample：', sample, ' Is Cut：', cut)
            print('Language：', self.metadata['language'])
            print('Class Num:', self.metadata['class_num'])
            print('Postive-Negtive Samples Portion：', np.sum(train_dataset[1], axis=0))
            print('****************************************************************************************')
            self.preprocess_features(train_dataset, eval_dataset)

        if self.done_training:
            return

        if self.START or self.FIRSTROUND or self.LASTROUND:
            print('Running in Sample Data Stage')
            self.max_epoch = self.FIRSTEPOCH
            print('TRAIN EPOCH:', self.epoch)
        else:
            self.max_epoch = 8
            print('\n--- Start Train: \n\tdata_id {} \n\tmodel_id {} \n\tdone_training {} \n\tepoch {}'.format(
                self.data_id, \
                self.model_id, self.done_training, self.epoch))
            models = ['CNN', 'GRU', '', '', '']
            methods = ['', 'char-level', 'word-level + pretrained embedding300dim', 'word-level + 64dim-embedding', '',
                       '', '']

            print('Current Model {}'.format(models[self.model_id]))
            print('Current Data Mode {}'.format(methods[self.data_id]))

        if self.START:
            self.Switch_to_New_Model = True

        elif self.FIRSTROUND:
            print('Train using sample data:', self.epoch)
            pass

        else:
            print('Start using data without cut')
            print('EPOCH:', self.epoch)
            print('START:', self.START, ' self.FIRSTROUND:', self.FIRSTROUND)
            if self.epoch == 1 and self.model_id == 0:
                # self.Switch_to_New_Model = True
                self.preprocess_features(train_dataset, eval_dataset)

        # Model creat and compile
        if self.epoch == 1:
            model = self.build_and_compile_model(train_dataset)
            if model is None:
                return
            # self.Switch_to_New_Model = False
        else:
            model = self.model

        callbacks, batch_size = self.update_train_parameters()
        start7 = time.time()
        history = model.fit([self.X_train], self.y_train,
                            epochs=1,
                            callbacks=callbacks,
                            verbose=1,
                            batch_size=batch_size,
                            shuffle=True)
        print("###training time:", time.time() - start7)

        max_auc, val_auc = self.evaluate_model(model)
        self.epoch += 1
        # If trigger early-stopping or reach the limit of the number of epochs
        # 如果早停条件满足 或者 epoch到达设置的数量。
        if self.k >= self.patience or self.epoch >= self.max_epoch:
            # Init model
            del model
            gc.collect()
            K.clear_session()
            model = None

            if self.FIRSTROUND:
                self.FIRSTROUND = False
                self.LASTROUND = True
            else:
                self.model_id += 1

            self.is_done()

            # Reset number of epoch and patience of early-stopping
            self.epoch = 1
            self.k = 0

            # The score in Sample Stage do not append in history score
            if not self.LASTROUND:
                self.his_scos.append(self.scos)
            else:   #0311:增加第一次输出
                self.his_scos.append(self.scos)
            self.scos = [-1]

        if self.model_num == 0:
            if self.his_scos:
                max_auc = np.max(self.his_scos[0])

            else:
                max_auc = np.max(self.scos)
            self.best_sco = max_auc
        else:
            if val_auc > self.best_sco:
                self.is_best = True
                self.best_sco = val_auc

        self.model = model

        if self.LASTROUND:
            # 0311:增加第一次输出
            # 不需要清空当前的best sco
            pass
            # self.best_sco = 0.02
        print('AFTER TRAIN best_sco:', self.best_sco, ' his_scos :', self.his_scos)

    def test(self, x_test, remaining_time_budget=None):
        print('****************************************************************************************')
        print('Length of test：', len(x_test))
        print('****************************************************************************************')

        if self.START or self.FIRSTROUND or self.LASTROUND:
            print('Running in Sample Stage ...')
            print('TEST EPOCH:', self.epoch)
            print(
                '\n--- Start Test:  \n\ttest_id: {} \n\tdata_id: {} \n\tbest_val_sco {} \n\tmodel_num {} \n\tis_best {}'.format(
                    self.test_id, self.data_id, \
                    self.best_sco, self.model_num, self.is_best))

        else:

            print(
                '\n--- Start Test:  \n\ttest_id: {} \n\tdata_id: {} \n\tbest_val_sco {} \n\tmodel_num {} \n\tis_best {}'.format(
                    self.test_id, self.data_id, \
                    self.best_sco, self.model_num, self.is_best))

        self.test_id += 1

        # If the current model has no better score or the model is finished
        if self.k != 0 or self.model == None or self.model == -1:
            print("check k {} or self.model {}".format(self.k, self.model))
            # If the model is finished and there is a history
            if self.k == 0 and self.model == None and len(self.his_scos) > 1:
                self.model = -1
                print('###update best result...')
                self.best_res = self.ensemble()

            self.is_best = False
            self.LASTROUND = False
            # print("check self.best_res {}".format(len(self.best_res)))

            return self.best_res

        model = self.model
        word_index = self.word_index
        max_length = self.max_length

        train_num, test_num = self.metadata['train_num'], self.metadata['test_num']
        class_num = self.metadata['class_num']

        start = time.time()
        print('START : ', self.START, ' FIRSTROUND:', self.FIRSTROUND, ' LASTROUND:', self.LASTROUND)
        if self.START:
            print('###data:', self.data_id)
            print('###Start Init TestData ')
            start = time.time()

            data_type = 1
            if self.metadata['language'] == 'ZH':
                data_lan = 0
            else:
                data_lan = 1
            deal_seg = 1
            sentence_len = self.FIRST_CUT
            x_test = np.array(x_test, dtype='object')
            print("deal x_test when sentence_len is {}".format(sentence_len))
            x_test = self.deal_data(x_test, data_lan, data_type, deal_seg, sentence_len)

            print("###initail:", time.time() - start)

            self.Xtest = x_test
            self.START = False

        elif self.FIRSTROUND:
            pass
        elif self.LASTROUND:
            self.LASTROUND = False
        else:
            if self.new_data:
                self.new_data = False
                if self.data_id == 3:
                    if self.Xtest is None:
                        print('###data:', self.data_id)
                        print('###Start Init TestData ')
                        start = time.time()

                        data_type = 1
                        if self.metadata['language'] == 'ZH':
                            data_lan = 0
                        else:
                            data_lan = 1
                        deal_seg = 2
                        sentence_len = self.SENTENCE_LEN
                        x_test = np.array(x_test, dtype='object')
                        print("deal x_test when sentence_len is {}".format(sentence_len))
                        x_test = self.deal_data(x_test, data_lan, data_type, deal_seg, sentence_len)

                        print("###initail:", time.time() - start)

                        self.Xtest = x_test
                    else:
                        x_test = self.Xtest

                else:
                    print('###data:', self.data_id)
                    print('###Start Init TestData ')
                    start = time.time()

                    data_type = 1
                    if self.metadata['language'] == 'ZH':
                        data_lan = 0
                    else:
                        data_lan = 1
                    deal_seg = self.data_id
                    sentence_len = self.SENTENCE_LEN
                    x_test = np.array(x_test, dtype='object')
                    print("deal x_test when sentence_len is {}".format(sentence_len))
                    x_test = self.deal_data(x_test, data_lan, data_type, deal_seg, sentence_len)

                    print("###initail:", time.time() - start)

                    self.Xtest = x_test

        x_test = self.Xtest
        print('###test data time:', time.time() - start)

        batch_size = 32 * int(len(x_test) / 2000)
        batch_size = min(batch_size, 2048)
        batch_size = max(batch_size, 32)

        batch_size = self.get_batch_size(len(x_test), batch_size)
        print('###test batch size:', batch_size)
        print("x_test shape is {}".format(x_test.shape[1]))
        result = model.predict(x_test, batch_size=batch_size)  #

        # print(result[:2])

        y_test = self.trasf_res(result, test_num, class_num)

        self.best_test_res[self.model_num] = y_test

        print("check best_test_res {}".format(len(self.best_test_res)))

        if self.model_num == 0:
            self.best_res = y_test
        else:
            if self.is_best:
                self.is_best = False
                self.best_res = y_test
                # self.best_res = self.ensemble()
            else:
                y_test = self.best_res

        return y_test
