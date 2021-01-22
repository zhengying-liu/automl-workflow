#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import random
from functools import partial

import numpy as np
from sklearn.preprocessing import StandardScaler

from CONSTANT import *
from CONSTANT import MAX_VALID_PERCLASS_SAMPLE
from data_augmentation import noise, shift, stretch, pitch, dyn_change, speed_npitch
from data_process import extract_mfcc_parallel, get_max_length, pad_seq, extract_melspectrogram_parallel, FEET_MODE
from data_process import lr_preprocess_update, pre_func_lr_prepro
from tools import log, timeit



class DataManager:
    def __init__(self, metadata, train_dataset):
        self._metadata = metadata

        # self._all_x, self._all_y = train_dataset
        # fixme: tuple to dict, for modifying and updating.
        self._all_x, self._all_y = train_dataset["x"], train_dataset["y"]
        self._all_x = np.array(self._all_x)

        self._train_x, self._train_y = None, None
        self._val_x, self._val_y = None, None

        self._pre_train_x, self._pre_train_y = None, None
        self._pre_val_x, self._pre_val_y = None, None

        self._lr_train_x, self._lr_train_y = None, None
        self._lr_val_x, self._lr_val_y = None, None

        self._each_class_index = []
        # self._even_class_index = []
        self._max_class_num, self._min_class_num = 0, 0
        self._pre_data = []

        self._num_classes = self._metadata[CLASS_NUM]
        self.fea_max_length = None
        self.raw_max_length = None
        self._start_nn = False

        self.need_30s = False
        self.crnn_first = False

        # zyadd:
        self.prev_even_class_index_list = [[] for i in range(self._num_classes)]
        self._even_class_index = [[] for i in range(self._num_classes)]
        self.train_x_lr_features_cache = list()
        self.update_train_dataset(train_dataset)

    # zyadd: receive updated train_dataset
    def update_train_dataset(self, new_train_dataset):
        cur_x, cur_y = new_train_dataset["x"], new_train_dataset["y"]
        cur_x = np.array(cur_x) # array for easy array slice.
        if self._val_x is None:
            cur_train_index_list, cur_val_index_list = self._train_val_split_index_by_y(cur_y)
            self._val_x = cur_x[cur_val_index_list]
            self._val_y = cur_y[cur_val_index_list]
            self._train_x = cur_x[cur_train_index_list]
            self._train_y = cur_y[cur_train_index_list]
            self.new_take_train_num = len(self._train_x)
        else:
            self.new_take_train_num = len(cur_x) - len(self._train_x)
            self._train_x = cur_x
            self._train_y = cur_y

        # update data-train-lr-features space.
        self.train_x_lr_features_cache.extend([None for i in range(self.new_take_train_num)])
        # update new each_class_index
        self._each_class_index = self._init_each_class_index_by_y(self._train_y)
        # get cur even_class_index
        cur_even_class_index_list = self._init_even_class_index_by_each(self._each_class_index)
        # update self._even_class_index_list, for sampling, 要计算更新的值，得update_even_index_list，用来进行更新的采样 in cur not in old
        for class_i in range(self._num_classes):
            update_even_class_index_list_class_i = [sample_id for sample_id in cur_even_class_index_list[class_i] if
                                                     sample_id not in self.prev_even_class_index_list[class_i]]
            self._even_class_index[class_i].extend(update_even_class_index_list_class_i)

        self.prev_even_class_index_list = cur_even_class_index_list



    # zyadd:
    def _train_val_split_index_by_y(self, cur_y_array, ratio=0.8):
        all_index, train_index, val_index = [], [], []
        for i in range(self._num_classes):
            all_index.append(
                list(np.where(cur_y_array[:, i] == 1)[0]))

        log_info = list()
        for i in range(self._num_classes):
            val_ratio_num = int(len(all_index[i]) * (1 - ratio))
            # sample_num = max(MIN_VALID_PER_CLASS, int(len(all_index[i]) * (1 - ratio)))
            sample_num = max(MIN_VALID_PER_CLASS, val_ratio_num)
            # log("sample_num={}, min_valid_per_class={}, sample_maxone={}".format(sample_num, MIN_VALID_PER_CLASS, val_ratio_num))
            log_info.append([sample_num, MIN_VALID_PER_CLASS, val_ratio_num])
            if sample_num <= len(all_index[i]):
                tmp = random.sample(all_index[i], sample_num)
            else:
                tmp = all_index[i]
            if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
                tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
            val_index += tmp
            differ_set = set(all_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            train_index += list(differ_set)

        log("Split Val all sample_num, min_valid_per_class, sample_maxone {}".format(log_info))
        log("note: cur_y_array len={}, train_index_len={}, val_index_len={}".format(len(cur_y_array), len(train_index), len(val_index)))
        return train_index, val_index

    # zyadd:
    def _init_each_class_index_by_y(self, cur_train_y):
        each_class_count = np.sum(np.array(cur_train_y), axis=0)
        self._max_class_num, self._min_class_num = int(
            np.max(each_class_count)), int(
            np.min(each_class_count))
        log('Raw train data: train_num(without val) {}; '.format(len(cur_train_y)) +
            'class_num {} ; max_class_num {}; min_class_num {}; '.format(self._num_classes, self._max_class_num, self._min_class_num))

        each_class_index = []
        for i in range(self._num_classes):
            each_class_index.append(
                list(np.where(cur_train_y[:, i] == 1)[0]))

        return each_class_index

    # zyadd:
    def _init_even_class_index_by_each(self, each_class_index_list):
        even_class_index = []
        sample_per_class = max(int(len(self._train_y) / self._num_classes), 1)
        log_info = list()
        for i in range(self._num_classes):
            class_cnt = len(each_class_index_list[i])
            tmp = []
            log_info.append([i, class_cnt])
            # log("init even class index, class_id={}, class_cnt={}".format(i, class_cnt))
            # fixme: bug, class_cnt 可能为0
            if class_cnt == 0:
                log("Init even class index, class_id={} cn=0".format(i))
                pass
            elif class_cnt < sample_per_class:
                tmp = each_class_index_list[i] * \
                    int(sample_per_class / class_cnt)
                tmp += random.sample(
                    each_class_index_list[i],
                    sample_per_class - len(tmp))
            else:
                tmp += random.sample(
                    each_class_index_list[i], sample_per_class)
            random.shuffle(tmp)
            even_class_index.append(tmp)

        log("Init even class index, class_id, class_cnt={}".format(log_info))
        return even_class_index



    def _init_each_class_index(self):
        each_class_count = np.sum(np.array(self._train_y), axis=0)
        self._max_class_num, self._min_class_num = int(
            np.max(each_class_count)), int(
            np.min(each_class_count))
        log('Raw train data: train_num(without val) {}; '.format(len(self._train_y)) +
            'class_num {} ; max_class_num {}; min_class_num {}; '.format(self._num_classes, self._max_class_num, self._min_class_num))

        self._each_class_index = []
        for i in range(self._num_classes):
            self._each_class_index.append(
                list(np.where(self._train_y[:, i] == 1)[0]))

    def _init_even_class_index(self):
        # self._even_class_index = []
        self._even_class_index = [[] for i in range(self._num_classes)]

        sample_per_class = max(int(len(self._train_y) / self._num_classes), 1)
        log_info = list()
        for i in range(self._num_classes):
            class_cnt = len(self._each_class_index[i])
            tmp = []
            log_info.append([i, class_cnt])
            # log("init even class index, class_id={}, class_cnt={}".format(i, class_cnt))
            # fixme: bug, class_cnt 可能为0
            if class_cnt == 0:
                pass
            elif class_cnt < sample_per_class:
                tmp = self._each_class_index[i] * \
                    int(sample_per_class / class_cnt)
                tmp += random.sample(
                    self._each_class_index[i],
                    sample_per_class - len(tmp))
            else:
                tmp += random.sample(
                    self._each_class_index[i], sample_per_class)
            random.shuffle(tmp)
            self._even_class_index[i].extend(tmp)

        log("init even class index, class_id, class_cnt={}".format(log_info))

    def _train_test_split_index(self, ratio=0.8):
        all_index, train_index, val_index = [], [], []
        for i in range(self._num_classes):
            all_index.append(
                list(np.where(self._all_y[:, i] == 1)[0]))
        for i in range(self._num_classes):
            sample_num = max(MIN_VALID_PER_CLASS, int(len(all_index[i]) * (1 - ratio)))
            log("sample_num={}, min_valid_per_class={}, sample_maxone={}".format(sample_num, MIN_VALID_PER_CLASS, int(len(all_index[i]) * (1 - ratio))))
            if sample_num <= len(all_index[i]):
                tmp = random.sample(all_index[i], sample_num)
            else:
                tmp = all_index[i]
            if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
                tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
            val_index += tmp
            differ_set = set(all_index[i]).difference(set(tmp))
            # avoid some classes only have one sample
            if len(differ_set) == 0:
                differ_set = set(tmp)
            train_index += list(differ_set)
        return train_index, val_index

    @timeit
    def _train_test_split(self, ratio=0.8):
        x = self._all_x
        y = self._all_y
        # ratio = max(1.0 - MAX_VALID_SET_SIZE / y.shape[0], ratio)
        train_index, valid_index = self._train_test_split_index(ratio)
        # self._metadata[TRAIN_NUM] = len(train_index)

        return x[train_index], x[valid_index], y[train_index], y[valid_index]

    def _get_samples_from_even_class(self, sample_num):
        per_class_num = max(int(sample_num / self._num_classes), 1)

        sample_indexs = []
        for i in range(self._num_classes):
            selected = self._even_class_index[i][:per_class_num]
            rest = self._even_class_index[i][per_class_num:]
            self._even_class_index[i] = rest
            sample_indexs += selected

        random.shuffle(sample_indexs)

        return sample_indexs

    def _get_samples_from_each_class(self, sample_num):
        per_class_num = max(int(sample_num / self._num_classes), 1)

        sample_indexs = []
        for i in range(self._num_classes):
            class_cnt = len(self._each_class_index[i])
            tmp = []
            if class_cnt < per_class_num:
                tmp = self._each_class_index[i] * \
                    int(per_class_num / class_cnt)
                tmp += random.sample(
                    self._each_class_index[i],
                    per_class_num - len(tmp))
            else:
                tmp += random.sample(self._each_class_index[i], per_class_num)
            random.shuffle(tmp)
            sample_indexs.extend(tmp)
        random.shuffle(sample_indexs)
        return sample_indexs

    def _get_preprocess_train(self, sample_index, pre_func):
        need_pre = set([i for i in sample_index if self._pre_data[i] is None])

        raw_data = [self._train_x[i] for i in need_pre]
        pre_data = pre_func(raw_data)
        log('Total {}, update {}'.format(len(sample_index), len(need_pre)))
        # update
        cnt = 0
        for i in need_pre:
            self._pre_data[i] = pre_data[cnt]
            cnt += 1

        train_x = [self._pre_data[i] for i in sample_index]
        train_y = [self._train_y[i] for i in sample_index]

        return train_x, train_y

    def _get_preprocess_train_for_lr(self, sample_index, pre_func):
        need_pre = set([i for i in sample_index if self.train_x_lr_features_cache[i] is None])

        raw_data = [self._train_x[i] for i in need_pre]
        pre_data = pre_func(raw_data)
        log('Total {}, update {}'.format(len(sample_index), len(need_pre)))
        # update
        cnt = 0
        for i in need_pre:
            self.train_x_lr_features_cache[i] = pre_data[cnt]
            cnt += 1

        train_x = [self.train_x_lr_features_cache[i] for i in sample_index]
        train_y = [self._train_y[i] for i in sample_index]

        return train_x, train_y


    # zyadd:
    def get_train_featured_data_for_lr(self, train_loop_num):
        # 1. 确定lr_sample_num
        if train_loop_num == MODEL_FIRST_MAX_RUN_LOOP:
            lr_sample_num = min(
                sum([len(self._even_class_index[i]) for i in range(self._num_classes)]), 4000)
        else:
            lr_sample_num = 500

        # 2. 确定 lr_sample_id_list
        train_samples = self._get_samples_from_even_class(
            sample_num=lr_sample_num)

        # 3. 确定 lr_sample_featured_list
        if len(train_samples) > 0:
            if FEET_MODE == "LIBROSA":
                pre_func = partial(self.lr_preprocess)
                train_x, train_y = self._get_preprocess_train_for_lr(train_samples, pre_func)
            else:
                train_x, train_y = self._get_preprocess_train_for_lr(train_samples, pre_func_lr_prepro)

            if self._lr_train_x is None:
                self._lr_train_x = np.asarray(train_x)
                self._lr_train_y = np.asarray(train_y)
            else:
                self._lr_train_x = np.concatenate(
                    (self._lr_train_x, train_x), axis=0)
                self._lr_train_y = np.concatenate(
                    (self._lr_train_y, train_y), axis=0).astype(
                    np.int)
        if self._lr_val_x is None:
            if FEET_MODE == "LIBROSA":
                self._lr_val_x = self.lr_preprocess(self._val_x)
            else:
                self._lr_val_x = lr_preprocess_update(self._val_x)

            self._lr_val_y = self._val_y
        return np.asarray(self._lr_train_x), np.asarray(
            self._lr_train_y), np.asarray(self._lr_val_x), np.asarray(self._lr_val_y)


    @timeit
    def get_train_data(self, train_loop_num, model_num,
                       round_num, use_new_train=False, use_mfcc=False):
        # split the valid dataset
        # if self._val_x is None:
        #     self._train_x, self._val_x, self._train_y, self._val_y = self._train_test_split(
        #         ratio=0.8)
        #     self._init_each_class_index()

        if use_new_train:
            if model_num == 0 or model_num == 1:
                self._init_even_class_index()
            if model_num > 0:
                self._lr_train_x = self._lr_train_y = self._lr_val_x = self._lr_val_y = None
            self._pre_train_x = self._pre_train_y = self._pre_val_x = self._pre_val_y = None
            self.raw_max_length = None
            self.fea_max_length = None
            self._pre_data = [None] * len(self._train_y)

        # for lr
        if model_num == 0:
            return self.get_train_featured_data_for_lr(train_loop_num)

        # for nn
        if round_num == 0 and model_num == 1:
            if not self._start_nn:
                self._start_nn = True
                sample_num = max(int(len(self._train_y) * 0.1), 200)
            elif train_loop_num < 10:
                sample_num = int(self._metadata[TRAIN_NUM] * 0.15)
            elif train_loop_num == 10:
                sample_num = sum([len(self._even_class_index[i])
                                  for i in range(self._num_classes)])
            else:
                sample_num = len(self._train_y)
        else:
            sample_num = len(self._train_y)

        # incremental
        if round_num == 0 and model_num == 1 and train_loop_num <= 10:
            train_samples = self._get_samples_from_even_class(sample_num)
            pre_func = partial(
                self.nn_preprocess,
                n_mfcc=96,
                max_duration=FIRST_ROUND_DURATION,
                is_mfcc=use_mfcc)
            if len(train_samples) > 0:
                train_x, train_y = self._get_preprocess_train(
                    train_samples, pre_func)
                if self._pre_train_x is None:
                    self._pre_train_x = np.array(train_x)
                    self._pre_train_y = np.array(train_y)
                else:
                    self._pre_train_x = np.concatenate(
                        (self._pre_train_x, np.array(train_x)), axis=0)
                    self._pre_train_y = np.concatenate(
                        (self._pre_train_y, np.array(train_y)), axis=0).astype(
                        np.int)
        else:
            train_samples = self._get_samples_from_each_class(sample_num)
            if round_num < 2:
                pre_func = partial(
                    self.nn_preprocess,
                    n_mfcc=96,
                    max_duration=FIRST_ROUND_DURATION,
                    is_mfcc=use_mfcc)
            else:
                pre_func = partial(
                    self.nn_preprocess,
                    n_mfcc=128,
                    max_duration=SECOND_ROUND_DURATION,
                    is_mfcc=use_mfcc)
            if len(train_samples) > 0:
                self._pre_train_x, self._pre_train_y = self._get_preprocess_train(
                    train_samples, pre_func)

        if self._pre_val_x is None:
            if round_num < 2:
                self._pre_val_x = np.array(
                    self.nn_preprocess(
                        self._val_x,
                        n_mfcc=96,
                        max_duration=FIRST_ROUND_DURATION,
                        is_mfcc=use_mfcc))
            else:
                self._pre_val_x = np.array(
                    self.nn_preprocess(
                        self._val_x,
                        n_mfcc=128,
                        max_duration=SECOND_ROUND_DURATION,
                        is_mfcc=use_mfcc))
            self._pre_val_y = np.array(self._val_y).astype(np.int)

        # data augmention
        # if loop_num > -1:
        #     cnt = len(train_x)
        #     train_x, train_y = self.get_augmention_data(train_x, train_y, ratio=0.3)
        #     log(   'Data augmention, from {cnt} to {len(train_x)} samples')

        return np.asarray(self._pre_train_x), np.asarray(self._pre_train_y),\
            np.asarray(self._pre_val_x), np.asarray(self._pre_val_y)

    def nn_preprocess(self, x, n_mfcc=96, max_duration=5, is_mfcc=True):
        if self.raw_max_length is None:
            self.raw_max_length = get_max_length(x)
            if self.raw_max_length > (MIDDLE_DURATION * AUDIO_SAMPLE_RATE):
                self.need_30s = True
                if len(self._train_y) < 1000 and self._num_classes < 30:
                    self.crnn_first = True
            self.raw_max_length = min(
                max_duration * AUDIO_SAMPLE_RATE,
                self.raw_max_length)
            self.raw_max_length = max(
                MAX_AUDIO_DURATION *
                AUDIO_SAMPLE_RATE,
                self.raw_max_length)
        x = [sample[0:self.raw_max_length] for sample in x]

        if is_mfcc:
            # extract mfcc
            x = extract_mfcc_parallel(x, n_mfcc=n_mfcc)
        else:
            x = extract_melspectrogram_parallel(
                x, n_mels=128, use_power_db=True)
        if self.fea_max_length is None:
            self.fea_max_length = get_max_length(x)
            self.fea_max_length = min(MAX_FRAME_NUM, self.fea_max_length)
        x = pad_seq(x, pad_len=self.fea_max_length)

        return x

    def lr_preprocess(self, x):
        x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in x]
        x_mel = extract_melspectrogram_parallel(
            x, n_mels=30, use_power_db=True)
        # x_contrast = extract_bandwidth_parallel(x)

        x_feas = []
        for i in range(len(x_mel)):
            mel = np.mean(x_mel[i], axis=0).reshape(-1)
            mel_std = np.std(x_mel[i], axis=0).reshape(-1)
            # contrast = np.mean(x_contrast[i], axis=0).reshape(-1)
            # contrast_std = np.std(x_contrast[i], axis=0).reshape(-1)
            # contrast, contrast_std
            x_feas.append(np.concatenate([mel, mel_std], axis=-1))
        x_feas = np.asarray(x_feas)

        scaler = StandardScaler()
        X = scaler.fit_transform(x_feas[:, :])
        return X

    @timeit
    def get_augmention_data(self, x, y, ratio=0.2):
        x_len = len(x)
        indices = range(x_len)
        sample_num = int(x_len * ratio)
        augmention_data = []

        # noise
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = np.array([noise(d) for d in x[idxs]])
        augmention_data.append((augmentions_x, y[idxs]))

        # shift
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = [shift(d) for d in x[idxs]]
        augmention_data.append((augmentions_x, y[idxs]))

        # stretch
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = [stretch(d) for d in x[idxs]]
        augmention_data.append((augmentions_x, y[idxs]))

        # pitch
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = [pitch(d) for d in x[idxs]]
        augmention_data.append((augmentions_x, y[idxs]))

        # dyn_change
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = [dyn_change(d) for d in x[idxs]]
        augmention_data.append((augmentions_x, y[idxs]))

        # speed_npitch
        idxs = np.random.choice(indices, sample_num, replace=False)
        augmentions_x = [speed_npitch(d) for d in x[idxs]]
        augmention_data.append((augmentions_x, y[idxs]))

        for x_a, y_a in augmention_data:
            # log(   'x {x.shape} y {y.shape} x_a {np.array(x_a).shape} y_a {y_a.shape}')
            x = np.concatenate((x, np.array(x_a)), axis=0)
            y = np.concatenate((y, y_a), axis=0).astype(np.int)

        return x, y
