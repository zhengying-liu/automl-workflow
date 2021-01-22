#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/5 10:35
# @Author:  Mecthew

import numpy as np
from sklearn.linear_model import logistic, SGDClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from CONSTANT import MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import (extract_melspectrogram_parallel)
from data_process import ohe2cat
from models.my_classifier import Classifier
from tools import log, timeit
import time

# model: lgb
import lightgbm as lgb



# Consider use LR as the first model because it can reach high point at
# first loop
class LogisticRegression(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        # clear_session()
        self.max_length = None
        self._model = None
        self._model_sag = None
        self.is_init = False
        self.class_num = None
        self.lr_call_num = 0

    def init_model(self,
                   kernel,
                   max_iter=200,
                   C=1.0,
                   class_num=0,
                   **kwargs):
        # base model libs.
        base_model_lr_liblinear = logistic.LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', multi_class='auto')
        base_model_lr_sag = logistic.LogisticRegression(C=C, max_iter=max_iter, solver='sag', multi_class='auto')
        base_model_lsvc = LinearSVC(random_state=0, tol=1e-5, max_iter=100)
        base_model_lsvc = CalibratedClassifierCV(base_model_lsvc)
        base_model_sgdc = SGDClassifier(loss="log")
        base_model_lgb = None
        # config for model12 and model3
        self.model_select_config = {
            "12": "lr_liblinear", #"lr_liblinear", # lgb
            "3": "lr_sag"  #"lr_sag"
        }

        self.lgb_params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            'num_class': class_num,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 2020,
            "num_threads": 4,
        }
        self.lgb_hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.08,
        }
        self.model_def_table = {
            "lgb": base_model_lgb,
            "lr_liblinear": base_model_lr_liblinear,
            "lr_sag": base_model_lr_sag,
            "sgdc": base_model_sgdc
        }

        # now condiiton: 1/2, lr_liblinear, 3: lr_sag
        # self._model = base_model_lr_liblinear
        # self._model_sag = base_model_lr_sag

        # try: 1/2 lsvc, 3 lsvc.
        # self._model = base_model_lsvc
        # self._model = base_model_lr_liblinear
        self._model = self.model_def_table[self.model_select_config["12"]]
        # self._model_sag = base_model_lsvc
        self._model_sag = self.model_def_table[self.model_select_config["3"]]

        # C = C, max_iter = max_iter, solver = 'liblinear', multi_class='auto', n_jobs=4)
            #C = C, max_iter = max_iter, solver = 'sag', multi_class = 'ovr', n_jobs = 4) # exp2
            # C = C, max_iter = max_iter, solver = 'sag', multi_class = 'ovr', n_jobs = 4, warm_start=True)  # exp3
            # C=C, max_iter=max_iter, solver='sag', multi_class='auto', n_jobs=4) # exp4
            # C = C, max_iter = max_iter, solver = 'sag', multi_class = 'auto', warm_start = True)  # exp5
            # C=C, max_iter=max_iter, solver='saga', multi_class='auto') # exp6
            # C = C, max_iter = max_iter, solver = 'lbfgs', multi_class = 'auto')  # exp7
            # C = C, max_iter = max_iter, solver = 'newton-cg', multi_class = 'auto')  # exp8

        self.class_num = class_num
        self.is_init = True

    @timeit
    def preprocess_data(self, x):
        # cut down
        x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in x]
        # extract mfcc
        # x_mfcc = extract_mfcc_parallel(x, n_mfcc=63)
        x_mel = extract_melspectrogram_parallel(
            x, n_mels=40, use_power_db=True)
        # x_chroma_stft = extract_chroma_stft_parallel(x, n_chroma=12)
        # x_rms = extract_rms_parallel(x)
        # x_contrast = extract_spectral_contrast_parallel(x, n_bands=6)
        # x_flatness = extract_spectral_flatness_parallel(x)
        # x_polyfeatures = extract_poly_features_parallel(x, order=1)
        # x_cent = extract_spectral_centroid_parallel(x)
        # x_bw = extract_bandwidth_parallel(x)
        # x_rolloff = extract_spectral_rolloff_parallel(x)
        # x_zcr = extract_zero_crossing_rate_parallel(x)

        x_feas = []
        for i in range(len(x_mel)):
            mel = np.mean(x_mel[i], axis=0).reshape(-1)
            mel_std = np.std(x_mel[i], axis=0).reshape(-1)
            # mel = np.mean(x_mel[i], axis=0).reshape(-1)
            # mel_std = np.std(x_mel[i], axis=0).reshape(-1)
            # chroma_stft = np.mean(x_chroma_stft[i], axis=0).reshape(-1)
            # chroma_stft_std = np.std(x_chroma_stft[i], axis=0).reshape(-1)
            # rms = np.mean(x_rms[i], axis=0).reshape(-1)
            # contrast = np.mean(x_contrast[i], axis=0).reshape(-1)
            # contrast_std = np.std(x_contrast[i], axis=0).reshape(-1)
            # flatness = np.mean(x_flatness[i], axis=0).reshape(-1)
            # polyfeatures = np.mean(x_polyfeatures[i], axis=0).reshape(-1)
            # cent = np.mean(x_cent[i], axis=0).reshape(-1)
            # cent_std = np.std(x_cent[i], axis=0).reshape(-1)
            # bw = np.mean(x_bw[i], axis=0).reshape(-1)
            # bw_std = np.std(x_bw[i], axis=0).reshape(-1)
            # rolloff = np.mean(x_rolloff[i], axis=0).reshape(-1)
            # zcr = np.mean(x_zcr[i], axis=0).reshape(-1)
            x_feas.append(np.concatenate([mel, mel_std], axis=-1))
            # x_feas.append(np.concatenate([mfcc, mel, contrast, bw, cent, mfcc_std, mel_std, contrast_std, bw_std, cent_std]))
        x_feas = np.asarray(x_feas)

        scaler = StandardScaler()
        X = scaler.fit_transform(x_feas[:, :])
        # log(   'x_feas shape: {X.shape}\n'
        #        'x_feas[0]: {X[0]}')
        return X

    def fit(self, x_train, y_train, *args, **kwargs):
        y_train_cats = ohe2cat(y_train)
        # log("note: LR model fit y_train_cats len={}, contents={}, ".format(len(y_train_cats), y_train_cats))
        # log("note: LR model fit y_train_cats len={}".format(len(y_train_cats)))
        cats_list = list(y_train_cats)
        cats_list_set = set(cats_list)
        cats_list_2 = list(cats_list_set)
        cats_list_2.sort()
        time_lr_fit_start = time.time()
        if self.lr_call_num < 2:
            if self.model_select_config["12"] == "lgb":
                lgb_train_ds = lgb.Dataset(x_train, label=y_train_cats)
                log("note: lgb fit build train_ds, shape={}".format(lgb_train_ds))
                self._model = lgb.train({**self.lgb_params, **self.lgb_hyperparams}, lgb_train_ds)
                log("note: lgb fit train done".format(lgb_train_ds))
            else:
                self._model.fit(x_train, y_train_cats)
        else:
            if self.model_select_config["12"] == "lgb":
                lgb_train_ds = lgb.Dataset(x_train, label=y_train_cats)
                self._model_sag = lgb.train({**self.lgb_params, **self.lgb_hyperparams}, lgb_train_ds)
            else:
                self._model_sag.fit(x_train, y_train_cats)

        time_lr_fit_end = time.time()
        self.lr_call_num += 1
        log("note: LR model, lr_call_num={}, y_train_num={}, y_train_cat_num={}, class_num={}, fit_cost_time={}".format( self.lr_call_num, len(y_train_cats), len(cats_list_2), self.class_num, round(time_lr_fit_end - time_lr_fit_start, 3)))


    # fixme: rebuild predict_prob_res, using classes_ and origin probs
    def rebuild_prob_res(self, cls_list, orig_prob_array):
        # re_init new prob array, which shape[1]==class_num
        # new_prob_arary = np.zeros((orig_prob_array.shape[0], max(cls_list) + 1))
        new_prob_arary = np.zeros((orig_prob_array.shape[0], self.class_num))
        # pprint.pprint(new_prob_arary)
        for i, cls in enumerate(cls_list):
            new_prob_arary[:, cls] = orig_prob_array[:, i]

        empty_cls_list = list()
        # for i in range(max(cls_list) + 1):
        for i in range(self.class_num):
            if i not in cls_list:
                empty_cls_list.append(i)

        # fill by median value.
        for sample_i in range(orig_prob_array.shape[0]):
            np_median_value = np.median(new_prob_arary[sample_i])
            for empty_cls in empty_cls_list:
                new_prob_arary[sample_i][empty_cls] = np_median_value

        return new_prob_arary

    def predict(self, x_test, batch_size=32):
        # lr_predict_res = self._model.predict(x_test)
        # log("LR predict_res = {}".format(lr_predict_res))
        # lr: preidct_proba, svc: decision_function, cv:predict,
        if self.lr_call_num < 3:
            if self.model_select_config["12"] in ["lr_liblinear", "lr_sag"]:
            # lr, sgdc
                lr_predict_proba_res = self._model.predict_proba(x_test)
                lr_classes = self._model.classes_
                new_lr_predict_proba_res = self.rebuild_prob_res(lr_classes, lr_predict_proba_res)
            elif self.model_select_config["12"] == "lgb":
                new_lr_predict_proba_res = self._model.predict(x_test)
            elif self.model_select_config["12"] == "lsvc":
                lr_predict_proba_res = self._model.decision_function(x_test)
            elif self.model_select_config["12"] == "lsvc_cv":
                lr_predict_proba_res = self._model.predict(x_test)
            else:
                print("Error: predict model error in moedel_selelct_config:{}".format(self.model_select_config))

        # svc:
        # svc: cv, predict
            # lr_predict_proba_res = self._model.predict(x_test)

        else:
            if self.model_select_config["3"] in ["lr_liblinear", "lr_sag"]:
            # lr, sgdc
                lr_predict_proba_res = self._model_sag.predict_proba(x_test)
                lr_classes = self._model_sag.classes_
                new_lr_predict_proba_res = self.rebuild_prob_res(lr_classes, lr_predict_proba_res)
                log("LR proba_res type={}, shape={}".format(type(lr_predict_proba_res), lr_predict_proba_res.shape))
            elif self.model_select_config["3"] == "lgb":
                new_lr_predict_proba_res = self._model_sag.predict(x_test)
            elif self.model_select_config["3"] == "lsvc":
                new_lr_predict_proba_res = self._model_sag.decision_function(x_test)
            elif self.model_select_config["3"] == "lsvc_cv":
                new_lr_predict_proba_res = self._model_sag.predict(x_test)
            else:
                print("Error: predict model error in moedel_selelct_config:{}".format(self.model_select_config))

            # lr:
            # lr_predict_proba_res = self._model_sag.predict_proba(x_test)
            # svc:
            # lr_predict_proba_res = self._model_sag.predict(x_test)
            # lr_classes = self._model_sag.classes_
            # new_lr_predict_proba_res = lr_predict_proba_res
            # new_lr_predict_proba_res = self.rebuild_prob_res(lr_classes, lr_predict_proba_res)


        # log("LR classes={}, type={}".format(lr_classes, type(lr_classes)))

        # new_lr_predict_proba_res = self.rebuild_prob_res(lr_classes, lr_predict_proba_res)

        log("LR new proba_res lr_call_num={}, type={}, shape={}".format(self.lr_call_num, type(new_lr_predict_proba_res), new_lr_predict_proba_res.shape))

        # return lr_predict_proba_res
        return new_lr_predict_proba_res
