#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
from functools import partial
from multiprocessing.pool import ThreadPool, Pool
import json

import librosa
import numpy as np
from tensorflow.python.keras.preprocessing import sequence

from CONSTANT import NUM_MFCC, FFT_DURATION, HOP_DURATION
import tools
print("tools file={}".format(tools.__file__))
from tools import timeit, log, info
import time
import tensorflow
info("tensorflow version = {}".format(tensorflow.__version__))


def ohe2cat(label):
    return np.argmax(label, axis=1)


@timeit
def get_max_length(x, ratio=0.95):
    """
    Get the max length cover 95% data.
    """
    lens = [len(_) for _ in x]
    max_len = max(lens)
    min_len = min(lens)
    lens.sort()
    # TODO need to drop the too short data?
    specified_len = lens[int(len(lens) * ratio)]
    log("Max length: {}; Min length {}; 95 length {}".format(max_len, min_len, specified_len))
    return specified_len


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)

# mfcc
@timeit
def extract_mfcc(data, sr=16000, n_mfcc=NUM_MFCC):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=n_mfcc)
        r = r.transpose()
        results.append(r)

    return results


def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    info("note: feee=librosa, extract r shape={}".format(r.shape))
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)

    info("note: feee=librosa, after power_to_db r shape={}".format(r.shape))
    r = r.transpose()
    info("note: feee=librosa, after transpose r shape={}".format(r.shape))
    return r, idx

# pool = ThreadPool(os.cpu_count())
pool = Pool(os.cpu_count())


@timeit
def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.mfcc, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    results = extract_parallel(data, extract)

    return results


# zero crossings

@timeit
def extract_zero_crossing_rate_parallel(data):
    extract = partial(extract_for_one_sample, extract=librosa.feature.zero_crossing_rate, pad=False)
    results = extract_parallel(data, extract)

    return results


# spectral centroid

@timeit
def extract_spectral_centroid_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_centroid, sr=sr,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_melspectrogram_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.melspectrogram,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    results = extract_parallel(data, extract)

    return results


# spectral rolloff
@timeit
def extract_spectral_rolloff_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_rolloff,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)  # data+0.01?
    # sklearn.preprocessing.scale()
    return results


# chroma stft
@timeit
def extract_chroma_stft_parallel(data, sr=16000, n_fft=None, hop_length=None, n_chroma=12):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_stft, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_bandwidth_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_bandwidth,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_contrast_parallel(data, sr=16000, n_fft=None, hop_length=None, n_bands=6):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_contrast,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_flatness_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_flatness,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_tonnetz_parallel(data, sr=16000):
    extract = partial(extract_for_one_sample, extract=librosa.feature.tonnetz, sr=sr)
    results = extract_parallel(data, extract)
    return results


@timeit
def extract_chroma_cens_parallel(data, sr=16000, hop_length=None, n_chroma=12):
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_cens, sr=sr,
                      hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_rms_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.rms,
                      frame_length=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_poly_features_parallel(data, sr=16000, n_fft=None, hop_length=None, order=1):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.poly_features,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    results = extract_parallel(data, extract)

    return results


from sklearn.preprocessing import StandardScaler
# speech features extraction using kapre.
SAMPLING_RATE = 16000
# N_MELS = 64
N_MELS = 30
# HOP_LENGTH = 512
# N_FFT = 1024  # 0.064 sec
HOP_LENGTH = int(SAMPLING_RATE * 0.04)
# N_FFT = int(SAMPLING_RATE * 0.1)# 0.064 sec
N_FFT = 1024 # 0.064 sec
FMIN = 20
FMAX = SAMPLING_RATE // 2

# from pasa

MAX_AUDIO_DURATION = 5  # limited length of audio, like 20s

AUDIO_SAMPLE_RATE = 16000

FEET_MODE = "KAPRE"  # KAPRE/LIBROSA
# FEET_MODE = "LIBROSA"  # KAPRE/LIBROSA

import keras
os.system('pip3 install kapre==0.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple')
info("install kapre, tf version={}".format(tensorflow.__version__))

from kapre.time_frequency import Melspectrogram

def make_extractor(input_shape, sr=SAMPLING_RATE):
    model = keras.models.Sequential()

    model.add(
        Melspectrogram(
            fmax=FMAX,
            fmin=FMIN,
            n_dft=N_FFT,
            n_hop=HOP_LENGTH,
            n_mels=N_MELS,
            name='melgram',
            image_data_format='channels_last',
            input_shape=input_shape,
            return_decibel_melgram=True,
            power_melgram=2.0,
            sr=sr,
            trainable_kernel=False
        )
    )

    return model

def get_fixed_array(X_list, len_sample=5, sr=SAMPLING_RATE):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample * sr:
            n_repeat = np.ceil(
                sr * len_sample / X_list[i].shape[0]
            ).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][:len_sample * sr]

    X = np.asarray(X_list)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)

    return X


CROP_SEC = 5
kapre_extractor = make_extractor((1, CROP_SEC * SAMPLING_RATE))
print("kapre_extractor done.")

def extract_features_old(X_list, model, len_sample=5, sr=SAMPLING_RATE):
    X = get_fixed_array(X_list, len_sample=len_sample, sr=sr)
    info("note: exft get fix done.")
    X = model.predict(X)
    info("note: exft model predict done.")
    X = X.transpose(0, 2, 1, 3)
    # squeeze.
    X = np.squeeze(X)
    # info("note: exft model transpose and squeeze done.")
    return X


def extract_features(X_list, len_sample=5, sr=SAMPLING_RATE, use_power_db=False):
    X = get_fixed_array(X_list, len_sample=len_sample, sr=sr)
    info("note: exft get fix done.")
    # X = model.predict(X)
    X = kapre_extractor.predict(X)
    info("note: exft model kapre_extractor predict done, kapre predict shape={}".format(X.shape))

    # basic: (147, 30, 125, 1) to (147, 125, 30, 1) to (147, 125, 30)
    # X = X.transpose(0, 2, 1, 3)
    # X = np.squeeze(X)

    # basic + powertodb. squeeze->powertodb->transpose
    # squeeze.
    X = np.squeeze(X)
    info("note: exft model transpose and squeeze done, shape={}".format(X.shape))
    if use_power_db:
        X = np.asarray([librosa.power_to_db(r) for r in X])
        info("note: exft model kapre_extractor power_to_db done.")

    X = X.transpose(0, 2, 1)
    # info("note: X transpose shape={}".format(X.shape))
    return X




def lr_preprocess_update( x):
    x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in x]
    info("note: lr_preprocess update, x type={}, len={}, x[0] shape={}, ele_type={}, value={}, x[-1] shape={}".format(type(x), len(x), x[0].shape, type(x[0][0]), x[0][0], x[-1].shape ))
    t1 = time.time()
    if FEET_MODE == "LIBROSA":
        x_mel = extract_melspectrogram_parallel(x, n_mels=30, use_power_db=True)
    elif FEET_MODE == "KAPRE":
        # x_mel = extract_features(x, model=kapre_extractor)
        x_mel = extract_features(x)

    # x_contrast = extract_bandwidth_parallel(x)
    t2 = time.time()
    info_log = list()
    x_feas = []
    for i in range(len(x_mel)):
        mel = np.mean(x_mel[i], axis=0).reshape(-1)
        mel_std = np.std(x_mel[i], axis=0).reshape(-1)
        # contrast = np.mean(x_contrast[i], axis=0).reshape(-1)
        # contrast_std = np.std(x_contrast[i], axis=0).reshape(-1)
        # contrast, contrast_std
        fea_item = np.concatenate([mel, mel_std], axis=-1)
        x_feas.append(fea_item)
        if i < 1:
            info_log.append("i={}, x_mel type={}, shape={}".format(i, type(x_mel[i]), x_mel[i].shape))
            info_log.append("i={}, mel type={}, shape={}".format(i, type(mel), mel.shape))
            info_log.append("i={}, mel_std type={}, shape={}".format(i, type(mel_std), mel_std.shape))
            info_log.append("i={}, fea_item type={}, shape={}".format(i, type(fea_item), fea_item.shape))

    x_feas = np.asarray(x_feas)
    scaler = StandardScaler()
    X = scaler.fit_transform(x_feas[:, :])

    info_log.append("FEET_MODE = {}, x_mel type={}, shape={}, cost_time={}s".format(FEET_MODE, type(x_mel), x_mel.shape, round(t2-t1, 3)))
    info_log.append("x_feas type={}, shape={}".format(type(x_feas), x_feas.shape))
    info_log.append("X type={}, shape={}".format(type(X), X.shape))
    info(json.dumps(info_log, indent=4))
    return X

warmup_size = 10
warmup_x = [np.array([np.random.uniform() for i in range(48000)], dtype=np.float32) for j in range(warmup_size)]
# warmup_x_mel = extract_features(warmup_x)
warmup_x_mel = lr_preprocess_update(warmup_x)


pre_func_lr_prepro = partial(lr_preprocess_update)



