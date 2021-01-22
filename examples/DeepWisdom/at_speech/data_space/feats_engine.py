import os
from functools import partial
from itertools import repeat
import json
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
import keras
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from kapre.time_frequency import Melspectrogram, Spectrogram

from at_toolkit.interface.adl_feats_maker import AbsFeatsMaker
from at_toolkit import info, as_timer


NCPU = multiprocessing.cpu_count()
SAMPLING_RATE = 16000
MAX_AUDIO_DURATION = 5  # limited length of audio, like 20s

AUDIO_SAMPLE_RATE = 16000
KAPRE_FMAKER_WARMUP = True


def wav_to_mag_old(wav, params, win_length=400, hop_length=160, n_fft=512):
    mode = params["mode"]
    wav = extend_wav(wav, params["train_wav_len"], params["test_wav_len"], mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    # mag, _ = librosa.magphase(linear_spect)
    mag, _ = librosa.magphase(np.asfortranarray(linear_spect))
    mag_T = mag.T
    if mode == "test":
        mag_T = load_data(mag_T, params["train_spec_len"], params["test_spec_len"], mode)
    return mag_T


def make_kapre_mag_maker(n_fft=1024, hop_length=128, audio_data_len=80000):
    stft_model = keras.models.Sequential()
    stft_model.add(Spectrogram(n_dft=n_fft, n_hop=hop_length, input_shape=(1, audio_data_len),
                               power_spectrogram=2.0, return_decibel_spectrogram=False,
                               trainable_kernel=False, name='stft'))
    return stft_model


def wav_to_mag(wav, params, win_length=400, hop_length=160, n_fft=512):
    mode = params["mode"]
    # info("ori_wav_len={}".format(len(wav)))
    wav = extend_wav(wav, params["train_wav_len"], params["test_wav_len"], mode=mode)
    # info("extend_wav_len={}".format(len(wav)))

    wav2feat_mode = 1

    if wav2feat_mode == 0:
        # 1. original:
        # linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
        # mag, _ = librosa.magphase(linear_spect)
        # mag_T = mag.T
        pass
    elif wav2feat_mode == 1:
        # 2. wav2linear_spectrogram: stft+magphase
        linear_sft = librosa.stft(np.asfortranarray(wav), n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram

        # simplify:
        mag_T = np.abs(linear_sft)

        # original:

        # info("linear_sft_shape={}".format(linear_sft.shape))
        # linear_spect = linear_sft.T
        # D = np.asfortranarray(linear_spect)
        # mag = np.abs(D)
        # mag_T = mag.T

        # 拆开 magphase
        # info("linear_sft_T_array={}".format(linear_spect.shape))
        # mag, _ = librosa.magphase()
        # mag **= 1
        # info("linear_sft_mag={}".format(mag.shape))
        # phase = np.exp(1.j * np.angle(D))
        # info("linear_sft_phase={}".format(phase.shape))

        # info("need_a_wav_map_shape={}".format(mag.shape))
        pass

    elif wav2feat_mode == 2:
        # 3. do not use mag.
        linear_sft = librosa.stft(np.asfortranarray(wav), n_fft=n_fft, win_length=win_length,
                                  hop_length=hop_length)  # linear spectrogram
        info("linear_sft_shape={}".format(linear_sft.shape))
        mag_T = linear_sft

    # using kapre.

    if mode == "test":
        mag_T = load_data(mag_T, params["train_spec_len"], params["test_spec_len"], mode)
    return mag_T




def get_fixed_array(X_list, len_sample=5, sr=SAMPLING_RATE):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample * sr:
            n_repeat = np.ceil(sr * len_sample / X_list[i].shape[0]).astype(np.int32)
            # info("X_list[i] type={}, len={}, lssr={}, n_repeat={}".format(type(X_list[i]), len(X_list[i]), len_sample*sr, n_repeat))
            X_list[i] = np.tile(X_list[i], n_repeat)
            # info("X_list[i] new len={}".format(len(X_list[i])))

        X_list[i] = X_list[i][: len_sample * sr]
        # info("xlist_i shape={}".format(X_list[i].shape))

    X = np.asarray(X_list)
    info("x shape={}".format(X.shape))
    X = np.stack(X)
    info("x shape={}".format(X.shape))
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)

    return X


def mel_feats_transform(x_mel):
    x_feas = []
    info_log = list()
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

    # info_log.append("FEET_MODE = {}, x_mel type={}, shape={}, cost_time={}s".format(FEET_MODE, type(x_mel), x_mel.shape, round(t2-t1, 3)))
    info_log.append("x_feas type={}, shape={}".format(type(x_feas), x_feas.shape))
    info_log.append("X type={}, shape={}".format(type(X), X.shape))
    info(json.dumps(info_log, indent=4))
    return X


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)


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


def extend_wav(wav, train_wav_len=40000, test_wav_len=40000, mode="train"):
    if mode == "train":
        div, mod = divmod(train_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        div, mod = divmod(test_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    # linear = librosa.stft(np.asfortranarray(wav), n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    linear = librosa.stft(
        np.asfortranarray(wav), n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )  # linear spectrogram
    return linear.T


def load_data(mag, train_spec_len=250, test_spec_len=250, mode="train"):
    freq, time = mag.shape
    if mode == "train":
        if time - train_spec_len > 0:
            randtime = np.random.randint(0, time - train_spec_len)
            spec_mag = mag[:, randtime : randtime + train_spec_len]
        else:
            spec_mag = mag[:, :train_spec_len]
    else:
        spec_mag = mag[:, :test_spec_len]
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


class KapreMelSpectroGramFeatsMaker(AbsFeatsMaker):
    # speech features extraction using kapre.
    SAMPLING_RATE = 16000
    # lr and cnn may use two params.
    # N_MELS = 64
    N_MELS = 30
    # HOP_LENGTH = 512
    # N_FFT = 1024  # 0.064 sec
    HOP_LENGTH = int(SAMPLING_RATE * 0.04)
    # N_FFT = int(SAMPLING_RATE * 0.1)# 0.064 sec
    N_FFT = 1024  # 0.064 sec
    FMIN = 20
    FMAX = SAMPLING_RATE // 2

    CROP_SEC = 5

    def __init__(self, feat_name, feat_tool="Kapre"):
        super().__init__(feat_tool, feat_name)
        self.kapre_melspectrogram_extractor = None
        self.kape_params = {
            "SAMPLING_RATE": self.SAMPLING_RATE,
            "N_MELS": self.N_MELS,
            "HOP_LENGTH": int(self.SAMPLING_RATE * 0.04),
            "N_FFT": self.N_FFT,  # 0.064 sec
            "FMIN": self.FMIN,
            "FMAX": self.SAMPLING_RATE // 2,
            "CROP_SEC": self.CROP_SEC,
        }
        self.init_kapre_melspectrogram_extractor()

    def make_melspectrogram_extractor(self, input_shape, sr=SAMPLING_RATE):
        model = keras.models.Sequential()
        model.add(
            Melspectrogram(
                fmax=self.kape_params.get("FMAX"),
                fmin=self.kape_params.get("FMIN"),
                n_dft=self.kape_params.get("N_FFT"),
                n_hop=self.kape_params.get("HOP_LENGTH"),
                n_mels=self.kape_params.get("N_MELS"),
                name="melgram",
                image_data_format="channels_last",
                input_shape=input_shape,
                return_decibel_melgram=True,
                power_melgram=2.0,
                sr=sr,
                trainable_kernel=False,
            )
        )
        return model

    def init_kapre_melspectrogram_extractor(self):
        self.kapre_melspectrogram_extractor = self.make_melspectrogram_extractor(
            (1, self.kape_params.get("CROP_SEC") * self.kape_params.get("SAMPLING_RATE"))
        )
        if KAPRE_FMAKER_WARMUP:
            warmup_size = 10
            warmup_x = [
                np.array([np.random.uniform() for i in range(48000)], dtype=np.float32) for j in range(warmup_size)
            ]
            # warmup_x_mel = extract_features(warmup_x)
            warmup_x_mel = self.make_features(warmup_x, feats_maker_params={"len_sample": 5, "sr": 16000})
            info("Kpare_featmaker warmup.")
            as_timer("Kpare_featmaker_warmup")

    def make_features(self, raw_data, feats_maker_params: dict):
        """
        :param raw_data:
        :param feats_maker_params:
            {
                "len_samples": 5,
                "sr": SAMPLING_RATE,

            }
        :return:
        """
        if isinstance(raw_data, list):
            info("raw_data, len={}, ele_type={}".format(len(raw_data), type(raw_data[0])))
        elif isinstance(raw_data, np.ndarray):
            info("raw_data, shape={}, ele_type={}".format(raw_data.shape, type(raw_data[0])))
        else:
            pass

        raw_data = [sample[0 : MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in raw_data]

        X = get_fixed_array(raw_data, len_sample=feats_maker_params.get("len_sample"), sr=feats_maker_params.get("sr"))
        info("note: exft get fix done.")
        # X = model.predict(X)
        X = self.kapre_melspectrogram_extractor.predict(X)
        info("note: exft model kapre_extractor predict done, kapre predict shape={}".format(X.shape))

        # basic: (147, 30, 125, 1) to (147, 125, 30, 1) to (147, 125, 30)
        # X = X.transpose(0, 2, 1, 3)
        # X = np.squeeze(X)

        # basic + powertodb. squeeze->powertodb->transpose
        # squeeze.
        X = np.squeeze(X)
        X = X.transpose(0, 2, 1)
        info("note: exft model transpose and squeeze done, shape={}".format(X.shape))

        # tranform melspectrogram features.
        X = mel_feats_transform(X)
        return X


class LibrosaMelSpectroGramFeatsMaker(AbsFeatsMaker):
    FFT_DURATION = 0.1
    HOP_DURATION = 0.04

    def __init__(self, feat_name, feat_tool="Librosa"):
        super().__init__(feat_tool, feat_name)

    def extract_melspectrogram_parallel(
        self, data, sr=16000, n_fft=None, hop_length=None, n_mels=30, use_power_db=False
    ):
        if n_fft is None:
            n_fft = int(sr * self.FFT_DURATION)
        if hop_length is None:
            hop_length = int(sr * self.HOP_DURATION)
        extract = partial(
            extract_for_one_sample,
            extract=librosa.feature.melspectrogram,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            use_power_db=use_power_db,
        )
        results = extract_parallel(data, extract)

        return results

    def make_features(self, raw_data, feats_maker_params: dict):
        x_mel = self.extract_melspectrogram_parallel(raw_data, n_mels=30, use_power_db=True)
        # tranform melspectrogram features.
        x_mel_transformed = mel_feats_transform(x_mel)
        return x_mel_transformed


pool = Pool(os.cpu_count())


class LbrsTr34FeatsMaker(AbsFeatsMaker):
    def __init__(self, feat_tool, feat_name):
        super().__init__(feat_tool, feat_name)
        self.feat_des = "for_TR34"

    def pre_trans_wav_update(self, wav_list, params):
        info("pre_trans_wav len={}, params={}".format(len(wav_list), params))
        if len(wav_list) == 0:
            return []
        # set=10, test for single CPU =10000
        elif len(wav_list) > NCPU * 2:
            info("note: using pool pre_trans_wav len={}".format(len(wav_list)))
            # with Pool(NCPU) as pool:
            mag_arr = pool.starmap(wav_to_mag, zip(wav_list, repeat(params)))
            # # mag_arr = pool.starmap(wav_to_mag, zip(np.asfortranarray(wav_list), repeat(params)))
            return mag_arr
        else:
            info("note: using no pool pre_trans_wav len={}".format(len(wav_list)))
            mag_arr = [wav_to_mag(wav, params) for wav in wav_list]
            info("note: using no pool pre_trans_wav done len={}".format(len(wav_list)))
            return mag_arr

    def make_features(self, raw_data, feats_maker_params: dict):
        # fixme: params must be done.
        tr34_data_features = self.pre_trans_wav_update(raw_data, feats_maker_params)
        return tr34_data_features
