# -*- coding: utf-8 -*-
# @Date    : 2020/1/17 14:19
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import numpy as np
import keras
from keras.preprocessing import sequence

from Auto_NLP.upwind_flys_update.utils import pad_sequence, clean_data, clean_en_text_parallel,clean_en_text_single


# MAX_SEQ_LENGTH = 1000

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train, labels, batch_size, mp_pooler, bpe_encoder, language, max_length, vocab, num_features,
                 tokenizer=None,
                 shuffle=True):
        self.indices_ = None
        self.batch_size = batch_size
        self.X = x_train
        self.labels = labels
        self.mp_pooler = mp_pooler
        self.bpe_encoder = bpe_encoder
        self.tokenizer = tokenizer
        self.language = language
        self.shuffle = shuffle
        self.max_length = max_length
        self.vocab = vocab
        self.num_features = num_features
        # print("check vocab size {}".format(len(self.vocab)))

        self.on_epoch_end()

    def __len__(self):
        if self.shuffle:
            return int(np.floor(len(self.X) / self.batch_size))
        else:
            # fixme：当给RNN时，这里要去掉+1？
            return int((len(self.X) / self.batch_size)+1)
            # return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices_[index * self.batch_size:min((index + 1) * self.batch_size, len(self.X))]
        # print("indexes", indexes)
        X_temp = [self.X[k] for k in indexes]
        if self.tokenizer is None:
            batch_x, batch_y = self._process_bpe(X_temp, indexes)
        else:
            batch_x, batch_y = self._process(X_temp, indexes)
        return batch_x, batch_y

    def on_epoch_end(self):
        self.indices_ = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices_)

    def _process_bpe(self, X_temp, indexes):
        data_ids = self.bpe_encoder.encode_ids(X_temp)
        padding_val = self.bpe_encoder.vectors.shape[0]
        max_length = self.max_length
        batch_x = sequence.pad_sequences(data_ids, maxlen=max_length, padding='post', value=padding_val)
        # print(batch_x[:4])
        batch_y = self.labels[indexes]
        return batch_x, batch_y

    def _process(self, X_temp, indexes):
        data_ids = self.tokenizer.texts_to_sequences(X_temp)
        max_length = self.max_length
        batch_x = sequence.pad_sequences(data_ids, maxlen=max_length, padding='post')
        batch_y = self.labels[indexes]
        return batch_x, batch_y
