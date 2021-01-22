import os
import tensorflow as tf
import numpy as np
import pprint
import time
from numba import cuda

from Auto_NLP.upwind_flys_update.time_utils import info, error
from nlp_autodl_config import IF_RESET_TFGRAPH_SESS_RUN, TF_DATASET_TO_NUMPY_MODE


class TfDatasetsConvertor(object):
    def __init__(self, if_train_shuffle=False, config=None):
        self.train_tfds = None
        self.test_tfds = None
        self.train_num = 0
        self.test_num = 0
        self.accum_train_x = list()
        self.accum_train_y = None
        self.accm_train_cnt = 0
        self.accum_test_x = list()
        self.accum_test_y = list()
        self.accm_test_cnt = 0

        self.tfds_train_os_iterator = None
        self.tfds_train_iter_next = None

        self.speech_train_dataset = {"x": None, "y": None}
        self.speech_test_dataset = None
        self.speech_x_test = None
        self.if_train_shuffle = if_train_shuffle

        self.tfds_convertor_sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.nlp_index_to_token_map = None
        self.nlp_sep = None

    def init_nlp_data(self, nlp_index_to_token_map, nlp_sep):
        if self.nlp_index_to_token_map is None or self.nlp_sep is None:
            self.nlp_index_to_token_map = nlp_index_to_token_map
            self.nlp_sep = nlp_sep

    def init_train_tfds(self, train_tfds, train_num, pad_num=30):

        if self.train_tfds is None or self.train_num == 0:

            # 原始20
            self.train_tfds = train_tfds.padded_batch(pad_num, padded_shapes=([None, 1, 1, 1], [None]),
                                                      padding_values=(
                                                          tf.constant(-1, dtype=tf.float32)
                                                          , tf.constant(-1, dtype=tf.float32)))

            # self.train_tfds = self.train_tfds.prefetch(tf.data.experimental.AUTOTUNE)


            self.train_num = train_num

    def init_test_tfds(self, test_tfds, pad_num=20):
        if self.test_tfds is None:
            # 原始20
            # self.test_tfds = self.test_tfds.prefetch(buffer_size=4)
            # self.test_tfds = test_tfds.padded_batch(pad_num, padded_shapes=([None, 1, 1, 1], [None]),
            #                                         padding_values=(
            #                                             tf.constant(-1, dtype=tf.float32)
            #                                             , tf.constant(-1, dtype=tf.float32)))

            # self.test_tfds = self.test_tfds.prefetch(buffer_size=1)
			

            self.test_tfds = test_tfds
    def get_train_numpy(self, update_train_num):
        if self.train_tfds is None:
            error("Error: train_tfds is None.")
            return self.accum_train_x, self.accum_train_y

        if self.tfds_train_os_iterator is None:
            self.tfds_train_os_iterator = self.train_tfds.make_one_shot_iterator()
            self.tfds_train_iter_next =self.tfds_train_os_iterator.get_next()

        X, Y = [], []
        cur_get_cnt = 0
        cur_data_y = list()

        if self.accm_train_cnt < self.train_num:
            info("show the accum_train_cnt {}".format(self.accm_train_cnt))
            time_train_np_start = time.time()
            while True:
                try:

                    example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)
                    example = np.squeeze(example, (2, 3))
                    example = np.squeeze(example, axis=-1)

                    # print("Note:time example shape={}".format(example.shape))
                    example = example.astype(np.int)
                    self.accum_train_x.extend(example)
                    # cur_data_y.extend(labels)
                    X.extend(example)
                    Y.extend(labels)
                    # cur_get_cnt += 1
                    # self.accm_train_cnt += 1
                    cur_get_cnt += example.shape[0]
                    self.accm_train_cnt += example.shape[0]
                    # example_batch_num += 1

                    if cur_get_cnt >= update_train_num or self.accm_train_cnt >= self.train_num:
                        time_train_np_end = time.time()
                        info("note: now take train update={}, accm_train_cnt={}, cost_time={}s".format(cur_get_cnt,
                                                                                                       self.accm_train_cnt,
                                                                                                       time_train_np_end - time_train_np_start))
                        break

                except tf.errors.OutOfRangeError:
                    break

            if self.accum_train_y is None:
                self.accum_train_y = np.array(cur_data_y)
            else:
                print("check accum_train_y shape {}, cur_data_y shape {}".format(self.accum_train_y.shape,
                                                                                 np.array(cur_data_y).shape))
                # self.accum_train_y = np.concatenate((self.accum_train_y, np.array(cur_data_y)), axis=0)
        else:
            self.tfds_convertor_sess.close()

        return X, Y

    def get_test_numpy(self):
        if self.test_tfds is None:
            error("Error: test_tfds is None.")
            return self.accum_test_x, self.accum_test_y
        X, Y = [], []

        if len(self.accum_test_x) == 0:
            time_test_np_start = time.time()
            tfds_test_os_iterator = self.test_tfds.make_one_shot_iterator()
            tfds_test_iter_next = tfds_test_os_iterator.get_next()

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

                while True:
                    try:
                        example, labels = sess.run(tfds_test_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)
                        X.extend(example)
                        Y.extend(labels)
                        self.accum_test_x.extend(example)
                        self.accum_test_y.extend(labels)
                        # self.accm_test_cnt += 1
                        self.accm_test_cnt += example.shape[0]

                    except tf.errors.OutOfRangeError:
                        break

            time_test_np_end = time.time()
            info("note: now take test accm_test_cnt={}, cost_time={}s".format(self.accm_test_cnt, round(
                time_test_np_end - time_test_np_start, 3)))
            # self.accum_test_y = np.array(self.accum_test_y)

        return X, Y

    def get_nlp_train_dataset(self, take_size=100):
        # fixme: need to be update.
        # update self.accum_train_x and self.accum_train_y
        # self.train_tfds = self.train_tfds.apply(tf.data.experimental.map_and_batch
        #                                         (map_func=lambda *x: (self.to_corpus(x[0],x[1])),
        #                                          batch_size=16)
        #                                         )
        # tf.map_fn(fn=lambda *x: )
        # self.train_tfds = self.train_tfds.take(take_size)
        X, Y = self.get_train_numpy(update_train_num=take_size)
        return X, Y

    def get_nlp_test_dataset(self, pad_num=20):
        self.test_tfds = self.test_tfds.padded_batch(pad_num, padded_shapes=([None, 1, 1, 1], [None]),
                                                     padding_values=(
                                                         tf.constant(-1, dtype=tf.float32)
                                                         , tf.constant(-1, dtype=tf.float32)))

        X, Y = self.get_test_numpy()
        return X, Y

    def to_corpus(self, id_lst, label):
        print("x:", id_lst)

        corpus = []
        # for index in id_lst:
        tokens = [self.nlp_index_to_token_map[i] for i in id_lst]
        document = self.nlp_sep.join(tokens)
        corpus.append(document)

        print(tokens)
        return document, label

    def get_nlp_train_numpy_prefetch(self, update_train_num):
        if self.train_tfds is None:
            error("Error: train_tfds is None.")
            return self.accum_train_x, self.accum_train_y

        if self.tfds_train_os_iterator is None:
            self.tfds_train_os_iterator = self.train_tfds.make_one_shot_iterator()
            self.tfds_train_iter_next = self.tfds_train_os_iterator.get_next()

        X, Y = [], []
        cur_get_cnt = 0
        cur_data_y = list()

        if self.accm_train_cnt < self.train_num:
            info("show the accum_train_cnt {}".format(self.accm_train_cnt))
            time_train_np_start = time.time()
            while True:
                try:

                    example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)
                    example = np.squeeze(example, (2, 3))
                    example = np.squeeze(example, axis=-1)

                    # print("Note:time example shape={}".format(example.shape))
                    example = example.astype(np.int)
                    self.accum_train_x.extend(example)
                    cur_data_y.extend(labels)
                    X.extend(example)
                    Y.extend(labels)
                    # cur_get_cnt += 1
                    # self.accm_train_cnt += 1
                    cur_get_cnt += example.shape[0]
                    self.accm_train_cnt += example.shape[0]
                    # example_batch_num += 1

                    if cur_get_cnt >= update_train_num or self.accm_train_cnt >= self.train_num:
                        time_train_np_end = time.time()
                        info("note: now take train update={}, accm_train_cnt={}, cost_time={}s".format(cur_get_cnt,
                                                                                                       self.accm_train_cnt,
                                                                                                       time_train_np_end - time_train_np_start))
                        break

                except tf.errors.OutOfRangeError:
                    break

            if self.accum_train_y is None:
                self.accum_train_y = np.array(cur_data_y)
            else:
                print("check accum_train_y shape {}, cur_data_y shape {}".format(self.accum_train_y.shape,
                                                                                 np.array(cur_data_y).shape))
                self.accum_train_y = np.concatenate((self.accum_train_y, np.array(cur_data_y)), axis=0)
        else:
            self.tfds_convertor_sess.close()

        return X, Y

    def get_nlp_test_numpy_prefetch(self):
        if self.test_tfds is None:
            error("Error: test_tfds is None.")
            return self.accum_test_x, self.accum_test_y
        X, Y = [], []

        if len(self.accum_test_x) == 0:
            time_test_np_start = time.time()
            tfds_test_os_iterator = self.test_tfds.make_one_shot_iterator()
            tfds_test_iter_next = tfds_test_os_iterator.get_next()

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        example, labels = sess.run(tfds_test_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)
                        X.extend(example)
                        Y.extend(labels)
                        self.accum_test_x.extend(example)
                        self.accum_test_y.extend(labels)
                        # self.accm_test_cnt += 1
                        self.accm_test_cnt += example.shape[0]

                    except tf.errors.OutOfRangeError:
                        break

            time_test_np_end = time.time()
            info("note: now take test accm_test_cnt={}, cost_time={}s".format(self.accm_test_cnt, round(
                time_test_np_end - time_test_np_start, 3)))
            # self.accum_test_y = np.array(self.accum_test_y)

        return X, Y


def main():
    pass


if __name__ == "__main__":
    main()
