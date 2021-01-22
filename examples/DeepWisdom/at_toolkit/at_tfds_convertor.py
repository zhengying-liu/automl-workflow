import tensorflow as tf
import numpy as np
import time

from at_toolkit.at_utils import info, error, as_timer
from at_toolkit.interface.adl_tfds_convertor import AbsTfdsConvertor

info("note, tf version={}".format(tf.__version__))


class TfdsConvertor(AbsTfdsConvertor):
    def __init__(self, if_train_shuffle=False, train_shuffle_size=100, if_pad_batch=False, padded_batch_size=20, domain=None):
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
        self.train_shuffle_size = train_shuffle_size
        self.train_max_shuffle_size = 1000
        self.if_padded_batch = if_pad_batch
        self.padded_batch_size = padded_batch_size

        self.tfds_convertor_sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.domain = domain

    def init_train_tfds(self, train_tfds, train_num, force_shuffle=False):
        if self.train_tfds is None or self.train_num == 0 or force_shuffle is True:
            self.train_num = train_num
            self.train_tfds = train_tfds

            if self.if_train_shuffle or force_shuffle is True:
                self.train_tfds = self.train_tfds.shuffle(buffer_size=min(self.train_max_shuffle_size, int(self.train_num * 0.6)))
                # self.train_tfds = self.train_tfds.shuffle(buffer_size=self.train_shuffle_size)

            if self.if_padded_batch:
                self.train_tfds = self.train_tfds.padded_batch(
                    self.padded_batch_size,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )
            # force reNone tfds_train_iterator.
            self.tfds_train_os_iterator = None

            info("note: train_tfds cache, if_train_shuffle={}, force_shuffle={}, reset tfds_train_os_iterator None.".format(self.if_train_shuffle, force_shuffle))

    def init_test_tfds(self, test_tfds):
        if self.test_tfds is None:
            self.test_tfds = test_tfds

            if self.if_padded_batch:
                self.test_tfds = test_tfds.padded_batch(
                    self.padded_batch_size,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )
            as_timer("tfds_cvtr_init_tfds")

    def get_train_np(self, take_size):
        as_timer("tfdscvtr_get_train_np_start")
        if self.train_tfds is None:
            error("Error: train_tfds is None.")
            return self.accum_train_x, self.accum_train_y

        if self.tfds_train_os_iterator is None:
            self.tfds_train_os_iterator = self.train_tfds.make_one_shot_iterator()
            as_timer("tfds_train_os_iterator_make")
            self.tfds_train_iter_next = self.tfds_train_os_iterator.get_next()

        cur_get_cnt = 0
        cur_data_y = list()
        cur_incre_train_x = list()

        if self.accm_train_cnt < self.train_num:
            # info("note: accm_train_cnt={}, train_num={}".format(self.accm_train_cnt, self.train_num))
            time_train_np_start = time.time()
            if self.if_padded_batch:
                info("note: domain={}".format(self.domain))
                while True:
                    example_batch_num = 0
                    try:
                        example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)
                        # fixme: 注意，这里example 和 labels都是batch, batch_size=20
                        cur_incre_train_x.extend(example)
                        cur_data_y.extend(labels)
                        # X.append(example)
                        # Y.append(labels)
                        cur_get_cnt += example.shape[0]
                        self.accm_train_cnt += example.shape[0]
                        example_batch_num += 1
                        # info("note: cur_get_cnt={}, accm_train_cnt={}, example_batch_num={}, a_example_shape={}".format(cur_get_cnt, self.accm_train_cnt, example_batch_num, example.shape))

                        if cur_get_cnt >= take_size or self.accm_train_cnt >= self.train_num:
                            time_train_np_end = time.time()
                            info(
                                "note: now text extend batch domain={} take train update={}, accm_train_cnt={}, cost_time={}s".format(
                                    self.domain,
                                    cur_get_cnt,
                                    self.accm_train_cnt,
                                    round(time_train_np_end - time_train_np_start, 3)
                                )
                            )
                            break

                    except tf.errors.OutOfRangeError:
                        info("train out of range, cur_get_cnt={}".format(cur_get_cnt))
                        break

            else:
                while True:
                    try:
                        example, labels = self.tfds_convertor_sess.run(self.tfds_train_iter_next)
                        # output:  Note:time example shape=(86401, 1, 1, 1)
                        # logger.info("Note:time example shape={}".format(example.shape))
                        # self.accum_train_x.append(example)
                        cur_incre_train_x.append(example)
                        cur_data_y.append(labels)
                        cur_get_cnt += 1
                        self.accm_train_cnt += 1
                        if cur_get_cnt >= take_size or self.accm_train_cnt >= self.train_num:
                            time_train_np_end = time.time()
                            info(
                                "note: now append domain={} take train update={}, accm_train_cnt={}, train_num={}, cost_time={}s".format(
                                    self.domain,
                                    cur_get_cnt,
                                    self.accm_train_cnt,
                                    self.train_num,
                                    round(time_train_np_end - time_train_np_start, 3)
                                )
                            )
                            as_timer("tfds_get_train_np_update={}".format(cur_get_cnt))
                            break

                    except tf.errors.OutOfRangeError:
                        break

            self.accum_train_x.extend(cur_incre_train_x)
            as_timer("tfds_get_train_np_accum_train_x_{}".format(len(self.accum_train_x)))

            if self.accum_train_y is None:
                self.accum_train_y = np.array(cur_data_y)
            else:
                self.accum_train_y = np.concatenate((self.accum_train_y, np.array(cur_data_y)))

            info("note: self.accum_train_x num_new={}, incre_train_num={}, self.accum_train_y shape={}, cur_data_y shape={}".format(
                len(self.accum_train_x),
                len(cur_incre_train_x),
                self.accum_train_y.shape,
                np.array(cur_data_y).shape
            ))

        else:
            self.tfds_convertor_sess.close()

        # return cur_incre_train_x, np.array(cur_data_y)
        return {"x": [np.squeeze(x) for x in cur_incre_train_x], "y": np.array(cur_data_y)}

    def get_train_np_accm(self, take_size) -> dict:
        self.get_train_np(take_size)
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_train_np_full(self):
        left_train_num = self.train_num - self.accm_train_cnt
        self.get_train_np(take_size=left_train_num)
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_test_np(self):
        if self.test_tfds is None:
            error("Error: test_tfds is None.")
            return self.accum_test_x, self.accum_test_y

        if len(self.accum_test_x) == 0:
            time_test_np_start = time.time()
            tfds_test_os_iterator = self.test_tfds.make_one_shot_iterator()
            as_timer("tfds_test_ositer")
            tfds_test_iter_next = tfds_test_os_iterator.get_next()

            # with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            if self.if_padded_batch:
                while True:
                    try:
                        # example, labels = sess.run(tfds_test_iter_next)
                        example, labels = self.tfds_convertor_sess.run(tfds_test_iter_next)
                        example = np.squeeze(example, (2, 3))
                        example = np.squeeze(example, axis=-1)
                        example = example.astype(np.int)

                        self.accum_test_x.extend(example)
                        self.accum_test_y.extend(labels)
                        self.accm_test_cnt += example.shape[0]
                        # X.append(example)
                        # Y.append(labels)
                    except tf.errors.OutOfRangeError:
                        break
            else:
                while True:
                    try:
                        # example, labels = sess.run(tfds_test_iter_next)
                        example, labels = self.tfds_convertor_sess.run(tfds_test_iter_next)
                        # output:  Note:time example shape=(86401, 1, 1, 1)
                        # info("Note:time example shape={}".format(example.shape))
                        self.accum_test_x.append(example)
                        self.accum_test_y.append(labels)
                        self.accm_test_cnt += 1

                    except tf.errors.OutOfRangeError:
                        as_timer("tfds_test_run_OOR_{}".format(self.accm_test_cnt))
                        break

            time_test_np_end = time.time()
            info(
                "note: now take test accm_test_cnt={}, cost_time={}s".format(
                    self.accm_test_cnt, round(time_test_np_end - time_test_np_start, 3)
                )
            )
            self.accum_test_y = np.array(self.accum_test_y)

        # return self.accum_test_x
        return [np.squeeze(x) for x in self.accum_test_x]

        # return self.accum_test_x, self.accum_test_y

    # def get_speech_train_dataset(self, take_size=100):
    #     self.get_train_np(take_size=take_size)
    #     return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    # def get_speech_train_dataset_full(self):
    #     self.get_train_numpy_full()
    #     return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    # def get_speech_test_dataset(self):
    #     self.get_test_numpy()
    #     return [np.squeeze(x) for x in self.accum_test_x], self.accum_test_x

