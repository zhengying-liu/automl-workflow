import tensorflow as tf
import numpy as np
import pprint
import time
import multiprocessing as mp
from multiprocessing.pool import Pool

NCPU = mp.cpu_count()

from log_utils import info, error, as_timer
from speech_autodl_config import IF_RESET_TFGRAPH_SESS_RUN, TF_DATASET_TO_NUMPY_MODE

NLP_2_CORPUS_MODE = 4  # 0/1/2
info("note, tf version={}".format(tf.__version__))


class NlpCorpusWorker(object):
    def __init__(self, index_to_token_map, sep):
        self.nlp_index_to_token_map = index_to_token_map
        self.nlp_sep = sep

    def nlp_2_corpus_item(self, x):
        x = x[x != -1]
        tokens = [self.nlp_index_to_token_map[int(i)] for i in x]
        document = self.nlp_sep.join(tokens)
        return document

    def nlp_2_corpus_list(self, x_list):
        s = list()
        for x in x_list:
            x = x[x != -1]
            tokens = [self.nlp_index_to_token_map[int(i)] for i in x]
            document = self.nlp_sep.join(tokens)
            s.append(document)

        return s


class TfDatasetsConvertor(object):
    def __init__(self, if_train_shuffle=False):
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
        self.nlp_train_x_corpus = None
        self.nlp_test_x_corpus = None
        self.domain = None
        self.nlp_corpus_worker = None

    def init_domain(self, domain):
        """
        Must be run in model.init
        :param domain:
        :return:
        """
        if self.domain is None:
            self.domain = domain

    def init_nlp_data(self, nlp_index_to_token_map, nlp_sep):
        if self.nlp_index_to_token_map is None or self.nlp_sep is None:
            self.nlp_index_to_token_map = nlp_index_to_token_map
            self.nlp_sep = nlp_sep
            self.nlp_corpus_worker = NlpCorpusWorker(nlp_index_to_token_map, nlp_sep)

    def init_train_tfds(self, train_tfds, train_num):
        if self.train_tfds is None or self.train_num == 0:
            self.train_num = train_num
            if self.domain == "text":
                self.train_tfds = train_tfds.padded_batch(
                    20,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )
            else:
                # if cache:
                self.train_tfds = train_tfds.cache()
                # self.train_tfds = self.train_tfds.map(lambda x, y: (np.array(x, dtype=np.float16), y), num_parallel_calls=4)
                info("note: train_tfds cache.")

    def init_test_tfds(self, test_tfds):
        if self.test_tfds is None:

            if self.domain == "text":
                self.test_tfds = test_tfds.padded_batch(
                    20,
                    padded_shapes=([None, 1, 1, 1], [None]),
                    padding_values=(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)),
                )

            else:
                # config: test_if_map_cutoff
                tfds_if_test_cutoff = False
                if tfds_if_test_cutoff:
                    self.test_tfds = test_tfds.map(lambda x,y: (x[:800000], y), num_parallel_calls=4)
                else:
                    self.test_tfds = test_tfds

            as_timer("tfds_cvtr_init_tfds")

    def get_train_numpy(self, update_train_num):
        # info(
        #     "note: get_train_numpy, update_train_num={}, domain={}, accm_train_cnt={}, train_num={}".format(
        #         update_train_num, self.domain, self.accm_train_cnt, self.train_num
        #     )
        # )
        as_timer("tfdscvtr_get_train_np_start")
        if self.train_tfds is None:
            error("Error: train_tfds is None.")
            return self.accum_train_x, self.accum_train_y

        if self.tfds_train_os_iterator is None:
            time_mosi_start = time.time()
            self.tfds_train_os_iterator = self.train_tfds.make_one_shot_iterator()
            as_timer("tfds_train_os_iterator_make")
            self.tfds_train_iter_next = self.tfds_train_os_iterator.get_next()
            time_mosi_end = time.time()
            info("note: train_os_iterator done, cost_time={}s".format(round(time_mosi_end - time_mosi_start, 3)))

        cur_get_cnt = 0
        cur_data_y = list()
        cur_incre_train_x = list()

        if self.accm_train_cnt < self.train_num:
            # info("note: accm_train_cnt={}, train_num={}".format(self.accm_train_cnt, self.train_num))
            time_train_np_start = time.time()
            if self.domain == "text":
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

                        if cur_get_cnt >= update_train_num or self.accm_train_cnt >= self.train_num:
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
                        if cur_get_cnt >= update_train_num or self.accm_train_cnt >= self.train_num:
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

            # 获取增量 train_x/y_numpy
            # info(
            #     "note: self.accum_train_x num = {}, cur_incre_train_x num={}".format(
            #         len(self.accum_train_x), len(cur_incre_train_x)
            #     )
            # )
            # update accum_train_x/accum_train_y
            self.accum_train_x.extend(cur_incre_train_x)
            as_timer("tfds_get_train_np_accum_train_x_{}".format(len(self.accum_train_x)))

            if self.accum_train_y is None:
                # info("note: np.array(cur_data_y) shape={}".format(np.array(cur_data_y).shape))
                self.accum_train_y = np.array(cur_data_y)
            else:
                # info(
                #     "note: self.accum_train_y shape={}, np.array(cur_data_y) shape={}".format(
                #         self.accum_train_y.shape, np.array(cur_data_y).shape
                #     )
                # )
                self.accum_train_y = np.concatenate((self.accum_train_y, np.array(cur_data_y)))

                # info(
                #     "note: self.accum_train_y shape={}, np.array(cur_data_y) shape={}".format(
                #         self.accum_train_y.shape, np.array(cur_data_y).shape
                #     )
                # )
            info("note: self.accum_train_x num_new={}, incre_train_num={}, self.accum_train_y shape={}, cur_data_y shape={}".format(
                len(self.accum_train_x),
                len(cur_incre_train_x),
                self.accum_train_y.shape,
                np.array(cur_data_y).shape
            ))

        else:
            self.tfds_convertor_sess.close()

        return cur_incre_train_x, np.array(cur_data_y)

        # return self.accum_train_x, self.accum_train_y

    def get_train_numpy_full(self):
        left_train_num = self.train_num - self.accm_train_cnt
        self.get_train_numpy(update_train_num=left_train_num)

    def get_test_numpy(self):
        if self.test_tfds is None:
            error("Error: test_tfds is None.")
            return self.accum_test_x, self.accum_test_y

        if len(self.accum_test_x) == 0:
            time_test_np_start = time.time()
            tfds_test_os_iterator = self.test_tfds.make_one_shot_iterator()
            as_timer("tfds_test_ositer")
            tfds_test_iter_next = tfds_test_os_iterator.get_next()
            time_test_os_iterator_end = time.time()
            info(
                "note: now take time_test_os_iterator_end cost_time={}s".format(
                    round(time_test_os_iterator_end - time_test_np_start, 3)
                )
            )

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                if self.domain == "text":
                    while True:
                        try:
                            example, labels = sess.run(tfds_test_iter_next)
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
                            example, labels = sess.run(tfds_test_iter_next)
                            # output:  Note:time example shape=(86401, 1, 1, 1)
                            # logger.info("Note:time example shape={}".format(example.shape))
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

        return self.accum_test_x

        # return self.accum_test_x, self.accum_test_y

    def get_speech_train_dataset(self, take_size=100):
        self.get_train_numpy(update_train_num=take_size)
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_speech_train_dataset_full(self):
        self.get_train_numpy_full()
        return {"x": [np.squeeze(x) for x in self.accum_train_x], "y": np.array(self.accum_train_y)}

    def get_speech_test_dataset(self):
        self.get_test_numpy()
        return [np.squeeze(x) for x in self.accum_test_x], self.accum_test_x

    def _nlp_2_corpus_item(self, x):
        x = x[x != -1]
        tokens = [self.nlp_index_to_token_map[int(i)] for i in x]
        document = self.nlp_sep.join(tokens)
        return document

    def _nlp_2_corpus_v0(self, x_np):
        nlp_x_corpus = []
        for x in x_np:  # each x in X is a list of indices (but as float)
            # fixme: x element?
            x = x[x != -1]
            tokens = [self.nlp_index_to_token_map[int(i)] for i in x]
            document = self.nlp_sep.join(tokens)
            nlp_x_corpus.append(document)

        info("Note: nlp2corpus mode=0, num={}".format(len(nlp_x_corpus)))
        return nlp_x_corpus

    def _nlp_2_corpus_v1(self, x_np):
        with Pool(NCPU) as pool:
            pool_res_list = pool.map(self._nlp_2_corpus_item, x_np)

        info("Note: nlp2corpus mode=1, num={}".format(len(pool_res_list)))
        return pool_res_list

    def _nlp_2_corpus_v4(self, x_np):
        with Pool(NCPU) as pool:
            # pool_res_list = pool.map(self.nlp_corpus_worker.nlp_2_corpus_item, x_np)
            pool_res_list = pool.map(self.nlp_corpus_worker.nlp_2_corpus_item, x_np)
            corpus_res = list()
            for res in pool_res_list:
                corpus_res.extend(corpus_res)

        info("Note: nlp2corpus mode=1, num={}".format(len(pool_res_list)))
        # return pool_res_list
        return corpus_res

    def _nlp_2_corpus_v2(self, x_np):
        pool_res_list = list()
        with Pool(NCPU) as pool:
            for d in x_np:
                res = pool.apply_async(self._nlp_2_corpus_item, d)
                pool_res_list.append(res)

        info("Note: nlp2corpus mode=2, num={}".format(len(pool_res_list)))
        return [r.get() for r in pool_res_list]

    def _nlp_2_corpus(self, x_np):
        """
        :param x_np:
        :return:
        """
        if NLP_2_CORPUS_MODE == 0:
            return self._nlp_2_corpus_v0(x_np)
        elif NLP_2_CORPUS_MODE == 1:
            return self._nlp_2_corpus_v1(x_np)
        elif NLP_2_CORPUS_MODE == 2:
            return self._nlp_2_corpus_v2(x_np)
        elif NLP_2_CORPUS_MODE == 4:
            return self._nlp_2_corpus_v4(x_np)
        else:
            error("ERROR: MODE is not found!")

    def get_nlp_train_dataset(self, take_size=100):
        # fixme: need to be update.
        # update self.accum_train_x and self.accum_train_y
        self.get_train_numpy(update_train_num=take_size)

        nlp_train_y_np = np.array(self.accum_train_y)
        # build nlp corpus.
        info("Note: get_nlp_train_dataset, accum_train_x_num={}".format(len(self.accum_train_x)))

        time_nlp_train_2_corpus_start = time.time()
        nlp_train_x_corpus = self._nlp_2_corpus(self.accum_train_x)
        time_nlp_train_2_corpus_end = time.time()
        info(
            "note: get_nlp_train_dataset, np2corpus MODE={}, train num={}, cost_time={}s".format(
                NLP_2_CORPUS_MODE,
                len(nlp_train_x_corpus),
                round(time_nlp_train_2_corpus_end - time_nlp_train_2_corpus_start, 3),
            )
        )

        return {"x": nlp_train_x_corpus, "y": nlp_train_y_np}

    def get_nlp_train_incre_dataset(self, take_size=100):
        # fixme: need to be update.
        # update self.accum_train_x and self.accum_train_y
        nlp_train_incre_x_list, nlp_train_incre_y_np = self.get_train_numpy(update_train_num=take_size)
        # build nlp corpus.
        nlp_train_incre_x_corpus = self._nlp_2_corpus(nlp_train_incre_x_list)
        return {"x": nlp_train_incre_x_corpus, "y": nlp_train_incre_y_np}

    def get_nlp_train_dataset_full(self):
        self.get_train_numpy_full()
        # build nlp corpus.
        if self.nlp_train_x_corpus is None:
            time_nlp_train_2_corpus_start = time.time()
            self.nlp_train_x_corpus = self._nlp_2_corpus(self.accum_train_x)
            time_nlp_train_2_corpus_end = time.time()
            info(
                "note: get_nlp_train_dataset, np2corpus train, num={}, cost_time={}s".format(
                    len(self.nlp_train_x_corpus), round(time_nlp_train_2_corpus_end - time_nlp_train_2_corpus_start, 3)
                )
            )

        return {"x": self.nlp_train_x_corpus, "y": np.array(self.accum_train_y)}

    def get_nlp_test_dataset(self):
        self.get_test_numpy()
        # build nlp corpus.

        if self.nlp_test_x_corpus is None:
            time_nlp_test_2_corpus_start = time.time()
            self.nlp_test_x_corpus = self._nlp_2_corpus(self.accum_test_x)
            time_nlp_test_2_corpus_end = time.time()
            info(
                "note: get_nlp_test_dataset, np2corpus test, num={}, cost_time={}s".format(
                    len(self.nlp_test_x_corpus), round(time_nlp_test_2_corpus_end - time_nlp_test_2_corpus_start, 3)
                )
            )
        return self.nlp_test_x_corpus, self.accum_test_x


"""
Des:
    convert tf.data.dataset to domain incrementally.

interface TfDatasetTransformer():

    def init_train_dataset(tf_train_dataset, train_num):
    
    def init_test_dataset(tf_test_dataset, test_num):
    
    def get_speech_train_dataset(take_size)
        return (x_train: list of arrays, y_train array of array)

    def get_speech_train_dataset_all()
        return (x_train: list of arrays, y_train array of array)

    def get_speech_test_dataset_()
        return (x_train: list of arrays, y_train array of array)
"""


def main():
    pass


if __name__ == "__main__":
    main()
