"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
from scipy import stats

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ""))

from nlp_autodl_config import autodl_g_conf_repr, AutoDlConf, META_SOLUS, DM_DS_PARAS
from meta_utils import feature_dict

from multiprocessing import Pool
import multiprocessing

NCPU = multiprocessing.cpu_count() - 1
print("NCPU:", NCPU)

model_dirs = ['',  # current directory
              'AutoCV/{}'.format(META_SOLUS.cv_solution),  # AutoCV/AutoCV2 winner model
              'AutoNLP/{}'.format(META_SOLUS.nlp_solution),  # AutoNLP 2nd place winner
              # 'AutoSpeech/PASA_NJU',    # AutoSpeech winner
              'AutoSpeech/{}'.format(META_SOLUS.speech_solution),  # AutoSpeech winner
              'tabular_Meysam']  # simple NN model
for model_dir in model_dirs:
    sys.path.append(os.path.join(here, model_dir))

seq_len = []
from nlp_dataset_convertor import TfDatasetsConvertor as TfDatasetTransformer
from log_utils import logger


def meta_domain_2_model(domain):
    if domain in ["image", "video"]:
        meta_solution_name = META_SOLUS.cv_solution
        if meta_solution_name == "DeepWisdom":
            from AutoCV.DeepWisdom.model import Model as AutoCVModel
        else:
            from AutoCV.kakaobrain.model import Model as AutoCVModel
        return AutoCVModel
    elif domain in ["text"]:
        meta_solution_name = META_SOLUS.nlp_solution
        if meta_solution_name == "DeepBlueAI":
            from Auto_NLP.DeepBlueAI.model import Model as AutoNLPModel
        else:
            # from AutoNLP.upwind_flys_update.run_model import Model as AutoNLPModel
            # from Auto_NLP.upwind_flys_update.run_model_clean_multi_svm import Model as AutoNLPModel
            from at_nlp.run_model import RunModel as AutoNLPModel
        return AutoNLPModel
    elif domain in ["speech"]:
        meta_solution_name = META_SOLUS.speech_solution
        if meta_solution_name == "PASA_NJU":
            from AutoSpeech.PASA_NJU.model import Model as AutoSpeechModel
        else:
            from AutoSpeech.rank_2_fuzhi.model import Model as AutoSpeechModel
        return AutoSpeechModel
    else:
        from tabular.model import Model as TabularModel
        return TabularModel


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
config.gpu_options.per_process_gpu_memory_fraction = 0.9

FIRST_SNOOP_DATA_NUM = 700

INDEX_TO_TOKENS = []
NLP_SEP = " "


def linear_sampling_func(call_num):
    return min(max(call_num * 0.08, 0), 1)


class Model():
    """A model that combine all winner solutions. Using domain inferring and
    apply winner solution in the corresponding domain."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        self.done_training = False
        self.metadata = metadata
        self.first_round_sample_maxnum = 200
        self.call_num = -1  # 0
        self.domain_dataset_train_dict = {"x": [],
                                          "y": np.array([])}
        # self.domain = infer_domain(metadata)
        self.domain = "text"
        logger.info("Note:The AutoDL_G_CONF: {}".format(autodl_g_conf_repr))
        logger.info("Note:The inferred domain of current dataset is: {}." \
                    .format(self.domain))

        # Domain识别及Model初始化
        # DomainModel = DOMAIN_TO_MODEL[self.domain]
        DomainModel = meta_domain_2_model(self.domain)

        self.domain_metadata = get_domain_metadata(metadata, self.domain)
        self.class_num = self.domain_metadata["class_num"]
        self.train_num = self.domain_metadata["train_num"]

        logger.info("Note:The domain metadata is {}".format(self.domain_metadata))
        self.domain_model = DomainModel(self.domain_metadata)
        # init for nlp
        self.nlp_index_to_token = None
        self.nlp_sep = None
        self.init_nlp()
        self.domain_model.vocab = self.vocabulary
        self.shuffle = False
        self.check_len = 0
        self.imbalance_level = -1
        # for tf_dataset.

        self.tf_dataset_trainsformer = TfDatasetTransformer(if_train_shuffle=False, config=config)
        self.tf_dataset_trainsformer.init_nlp_data(self.nlp_index_to_token, self.nlp_sep)
        self.time_record = {}
        self.seq_len = []
        self.first_round_X = []
        self.first_round_Y = np.array([])
        self.X_test_raw = None

    def init_nlp(self):
        # Retrieve vocabulary (token to index map) from metadata and construct
        # the inverse map
        self.vocabulary = self.metadata.get_channel_to_index_map()
        self.index_to_token = [None] * len(self.vocabulary)
        # self.index_to_token = {}
        for token in self.vocabulary:
            index = self.vocabulary[token]
            self.index_to_token[index] = token

        # Get separator depending on whether the dataset is in Chinese
        if self.domain_metadata["language"] == "ZH":
            self.nlp_sep = ""
            # self.nlp_sep = "" # char-level
        else:
            self.nlp_sep = " "

    def train(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train
        logger.info("Note:train_process  model.py starts train")
        # if self.call_num==0:
        #     dataset = dataset.shuffle(min(1000, self.train_num))
        start = time.time()
        self.tf_dataset_trainsformer.init_train_tfds(dataset, self.train_num)
        end = time.time()
        self.time_record["init_train_tfds"] = end - start

        if "train_num" not in self.domain_model.feature_dict:
            self.domain_model.feature_dict["train_num"] = self.train_num
            self.domain_model.feature_dict["class_num"] = self.class_num
            self.domain_model.feature_dict["language"] = self.domain_metadata['language']

        self.set_domain_dataset(dataset, is_training=True)
        logger.info("Note:train_process  model.py set domain dataset finished, domain_model train starts.")
        self.domain_model.time_record = self.time_record
        # Train the model

        # print("check domain_y", self.domain_dataset_train_dict["y"].shape)
        if self.call_num == -1:
            # self.domain_model.train_first_svm(self.domain_dataset_train_dict["x"], self.domain_dataset_train_dict["y"],
            #                     remaining_time_budget=remaining_time_budget)
            self.domain_model.train(self.domain_dataset_train_dict["x"], self.domain_dataset_train_dict["y"],
                                    remaining_time_budget=remaining_time_budget)
        else:

            self.domain_model.train(self.domain_dataset_train_dict["x"], self.domain_dataset_train_dict["y"],
                                    remaining_time_budget=remaining_time_budget)
            self.call_num += 1

        logger.info("Note:train_process  model.py domain_model train finished.")

        # Update self.done_training
        self.done_training = self.domain_model.done_training

    def test(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        start = time.time()

        self.tf_dataset_trainsformer.init_test_tfds(dataset)
        end = time.time()
        self.domain_model.time_record["init_test_tfds"] = end - start

        self.set_domain_dataset(dataset, is_training=False)

        # As the original metadata doesn't contain number of test examples, we
        # need to add this information
        if self.domain in ['text', 'speech'] and \
                (not self.domain_metadata['test_num'] >= 0):
            self.domain_metadata['test_num'] = len(self.X_test)
        logger.info("Note:test_process test domain metadata is {}".format(self.domain_metadata))

        # Make predictions
        logger.info("call num is {}".format(self.call_num))
        if self.call_num == -1:
            # Y_pred = self.domain_model.test_first_svm(self.domain_dataset_test,
            #                             remaining_time_budget=remaining_time_budget)
            Y_pred = self.domain_model.test(self.domain_dataset_test,
                                            remaining_time_budget=remaining_time_budget)
            self.call_num += 1
        else:
            Y_pred = self.domain_model.test(self.domain_dataset_test,
                                            remaining_time_budget=remaining_time_budget)
        if "test_num" not in self.domain_model.feature_dict:
            self.domain_model.feature_dict["test_num"] = self.domain_metadata['test_num']

        # Update self.done_training
        self.done_training = self.domain_model.done_training

        return Y_pred

    def check_label_coverage(self, input_y):
        _label_distribution = np.sum(np.array(input_y), 0)
        empty_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] == 0]
        normal_std = np.std(_label_distribution) / np.sum(_label_distribution)
        if empty_class_:
            label_coverage = 1-float(len(empty_class_)) / float(self.class_num)
        else:
            label_coverage = 1.0
        return label_coverage, normal_std

    def check_label_distribution(self, input_y):
        _label_distribution = np.sum(np.array(input_y), 0)
        empty_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] == 0]  # 包含样本量为0的类别
        self.kurtosis = stats.kurtosis(_label_distribution)
        self.normal_std = np.std(_label_distribution) / np.sum(_label_distribution)
        logger.info("check input_y kurtosis {}".format(self.kurtosis))
        logger.info("check input_y class: {} and normal_std is {}".format(empty_class_, self.normal_std))
        if len(empty_class_) == 0:  # No empty label, all label covered!
            self.shuffle = False
        else:
            self.shuffle = True
        if self.normal_std > 0.3:
            # fixme: 针对类别极不均衡
            self.imbalance_level = 2  #
        elif self.normal_std > 0.07:
            # fixme：针对类别不均衡
            self.imbalance_level = 1
        else:
            # fixme: 类别均衡
            self.imbalance_level = 0

        # if (float(len(empty_class_)) / float(self.class_num))<1.0:
        #     self.shuffle = True

    def check_input_length(self, input_x):
        check_seq_len = []
        for x in input_x:
            x = x[x != -1]
            check_seq_len.append(x.shape[0])
        self.check_len = np.average(check_seq_len)
        self.check_len_std = np.std(check_seq_len)

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################
    def decide_first_num(self):
        snoop_data_num = min(0.01 * self.train_num, FIRST_SNOOP_DATA_NUM)  # 第一次最多取700
        snoop_X, snoop_Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(snoop_data_num)
        label_coverage, normal_std = self.check_label_coverage(snoop_Y)
        self.check_input_length(snoop_X[:FIRST_SNOOP_DATA_NUM])
        logger.info("label_coverage is {}".format(label_coverage))
        if normal_std>0.3:
            dataset_read_num = min(5000, int(0.1 * self.train_num))
        else:
            if self.class_num == 2 and self.train_num <= 50000:
                if label_coverage == 1.0:  # 类别均匀覆盖
                    dataset_read_num = max(int(0.01 * self.train_num), 500)
                    # 设置小样本下限
                    if self.train_num <= 10000:
                        dataset_read_num = min(5000, self.domain_metadata["class_num"] * 3000)
                else:  # snoop类别有缺失, 可能为顺序进样
                    dataset_read_num = min(5000, int(0.1 * self.train_num))

            elif self.class_num == 2 and self.train_num > 50000:
                if label_coverage == 1.0:  # 类别均匀覆盖
                    # 不超过10w的数据集，取1%, 超过10w的数据集，取1000上限
                    dataset_read_num = min(int(0.01 * self.train_num), 1000)
                else:  # snoop类别有缺失, 可能为顺序进样
                    dataset_read_num = min(5000, int(0.1 * self.train_num))

            ########################### 多分类 ######################################
            elif self.class_num > 2 and self.train_num <= 50000:
                if label_coverage == 1.0:  # 类别均匀覆盖
                    dataset_read_num = min(int((2 / self.class_num) * self.train_num), 1000)
                    # 设置小样本下限
                    if self.train_num <= 10000:
                        dataset_read_num = min(5000, self.domain_metadata["class_num"] * 3000)
                else:
                    dataset_read_num = min(5000, int(0.1 * self.train_num))
            elif self.class_num > 2 and self.train_num > 50000:
                if label_coverage == 1.0:  # 类别均匀覆盖
                    # 不超过10w的数据集，取1%, 超过10w的数据集，取1500上限
                    dataset_read_num = min(int((2 / self.class_num) * self.train_num), 1500)
                else:  # snoop类别有缺失, 可能为顺序进样
                    dataset_read_num = min(5000, int(0.1 * self.train_num))
            ########################### 多分类 ######################################
                if self.domain_metadata["language"] == "ZH" and self.check_len<=40:
                    dataset_read_num += min(2000, 0.1*self.train_num)

        X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
        X = X + snoop_X
        Y = np.concatenate([Y, snoop_Y], axis=0)
        return dataset_read_num, X, Y

    def set_domain_dataset(self, dataset, is_training=True):
        """Recover the dataset in corresponding competition format (esp. AutoNLP
        and AutoSpeech) and set corresponding attributes:
          self.domain_dataset_train
          self.domain_dataset_test
        according to `is_training`.
        """
        # self.dataset = None
        if is_training:
            subset = 'train'
        else:
            subset = 'test'
        attr_dataset = 'domain_dataset_{}'.format(subset)

        if not hasattr(self, attr_dataset):
            logger.info("Note: Begin recovering dataset format in the original " +
                        "competition for the subset: {}...".format(subset))
            if self.domain == 'text':
                if DM_DS_PARAS.text.if_sample and is_training:

                    # dataset_read_num = min(5000, self.domain_metadata["class_num"] * 3000)
                    # if self.train_num >= 10000:
                    #     dataset_read_num = min(dataset_read_num, int(0.1 * self.train_num))
                    dataset_read_num, X, Y = self.decide_first_num()
                    logger.info(
                        "Note: set_domain_dataset text, dataset sampling, shuffle and take starts, train_read_num = {}".format(
                            dataset_read_num))

                    # Get X, Y as lists of NumPy array
                    start = time.time()
                    self.check_label_distribution(np.array(Y))
                    end = time.time()
                    self.time_record["check_label_distribution"] = end - start

                    self.domain_model.imbalance_level = self.imbalance_level

                    feature_dict["check_len"] = float(self.check_len)
                    feature_dict["kurtosis"] = float(self.kurtosis)
                    feature_dict["first_detect_normal_std"] = float(self.normal_std)
                    feature_dict["imbalance_level"] = self.imbalance_level
                    feature_dict["is_shuffle"] = self.shuffle

                    logger.info("Note: update domain model imbalace level after first detect!")
                    # if self.check_len <= 40 or self.normal_std>=0.2:
                    #     dataset_read_num += min(0.2*self.train_num, 12000)

                    if self.shuffle and self.domain_metadata["language"] == "ZH":
                        self.shuffle = False
                        dataset_read_num = int(0.4 * self.train_num)
                        start = time.time()
                        _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                        X = X + _X
                        Y = np.concatenate([Y, _Y], axis=0)
                        end = time.time()
                        self.time_record["get_nlp_train_dataset_new"] = end - start

                        _label_distribution = np.sum(Y, 0)
                        occu_class_ = [i for i in range(_label_distribution.shape[0]) if _label_distribution[i] != 0] # 已经拿到的label类别

                        if len(occu_class_)>=2:
                            pass
                        else:
                            # 再多取20%
                            dataset_read_num = int(0.2 * self.train_num)
                            logger.info("Use extra 20% sample: Class num < 2 for ZH data!")
                            _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                            X = X + _X
                            Y = np.concatenate([Y, _Y], axis=0)
                            _label_distribution = np.sum(Y, 0)
                            occu_class_ = [i for i in range(_label_distribution.shape[0]) if
                                           _label_distribution[i] != 0]
                            if len(occu_class_)<2:
                                logger.info("Use extra 100% sample: Class num < 2!")
                                dataset_read_num = int(self.train_num)
                                _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                                X = X + _X
                                Y = np.concatenate([Y, _Y], axis=0)

                    ######################### 原始英文shuffle逻辑 ####################
                    if self.shuffle:
                        logger.info("Note: start shuffle dataset due to not enough labels!")
                        # redo take
                        start = time.time()
                        del self.tf_dataset_trainsformer
                        self.tf_dataset_trainsformer = TfDatasetTransformer(if_train_shuffle=True, config=config)
                        end = time.time()
                        self.time_record["del trainsformer and init"] = end - start

                        start = time.time()
                        shuffle_size = max(int(0.5 * (self.train_num)), 10000)

                        shuffle_dataset = dataset.shuffle(shuffle_size)
                        end = time.time()
                        self.time_record["shuffle dataset"] = end - start

                        start = time.time()
                        self.tf_dataset_trainsformer.init_train_tfds(shuffle_dataset, self.train_num, pad_num=20)
                        end = time.time()
                        self.time_record["init_new_train_tfds"] = end - start

                        start = time.time()
                        X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)

                        _label_distribution = np.sum(Y, 0)
                        occu_class_ = [i for i in range(_label_distribution.shape[0]) if
                                       _label_distribution[i] != 0]  # 已经拿到的label类别
                        if len(occu_class_) >= 2:
                            pass
                        else:
                            logger.info("Use extra 100% sample: Class num < 2 for EN data!")
                            dataset_read_num = int(1 * (self.train_num))
                            _X, _Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                            X = X + _X
                            Y = np.concatenate([Y, _Y], axis=0)
                        end = time.time()
                        self.time_record["get_nlp_train_dataset_new"] = end - start

                        logger.info("Note: finish take after shuffle dataset")
                    ###################################################################

                    logger.info(
                        "Note: set_domain_dataset text, dataset sampling, shuffle and take ends, train_read_num = {}".format(
                            dataset_read_num))
                    # self.domain_model.vocab = self.vocabulary
                    self.domain_model.avg_word_per_sample = float(
                        len(self.vocabulary) / self.domain_metadata["train_num"])
                    if "avg_word_per_sample" not in feature_dict:
                        feature_dict["avg_word_per_sample"] = self.domain_model.avg_word_per_sample
                    self.domain_model.feature_dict = feature_dict
                    logger.info(
                        "Note: vocab size is {} and avg_word_per_sample is {}".format(len(self.domain_model.vocab),
                                                                                      self.domain_model.avg_word_per_sample))
                elif not is_training:
                    start = time.time()
                    pad_num = 20
                    logger.info("pad num is {}".format(pad_num))
                    X, Y = self.tf_dataset_trainsformer.get_nlp_test_dataset(pad_num=pad_num)
                    # self.X_test_raw = X
                    end = time.time()
                    self.domain_model.time_record["get_nlp_test_dataset_numpy_test"] = end - start

                if is_training:
                    self.first_round_X = X
                    self.first_round_Y = Y
                # Construct the corpus
                start = time.time()

                # 不转 corpus
                if self.call_num == 0:
                    corpus = []
                    seq_len = []

                    for _x in X:
                        _x = _x[_x != -1]
                        num_words = max(int(_x.shape[0] * 0.1), 301)
                        _x = _x[:num_words]
                        _x = _x.astype(str)
                        tokens = _x.tolist()
                        document = self.nlp_sep.join(tokens)
                        corpus.append(document)

                    logger.info("USE id as corpus {}")
                else:
                    corpus, seq_len = to_corpus(X, self.index_to_token, self.nlp_sep)
                    logger.info("USE word as corpus {}")

                end = time.time()
                self.seq_len = seq_len
                if is_training:
                    logger.info("to_corpus_train cost {}".format(end - start))
                    self.domain_model.time_record["to_corpus_train"] = end - start
                else:
                    logger.info("to_corpus_test cost {}".format(end - start))
                    self.domain_model.time_record["to_corpus_test"] = end - start
                # Construct the dataset for training or test
                if is_training:
                    labels = np.array(Y)
                    cnt = np.sum(np.count_nonzero(labels, axis=1), axis=0)
                    print("Check multi-label cnt {}".format(cnt))
                    if cnt > labels.shape[0]:
                        print("Check multi-label: True")
                        self.domain_model.multi_label = True
                        # self.domain_model.fasttext_embeddings_index = None
                        self.domain_model.db_model = None
                        self.domain_model.ft_model = None
                    domain_dataset = corpus, labels
                    # Set the attribute
                    self.domain_dataset_train_dict["x"] = corpus
                    self.domain_dataset_train_dict["y"] = labels
                else:
                    domain_dataset = corpus
                    # Set the attribute
                    self.domain_dataset_train_dict["x"] = corpus
                    self.X_test = corpus

                setattr(self, attr_dataset, domain_dataset)

            elif self.domain == 'speech':
                # Set the attribute
                setattr(self, attr_dataset, dataset)

            elif self.domain in ['image', 'video', 'tabular']:
                setattr(self, attr_dataset, dataset)
            else:
                raise ValueError("The domain {} doesn't exist.".format(self.domain))

        else:
            if subset == 'test':
                if self.X_test_raw:
                    self.domain_dataset_test, test_seq_len = to_corpus(self.X_test_raw, self.index_to_token,
                                                                       self.nlp_sep)
                self.X_test_raw = None
                return

            if self.domain == 'text':
                if DM_DS_PARAS.text.if_sample and is_training:
                    if self.domain_model.multi_label:
                        self.domain_model.use_multi_svm = True
                        self.domain_model.start_cnn_call_num = 2
                        dataset_read_num = self.train_num
                        if dataset_read_num>50000:
                            dataset_read_num = 50000
                            logger.info(" Set Upper limit!")
                    else:
                        if self.imbalance_level >= 1:
                            dataset_read_num = self.train_num
                            self.domain_model.use_multi_svm = False
                            self.domain_model.start_cnn_call_num = 1
                            if dataset_read_num > 50000:
                                dataset_read_num = 50000
                                logger.info(" Set Upper limit!")
                        else:
                            self.domain_model.use_multi_svm = True
                            if self.call_num <= self.domain_model.start_first_stage_call_num - 1:
                                dataset_read_num = 3000
                                if self.check_len <= 40 or self.normal_std >= 0.2:
                                    dataset_read_num += min(int(0.2 * self.train_num), 12000)
                            else:
                                # dataset_read_num = int(self.domain_metadata["train_num"] * linear_sampling_func(self.call_num))
                                if self.call_num == self.domain_model.start_first_stage_call_num:
                                    dataset_read_num = int(0.9 * self.domain_metadata["train_num"])
                                    if dataset_read_num > 50000:
                                        dataset_read_num = 50000
                                else:
                                    if self.train_num <= 55555:
                                        dataset_read_num = 4000
                                    else:
                                        dataset_read_num = 5500

                    logger.info(
                        "Note: set_domain_dataset text, dataset sampling, shuffle and take starts, train_read_num = {}".format(
                            dataset_read_num))
                    # Get X, Y as lists of NumPy array
                    start = time.time()
                    X, Y = self.tf_dataset_trainsformer.get_nlp_train_dataset(dataset_read_num)
                    end = time.time()
                    # if self.call_num == 0:
                    #     logger.info("Use first round data!")
                    #     X = self.first_round_X + X
                    #     Y = np.concatenate([self.first_round_Y, Y], axis=0)
                    if self.call_num == 1:
                        self.time_record["get_nlp_train_dataset_to_numpy call_num=1"] = end - start
                    logger.info(
                        "Note: set_domain_dataset text, dataset sampling, shuffle and take ends, train_read_num = {}".format(
                            dataset_read_num))

                # Construct the corpus
                corpus = []
                start = time.time()

                corpus, seq_len = to_corpus(X, self.index_to_token, self.nlp_sep)
                end = time.time()
                self.seq_len.extend(seq_len)
                # self.time_record["to_corpus when call_num=1"] = end-start
                if "avg_length" not in self.domain_model.feature_dict:
                    self.domain_model.feature_dict["avg_length"] = int(np.average(self.seq_len))
                    self.domain_model.feature_dict["max_length"] = int(np.max(self.seq_len))
                    self.domain_model.feature_dict["min_length"] = int(np.min(self.seq_len))
                    self.domain_model.feature_dict["seq_len_std"] = int(np.std(self.seq_len))

                if self.domain_model.max_length == 0:
                    if int(np.max(self.seq_len)) <= 301:
                        self.domain_model.max_length = int(np.max(self.seq_len))
                        self.domain_model.bert_check_length = int(np.max(self.seq_len))

                    else:
                        self.domain_model.max_length = int(np.average(self.seq_len))
                        self.domain_model.bert_check_length = int(np.average(self.seq_len))

                    self.domain_model.seq_len_std = int(np.std(self.seq_len))
                if self.seq_len:
                    logger.info("Note: set domain_model max_length = {}".format(
                        self.domain_model.max_length))

                    logger.info("Note: check domain_model max_length = {}".format(
                        int(np.max(self.seq_len))))
                    logger.info("Note: check domain_model max_length std = {}".format(
                        int(np.std(self.seq_len))))
                # Construct the dataset for training or test
                if is_training:
                    labels = np.array(Y)
                    domain_dataset = corpus, labels
                    print("\n upadte domain_dataset \n")
                    print("check domain_dataset_train_dict y:", labels.shape)
                    self.domain_dataset_train_dict["x"] = corpus
                    self.domain_dataset_train_dict["y"] = labels
                    # print(self.domain_dataset_train)
                    # self.domain_dataset_train = domain_dataset
                else:
                    domain_dataset = corpus


def parallel_to_corpus(dat, worker_num=NCPU, partition_num=100, index_to_token=INDEX_TO_TOKENS, nlp_sep=NLP_SEP):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)
    data = [p.apply_async(func=to_corpus, args=(x, index_to_token, nlp_sep)) for x in sub_data_list]
    p.close()
    flat_data = [p.get() for p in data]

    seq_len = [size for chunk, word_list in flat_data for size in word_list]
    flat_data = [_ for chunk, word_list in flat_data for _ in chunk]
    return flat_data, seq_len


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
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def to_corpus(x, index_to_token=INDEX_TO_TOKENS, nlp_sep=NLP_SEP):
    corpus = []
    seq_len = []
    for _x in x:
        _x = _x[_x != -1]
        tokens = [index_to_token[i] for i in _x]
        seq_len.append(len(tokens))
        document = nlp_sep.join(tokens)
        corpus.append(document)
    return corpus, seq_len


def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor.

    Args:
      metadata: an AutoDLMetadata object.
    """
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = metadata.get_channel_to_index_map()
    domain = None
    if sequence_size == 1:
        if row_count == 1 or col_count == 1:
            domain = "tabular"
        else:
            domain = "image"
    else:
        if row_count == 1 and col_count == 1:
            if len(channel_to_index_map) > 0:
                domain = "text"
            else:
                domain = "speech"
        else:
            domain = "video"
    return domain


def is_chinese(tokens):
    """Judge if the tokens are in Chinese. The current criterion is if each token
    contains one single character, because when the documents are in Chinese,
    we tokenize each character when formatting the dataset.
    """
    is_of_len_1 = all([len(t) == 1 for t in tokens[:100]])
    # fixme: use ratio instead of 'all'
    num = [1 for t in tokens[:100] if len(t) == 1]
    ratio = np.sum(num) / 100
    if ratio > 0.95:
        return True
    else:
        return False
    # return is_of_len_1


def get_domain_metadata(metadata, domain, is_training=True):
    """Recover the metadata in corresponding competitions, esp. AutoNLP
    and AutoSpeech.

    Args:
      metadata: an AutoDLMetadata object.
      domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
    """
    if domain == 'text':
        # Fetch metadata info from `metadata`
        class_num = metadata.get_output_size()
        num_examples = metadata.size()
        channel_to_index_map = metadata.get_channel_to_index_map()
        revers_map = {v: k for k, v in channel_to_index_map.items()}
        tokens = [revers_map[int(id)] for id in range(100)]
        language = 'ZH' if is_chinese(tokens) else 'EN'
        time_budget = 1200  # WARNING: Hard-coded

        # Create domain metadata
        domain_metadata = {}
        domain_metadata['class_num'] = class_num
        if is_training:
            domain_metadata['train_num'] = num_examples
            domain_metadata['test_num'] = -1
        else:
            domain_metadata['train_num'] = -1
            domain_metadata['test_num'] = num_examples
        domain_metadata['language'] = language
        domain_metadata['time_budget'] = time_budget

        return domain_metadata

    else:
        return metadata
