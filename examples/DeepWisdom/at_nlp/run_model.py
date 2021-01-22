import numpy as np
import math
import gc

import tensorflow as tf
from keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, LearningRateScheduler

from at_toolkit.at_utils import autodl_nlp_install_download

autodl_nlp_install_download()

from log_utils import info
from at_nlp.data_manager.data_sampler import DataGenerator
from at_nlp.data_manager.sample_utils import downsampling_input_data
from at_nlp.utils import color_msg, ohe2cat
from at_nlp.model_manager.emb_utils import _load_emb
from at_nlp.data_manager.sample_config import sample_strategy
from at_nlp.generators.model_generator import ModelGenerator
from at_nlp.generators.feature_generator import FeatureGenerator
from at_nlp.generators.data_generator import DataGenerator as BatchDataGenerator
from at_nlp.data_manager.preprocess_utils import _tokenize_chinese_words
from at_nlp.evaluator import Evaluator
from at_toolkit.at_evalator import ATEvaluator

from Auto_NLP.deepWisdom.pytf_finetune_bert import FineTuneBertModel
from Auto_NLP.DeepBlueAI.model_iter_db import Model as DB_Model

INIT_BATCH_SIZE = 32
MAX_SVM_FIT_NUM = 20000
MAX_BERT_FIT_NUM = 3000
MAX_EPOCH_NUM = 120

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)


class RunModel(object):
    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.multi_label = False
        self.multi_label_cnt_thred = 10
        self.fasttext_embeddings_index = None
        # self.load_pretrain_emb = False
        self.load_pretrain_emb = True
        if self.load_pretrain_emb:
            self.fasttext_embeddings_index = _load_emb(self.metadata["language"])

        self.model = None
        self.accu_nn_tokenizer_x = True

        self.db_model = DB_Model(self.metadata, fasttext_emb=self.fasttext_embeddings_index)

        ################# BERT 相关 #########################
        finetune_classifer = FineTuneBertModel(metadata=self.metadata)
        self.ft_model = finetune_classifer
        self.bert_check_length = 0
        self.start_ft_bert = False
        self.start_ft_bert_call_num = 0
        self.bert_output_patience = 3
        self.bert_auc = 0.0
        self.best_val_auc = 0.0
        self.update_bert = False
        self.best_bert_pred = []
        self.bert_result = []
        self.use_pretrain_model = False
        ################# BERT 相关 #########################

        ################# db model 相关 #########################
        self.use_db_model = False
        self.start_db_model = False
        self.db_patience = 0
        self.db_patience_max_num = 8
        ################# db model 相关 #########################

        self.call_num = 0
        ## run model 配置有一个采样器，一个模型选择器
        self.data_sampler = None
        self.model_selector = None
        self.imbalance_level = -1
        self.start_cnn_call_num = 3

        self.split_val_x = False
        self.imbalance_flow_control = 1
        self.feature_dict = {}
        self.vocab = {}
        self.time_record = {}
        self.imbalance_level = -1
        self.avg_word_per_sample = 0
        self.use_multi_svm = True
        self.max_length = 0
        self.bert_check_length = 0
        self.seq_len_std = 0

        self.val_x = []
        self.clean_val_x = []
        self.val_y = np.array([])
        self.build_tokenizer_x = []
        self.build_tokenizer_y = np.array([])

        self.first_stage_done = False
        self.second_stage_done = False
        self.third_stage_done = False
        self.start_first_stage_call_num = 3  # 3
        self.tokenize_test = False

        self.switch_new_model = True
        self.switch_new_feature_mode = False
        self.cur_model_train_start = False
        self.train_epoch = 0
        self.max_train_epoch = 3
        self.model_id = 0
        self.train_model_id = -1
        self.feature_id = -1

        self.best_auc = 0.0

        self.model_lib = ['text_cnn']

        self.feature_lib = ['char-level + 300dim-embedding',
                            'word-level + pretrained embedding300dim']

        self.use_ft_model = False

        self.callbacks = []
        normal_lr = LearningRateScheduler(self.lr_decay)
        step_lr = LearningRateScheduler(self.step_decay)
        self.callbacks_ = [normal_lr, step_lr]
        self.cur_lr = None

        self.test_result_list = [0] * 30
        self.cur_model_test_res = []
        self.svm_test_result = []
        self.model_weights_list = [[]] * 20
        self.hist_test = [[]] * 20

        self.is_best = False
        self.best_call_num = 0
        self.best_svm = 0.0
        self.best_cnt = []
        self.best_svm_scores = []

        self.batch_size = INIT_BATCH_SIZE

    def _show_runtime_info(self):
        info(color_msg("********************************************************"))
        info(color_msg("current model_id is {}, model_name is {}".format(self.model_id, self.model_lib[self.model_id])))
        info(color_msg(
            "current feature_id is {}, feature_name is {}".format(self.feature_id, self.feature_lib[self.feature_id])))
        info(color_msg("train_model_id is {}".format(self.train_model_id)))
        info(color_msg("********************************************************\n"))

    def _show_data_info(self, train_diff_x, val_diff_x):
        if val_diff_x:
            info(color_msg("train_x size is {} and val_x size is {}".format(len(train_diff_x),
                                                                            len(val_diff_x)), color='green'))
        else:
            info(color_msg("train_x size is {} and val_x size is {}".format(len(train_diff_x), 0), color='green'))
        info(color_msg("Note: current full val_x size is {}".format(self.val_y.shape[0]), color='green'))

    def _show_ensemble_info(self):
        if self.is_best:
            info(color_msg("ensemble new model!", color='yellow'))
        else:
            info(color_msg("do not ensemble new model!", color='yellow'))
        info(color_msg("use models : {}".format(self.model_lib)))

    def _set_sampling_strategy(self, y_train):
        strategy = sample_strategy['sample_iter_incremental_no_train_split']

        if y_train.shape[0] > 0:  # 如果当前有增量数据进样
            if self.call_num == 0 or self.call_num >= self.start_cnn_call_num:
                strategy = sample_strategy['sample_iter_incremental_no_train_split']
                info(color_msg("strategy is sample_iter_incremental_no_train_split"))

            elif self.call_num < self.start_cnn_call_num:
                strategy = sample_strategy["sample_iter_incremental_with_train_split"]
                info(color_msg("strategy is sample_iter_incremental_with_train_split"))

            if self.start_cnn_call_num == self.imbalance_flow_control and not self.split_val_x:
                strategy = sample_strategy["sample_from_full_data"]
                info(color_msg("strategy is sample_from_full_data"))

        else:  # 当前已无增量数据
            if self.val_y.shape[0] > 0:  # 如果已经有val数据集
                strategy = sample_strategy["sample_from_full_train_data"]
                info(color_msg("strategy is sample_from_full_train_data"))
            else:
                strategy = sample_strategy["sample_from_full_data"]
                info(color_msg("strategy is sample_from_full_data"))

        info(color_msg("call num is {}: add_val_to_train is {}, "
                       "update_train is {}, use_full is {}".format(self.call_num,
                                                                   strategy["add_val_to_train"],
                                                                   strategy["update_train"],
                                                                   strategy["use_full"])))
        return strategy

    def do_data_sampling(self, y_train):
        strategy = self._set_sampling_strategy(y_train)
        return self.data_manager.sample_dataset_iter(add_val_to_train=strategy["add_val_to_train"],
                                                     update_train=strategy["update_train"],
                                                     use_full=strategy["use_full"])

    def run_svm(self, model_name, train_x, train_y):
        self.feature_generator.tokenizer = None
        # if self.metadata["language"] == "ZH" and self.call_num == 0:
        if self.metadata["language"] == "ZH" and self.call_num <= 1:
            analyzer = "char"
        else:
            analyzer = "word"
        self.feature_generator.build_tokenizer(train_x, 'svm', analyzer)
        # 后处理，将文本数据转换为 fidf feature
        if len(train_x) > MAX_SVM_FIT_NUM:
            info("Redo sample size limitation for SVM! Use up to 20000 samples")
            train_x = train_x[:MAX_SVM_FIT_NUM]
            train_y = train_y[:MAX_SVM_FIT_NUM, :]

        train_data = self.feature_generator.postprocess_data(train_x)
        info(color_msg("model type is {}, train_feature.shape:{}".format('svm', train_data.shape)))
        classifier = self.model_manager.select_classifier(model_name=model_name, feature_mode=None,
                                                          data_feature=self.feature_generator.data_feature)

        self.svm_token = self.feature_generator.tokenizer
        if self.multi_label:
            classifier.fit(train_data, train_y)
        else:
            classifier.fit(train_data, ohe2cat(train_y))
        return classifier

    def prepare_nn_tokenizer(self, train_x):
        self.feature_generator.build_tokenizer(train_x, model_name='nn')
        self.feature_generator.set_data_feature()
        self.tokenizer = self.feature_generator.tokenizer

    def model_fit(self, train_x, train_y, model):
        num_epochs = 1
        bs_training_generator = BatchDataGenerator(train_x, train_y, batch_size=self.batch_size,
                                                   language=self.metadata["language"],
                                                   max_length=self.feature_generator.max_length
                                                   if self.feature_generator.max_length else 100,
                                                   vocab=None,
                                                   tokenizer=self.tokenizer,
                                                   num_features=self.feature_generator.num_features)

        model.fit_generator(generator=bs_training_generator, verbose=1,
                            epochs=num_epochs,
                            callbacks=self.callbacks,
                            shuffle=True)
        return model

    def init_generators(self, x_train, y_train):
        self.data_manager = DataGenerator(x_train, y_train, self.metadata, self.imbalance_level, self.multi_label)
        self.model_manager = ModelGenerator(load_pretrain_emb=self.load_pretrain_emb,
                                            fasttext_embeddings_index=self.fasttext_embeddings_index,
                                            multi_label=self.multi_label)
        # ZH 前期默认不切分词，后期需切词
        self.feature_generator = FeatureGenerator(self.metadata["language"], do_seg=False,
                                                  num_class=self.metadata["class_num"])

        self.evaluator = Evaluator(self.clean_val_x, self.val_y)

    def process_val_data(self, val_diff_x, val_diff_y):
        # 处理增量val数据
        if val_diff_x:  # 如果有val数据，添加到全局的val数据里
            # todo: 切换前处理方式时，清空clean_val_x, 重新进行前处理
            clean_val_x = self.feature_generator.preprocess_data(val_diff_x)
            if self.val_y.shape[0] > 0:
                self.val_x = np.concatenate([self.val_x, val_diff_x], axis=0)
                self.val_y = np.concatenate([self.val_y, val_diff_y], axis=0)
                self.clean_val_x = np.concatenate([self.clean_val_x, clean_val_x], axis=0)

            else:
                self.val_x = val_diff_x
                self.val_y = val_diff_y
                self.clean_val_x = clean_val_x
            return clean_val_x
        else:
            return None

    def _reset_train_status(self):
        self.cur_model_train_start = False
        self.switch_new_model = True
        self.cur_lr = None
        self.train_epoch = 0
        self.best_cnt = []
        self.model_weights_list[self.train_model_id] = self.evaluator.model_weights_list
        self.cur_model_test_res = []
        self.evaluator.max_epoch = 5
        self.evaluator._reset()

    def _clear_train_space(self):
        del self.model
        gc.collect()
        K.clear_session()

    def _init_nn_train_process(self):
        self.feature_id += 1
        self.train_model_id = self.model_id + self.feature_id * len(self.model_lib)
        self._show_runtime_info()

        self.model_weights_list[self.train_model_id] = []
        self.cur_model_train_start = True
        self.switch_new_model = False
        self.train_epoch = 0
        self.callbacks = []

    def do_evaluation(self, eval_svm=True, update_val=True, update_setting=True):
        if eval_svm:
            tokenizer = self.svm_token
        else:
            tokenizer = self.tokenizer

        if update_val:
            self.evaluator.update_val_data(self.clean_val_x, self.val_y)
        if update_setting:
            self.evaluator.update_setting(self.metadata["language"],
                                          self.feature_generator.max_length,
                                          self.feature_generator.num_features,
                                          tokenizer)

        self.evaluator.valid_auc(is_svm=eval_svm, model=self.model, use_autodl_auc=True)

    def _train_nn_process(self, train_preprocessed_data, train_diff_y):
        self.model = self.model_fit(train_preprocessed_data, train_diff_y, self.model)
        if self.cur_model_train_start:  # 如果当前模型正在训练，评估每一次模型结果
            if self.call_num == self.start_first_stage_call_num:
                self.do_evaluation(eval_svm=False, update_val=True, update_setting=True)
            else:
                self.do_evaluation(eval_svm=False, update_val=False, update_setting=False)
            self.evaluator.update_model_weights(self.model, self.train_epoch)

            self.train_epoch += 1

    def _ensemble_multi_models(self):
        info(color_msg("cur best_auc is {} and val_auc is {}".format(self.best_auc, self.evaluator.best_auc),
                       color='yellow'))
        # 对于SVM及第一个NN模型结果，要求比前一次评估高
        if self.call_num <= self.start_first_stage_call_num:
            ensemble_condition = self.evaluator.best_auc > self.best_auc
        else:
            # 对于第二个NN模型结果，要求达到前一次NN模型效果的97%
            ensemble_condition = self.evaluator.best_auc > 0.97 * self.best_auc

        if ensemble_condition:
            self.is_best = True  # 允许多模型融合
            self.best_auc = max(self.evaluator.best_auc, self.best_auc)

        else:
            self.is_best = False

        self.best_cnt.append(self.is_best)

        if self.call_num < self.start_first_stage_call_num:
            self.best_svm = self.best_auc
            self.best_svm_scores.append(self.best_svm)

        self._show_ensemble_info()

    def is_done(self):
        if self.model_id == len(self.model_lib) - 1:
            # 流程：检查是否结束全流程 --> 检查是否进入bert阶段 -->
            if self.feature_id == len(self.feature_lib) - 1:
                self.first_stage_done = True

            else:
                self.model_id = -1
                self.switch_new_feature_mode = True

        if self.switch_new_feature_mode:
            self.feature_id += 1
            self.switch_new_feature_mode = False

    def is_stage_done(self):
        info(color_msg("check stage: model id is {}, feature_id is {}".format(self.model_id, self.feature_id)))
        if self.model_id == len(self.model_lib) - 1:
            if self.feature_id == len(self.feature_lib) - 1:
                self.first_stage_done = True

    def meta_strategy(self):

        if self.max_length <= 156 and self.avg_word_per_sample <= 12:
            self.use_ft_model = True
            self.evaluator.max_epoch = 8

        elif self.max_length > 156:
            self.use_ft_model = False
            self.db_patience_max_num = 8
            self.feature_lib = ['char-level + 300dim-embedding',
                                'word-level + pretrained embedding300dim']
            # self.feature_lib.pop(1)

        if self.imbalance_level == 2 and self.metadata["language"] == "EN":
            self.feature_lib = ['char-level + 300dim-embedding']
            # self.feature_lib.pop(1)

        if self.metadata["language"] == "ZH":
            self.feature_lib = ['word-level + pretrained embedding300dim',
                                'char-level + 300dim-embedding']
            self.db_patience_max_num = 8

        if self.multi_label:
            self.feature_lib = ['char-level + 300dim-embedding']

    def _update_build_tokenizer_data(self):
        pass

    def prepare_clean_data(self, train_diff_x, val_diff_x, val_diff_y):
        # 前处理：根据给定的前处理方式 清洗文本数据, 先默认default
        train_preprocessed_data = self.feature_generator.preprocess_data(train_diff_x)
        self.process_val_data(val_diff_x, val_diff_y)
        if self.call_num == 2 and self.metadata["language"] == "ZH":
            info(color_msg("do tokenization for val data!"))
            self.clean_val_x = list(map(_tokenize_chinese_words, self.clean_val_x))

        if self.accu_nn_tokenizer_x:
            # if self.call_num == 1 and self.metadata["language"] == "ZH":  # 前2次都不用
            if self.call_num == 2 and self.metadata["language"] == "ZH":
                # 不用ZH第一次的数据(char-level)
                self.build_tokenizer_x = train_preprocessed_data
            else:

                self.build_tokenizer_x = self.build_tokenizer_x + train_preprocessed_data

        return train_preprocessed_data

    def _train_bert_process(self, train_x, train_y):
        if self.bert_check_length <= 64:
            self.ft_model.finetune_config.max_seq_length = 64
        else:
            self.ft_model.finetune_config.max_seq_length = 128

        if not self.start_ft_bert:
            self.ft_model.finetune_config.num_train_epochs = 2
            self.ft_model.finetune_config.per_gpu_train_batch_size = 16
            self.ft_model.finetune_config.warmup_steps = int(0.1 * (3000 // 16 * 2))
            if self.ft_model.finetune_config.max_seq_length==128:
                self.ft_model.finetune_config.learning_rate = 2e-5
                self.ft_model.finetune_config.per_gpu_train_batch_size = 8
                self.ft_model.finetune_config.warmup_steps = int(0.05 * (3000 // 8 * 2))

            gc.collect()
            K.clear_session()
            #
            self.db_model = None
            self.start_ft_bert = True
            self.start_ft_bert_call_num = self.call_num
            info("start_ft_bert_call_num is {}".format(self.start_ft_bert_call_num))

        if self.call_num >= self.start_ft_bert_call_num + 2:  #
            self.ft_model.finetune_config.learning_rate = max(3 * self.ft_model.finetune_config.learning_rate / 5, 1e-5)
            self.ft_model.finetune_config.num_train_epochs = 1
            self.ft_model.finetune_config.warmup_steps = 0

        if len(train_x) > MAX_BERT_FIT_NUM:
            max_num = int(float(MAX_BERT_FIT_NUM) / float(self.metadata["class_num"]))
            _x_train, _y_train = downsampling_input_data(train_x, train_y, self.metadata["class_num"],
                                                         max_sample_num=max_num)
        else:
            _x_train, _y_train = train_x, train_y

        ft_model = self.ft_model.train_model_process(_x_train, ohe2cat(_y_train), self.ft_model.model)
        y_eval = self.ft_model.model_eval_process(self.clean_val_x, ohe2cat(self.val_y), ft_model)
        bert_auc = ATEvaluator.autodl_auc(solution=self.val_y, prediction=y_eval)
        info(color_msg("bert_auc is {} and best bert_auc is {}".format(bert_auc, self.bert_auc)))
        if bert_auc > self.bert_auc:
            info("update bert ft model!\n ")
            # 仅考虑连续auc不上升的case，当auc出现更优结果，又重新计算patience
            self.bert_output_patience = 3
            self.update_bert = True
            self.bert_auc = bert_auc
        else:
            self.bert_output_patience -= 1
            self.update_bert = False

        if self.bert_auc > self.best_auc:
            self.use_pretrain_model = True
            self.selcet_svm = False
            return
        else:
            info("update: model save and reload!")
            self.use_pretrain_model = False
            return

    def run_second_stage(self, remaining_time_budget):
        if not self.start_db_model:
            # 第一次进入db时，先清空现有的sess
            self._clear_train_space()
            self.start_db_model = True
            if self.imbalance_level == 2:
                info(color_msg("use bs sampling val for db solution!"))
                self.db_model.split_val = False

        if self.db_model.model_id == len(
                self.db_model.cand_models) and self.db_model.data_id == self.db_model.max_data:
            self.second_stage_done = True
            info("finish second stage!")
            return
        if self.db_patience >= self.db_patience_max_num:
            self.db_model.epoch = 1
            info(color_msg("Change DB model due to low auc!!!!"))
            self.db_patience = 0
            do_clean = True
        else:
            do_clean = False

        if self.db_model.split_val:
            self.db_model.train_iter((self.data_manager.meta_data_x, self.data_manager.meta_data_y),
                                     # eval_dataset=(self.clean_val_x, self.val_y),
                                     eval_dataset=(self.val_x, self.val_y),
                                     remaining_time_budget=remaining_time_budget,
                                     do_clean=do_clean)
        else:
            self.db_model.train_iter((self.data_manager.meta_train_x, self.data_manager.meta_train_y),
                                     # eval_dataset=(self.clean_val_x, self.val_y),
                                     eval_dataset=(self.val_x, self.val_y),
                                     remaining_time_budget=remaining_time_budget,
                                     do_clean=do_clean)

        db_auc = self.db_model.best_sco  # 本身是一个集成结果
        self.evaluator.best_auc = db_auc
        if db_auc == -1 or db_auc == 0.02:
            db_auc = 0.0
        if db_auc >= self.best_auc * 0.97 and db_auc > 0.0:
            info("Use db_model when db_auc is {} and best_val_auc is {}".format(db_auc, self.best_auc))
            self.use_db_model = True
            if self.db_model.Xtest is None and self.db_model.FIRSTROUND:
                info(color_msg("Use START STAGE MODEL"))
                self.db_model.START = True
            elif self.db_model.Xtest is None and self.db_model.new_data:
                info(color_msg("Use NEW DATA STAGE MODEL"))
                # if self.db_model.data_id == 3:
                #     self.db_model.data_id = 2
                self.db_model.START = False
            return
        else:
            info("Do not Use db_model when db_auc is {} and best_val_auc is {}".format(db_auc, self.best_auc))
            # 这里需要保持db model内部的状态，不然会漏状态
            if self.db_model.START == False and self.db_model.FIRSTROUND == False and self.db_model.LASTROUND:
                self.db_model.is_best = False
                self.db_model.LASTROUND = False
            elif self.db_model.START == True:  # 如果db START模型没有超过当前
                self.db_model.START = False

            info("update: model save and reload!")
            self.use_db_model = False
            return

    def run_first_stage_model(self, train_preprocessed_data, train_diff_y):
        if self.switch_new_model and not self.cur_model_train_start:  # 如果切换模型，且当前模型没有开始训练
            self._clear_train_space()
            self._init_nn_train_process()
            self.model = self.model_manager.select_classifier(model_name=self.model_lib[self.model_id],
                                                              feature_mode=self.feature_lib[self.feature_id],
                                                              data_feature=self.feature_generator.data_feature)
            info(color_msg("start new nn model training!"))

            if self.model_lib[self.model_id] == "text_cnn":
                if self.imbalance_level == 2 or self.metadata["class_num"] >= 5:
                    self.callbacks = []
                else:
                    self.callbacks = [self.callbacks_[0]]
            else:
                self.callbacks = [self.callbacks_[1]]

        self._train_nn_process(train_preprocessed_data, train_diff_y)

        if self.train_model_id >= 1:  # 训练了至少2个模型
            self._ensemble_multi_models()
        else:
            self._ensemble_multi_models()

        #  达到结束的条件
        if self.evaluator.decide_stop(train_epoch=self.train_epoch):
            self._reset_train_status()

    def train(self, x_train, y_train, remaining_time_budget=None):
        if self.done_training:
            return
        if not self.use_multi_svm and self.metadata["language"] == "EN":
            self.start_first_stage_call_num = 1
        if not self.use_multi_svm and self.metadata["language"] == "ZH":
            self.start_first_stage_call_num = 2
        if self.use_multi_svm and self.multi_label:
            self.start_first_stage_call_num = 2

        info(color_msg("use multi_svm {} and start_first_stage_call_num is {}".format(self.use_multi_svm,
                                                                                      self.start_first_stage_call_num)))

        if self.call_num == 0:
            self.init_generators(x_train, y_train)
            if self.use_multi_svm:
                self.evaluator.max_epoch = 15
            if self.multi_label:
                self.evaluator.max_epoch = 18

        else:
            # if self.call_num == 1 and self.metadata["language"] == "ZH":
            if self.call_num == 2 and self.metadata["language"] == "ZH":
                self.feature_generator.do_seg = True

            if y_train.shape[0] > 0:
                self.data_manager.update_meta_data(x_train, y_train)

        info(color_msg("check imbalance level {}".format(self.imbalance_level)))

        # 数据采样
        train_diff_x, train_diff_y, val_diff_x, val_diff_y = self.do_data_sampling(y_train)
        self._show_data_info(train_diff_x, val_diff_x)

        ########################## 设定采用的预处理方式###########################################
        # 是否截断
        # 是否分词
        ########################## 数据前处理 ####################################################
        train_preprocessed_data = self.prepare_clean_data(train_diff_x, val_diff_x, val_diff_y)

        ############################ 进入 SVM 阶段模型 ############################
        if self.call_num < self.start_first_stage_call_num:
            self.model = self.run_svm('svm', train_preprocessed_data, train_diff_y)

            if self.val_y.shape[0] > 0:
                self.do_evaluation(eval_svm=True)
                self.evaluator.update_model_weights(self.model, self.train_epoch, is_svm=True)
            self._ensemble_multi_models()

        else:

            if self.call_num == self.start_first_stage_call_num:
                # 进行模型预选择
                self.meta_strategy()
                self.feature_generator.reset_tokenizer()
                # 设定文本长度及文本长度std，影响后处理pad长度设置
                self.feature_generator.max_length = self.max_length
                self.feature_generator.seq_len_std = self.seq_len_std
                self.prepare_nn_tokenizer(train_x=self.build_tokenizer_x)
                self.accu_nn_tokenizer_x = False

            ############################ 进入第三阶段模型 ############################
            if self.second_stage_done and self.use_ft_model:
                if self.start_ft_bert and not self.use_pretrain_model:
                    self.use_pretrain_model = False
                    return
                else:
                    if self.bert_output_patience > 0:
                        return self._train_bert_process(train_preprocessed_data, train_diff_y)
                    else:
                        self.use_pretrain_model = False
                        return

            ############################ 进入第二阶段模型 ############################

            elif self.first_stage_done:
                if not self.multi_label:
                    self.run_second_stage(remaining_time_budget)
                else:
                    info(color_msg("do not run db when multi_label is {}".format(self.multi_label)))
                    self.second_stage_done = False
            ############################ 进入第一阶段模型 ############################
            else:
                if self.switch_new_model and not self.cur_model_train_start:
                    self.is_stage_done()
                if not self.first_stage_done:
                    self.run_first_stage_model(train_preprocessed_data, train_diff_y)

        return

    def _update_multi_model_result(self, pred):
        if self.is_best:
            self.test_result_list[self.train_model_id] = pred
            result = np.mean(self.test_result_list[:self.train_model_id + 1], axis=0)
        else:
            info(color_msg("do not ensemble cur model!!!!\n"))
            if isinstance(self.test_result_list[self.train_model_id], int):
                result = self.test_result_list[0]
            else:
                result = np.mean(self.test_result_list[:self.train_model_id + 1], axis=0)
        return result

    def transform_test(self):
        if self.call_num < self.start_first_stage_call_num:
            x_test_feature = self.svm_token.transform(self.x_test_clean)
            return x_test_feature

        else:
            if not self.tokenize_test:
                x_test_feature = self.tokenizer.texts_to_sequences(self.x_test_clean)
                x_test = sequence.pad_sequences(x_test_feature,
                                                maxlen=self.feature_generator.max_length,
                                                padding='post')
                self.tokenize_test = True
            else:
                x_test = self.x_test
            return x_test

    def output_svm_result(self):
        result = self.model.predict_proba(self.x_test)
        if self.is_best:
            self.svm_test_result.append(result)
        elif self.call_num > 0:
            result = self.svm_test_result[-1]
        return result

    def output_second_stage_result(self):
        info("Output in second stage!")
        # 第二阶段没有结束：只有两个选择：db 模型 or 第一阶段最优模型
        if self.use_db_model:
            self.db_patience = 0
            info(color_msg("Use db Model!!"))
            db_result = self.db_model.test(self.x_test_raw)

            info(color_msg("db result is {}".format(type(db_result))))
            if isinstance(db_result, list):
                info(color_msg("db result is {}".format(len(db_result))))
            if isinstance(db_result, np.ndarray):
                info(color_msg("db result is {}".format(db_result.shape[0])))
            if isinstance(db_result, np.float):
                info(color_msg("db result is {}".format(db_result)))

            # 如果db输出为空，返回第一个阶段结果
            if db_result.shape[0] == 0:
                if isinstance(self.test_result_list[2], int):
                    result = np.mean(self.test_result_list[:2], axis=0)
                else:
                    result = np.mean(self.test_result_list[:3], axis=0)
                return result
            else:
                self.test_result_list[2] = db_result
                result = np.mean(self.test_result_list[:3], axis=0)
                return result
        else:
            info(color_msg("Do Not Use db Model!! db_patience is {}".format(self.db_patience)))
            self.db_patience += 1
            if self.start_db_model:
                if isinstance(self.test_result_list[2], int):
                    info(color_msg("Do Not Use db Model and use First Stage Result!!!"))
                    info(color_msg("check test_result_list 2:{}".format(len(self.test_result_list[:2])),
                                   color='yellow'))
                    result = np.mean(self.test_result_list[:2], axis=0)
                else:
                    result = np.mean(self.test_result_list[:3], axis=0)
            else:
                info(color_msg("Start db, but use First Stage Result!"))
                if self.train_model_id == 0:
                    info(color_msg("Check test_Result_list:{}".format(self.test_result_list[1])))
                    info(color_msg("Start db, but use First Stage Result when train_model_id==0"))
                    result = self.test_result_list[0]
                else:
                    info(color_msg("Start db, but use First Stage Result when train_model_id==1"))
                    result = np.mean(self.test_result_list[:2], axis=0)
            return result

    def output_third_stage_result(self):
        info(color_msg("First stage finish! Output Bert Result!!"))
        if self.use_pretrain_model:
            if self.update_bert:  # 如果更新了bert模型，采用更新的参数进行预测
                # self.ft_model.
                result = self.ft_model.model_predict_process(self.x_test_clean, self.ft_model.model)
                self.best_bert_pred = result
                self.bert_result.append(result)
                if len(self.bert_result) > 0:  # ensemble前N次 bert结果
                    result = np.mean(self.bert_result, axis=0)

            else:  # 否则，用历史结果出点
                result = np.mean(self.bert_result, axis=0)

        else:
            if self.bert_auc * 0.98 > self.best_auc:  # 表明已经存在训练过的bert模型且结果远高于前两阶段
                result = np.mean(self.bert_result, axis=0)
            elif self.bert_auc > 0.0:  # 表面已存在训练过的bert模型，但结果没有远超过前两阶段
                if isinstance(self.test_result_list[3], int):

                    result = np.mean(self.test_result_list[:3], axis=0)
                else:
                    result = np.mean(self.test_result_list[:3], axis=0)
            else:  # 表面当前只有CNN模型
                result = np.mean(self.test_result_list[:3], axis=0)
        return result

    def output_first_stage_result_with_svm(self, result):
        if self.is_best:  # 包括svm 结果
            info(color_msg("std is {}".format(np.std([np.max(self.best_svm_scores), self.best_auc]))))
            if np.std([np.max(self.best_svm_scores), self.best_auc]) < 0.005:
                self.svm_test_result.append(result)
                result = np.mean(self.svm_test_result, axis=0)
                # self.svm_test_result.pop(-1)
            else:
                self.multi_label_cnt_thred-=1
                result = result
        else:
            if np.std([np.max(self.best_svm_scores), self.best_auc]) < 0.02:
                self.svm_test_result.append(result)
                result = np.mean(self.svm_test_result, axis=0)
                self.svm_test_result.pop(-1)
            else:
                result = self.svm_test_result[-1]
        return result

    def output_first_stage_result(self):
        result = self.model.predict(self.x_test,
                                    batch_size=self.batch_size * 16)
        self.cur_model_test_res.append(result)
        if self.train_model_id == 0 and not self.cur_model_train_start:
            info(color_msg("finish current model train", color='yellow'))
            # self.test_result_list[0] = np.mean(self.cur_model_test_res, axis=0)
            self.test_result_list[0] = self.cur_model_test_res[-1]

        if self.train_model_id >= 1:
            result = self._update_multi_model_result(result)

        if self.call_num == self.start_first_stage_call_num:
            info(color_msg("std is {}".format(np.std([np.max(self.best_svm_scores), self.best_auc]))))
            if self.is_best:
                if np.std([np.max(self.best_svm_scores), self.best_auc]) < 0.008:
                    self.svm_test_result.append(result)
                    result = np.mean(self.svm_test_result, axis=0)
                else:
                    result = result
            else:
                if np.std([np.max(self.best_svm_scores), self.best_auc]) < 0.02:
                    self.svm_test_result.append(result)
                    result = np.mean(self.svm_test_result, axis=0)
                    self.svm_test_result.pop(-1)
                else:
                    result = self.svm_test_result[-1]

        if self.multi_label:
            if self.train_model_id >= 1:
                result = self._update_multi_model_result(result)
            else:
                result = self.output_first_stage_result_with_svm(result)

        return result

    def test(self, x_test, remaining_time_budget):
        if self.call_num == 0:
            self.x_test_raw = x_test
            self.x_test_clean = self.feature_generator.preprocess_data(self.x_test_raw)

        # if self.metadata["language"] == "ZH" and self.call_num == 1:
        if self.metadata["language"] == "ZH" and self.call_num == 2:
            # feature.do_seg 已经更新
            self.x_test_clean = self.feature_generator.preprocess_data(self.x_test_raw)

        self.x_test = self.transform_test()

        # 输出svm 结果
        if self.call_num < self.start_first_stage_call_num:
            result = self.output_svm_result()

        elif self.second_stage_done:
            result = self.output_third_stage_result()

        elif self.first_stage_done:
            if self.multi_label:
                if self.multi_label_cnt_thred<0:
                    result = self.cur_model_test_res[-1]
                    return result
                else:
                    result = self.svm_test_result[-1]
            else:
                result = self.output_second_stage_result()

        else:
            ## 当前为NN模型输出结果
            result = self.output_first_stage_result()

        if self.first_stage_done:
            info(color_msg("check test_result_list 0:{}".format(self.test_result_list[0]),
                           color='yellow'))

        else:
            print(self.train_model_id)
            info(color_msg("check test_result_list:{}".format(self.test_result_list[self.train_model_id]),
                           color='yellow'))

        self.done_training = False
        self.call_num += 1
        if self.call_num == MAX_EPOCH_NUM:
            self.done_training = True
            self.ft_model = None
        return result

    def lr_decay(self, epoch):
        if self.call_num == 1 or self.cur_lr is None:
            self.cur_lr = self.model_manager.lr
        if self.train_epoch % 3 == 0 and self.train_epoch > 0:
            self.cur_lr = 3 * self.cur_lr / 5
        self.cur_lr = max(self.cur_lr, 0.0001)
        info(color_msg("recompile lr {}".format(self.cur_lr), color="blue"))
        lr = self.cur_lr
        return lr

    def step_decay(self, epoch):
        epoch = self.train_epoch // 3
        initial_lrate = self.model_manager.lr  # 0.016 #0.0035 #
        drop = 0.65  # 0.65
        epochs_drop = 1.0  # 2.0
        if (self.train_epoch) <= 2:
            lrate = initial_lrate
        else:
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        lrate = max(lrate, 0.0001)
        info(color_msg("recompile lr {}".format(lrate), color="blue"))
        return lrate
