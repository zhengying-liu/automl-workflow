# -*- coding: utf-8 -*-
# @Date    : 2020/3/4 16:11
import numpy as np
from scipy import stats

from log_utils import info
from at_nlp.utils import color_msg, ohe2cat
from at_toolkit.at_evalator import ATEvaluator
from at_nlp.generators.data_generator import DataGenerator as BatchDataGenerator


class Evaluator():
    def __init__(self, x, y):
        self.x = x
        self.label = y
        self.pred = None
        self.tokenizer = None
        self.eval_generator = None
        self.batch_size = 32
        self.language = "EN"
        self.max_length = 0.0
        self.num_features = 20000
        self.cur_val_auc = 0.0
        self.best_auc = 0.0
        self.is_best = False
        self.model_weights_list = []
        self.val_auc_list = []
        self.k = 0  # 连续情况：val auc低于阈值次数统计
        self.patience = 3
        self.max_epoch = 5
        self.last_val_auc = 0.0
        self.stop_criteria = False
        self.best_call_num = 0

    def _reset(self):
        self.model_weights_list = []
        self.val_auc_list = []
        self.best_auc = 0.0
        self.last_val_auc = 0.0
        self.is_best = False
        self.best_call_num = 0
        self.k = 0
        self.stop_criteria = False


    def update_val_data(self, new_x, new_y):
        self.x = new_x
        self.label = new_y

    def update_setting(self, language, max_length, num_features, tokenizer):
        self.language = language
        self.max_length = max_length
        self.num_features = num_features
        self.tokenizer = tokenizer
        self.eval_generator = BatchDataGenerator(self.x, self.label,
                                                 batch_size=self.batch_size,
                                                 language=self.language,
                                                 max_length=self.max_length
                                                 if self.max_length else 100,
                                                 vocab=None,
                                                 tokenizer=self.tokenizer,
                                                 num_features=self.num_features,
                                                 shuffle=False)

    def valid_auc(self, is_svm=False, model=None, use_autodl_auc=True):
        if is_svm:
            x_valid = self.tokenizer.transform(self.x)

            result = model.predict_proba(x_valid)
            result = self.rebuild_predict_prob(result)
            info(color_msg("self.x shape {}".format(self.label.shape)))
            info(color_msg("result shape {}".format(result.shape)))
        else:

            result = model.predict_generator(self.eval_generator)

        if use_autodl_auc:
            self.cur_val_auc = ATEvaluator.autodl_auc(solution=self.label, prediction=result)
        else:
            self.cur_val_auc = ATEvaluator.auc_metric(solution=self.label, prediction=result)

        info(color_msg("Note: cur_val_auc is {}".format(self.cur_val_auc), color='blue'))

    def _reset_pred(self, prediction):
        new_prob_arary = prediction
        for sample_i in range(prediction.shape[0]):
            np_median_value = np.median(prediction[sample_i])
            for empty_cls in self.empty_class_:
                new_prob_arary[sample_i][empty_cls] = np_median_value
        return new_prob_arary

    def rebuild_predict_prob(self, prediction):
        new_prob_arary = prediction
        val_label_distribution = np.sum(np.array(self.label), 0)

        self.empty_class_ = [i for i in range(val_label_distribution.shape[0]) if val_label_distribution[i] == 0]
        self.kurtosis = stats.kurtosis(val_label_distribution)
        self.nomalized_std = np.std(val_label_distribution) / np.sum(val_label_distribution)
        info("check empty_class {} and kurtosis is {}".format(self.empty_class_, self.kurtosis))
        if self.empty_class_:
            new_prob_arary = self._reset_pred(prediction)

        return new_prob_arary

    def check_early_stop_criteria(self, train_epoch):
        # db方案早停条件: 出现k次低于最佳auc的情况: 针对模型比较震荡
        early_stop_criteria_1 = self.k >= self.patience or train_epoch > self.max_epoch
        # upwind 方案早停条件1: 当前评估auc足够高且训练次数足够大，出现一次下降即停：针对模型后期，避免下降
        early_stop_criteria_2 = self.cur_val_auc < self.last_val_auc and self.cur_val_auc > 0.96 and train_epoch > self.max_epoch
        # upwind 方案早停条件2: 当前训练次数达到阈值，且连续下降次数达到阈值即停：针对模型难收敛/过拟合
        early_stop_criteria_3 = train_epoch>= 5 and self.k >= 2
        info(color_msg("stop criteria 1 is {}, 2 is {}, 3 is {}".format(early_stop_criteria_1, early_stop_criteria_2, early_stop_criteria_2), color='blue'))
        return (early_stop_criteria_1 or early_stop_criteria_2 or early_stop_criteria_3)

    def update_early_stop_params(self):
        if self.val_auc_list:
            max_auc = np.max(self.val_auc_list)
            self.last_val_auc = self.val_auc_list[-1]
        else:
            max_auc = 0.0
            self.last_val_auc = 0.0

        if self.cur_val_auc > max_auc and max_auc!=0.0:
            self.k = 0
        else:
            self.k += 1

    def update_model_weights(self, model, train_epoch, is_svm=False):
        info(color_msg("train_epoch is {}， cur model: model_weight_list is {}\n".format(train_epoch,
                                                                                        len(self.model_weights_list))))
        if self.model_weights_list:
            if self.cur_val_auc > self.best_auc:
                info(color_msg("Update best result!"))
                self.best_auc = self.cur_val_auc
                self.is_best = True
                self.best_call_num = train_epoch  # 从0开始算

            else:
                info(color_msg("Do not update best result!"))
                self.is_best = False
                model.set_weights(self.model_weights_list[self.best_call_num])

        else: # 新增第一个模型权值
            self.is_best = True
            self.best_auc = self.cur_val_auc
            self.best_call_num = train_epoch
        if is_svm:
            # self.model_weights_list.append(model)
            pass
        else:
            model_weights = model.get_weights()
            self.model_weights_list.append(model_weights)

    def decide_stop(self, train_epoch):
        self.update_early_stop_params()
        self.val_auc_list.append(self.cur_val_auc)
        self.stop_criteria = self.check_early_stop_criteria(train_epoch)
        info(color_msg("Note: stop condition is {}".format(self.stop_criteria), color='blue'))
        return self.stop_criteria