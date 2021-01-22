from __future__ import absolute_import
import os
import numpy as np

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# from at_toolkit.interface.adl_classifier import AdlOfflineClassifier, AdlOnlineClassifier
from at_toolkit import info, error, as_timer, AdlOfflineClassifier, AdlOnlineClassifier, ATEvaluator
from at_speech import DNpAugPreprocessor, TTAGenerator, MixupGenerator, cnn_load_pretrained_model


class CNNClassifier(AdlOnlineClassifier):
    def init(self, class_num: int, init_params: dict):
        """
        :param class_num:
        :param init_params:
            - n_mels
            - pretrain_path
        :return:
        """
        self.class_num = class_num
        self.clf_name = "cnn_pret"
        self.n_mels = init_params.get("n_mels")  # 64, fixed, as pretrained.
        # self.model = self._load_pretrained_model(input_shape=(self.n_mels, self.n_mels, 1), n_classes=self.class_num)
        self.model = cnn_load_pretrained_model(input_shape=(self.n_mels, self.n_mels, 1), n_classes=self.class_num)
        info(
            "Backbone classifier={} is init, class_num={}, init_params={}".format(
                self.clf_name, self.class_num, init_params
            )
        )
        as_timer("clf_{}_init".format(self.clf_name))

        self.train_batch_size = init_params.get("train_batch_size")
        self.predict_batch_size = init_params.get("predict_batch_size")
        self.n_iter = 0

        # option:
        self.img_freqmasking_datagen = ImageDataGenerator(preprocessing_function=DNpAugPreprocessor.frequency_masking)

    # fixme: need update.
    def _fit_params_decision(self, fit_params):
        self.fit_params_res = dict()
        # fixme: how to setup? total_train_num or cur_train_num?
        self.fit_params_res["steps_per_epoch"] = self.cur_train_num // self.train_batch_size
        # fixme: how to setup epochs?
        self.fit_params_res["epochs"] = self.n_iter + 5
        self.fit_params_res["initial_epoch"] = self.n_iter

    def online_fit(self, train_examples_x: np.ndarray, train_examples_y: np.ndarray, fit_params: dict):
        self.cur_train_num = len(train_examples_x)
        cur_training_generator = MixupGenerator(
            train_examples_x, train_examples_y, batch_size=self.train_batch_size, datagen=self.img_freqmasking_datagen
        )()
        # update
        self._fit_params_decision(fit_params)

        self.model.fit_generator(
            cur_training_generator,
            steps_per_epoch=self.fit_params_res["steps_per_epoch"],
            epochs=self.fit_params_res["epochs"],
            initial_epoch=self.fit_params_res["initial_epoch"],
            shuffle=True,
            verbose=1,
        )
        self.n_iter += 5
        as_timer(
            "CNNCls_fit_{}_{}_{}_{}".format(
                self.cur_train_num,
                self.fit_params_res["initial_epoch"],
                self.fit_params_res["epochs"],
                self.fit_params_res["steps_per_epoch"],
            )
        )

    def eval_val(self, val_examples_x, val_examples_y):
        valid_generator = TTAGenerator(val_examples_x, batch_size=self.predict_batch_size)()
        valid_size = len(val_examples_x)
        valid_probas = self.model.predict_generator(
            valid_generator, steps=int(np.ceil(valid_size / self.predict_batch_size))
        )
        # val_auc = ATEvaluator.autodl_auc(val_examples_y, valid_probas)
        val_auc = ATEvaluator.skl_auc_macro(val_examples_y, valid_probas)
        as_timer("CNNCls_evalval_{}_{}".format(valid_size, val_auc))
        return val_auc

    def predict_proba(self, test_examples: np.ndarray, predict_prob_params: dict) -> np.ndarray:
        cur_test_generator = TTAGenerator(test_examples, batch_size=self.predict_batch_size)()
        test_size = len(test_examples)
        pred_probs = self.model.predict_generator(
            cur_test_generator, steps=int(np.ceil(test_size / self.predict_batch_size))
        )
        as_timer("CNNCls_testpred")
        return pred_probs
