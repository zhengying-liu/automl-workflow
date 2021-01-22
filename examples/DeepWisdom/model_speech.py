"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import copy

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ""))

from Auto_Speech.rank_2_fuzhi.model import Model as ASpeechWidsomModel
from speech_dataset_convertor import TfDatasetsConvertor as TfDatasetTransformer
from speech_autodl_config import (
    autodl_g_conf_repr,
    AutoDlConf,
    META_SOLUS,
    DM_DS_PARAS,
    speech_ds_tds_conf,
    speech_ms_mlp_conf,
)
from log_utils import logger, as_timer


# Config data, temp and will be moved to autodl_config
ds_config_sample_mode = "iter"  # first or iter


def meta_domain_2_model(domain):
    """
    Task for Speech.
    :param domain:
    :return:
    """
    assert domain == "speech", "Error: Task is not Speech."
    meta_solution_name = META_SOLUS.speech_solution
    if meta_solution_name == "PASA_NJU":
        from Auto_Speech.PASA_NJU.model import Model as AutoSpeechModel
    else:
        from Auto_Speech.rank_2_fuzhi.model import Model as AutoSpeechModel
    return AutoSpeechModel


class Model:
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
        # self.domain = infer_domain(metadata)
        self.domain = "speech"
        # logger.info("Note:The AutoDL_G_CONF: {}".format(autodl_g_conf_repr))
        logger.info("Note:The inferred domain of current dataset is: {}.".format(self.domain))
        # Domain识别及Model初始化
        # DomainModel = DOMAIN_TO_MODEL[self.domain]
        DomainModel = meta_domain_2_model(self.domain)
        self.domain_metadata = get_domain_metadata(metadata, self.domain)
        self.class_num = self.domain_metadata["class_num"]
        self.train_num = self.domain_metadata["train_num"]

        logger.info("Note:The domain metadata is {}".format(self.domain_metadata))
        self.domain_model = DomainModel(self.domain_metadata)

        # fixme: 增加更新数据.
        self.speech_widsom_model = ASpeechWidsomModel(self.domain_metadata)
        self.speech_wisdom_dataset_train = None
        logger.info("Note:Init Speech Wisdom solution, is {}".format(self.domain_metadata))
        self.main_train_loop_num = 0
        self.main_test_loop_num = 0
        #
        self.raw_tf_train_dataset = None
        self.dataset_sample_size = None
        self.dataset_read_num_second = None
        self.data_all_np_x_list = list()
        self.data_all_np_y_array = None
        self.ds_incr_flag = True  # dataset sampling if still remain to be sampled incrementally.
        self.domain_dataset_train = None
        self.domain_dataset_test = None

        # for tf_dataset.
        self.tf_dataset_trainsformer = TfDatasetTransformer(if_train_shuffle=speech_ds_tds_conf.if_shuffle)
        as_timer("model_speech_init")


    def train(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train
        logger.info("Note: speech_train_process  model.py starts train")
        as_timer("train_start")

        # load tf_train_dataset for first time.
        self.tf_dataset_trainsformer.init_train_tfds(dataset, self.train_num)

        if self.domain in ["speech"]:
            # Train the model with light model.
            if self.main_train_loop_num < speech_ms_mlp_conf.lightwei_train_end_loop:
                # fixme: need to be autotuned.
                ds_take_size = min(int(self.train_num * speech_ds_tds_conf.sample_ratio[self.main_train_loop_num]), self.class_num * 50)

                # self.domain_dataset_train = self.tf_dataset_trainsformer.get_speech_train_dataset(ds_take_size)

                # self.domain_model.train(self.domain_dataset_train, remaining_time_budget=remaining_time_budget)
                self.domain_model.train(self.tf_dataset_trainsformer.get_speech_train_dataset(ds_take_size), remaining_time_budget=remaining_time_budget)

                logger.info(
                    "Note: domain={}, main_train_loop_num={}, light_model train finished.".format(
                        self.domain, self.main_train_loop_num
                    )
                )
                as_timer("speech_model_basic_train")

            if self.main_train_loop_num >= speech_ms_mlp_conf.midwei_train_start_loop:
                self.speech_widsom_model.train(
                    # (self.domain_dataset_train["x"], self.domain_dataset_train["y"]), remaining_time_budget
                    self.tf_dataset_trainsformer.get_speech_train_dataset_full(), remaining_time_budget
                )
                logger.info("Note: start wisdom at np, main_train_loop_num={}".format( self.main_train_loop_num))
                as_timer("speech_tr34_train")

            logger.info("Note:time_train model.py domain_model train finished.")

            # Update self.done_training
            self.done_training = self.domain_model.done_training
            self.main_train_loop_num += 1
            # print(as_timer)
            as_timer("train_end")
            logger.info(as_timer)
        else:
            logger.error("Note: Domain is not Speech!")


    def test(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        # self.set_domain_dataset(dataset, is_training=False)

        as_timer("test_start")
        # init tf_test_dataset for the first time.
        self.tf_dataset_trainsformer.init_test_tfds(dataset)

        self.domain_dataset_test, self.X_test = self.tf_dataset_trainsformer.get_speech_test_dataset()

        # As the original metadata doesn't contain number of test examples, we
        # need to add this information
        if self.domain in ["text", "speech"] and (not self.domain_metadata["test_num"] >= 0):
            self.domain_metadata["test_num"] = len(self.X_test)
        logger.info("Note:test_process test domain metadata is {}".format(self.domain_metadata))

        # Make predictions
        if self.domain in ["speech"]:
            if (
                self.main_train_loop_num
                <= speech_ms_mlp_conf.midwei_train_start_loop + speech_ms_mlp_conf.midwei_predict_block_loop
            ):
                Y_pred = self.domain_model.test(self.domain_dataset_test, remaining_time_budget=remaining_time_budget)
                logger.info(
                    "Note: speech pasa_model, speech_main_train_loop={}, speech_main_test_loop={}".format(
                        self.main_train_loop_num, self.main_test_loop_num
                    )
                )
                # Update self.done_training
                self.done_training = self.domain_model.done_training

            else:
                Y_pred = self.speech_widsom_model.test(
                    self.domain_dataset_test, remaining_time_budget=remaining_time_budget
                )
                logger.info(
                    "Note: speech dw_model, train_loop={}, test_loop={}".format( self.main_train_loop_num, self.main_test_loop_num)
                )
                # Update self.done_training
                self.done_training = self.speech_widsom_model.done_training

            as_timer("test_end")
            logger.info(as_timer)
        else:
            logger.error("Note: Domain is not Speech!")

        self.main_test_loop_num += 1

        return Y_pred


def get_domain_metadata(metadata, domain, is_training=True):
    # Specific for Speech.
    """Recover the metadata in corresponding competitions, esp. AutoNLP
  and AutoSpeech.

  Args:
    metadata: an AutoDLMetadata object.
    domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
  """

    # Fetch metadata info from `metadata`
    class_num = metadata.get_output_size()
    num_examples = metadata.size()

    # WARNING: hard-coded properties
    file_format = "wav"
    # sample_rate = 16000
    sample_rate = 8000

    # Create domain metadata
    domain_metadata = {}
    domain_metadata["class_num"] = class_num
    if is_training:
        domain_metadata["train_num"] = num_examples
        domain_metadata["test_num"] = -1
    else:
        domain_metadata["train_num"] = -1
        domain_metadata["test_num"] = num_examples
    domain_metadata["file_format"] = file_format
    domain_metadata["sample_rate"] = sample_rate

    return domain_metadata

