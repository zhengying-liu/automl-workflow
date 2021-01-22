"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES']='3'
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

here = os.path.dirname(os.path.abspath(__file__))
import_dir = os.path.abspath(os.path.join(here, ".."))
print("import_dir={}".format(import_dir))
sys.path.append(import_dir)

sys.path.append(os.path.join(here, ""))
from at_toolkit.at_utils import autodl_install_download

autodl_install_download("speech")

from at_speech.policy_space.model_executor import ModelExecutor
from at_toolkit import logger, info, error, as_timer
from at_speech.at_speech_config import IF_TRAIN_BREAK_CONDITION
from at_speech.at_speech_cons import CLS_TR34


# Config data, temp and will be moved to autodl_config
# ds_config_sample_mode = "iter"  # first or iter


EVAL_TLOSS_TAIL_SIZE = 8
EVAL_TLOSS_GODOWN_RATE_THRES = 0.7


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
        self.domain = "speech"
        test_metadata_filename = self.metadata.get_dataset_name().replace('train', 'test') + '/metadata.textproto'
        self.test_num = [int(line.split(':')[1]) for line in open(test_metadata_filename, 'r').readlines()[:3] if 'sample_count' in line][0]

        # DomainModel = meta_domain_2_model(self.domain)
        self.domain_metadata = get_domain_metadata(metadata, self.domain)
        self.domain_metadata["test_num"] = self.test_num
        self.class_num = self.domain_metadata["class_num"]
        self.train_num = self.domain_metadata["train_num"]

        logger.info("Note:domain={}, domain_metadata is {}".format(self.domain, self.domain_metadata))
        # self.domain_model = DomainModel(self.domain_metadata)
        self.domain_model = ModelExecutor(self.domain_metadata)
        self.ensemble_val_record_list = list()
        self.ensemble_val_nauc_list = list()
        self.cur_cls_name = None
        self.cur_train_his_report = dict()
        self.g_predevel_space = list()
        self.g_train_loss_list = list()

        as_timer("model_speech_init")

    def get_accept_nauc(self):
        if len(self.ensemble_val_nauc_list) == 0:
            return 0
        if max(self.ensemble_val_nauc_list) <= 0.9:
            return max(self.ensemble_val_nauc_list)*0.98
        elif 0.9 < max(self.ensemble_val_nauc_list) <= 0.95:
            return max(self.ensemble_val_nauc_list) * 0.99
        elif 0.95 < max(self.ensemble_val_nauc_list) <= 1:
            return max(self.ensemble_val_nauc_list) * 0.996
        else:
            return 0

    def train(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        logger.info("Note: speech_train_process  model.py starts train")
        as_timer("train_start")

        if IF_TRAIN_BREAK_CONDITION:
            while True:
                self.cur_train_his_report = self.domain_model.train_pipeline(dataset)
                self.cur_cls_name = self.cur_train_his_report.get("cls_name")

                cur_val_nauc = self.cur_train_his_report["val_nauc"]
                self.ensemble_val_record_list.append([self.cur_cls_name, cur_val_nauc])
                self.ensemble_val_nauc_list.append(cur_val_nauc)
                if cur_val_nauc == -1 or cur_val_nauc > self.get_accept_nauc():
                    info("Decision=Yes, cur_cls_name={}, cur_val_nauc={}, his_top_nauc={}".format(self.cur_cls_name, cur_val_nauc, max(self.ensemble_val_nauc_list)))
                    break
                else:
                    info("Decision=No, cur_cls_name={}, cur_val_nauc={}, his_top_nauc={}".format(self.cur_cls_name, cur_val_nauc, max(self.ensemble_val_nauc_list)))

        else:
            self.cur_train_his_report = self.domain_model.train_pipeline(dataset)
            self.cur_cls_name = self.cur_train_his_report.get("cls_name")

            cur_t_loss = self.cur_train_his_report.get("t_loss")
            if cur_t_loss is None:
                self.g_train_loss_list.append(100000)
            else:
                self.g_train_loss_list.append(cur_t_loss)

            info("train_his_report={}".format(self.cur_train_his_report))
            cur_val_nauc = self.cur_train_his_report["val_nauc"]

            self.ensemble_val_record_list.append([self.cur_cls_name, cur_val_nauc])
            self.ensemble_val_nauc_list.append(cur_val_nauc)


        as_timer("speech_model_basic_train")

    def test(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        cur_y_pred = self.domain_model.test_pipeline(dataset)

        self.cur_train_his_report["pred_probas"] = cur_y_pred

        if self.cur_cls_name == CLS_TR34 and self.domain_model.tr34_cls_train_pip_run >= 8:
            loss_godown_rate = self.domain_model.decision_maker.ensemble_learner.get_loss_godown_rate(self.g_train_loss_list, EVAL_TLOSS_TAIL_SIZE)
            if loss_godown_rate >= EVAL_TLOSS_GODOWN_RATE_THRES:
                self.domain_model.decision_maker.ensemble_learner.add_eval_pred_item(self.cur_train_his_report)

        if self.domain_model.tr34_cls_train_pip_run >= 15:
            pred_rule = {
                "t_loss": 5,
                # "t_acc": 1
            }
            pred_ensemble = self.domain_model.decision_maker.ensemble_learner.softvoting_ensemble_preds(pred_rule)
        else:
            pred_ensemble = cur_y_pred
        # if pred ensemble.
        # real_y_pred =
        self.done_training = False
        as_timer("test_start")
        # return cur_y_pred
        return pred_ensemble


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

