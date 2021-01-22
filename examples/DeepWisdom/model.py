"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import copy
import tensorflow as tf

from log_utils import logger, error



here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ""))

# for import ac.pyx
# os.system("pip3 install cython -i https://pypi.tuna.tsinghua.edu.cn/simple")


# Config for Solution
META_SOLUS_NLP = "upwind_flys"
META_SOLUS_SPEECH = "PASA_NJU"
VIDEO_SOLUTION_FLAG = "2d"  # 2d/3d


model_dirs = [
    "",  # current directory
    # "Auto_Image",  # AutoCV/AutoCV2 winner model
    # "Auto_Video",  # AutoCV/AutoCV2 winner model
    "Auto_Video_",  # AutoCV/AutoCV2 winner model
    "Auto_NLP/{}".format(META_SOLUS_NLP),  # AutoNLP 2nd place winner
    # 'AutoSpeech/PASA_NJU',    # AutoSpeech winner
    "Auto_Speech/{}".format(META_SOLUS_SPEECH),  # AutoSpeech winner
    "Auto_Speech/{}".format("rank_2_fuzhi"),  # AutoSpeech winner
    "Auto_Tabular",
]
for model_dir in model_dirs:
    sys.path.append(os.path.join(here, model_dir))


def meta_domain_2_model(domain):
    if domain in ["image"]:
        model_dir = "Auto_Image"
        sys.path.append(os.path.join(here, model_dir))
        from Auto_Image.model import Model as AutoImageModel

        return AutoImageModel

    elif domain in ["video"]:
        if VIDEO_SOLUTION_FLAG == "2d":
            model_dir = "Auto_Video"
            sys.path.append(os.path.join(here, model_dir))
            from Auto_Video.model import Model as AutoVideoModel
        elif VIDEO_SOLUTION_FLAG == "3d":
            model_dir = "Auto_Video_3d"
            sys.path.append(os.path.join(here, model_dir))
            # from Auto_Video_3d.model import Model as AutoVideoModel
            from Auto_Video.model import Model as AutoVideoModel
        else:
            print("Error, Wrong Auto_Video Solution flag.")
        return AutoVideoModel

    elif domain in ["text"]:
        from model_nlp import Model as AutoNlpModel

        return AutoNlpModel

    elif domain in ["speech"]:
        # from model_speech import Model as AutoSpeechModel
        from at_speech.model import Model as AutoSpeechModel

        return AutoSpeechModel

    else:
        from Auto_Tabular.model import Model as TabularModel

        return TabularModel


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
        self.domain = infer_domain(metadata)
        train_data_dir = self.metadata.get_dataset_name() + "/*"
        test_data_dir = self.metadata.get_dataset_name().replace("train", "test") + "/*"
        test_metadata_filename = self.metadata.get_dataset_name().replace("train", "test") + "/metadata.textproto"
        warmup_train_cmd = "cat {} >/dev/null".format(train_data_dir)
        warmup_test_cmd = "cat {} >/dev/null".format(test_data_dir)
        # logger.info("Note: test_metadata_filename={}, cmd={},{} AutoDL_G_CONF: {}".format(test_metadata_filename, warmup_train_cmd, warmup_test_cmd, autodl_g_conf_repr))
        logger.info(
            "Note: test_metadata_filename={}, cmd={},{}".format(
                test_metadata_filename, warmup_train_cmd, warmup_test_cmd
            )
        )
        os.system(warmup_train_cmd)
        os.system(warmup_test_cmd)
        logger.info("Note:The inferred domain of current dataset is: {}.".format(self.domain))
        # Domain识别及Model初始化
        # DomainModel = DOMAIN_TO_MODEL[self.domain]
        DomainModel = meta_domain_2_model(self.domain)
        # self.domain_metadata = get_domain_metadata(metadata, self.domain)
        # logger.info("Note:The domain metadata is {}".format(self.domain_metadata))
        self.domain_model = DomainModel(self.metadata)
        self.has_exception = False
        self.y_pred_last = None

    def train(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train
        logger.info("Note:train_process  model.py starts train")

        try:
            # Train the model
            self.domain_model.train(dataset, remaining_time_budget)
            # Update self.done_training
            self.done_training = self.domain_model.done_training

        except Exception as exp:
            self.has_exception = True
            self.done_training = True
            error("Error, model_train exp={}, done_traning={}".format(exp, self.done_training))


    def test(self, dataset, remaining_time_budget=None):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        # Make predictions

        if self.done_training is True or self.has_exception is True:
            return self.y_pred_last

        try:
            Y_pred = self.domain_model.test(dataset, remaining_time_budget=remaining_time_budget)

            self.y_pred_last = Y_pred
            # Update self.done_training
            self.done_training = self.domain_model.done_training

        except MemoryError as mem_error:
            self.has_exception = True
            self.done_training = True
            error("Error, model_test OutOfMemoryError={}, done_traning={}".format(mem_error, self.done_training))
        except Exception as exp:
            self.has_exception = True
            self.done_training = True
            error("Error, model_test exp={}, done_traning={}".format(exp, self.done_training))

        # return Y_pred
        return self.y_pred_last


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
