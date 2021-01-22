# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from keras import backend as K
import tensorflow as tf
import torch
import numpy as np



from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def fauc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return (auc * 2) - 1
#r = auc(tf.zeros((1,3)),tf.zeros((1,3)))
# r = auc(tf.constant([[0,1,0],[0,0,1],[0,0,1]]),tf.constant([[0,0,1],[0,0,1],[0,0,1]]))
# print(K.get_session().run(r))

def np_array_auc(y_true,y_pred):
    _y_true = []
    _y_pred = []
    nums = 20
    for y in y_true:
        row = [0]*nums
        row[y] = 1
        _y_true.append(row)

    for y in y_pred:
        row = [0] * nums
        row[y] = 1
        _y_pred.append(row)

    y_true = tf.constant(_y_true)
    y_pred = tf.constant(_y_pred)
    r = auc(y_true,y_pred)
    return float(K.get_session().run(r))

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class O1Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        input_file_data = os.path.join(data_dir,'O1','O1.data','train.data')
        input_file_label = os.path.join(data_dir,'O1','O1.data','train.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return  self._create_examples(pair, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        input_file_data = os.path.join(data_dir,'O1','O1.data','test.data')
        input_file_label = os.path.join(data_dir,'O1','O1.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return self._create_examples(pair, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class FinetuneDataProcessor():
    def __init__(self, x_train, y_train, set_type="train"):
        """
        接受model输入 x_train, y_train or x_test, y_test.
        :param x_train: 类型为 list of string.
        :param y_train: 类型可以为 numpy of int.
        :param set_type: train的时候可以为 train, test 的时候为 eval.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.y_train = [str(y) for y in self.y_train]
        # 先转int
        self.y_train_labels = [int(y) for y in list(set(self.y_train))]
        # 第二，sort
        self.y_train_labels.sort()
        # 第三，再 string.
        self.y_train_labels = [str(y) for y in self.y_train_labels]

        self.set_type = set_type
        pass

    def get_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        # y_train must be type of list of str
        pair = [[x, y] for x,y in zip(self.x_train, self.y_train)]

        return  self._create_examples(pair, self.set_type)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.y_train_labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = None  # 对于句子分类任务，text_b = None.
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class O2Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        input_file_data = os.path.join(data_dir,'O2','O2.data','train.data')
        input_file_label = os.path.join(data_dir,'O2','O2.data','train.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return  self._create_examples(pair, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        input_file_data = os.path.join(data_dir,'O2','O2.data','test.data')
        input_file_label = os.path.join(data_dir,'O2','O2.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return self._create_examples(pair, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(i) for i in range(20)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class O3Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        input_file_data = os.path.join(data_dir,'O3','O3.data','train.data')
        input_file_label = os.path.join(data_dir,'O3','O3.data','train.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return  self._create_examples(pair, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        input_file_data = os.path.join(data_dir,'O3','O3.data','test.data')
        input_file_label = os.path.join(data_dir,'O3','O3.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return self._create_examples(pair, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(i) for i in range(2)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class O4Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        input_file_data = os.path.join(data_dir,'O4','O4.data','train.data')
        input_file_label = os.path.join(data_dir,'O4','O4.data','train.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return  self._create_examples(pair, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        input_file_data = os.path.join(data_dir,'O4','O4.data','test.data')
        input_file_label = os.path.join(data_dir,'O4','O4.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return self._create_examples(pair, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(i) for i in range(10)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class O5Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        input_file_data = os.path.join(data_dir,'O5','O5.data','train.data')
        input_file_label = os.path.join(data_dir,'O5','O5.data','train.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return  self._create_examples(pair, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        input_file_data = os.path.join(data_dir,'O5','O5.data','test.data')
        input_file_label = os.path.join(data_dir,'O5','O5.solution')
        with open(input_file_data, "r", encoding="utf-8-sig") as f:
            xs = f.read().splitlines()
        with open(input_file_label, "r", encoding="utf-8-sig") as f:
            ys = f.read().splitlines()
            ys =[y.split(' ').index('1') for y in ys]
            ys = [str(y) for y in ys]
        pair = [[x,y] for x,y in zip(xs,ys)]
        return self._create_examples(pair, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(i) for i in range(18)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    print('you cannot go inside')
    label_map = {label : i for i, label in enumerate(label_list)}
    print(label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info([str(x) for x in tokens])
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

# key
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "o1":
        return acc_and_f1(preds, labels)
    elif task_name == "o3":
        return acc_and_f1(preds, labels)
    elif task_name == "o2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "o4":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "o5":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "o1": O1Processor,
    "o2": O2Processor,
    "o3": O3Processor,
    "o4": O4Processor,
    "o5": O5Processor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "o1": "classification",
    "o2": "classification",
    "o3": "classification",
    "o4": "classification",
    "o5": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "o1": 2,
    "o2": 20,
}



# 记录项目各种路径位置的Config.
class ProjectPathConfig(object):
    Code_Utils_Path = os.path.abspath(os.path.dirname(__file__))
    Code_Sub_Path = os.path.abspath(os.path.join(Code_Utils_Path, "..", ".."))
    Pretrained_Models_Path = os.path.join(Code_Sub_Path, "pretrain_models")
    Pretrained_En_Bert_Path = os.path.join(Pretrained_Models_Path, "en")
    Pretrained_En_RoBerta_Path = os.path.join(Pretrained_Models_Path, "en-roberta")
    Pretrained_En_XLNet_Path = os.path.join(Pretrained_Models_Path, "en-xlnet")
    Pretrained_Zh_Bert_Path = os.path.join(Pretrained_Models_Path, "zh")
    Pretrained_Zh_Albert_Path = os.path.join(Pretrained_Models_Path, "zh_albert_base")
    Pretrained_MultiLingual_Distil_Bert_Path =  os.path.join(Pretrained_Models_Path, "multilngual_distil_bert")
    Pretrained_Zh_Char_Path = os.path.join(Pretrained_Models_Path, "zh-char")
    Pretrained_Zh_XLNet_Path = os.path.join(Pretrained_Models_Path, "zh-xlnet")
    Pretrained_En_Distil_Roberta_Path = os.path.join(Pretrained_Models_Path, "en_distilroberta")


    # pretrained use path.
    Pretrained_En_RoBerta_Model_Path = os.path.join(Pretrained_En_RoBerta_Path, "pytorch_model.bin")
    Pretrained_Zh_Bert_Model_Path = os.path.join(Pretrained_Zh_Bert_Path, "pytorch_model.bin")
    Pretrained_En_Distil_Roberta_Model_Path = os.path.join(Pretrained_En_Distil_Roberta_Path, "pytorch_model.bin")
    #Pretrained_Zh_XLNet_Model_Path = os.path.join(Pretrained_Zh_XLNet_Path, "pytorch_model.bin")
    Pretrained_Zh_Albert_Model_Path = os.path.join(Pretrained_Zh_Albert_Path, "pytorch_model.bin")
    Pretrained_MultiLingual_Distil_Bert_Model_Path = os.path.join(Pretrained_MultiLingual_Distil_Bert_Path, "pytorch_model.bin")

    # model type config.
    Pretrained_Model_Type_Bert = "bert"
    Pretrained_Model_Type_XLNet = "xlnet"
    Pretrained_Model_Type_Xlm = "xlm"
    Pretrained_Model_Type_RoBerta = "roberta"
    Pretrained_Model_Type_Distil_Roberta = "distil_roberta"
    Pretrained_Model_Type_Albert = "albert"
    Pretrained_Model_Type_Distil_Bert = "distil_bert"


    # config for cache and output_dir.
    If_Overwrite_cache = True
    If_Overwrite_outputdir = True


    pass


def main():
    print(ProjectPathConfig.Pretrained_Models_Path)
    pass

if __name__ == '__main__':
    main()
