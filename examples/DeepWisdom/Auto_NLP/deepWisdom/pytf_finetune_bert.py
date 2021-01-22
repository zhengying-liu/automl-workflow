# coding:utf-8
# @Time    : 2019/8/27 2:50 PM
# @Author  : youngz
# @Email    : csyang.zhang@gmail.com
# @File    : pytf_finetune_classifier.py
# @Software: PyCharm
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import shutil
import sys
import random
import time
import numpy as np
import torch
import json
# import tensorflow as tf
# from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
# from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
#                                   BertForSequenceClassification, BertTokenizer,
#                                   RobertaConfig,
#                                   RobertaForSequenceClassification,
#                                   RobertaTokenizer,
#                                   XLMConfig, XLMForSequenceClassification,
#                                   XLMTokenizer, XLNetConfig,
#                                   XLNetForSequenceClassification,
#                                   XLNetTokenizer,
#                                   DistilBertConfig, DistilBertForSequenceClassification,
#                                   DistilBertModel, DistilBertTokenizer,
#                                   )
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForSequenceClassification,
                                  DistilBertModel, DistilBertTokenizer,
                                  )#AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer

# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
#from pytorch_transformers.modeling_bert import n_layers
print("add albert2 ")
print('************')
dir  = os.path.dirname(os.path.abspath(__file__))
print(dir)
full_path = os.path.join(dir,"transformers_")
print('-------------',full_path)
print('************')
sys.path.append(dir)
from transformers_.configuration_bert import BertConfig as AlbertConfig
from transformers_.tokenization_bert import BertTokenizer as AlbertTokenizer
from transformers_.modeling_albert import AlbertForSequenceClassification
# add sys.path.
cur_file_path = os.path.dirname(__file__)
cur_sample_code_submission_dir = os.path.abspath(os.path.join(cur_file_path, ".."))
sys.path.append(cur_sample_code_submission_dir)
print("Cur sys path = %s" %(sys.path))


from Auto_NLP.deepWisdom.pytf_finetune_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, np_array_auc, FinetuneDataProcessor, ProjectPathConfig)
from Auto_NLP.deepWisdom.time_utils import TimerD
from Auto_NLP.deepWisdom.log_utils import info
from Auto_NLP.deepWisdom.classification_config import FT_MAX_SEQ_LEN, FT_TRAIN_BATCH_SIZE, FT_EVAL_BATCH_SIZE, IF_Down_Pretrained_Mode
import json


"""
    直接接受 x_train, y_train 来构建 processor..
"""
en_model_path = ProjectPathConfig.Pretrained_En_Distil_Roberta_Path
zh_model_path = ProjectPathConfig.Pretrained_Zh_Albert_Path

en_model_file = ProjectPathConfig.Pretrained_En_Distil_Roberta_Model_Path
zh_model_file = ProjectPathConfig.Pretrained_Zh_Albert_Model_Path

def pretrained_models_download():
    if not os.path.exists(en_model_file):
        # os.makedirs(en_model_path)
        info("download en model...")
        os.system("wget -P {} http: // 120.27.216.109:8011/nlp/en_distilroberta/pytorch_model.bin".format(en_model_path))


    if not os.path.exists(zh_model_file):
        info("download zh model...")
        os.system("wget -P {} http://120.27.216.109:8011/nlp/zh_albert_base/pytorch_model.bin".format(zh_model_path))


    pass


if IF_Down_Pretrained_Mode:
    pretrained_models_download()
    pass


def save_dict(file,json_dict):
    string = json.dumps(json_dict)
    #with open('o2train_logs_{}_frozentop{}.json'.format(file,n_layers),'w')as f:
    with open('o2train_logs_{}_frozentop{}.json'.format(file, 0),'w')as f:
        f.write(string)
        f.close()
    return

def softmax(x):
    """Compute the softmax of vector x."""

    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distil_roberta':(RobertaConfig, RobertaForSequenceClassification,RobertaTokenizer),
    'distil_bert':(DistilBertConfig, DistilBertForSequenceClassification, DistilBertConfig),
    'albert':(AlbertConfig, AlbertForSequenceClassification,AlbertTokenizer)

}
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class FineTuneBertConfig(object):
    def __init__(self, metadata, model_type="roberta"):
        # must paras
        self.metadata = metadata
        self.language = self.metadata.get("language")
        self.model_type = model_type
        self.model_name_or_path = None
        # 根据language来确定 model_type 和 pretrained_model_path.
        self.setup_model_name_path()
        self.task_name = "autonlp_data_{}".format(self.metadata.get("train_num"))
        self.output_dir = "tmp_{}_output".format(self.task_name)

        # optional paras.
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.max_seq_length = FT_MAX_SEQ_LEN # 256
        self.do_train = True
        self.do_eval = False # 开始只train, test, 不 eval.
        self.evaluate_during_training = False # 开始只train/test，不eval
        self.do_lower_case = True
        self.per_gpu_train_batch_size = FT_TRAIN_BATCH_SIZE
        self.per_gpu_eval_batch_size = FT_EVAL_BATCH_SIZE
        self.gradient_accumulation_steps = 1
        # self.learning_rate = 2e-5
        self.learning_rate = 5e-5
        self.weight_decay = 5e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0 # maybe changed.
        self.num_train_epochs = 1.0 # fixme: 1 or more.
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 150
        self.save_steps = 300
        self.eval_all_checkpoints = True#fixme: checkpoint 的 global-steps, 在 eval 的时候怎么选择?
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.local_rank = -1
        self.server_ip = ""
        self.server_port = ""

        self.task_name = self.task_name.lower()
        self.output_mode = "classification"

        self.num_labels = self.metadata.get("class_num")

        # 对output_dir目录，保证最新.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            shutil.rmtree(self.output_dir)  # Removes all the subdirectories!
            os.makedirs(self.output_dir)


    def to_string(self):
        finetune_config_json = {
            # "data_dir": self.data_dir,
            "model_type": self.model_type,
            "model_name_or_path": self.model_name_or_path,
            "task_name": self.task_name,
            "output_dir": self.output_dir
        }
        # self.logger.info("Finetune bert config info = %s" %(finetune_config_json))
        info("Finetune bert config info = %s" %(finetune_config_json))
        return finetune_config_json


    def setup_model_name_path(self):
        """
        根据当前 language 和 model_type 来确定 model_path.
        :param language:
        :return:

        """
        if self.language == "EN":
            self.model_type = ProjectPathConfig.Pretrained_Model_Type_Distil_Roberta
            self.model_name_or_path = ProjectPathConfig.Pretrained_En_Distil_Roberta_Path

        else:
            self.model_type = ProjectPathConfig.Pretrained_Model_Type_Albert
            self.model_name_or_path = ProjectPathConfig.Pretrained_Zh_Albert_Path


    def setup_cuda(self):
        if self.local_rank == -1 or self.no_cuda:
            print("torch.cuda.is_available()",torch.cuda.is_available())
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
            info("local_rank = %d, and n_gpu = %d" %(self.local_rank, self.n_gpu))
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.n_gpu = 1
            info("local_rank = %d, and n_gpu = %d, will use nccl backend." %(self.local_rank, self.n_gpu))
        self.device = device
        info("Device original n_gpu = %d" %(self.n_gpu))
        # must be 1.
        self.n_gpu = 1
        info("Change n_gpu = 1 = %d" %(self.n_gpu))
        self.logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   self.local_rank, self.device, self.n_gpu, bool(self.local_rank != -1), self.fp16)


    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)


    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)


    def setup_processors(self, x_data, y_data, data_mode="train"):
        """
        要求：setup_processors 的模式必须在前.
        :param x_data:
        :param y_data:
        :param data_mode:
        :return:
        """
        # Prepare GLUE task
        # 用固定的 processor.

        # if self.task_name not in processors:
        #     raise ValueError("Task not found: %s" % (self.task_name))
        if data_mode == "train":
            self.train_data_processor = FinetuneDataProcessor(x_data, y_data, set_type="train")
            self.label_list = self.train_data_processor.get_labels()
            pass
        elif data_mode == "dev":
            self.dev_data_processor = FinetuneDataProcessor(x_data, y_data, set_type="dev")
            pass
        elif data_mode == "test":
            self.test_data_processor = FinetuneDataProcessor(x_data, y_data, set_type="test")
            pass
        # self.output_mode = output_modes[self.task_name]
        # model 固定.

    def load_prertrained_models_tokenizer(self):
        info("Finetuen Training/evaluation parameters %s" %(self.to_string()))
        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model_type = self.model_type.lower()
        print("[info]model type",self.model_type)
        config_class, self.model_class, tokenizer_class = MODEL_CLASSES[self.model_type]

        info("load pretrained model, config class type = %s, model_class type = %s, tokenizer_class = %s" %(type(config_class),  type(self.model_class), type(tokenizer_class)))
        info("load pretrained model, config class = %s, model_class = %s, tokenizer_class = %s" %(config_class, self.model_class, tokenizer_class))

        self.config = config_class.from_pretrained(self.config_name if self.config_name else self.model_name_or_path,
                                              num_labels=self.num_labels, finetuning_task=self.task_name)

        info("load pretrained model, config type = %s, config = %s" %(type(self.config), self.config))

        self.tokenizer = tokenizer_class.from_pretrained(self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
                                                    do_lower_case=self.do_lower_case)

        info("load pretrained model, tokenizer type = %s, content = %s" %(type(self.tokenizer), self.tokenizer))

        self.model = self.model_class.from_pretrained(self.model_name_or_path, from_tf=bool('.ckpt' in self.model_name_or_path),
                                            config=self.config)
        info("load pretrained model, model type = %s" %(type(self.model)))

        ############################## 定义哪些层冻结 ######################################
        if self.model_type == 'distil_roberta':
            for param in self.model.roberta.encoder.layer[:3].parameters():
                param.requires_grad = False

        if self.local_rank == 0:
            info("load pretrained model, local_rank = 0 ")
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(self.device)

        info("Training/evaluation parameters %s" %(self.to_string()))


    def setup_init_load(self):
        """
        注意：顺序不能变.
        :return:
        """
        self.setup_logger()
        self.setup_cuda()
        self.set_seed()



class FineTuneBertModel(object):
    def __init__(self, metadata):
        self.metadata = metadata
        self.finetune_config = FineTuneBertConfig(self.metadata)
        self.finetune_config.setup_init_load()
        # 顺序需要，load 中要用到 num_labels
        self.finetune_config.load_prertrained_models_tokenizer()
        self.logger = self.finetune_config.logger

        # need to assgin.
        self.train_dataset = None

        self.time_checker = TimerD()

        # record info need to be summaryed.
        self.finetune_summary_info = dict()
        self.finetune_summary_info["metadata"] = self.metadata
        self.finetune_summary_info["max_seq_len"] = self.finetune_config.max_seq_length
        self.finetune_summary_info["train_batch_size"] = self.finetune_config.per_gpu_train_batch_size
        self.finetune_summary_info["predict_batch_size"] = self.finetune_config.per_gpu_eval_batch_size

        self.model = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    #def load_and_cache_examples(self, task, tokenizer, evaluate=False):
    def load_and_cache_examples(self, task, tokenizer, data_tag="train"):
        """
        目的是返回 TensorDataset, 可能是Train数据，可能是 Dev/Test数据.
        tag可选为 train/dev/test
        :param task:
        :param tokenizer:
        :param evaluate:
        :return: TensorDataset类型。如果第一次，读取features的同时保存features到cache。如果是第一次之后，直接从cache中读取.
        """
        if self.finetune_config.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # processor = processors[task]()
        output_mode = self.finetune_config.output_mode
        # Load data features from cache or dataset file. 第一次是从数据processor的examples来，后面都从 cache来.
        #cached_features_file = os.path.join(self.finetune_config.data_dir, 'cached_{}_{}_{}_{}'.format(
        # 通过data_tag来区分, train/dev/test, 分别在不同的cache_file.
        cached_features_file = os.path.join(os.path.dirname(__file__), 'cached_{}_{}_{}_{}'.format(
            # 'dev' if evaluate else 'train',
            data_tag,
            self.finetune_config.model_type,
            str(self.finetune_config.max_seq_length),
            str(task)))

        # 如果第一次，走else流程，从xx_processors中先拿到examples，再转换生成features. 第一次之后只走if流程.
        if os.path.exists(cached_features_file) and not ProjectPathConfig.If_Overwrite_cache:
            print("!!!!!!!!!!!!\n\n")
            info("Data_tag = %s, Loading features from cached file %s" %(data_tag, cached_features_file))
            features = torch.load(cached_features_file)
        else:
            # self.logger.info("Creating features from dataset file at %s", self.finetune_config.data_dir)
            # 注意，label_list只从 train_processor中获取.
            label_list = self.finetune_config.train_data_processor.get_labels()
            # 不会用到,下面的路不会走到.
            if task in ['mnli', 'mnli-mm'] and self.finetune_config.model_type in ['roberta']:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]

            # 第一遍eval是False，因为来train.
            if data_tag == "train":
                examples = self.finetune_config.train_data_processor.get_examples()
                info("Datatag = train, get examples from train_data_processor.")
                pass
            elif data_tag == "dev":
                examples = self.finetune_config.dev_data_processor.get_examples()
                info("Datatag = dev, get examples from dev_data_processor.")
                pass
            elif data_tag == "test":
                examples = self.finetune_config.test_data_processor.get_examples()
                info("Datatag = test, get examples from test_data_processor.")
                pass
            else:
                pass
            # examples = processor.get_dev_examples(self.finetune_config.data_dir) if evaluate else processor.get_train_examples(
            #     self.finetune_config.data_dir)
            # 从 example到features.

            # fixme: 不用预设的最大文本长度，用当前文本最大长度(动态文本长度）

            features = convert_examples_to_features(examples, label_list, self.finetune_config.max_seq_length, tokenizer, output_mode,
                                                    cls_token_at_end=bool(self.finetune_config.model_type in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if self.finetune_config.model_type in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=False, # bool(self.finetune_config.model_type in ['roberta'])
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(self.finetune_config.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.encoder[tokenizer.pad_token] if self.finetune_config.model_type in [
                                                        'roberta'] else tokenizer.encode(tokenizer.pad_token)[0],
                                                        #'roberta'] else tokenizer.vocab[tokenizer.pad_token],
                                                    pad_token_segment_id=4 if self.finetune_config.model_type in ['xlnet'] else 0,
                                                    )
            if self.finetune_config.local_rank in [-1, 0]:
                info("Saving features into cached file %s", cached_features_file)
                # 第一次的时候，把所有features都保存到缓存cache里。注意：train的cache和dev/test的cache是分开的.
                torch.save(features, cached_features_file)

        if self.finetune_config.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset



    def model_predict(self, model, tokenizer, prefix="", data_tag="test"):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        # 在这里task_name的变化没用.
        # eval_task_names = ("mnli", "mnli-mm") if self.finetune_config.task_name == "mnli" else (self.finetune_config.task_name,)
        predict_task = self.finetune_config.task_name
        # eval_outputs_dirs = (self.finetune_config.output_dir, self.finetune_config.output_dir + '-MM') if self.finetune_config.task_name == "mnli" else (self.finetune_config.output_dir,)
        predict_output_dir = self.finetune_config.output_dir

        # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # 获取待predict的数据dataset，通过 data_tag 控制操作，前面需要事前设置 processor.
        # 如果是test，为了喂数据的格式符合，labels还是要存在，只是后面不用，和train/eval不同.

        time_load_cache_example_start = time.time()
        test_dataset = self.load_and_cache_examples(predict_task, tokenizer, data_tag = data_tag)
        time_load_cache_example_end = time.time()
        self.finetune_summary_info["test_loadcache_time"] = time_load_cache_example_end - time_load_cache_example_start
        self.time_checker.check("Test predict, load cache examples done.")

        # 逻辑就是 output 要新建，如果存在则删除新建.
        if not os.path.exists(predict_output_dir) and self.finetune_config.local_rank in [-1, 0]:
            os.makedirs(predict_output_dir)
            pass
        # 这里n_gpu=1，所以没用.
        self.finetune_config.eval_batch_size = self.finetune_config.per_gpu_eval_batch_size * max(1, self.finetune_config.n_gpu)
        # Note that DistributedSampler samples randomly
        # 对原始数据采样，得到分batch之后结果.
        test_sampler = SequentialSampler(test_dataset) if self.finetune_config.local_rank == -1 else DistributedSampler(test_dataset)
        # dataloader, 有 list 类型操作.
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.finetune_config.eval_batch_size)

        # Predict for Test. Eval!
        info("***** Running prediction, for test{} *****".format(prefix))
        info("  Num examples = %d", len(test_dataset))
        info("  Batch size = %d", self.finetune_config.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        bert_embedding = None

        time_predict_batch_start = time.time()
        for batch in tqdm(test_dataloader, desc="Predicting"):
            # 注意这里和 model.train不同，test的时候也是 model.eval().
            model.eval()
            # 把待预测的数据，分别写到device GPU里.
            batch = tuple(t.to(self.finetune_config.device) for t in batch)

            with torch.no_grad():
                # inputs = {'input_ids': batch[0],
                #           'attention_mask': batch[1],
                #           'token_type_ids': batch[2] if self.finetune_config.model_type in ['bert', 'xlnet'] else None,
                #           # XLM and RoBERTa don't use segment_ids
                #           'labels': batch[3]}
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.finetune_config.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.finetune_config.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                info("model type = %s" %(type(model)))

                # 这里只是为了获得 bert 的导数第二层的embedding.
                # bert_model = model.bert
                # a = bert_model(batch[0])
                # batch_hidden_state = a[1]

                # 直接根据model进行预测.
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                # 预测的结果从 gpu到cpu到numpy内容形式.
                preds = logits.detach().cpu().numpy()
                # 在test的时候这个没用，也不评估eval, 不用算loss.
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                # bert_embedding = batch_hidden_state.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                # bert_embedding = np.append(bert_embedding, batch_hidden_state.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        time_predict_batch_end = time.time()
        self.finetune_summary_info["test_predict_batch_time"] = time_predict_batch_end - time_predict_batch_start

        self.time_checker.check("Test predict, predict batches done.")

        # need pred float point ad out_label_ids

        # 查看直接预测的结果: preds 是预测分数，numpy形式；out_label_ids 是预测的labels的编号，numpy类型.
        #info("Predict preds, origin scores type = %s, shape = %s, content-top3= %s" %(type(preds), preds.shape, str(preds[:3])))
        #info("Only Show preparetion, do not train or eval, output_label_ids, type = %s, shape = %s, content-top3 = %s" %(type(out_label_ids), out_label_ids.shape, str(out_label_ids[:3])))
        # eval_loss = eval_loss / nb_eval_steps
        if self.finetune_config.output_mode == "classification":
            pass


        elif self.finetune_config.output_mode == "regression":
            preds = np.squeeze(preds)

        # 返回predict的label所在索引号结果.
        return preds

    # 先不走 evaluate, train完直接predict，给外部 eval.
    def evaluate_model(self, model, tokenizer, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if self.finetune_config.task_name == "mnli" else (self.finetune_config.task_name,)
        eval_outputs_dirs = (self.finetune_config.output_dir, self.finetune_config.output_dir + '-MM') if self.finetune_config.task_name == "mnli" else (self.finetune_config.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(eval_task, tokenizer, data_tag = "dev")

            if not os.path.exists(eval_output_dir) and self.finetune_config.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            self.finetune_config.eval_batch_size = self.finetune_config.per_gpu_eval_batch_size * max(1, self.finetune_config.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset) if self.finetune_config.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.finetune_config.eval_batch_size)

            # Eval!
            info("***** Running evaluation {} *****".format(prefix))
            info("  Num examples = %d", len(eval_dataset))
            info("  Batch size = %d", self.finetune_config.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            bert_embedding = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(self.finetune_config.device) for t in batch)

                with torch.no_grad():
                    # inputs = {'input_ids': batch[0],
                    #           'attention_mask': batch[1],
                    #           'token_type_ids': batch[2] if self.finetune_config.model_type in ['bert', 'xlnet'] else None,
                    #           # XLM and RoBERTa don't use segment_ids
                    #           'labels': batch[3]}
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if self.finetune_config.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if self.finetune_config.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    info("model type = %s" %(type(model)))

                    # bert_model = model.bert
                    # a = bert_model(batch[0])
                    # batch_hidden_state = a[1]

                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                    # bert_embedding = batch_hidden_state.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    # bert_embedding = np.append(bert_embedding, batch_hidden_state.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            # need pred float point ad out_label_ids

            eval_loss = eval_loss / nb_eval_steps
            if self.finetune_config.output_mode == "classification":
                pass

            elif self.finetune_config.output_mode == "regression":
                preds = np.squeeze(preds)


            # 返回predict的label所在索引号结果.
            return preds

    def eval_bert_simple(self, simple_model, model, tokenizer, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if self.finetune_config.task_name == "mnli" else (self.finetune_config.task_name,)
        eval_outputs_dirs = (self.finetune_config.output_dir, self.finetune_config.output_dir + '-MM') if self.finetune_config.task_name == "mnli" else (self.finetune_config.output_dir,)

        results = {}
        results['stage']='eval_bert_simple'
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(eval_task, tokenizer, evaluate=True)

            if not os.path.exists(eval_output_dir) and self.finetune_config.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            self.finetune_config.eval_batch_size = self.finetune_config.per_gpu_eval_batch_size * max(1, self.finetune_config.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset) if self.finetune_config.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.finetune_config.eval_batch_size)

            # Eval!
            info("***** Running bert simple evaluation {} *****".format(prefix))
            info("  Num examples = %d", len(eval_dataset))
            info("  Batch size = %d", self.finetune_config.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            bert_embedding = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(self.finetune_config.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if self.finetune_config.model_type in ['bert', 'xlnet'] else None,
                              # XLM and RoBERTa don't use segment_ids
                              'labels': batch[3]}

                    bert_model = model.bert
                    a = bert_model(batch[0])
                    batch_hidden_state = a[1]

                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                    bert_embedding = batch_hidden_state.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    bert_embedding = np.append(bert_embedding, batch_hidden_state.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            # need pred float point ad out_label_ids

            eval_loss = eval_loss / nb_eval_steps
            if self.finetune_config.output_mode == "classification":
                probs = self.sigmoid(preds)
                # preds = np.argmax(preds, axis=1)
                preds = np.argmax(probs, axis=1)

                #bert_classificar_auc = roc_auc_score(out_label_ids, preds)
                bert_classificar_auc = np_array_auc(out_label_ids, preds)
                #bert_classificar_auc = roc_auc_score(out_label_ids, probs[:, 1])

                #bert_probs = simple_model.predict_proba(bert_embedding)
                bert_probs = simple_model.predict(bert_embedding)
                #bert_simple_auc_on_eval = roc_auc_score(out_label_ids, bert_probs)#[:, 1]
                bert_simple_auc_on_eval = np_array_auc(out_label_ids, bert_probs)#[:, 1]
                info("bert auc on evaluate is {}".format(bert_simple_auc_on_eval))
                # print(auc)

            elif self.finetune_config.output_mode == "regression":
                preds = np.squeeze(preds)

            result = compute_metrics(eval_task, preds, out_label_ids)
            if self.finetune_config.output_mode == "classification":
                result['bert_cls_auc_on_eval'] = bert_classificar_auc
                result['bert_simple_auc_on_eval'] = bert_simple_auc_on_eval
            results.update(result)
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results
        pass

    def train_model(self, train_dataset, model, tokenizer):
        """ Train the model """
        eval_logs = []
        if self.finetune_config.local_rank in [-1, 0]:
            pass

        # 原来支持多GPU，但是现在强制 n_gpu=1. 不影响.
        self.finetune_config.train_batch_size = self.finetune_config.per_gpu_train_batch_size * max(1, self.finetune_config.n_gpu)
        # 还是本地，不分布式.
        train_sampler = RandomSampler(train_dataset) if self.finetune_config.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.finetune_config.train_batch_size)

        if self.finetune_config.max_steps > 0:
            # 表示设置最大次数上限.
            t_total = self.finetune_config.max_steps
            self.finetune_config.num_train_epochs = self.finetune_config.max_steps // (len(train_dataloader) // self.finetune_config.gradient_accumulation_steps) + 1
        else:
            # 表示不设置最大上限，完全根据 train_dataloader长度来.
            t_total = len(train_dataloader) // self.finetune_config.gradient_accumulation_steps * self.finetune_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.finetune_config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.finetune_config.learning_rate, eps=self.finetune_config.adam_epsilon)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.finetune_config.warmup_steps, t_total=t_total)
        scheduler =get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.finetune_config.warmup_steps, num_training_steps=t_total
        )
        if self.finetune_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.finetune_config.fp16_opt_level)

        # 不做多GPU支持.
        # multi-gpu training (should be after apex fp16 initialization)
        info("n_gpu = %d, model origin type = %s" %(self.finetune_config.n_gpu, type(model)))
        if self.finetune_config.n_gpu > 1:
            info("n_gpu = %d, model origin type = %s" %(self.finetune_config.n_gpu, type(model)))
            model = torch.nn.DataParallel(model)
            info("n_gpu = %d, model after type = %s" %(self.finetune_config.n_gpu, type(model)))

        # -1, 表示就用本地模式.
        # Distributed training (should be after apex fp16 initialization)
        if self.finetune_config.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.finetune_config.local_rank],
                                                              output_device=self.finetune_config.local_rank,
                                                              find_unused_parameters=True)

        # Train!
        info("***** Running training *****")
        info("  Num examples = %d", len(train_dataset))
        info("  Num Epochs = %d", self.finetune_config.num_train_epochs)
        info("  Instantaneous batch size per GPU = %d", self.finetune_config.per_gpu_train_batch_size)
        info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.finetune_config.train_batch_size * self.finetune_config.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if self.finetune_config.local_rank != -1 else 1))
        info("  Gradient Accumulation steps = %d", self.finetune_config.gradient_accumulation_steps)
        info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(self.finetune_config.num_train_epochs), desc="Epoch", disable=self.finetune_config.local_rank not in [-1, 0])
        self.finetune_config.set_seed()  # Added here for reproductibility (even between python 2 and 3)
        start_time = time.time()
        time_spent_evals = []
        i = 0

        time_train_iterator_start = time.time()
        for _ in train_iterator:
            i+=1
            info("Finetune bert, train_model train_iterator i = %d" %(i))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.finetune_config.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.finetune_config.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.finetune_config.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.finetune_config.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.finetune_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.finetune_config.gradient_accumulation_steps > 1:
                    loss = loss / self.finetune_config.gradient_accumulation_steps

                if self.finetune_config.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.finetune_config.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.finetune_config.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % self.finetune_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logging_loss = tr_loss
                if self.finetune_config.max_steps > 0 and global_step > self.finetune_config.max_steps:
                    epoch_iterator.close()
                    break
            if self.finetune_config.max_steps > 0 and global_step > self.finetune_config.max_steps:
                train_iterator.close()
                break

        if self.finetune_config.local_rank in [-1, 0]:
            pass

        time_train_iterator_end = time.time()
        self.finetune_summary_info["train_iterator_time"] = time_train_iterator_end - time_train_iterator_start

        # fixme: 最后一次一定保存 checkpoint. Save model checkpoint
        output_dir = os.path.join(self.finetune_config.output_dir, 'checkpoint-{}'.format(global_step))
        import pathlib
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        info("Finally Saving model checkpoint to %s", output_dir)
        return global_step, tr_loss / global_step , eval_logs, model


    def train_model_process(self, train_x_data: list, train_y_labels: list, model):
        train_start = time.time()
        # 初始化 train_data_processor.喂进去 train_data, 得到对应 train_processors.
        self.finetune_config.setup_processors(train_x_data, train_y_labels, data_mode="train")

        # train 的时候，通过 evaluate 字段来控制.
        self.train_dataset = self.load_and_cache_examples(self.finetune_config.task_name, self.finetune_config.tokenizer, data_tag="train")
        info("self.train_dataset type = %s " %(type(self.train_dataset)))
        if model is None:
            info("finetune fisrt time on pretrain_model!")
            global_step, tr_loss, eval_logs, model  = self.train_model(self.train_dataset, self.finetune_config.model, self.finetune_config.tokenizer)
            self.model = model
        else:
            info("finetune Not fisrt time on finetune_model!")
            global_step, tr_loss, eval_logs, model = self.train_model(self.train_dataset, model,
                                                                      self.finetune_config.tokenizer)


        info(" global_step = %s, average loss = %s", global_step, tr_loss)

        train_end = time.time()
        info(eval_logs)
        info('train used {}sec'.format(train_end - train_start))
        return model


    def eval_model_process(self, test_x_data: list):

        results = {}
        checkpoints = [self.finetune_config.output_dir]
        if self.finetune_config.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(self.finetune_config.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            path_abs = os.path.abspath(checkpoint)
            info(path_abs)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = self.finetune_config.model_class.from_pretrained(checkpoint)
            model.to(self.finetune_config.device)
            result,lr = self.evaluate_model( model, self.finetune_config.tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    def model_eval_process(self, eval_x_data, eval_y_label, model):
        # 初始化 train_data_processor.喂进去 train_data, 得到对应 train_processors.
        self.finetune_config.setup_processors(eval_x_data, eval_y_label, data_mode="dev")
        eval_results = self.model_predict(model=model, tokenizer=self.finetune_config.tokenizer, data_tag="dev")
        return eval_results

    def model_predict_process(self, test_x_data: list, model):
        """
        :param test_x_data:
        :return: 针对样本预测的label列表.
        """
        # 初始化 test_data_processor.喂进去 test_data, 得到对应 test_processors.
        # test 模式下 test_y_labels 只是桩数据.
        test_y_labels = [1 for i in range(len(test_x_data))]
        self.finetune_config.setup_processors(test_x_data, test_y_labels, data_mode="test")

        # train 的时候，通过 evaluate 字段来控制.
        results = {}
        checkpoints = [self.finetune_config.output_dir]
        if self.finetune_config.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(self.finetune_config.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # 默认情况下只走一个.
        info("Evaluate the following checkpoints: %s", checkpoints)
        model.to(self.finetune_config.device)
        # 这里用 model_predict.
        predict_results = self.model_predict(model= model, tokenizer=self.finetune_config.tokenizer)
        return predict_results

