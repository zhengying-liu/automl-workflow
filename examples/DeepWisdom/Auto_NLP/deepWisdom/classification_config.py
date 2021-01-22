# -*- encoding: utf-8 -*-
"""
@Time    : 2019-08-17 10:49
@Author  : alexanderwu
@Email   : alexanderwu@fuzhi.ai
@Software: PyCharm
"""

import os

# config for final.
FT_GPU_ENV = "PROD" # DEV or PROD
IF_Sniffer = False # Ture or False
IF_Down_Pretrained_Mode = True


# common
MAX_SEQ_LEN = 64
bs = 16
MAX_VOCAB_SIZE = 30000

# fasttext keras
ngram_range = 1
max_features = 20000
# maxlen = 128
maxlen = 400
batch_size = 10 #32
embedding_dims = 100
epochs = 100
EARLY_STOPPING_EPOCH = 5

# glove
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# bert
WITH_TAIL = False
BERT_CHINESE_CHAR = True

USE_ROBERTA = True
mask_padding_with_zero = True


# 分词
USE_CPPJIEBA_PY = False #不用CPP版本.
ChZh_Wordseg_Method = "jieba_fast" # "cppjieba-py, jieba_fast"


# for autosampling
SVM_MAX_AUTOSAMPLE_NUM = 20000
FINETUNE_MAX_AUTOSAMPLE_NUM = 200#2500
Min_Sample_Num_Per_Label = 300
Total_Sample_Num_for_SVM = 30000
Lowbound_Fold_for_Binary = 2



# for finetune
FT_MAX_SEQ_LEN = 128
FT_TRAIN_BATCH_SIZE = 4
FT_EVAL_BATCH_SIZE = 128 #512


# if FT_GPU_ENV == "DEV":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# for sniffer.
Test_Sniffer_Num = 50


class Config(object):
    def __init__(self):
        from keras.optimizers import Adam
        # input configuration
        self.level = 'word'
        self.input_level = 'word'
        self.word_max_len = 400
        self.char_max_len = 200
        self.max_len = {'word': self.word_max_len,
                        'char': self.char_max_len
                        }
        self.han_max_sent = 10
        self.word_embed_dim = 100
        self.word_embed_type = 'glove'
        self.word_embed_trainable = True
        self.word_embeddings = None

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.rnn_units = 300
        self.dense_units = 512

        # model training configuration
        self.batch_size = 128
        self.n_epoch = 50
        self.learning_rate = 0.01
        self.optimizer = Adam(self.learning_rate)
        self.dropout = 0.5
        self.l2_reg = 0.001

        # output configuration
        self.n_class = 3

        # checkpoint configuration
        self.checkpoint_dir = 'ckpt'
        self.checkpoint_monitor = 'val_loss'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stopping configuration
        self.early_stopping_monitor = 'val_loss'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1


class AutoNLPPathConfig(object):
    AutoNLP_Config_File_Dir = os.path.dirname(__file__)
    AutoNLP_Src_Script_Dir = os.path.abspath(os.path.join(AutoNLP_Config_File_Dir, ".."))
    AutoNLP_Pro_Dir = os.path.abspath(os.path.join(AutoNLP_Src_Script_Dir, "..")) # 向上要abspath, 向下不用.
    AutoNLP_Pro_Log_Dir = os.path.join(AutoNLP_Pro_Dir, "logs")
    AutoNLP_Model_Warehouse_Dir = os.path.join(AutoNLP_Pro_Dir, "models_warehouses")


autonlp_pro_log_dir = AutoNLPPathConfig.AutoNLP_Pro_Log_Dir
models_warehouses_dir = AutoNLPPathConfig.AutoNLP_Model_Warehouse_Dir


class AutoNlpLoggerConfig(object):
    LOG_DIR = autonlp_pro_log_dir
    AutoNLP_OFFLINE_LOGNAME = "autonlp_offline"
    AutoNLP_ONLINE_LOGNAME = "autonlp_online"


    AutoNLP_OFFLINE_LOGFILE = os.path.join(LOG_DIR, "autonlp_offline.log")
    AutoNLP_ONLINE_LOGFILE = os.path.join(LOG_DIR, "autonlp_online.log")

    # OFFLINE_LOG_LEVEL = logging.INFO
    # ONLINE_LOG_LEVEL = logging.INFO




class AutoClassificationServiceConfig(object):
    #autonlp_demo_host = "192.168.21.69" # 顶层外部agent_server的host.
    autonlp_demo_host = "0.0.0.0" # 顶层外部agent_server的host.
    autonlp_demo_port = 38008 # 顶层外部agent_server的port.
    # bot_agent_port = 8080# 顶层外部agent_server的port.
    autonlp_demo_route_path = "autonlp_classification"
    bot_agent_server_url = "http://{}:{}/{}".format(autonlp_demo_host, autonlp_demo_port, autonlp_demo_route_path)
    print(bot_agent_server_url)



