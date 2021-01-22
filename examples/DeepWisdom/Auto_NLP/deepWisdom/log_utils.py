# -*- coding: utf-8 -*-
# @Date    : 2019/8/15 11:16
# @Author  : alexanderwu (alexanderwu@fuzhi.ai)
# @Desc    :

from __future__ import absolute_import
from Auto_NLP.deepWisdom.time_utils import logger, debug, info, warning, error, timeit, timeit_ol, log, colored, TimerD, timeit_endl


import logging
from logging.handlers import RotatingFileHandler
from Auto_NLP.deepWisdom.autnlp_config import AutoNlpLoggerConfig


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)

    fmt = "[%(asctime)-15s] [%(levelname)s] %(filename)s [line=%(lineno)d] [PID=%(process)d] %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"

    formatter = logging.Formatter(fmt, datefmt)

    fileHandler = RotatingFileHandler(filename=log_file, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
                             encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志

    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)



# setup_logger(AutoNlpLoggerConfig.AutoNLP_OFFLINE_LOGNAME, AutoNlpLoggerConfig.AutoNLP_OFFLINE_LOGFILE)
# setup_logger(AutoNlpLoggerConfig.AutoNLP_ONLINE_LOGNAME, AutoNlpLoggerConfig.AutoNLP_ONLINE_LOGFILE)

# autonlp_offline_logger = logging.getLogger(AutoNlpLoggerConfig.AutoNLP_OFFLINE_LOGNAME)
# autonlp_online_logger = logging.getLogger(AutoNlpLoggerConfig.AutoNLP_OFFLINE_LOGNAME)

