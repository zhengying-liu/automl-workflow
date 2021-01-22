#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import time
from typing import Any
import logging
import sys

nesting_level = 0


def log_old(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print("{}{}".format(space, entry))

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  # formatter = logging.Formatter(
  #   fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')

  formatter = logging.Formatter('%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s')

  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')
log = logger.info
log_warning = logger.warning
info = logger.info


def timeit(method, start_log=None):
    def wrapper(*args, **kw):
        global nesting_level

        log("Start [{}]:" + (start_log if start_log else "").format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End   [{}]. Time elapsed: {:0.2f} sec.".format(method.__name__, end_time - start_time))
        return result

    return wrapper

