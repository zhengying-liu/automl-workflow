import re
import jieba_fast as jieba

MAX_SEQ_LENGTH = 301
MAX_SEQ_LENGTH_PRE = 128
MAX_SEQ_LENGTH_POST = 256

def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))

def _tokenize_chinese_words_char(text):
    return ' '.join(jieba.cut(text, cut_all=False))


import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Pool
import scipy.sparse as sp




def cut_sentence(cut_type, line_split, ratio=0.1, is_ratio=True):
    """
    截取sentence方式
    :param cut_type:
        0: 默认方式，截取sentence前N个词
        1：前后各截取一定比例
        2：截取关键词
    :return:
    """
    if cut_type == 0:
        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            clean_line = line_split[0:NUM_WORD]
        else:
            clean_line = line_split
        return clean_line

    elif cut_type == 1:
        if is_ratio:
            NUM_WORD_PRE = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH_PRE)
            NUM_WORD_POST = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH_POST)
            NUM_WORD = NUM_WORD_PRE + NUM_WORD_POST
        else:
            NUM_WORD_PRE = MAX_SEQ_LENGTH_PRE
            NUM_WORD_POST = MAX_SEQ_LENGTH_POST
            NUM_WORD = NUM_WORD_PRE + NUM_WORD_POST

        if len(line_split)>NUM_WORD:
            clean_line_pre = line_split[:NUM_WORD_PRE]
            clean_line_post = line_split[-NUM_WORD_POST:]
            clean_line = clean_line_pre+clean_line_post
        else:
            clean_line = line_split
        return clean_line

    elif cut_type==2:
        pass

def clean_en_with_different_cut(dat, ratio=0.1, is_ratio=True, cut_type=1):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;-]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        line_split = line.split()

        cut_line_split = cut_sentence(cut_type, line_split, ratio=ratio, is_ratio=is_ratio)
        line = " ".join(cut_line_split)
        ret.append(line)
    return ret


def clean_en_original(dat, ratio=0.1, is_ratio=True):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;-]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])

        ret.append(line)
    return ret