# -*- coding: utf-8 -*-
# @Date    : 2020/1/17 14:32
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import os
import re
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
# print(stopwords)
from keras.preprocessing import text
from keras.preprocessing import sequence
from nltk.stem.snowball import EnglishStemmer, SnowballStemmer
stemmer = SnowballStemmer('english')

# MAX_SEQ_LENGTH = 301
MAX_SEQ_LENGTH = 601
MAX_CHAR_LENGTH = 96
MAX_EN_CHAR_LENGTH = 35
import multiprocessing
from multiprocessing import Pool
with open(os.path.join(os.path.dirname(__file__), "en_stop_words_nltk.txt"), "r+",encoding='utf-8') as fp:
    nltk_english_stopwords = fp.readlines()
    nltk_english_stopwords = [word.strip() for word in nltk_english_stopwords]

full_stop_words = list(stopwords)+nltk_english_stopwords
NCPU = multiprocessing.cpu_count() - 1

def set_mp(processes=4):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except BaseException:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret

def clean_en_text(dat, ratio=0.1, is_ratio=True, vocab=None, rmv_stop_words=True):

    trantab = str.maketrans(dict.fromkeys(string.punctuation+"@!#$%^&*()-<>[]<=>;:?.\/+[\\]^_`{|}~\t\n"+'0123456789'))
    # '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    ret = []
    # print("utils check vocab size {}".format(len(vocab)))
    # print(dat[:5])
    for line in dat:
        # print("original:", line)
        line = line.strip()
        line = line.translate(trantab)
        line_split = line.split()
        line_split = [word.lower() for word in line_split if (len(word)<MAX_EN_CHAR_LENGTH and len(word)>1)]
        if vocab is not None:
            # print("use tfidf vocab!")
            _line_split = list(set(line_split).intersection(vocab))
            _line_split.sort(key=line_split.index)
            line_split = _line_split

        # fixme: 是否要去除stopwords
        #line_split = [tok for tok in line_split if (tok not in stopwords and tok not in punctuations) ]
        #line_split = [tok for tok in line_split if (tok not in punctuations)]
        if rmv_stop_words:
            # print("original:",line_split)
            new_line_split = list(set(line_split).difference(set(full_stop_words)))
            new_line_split.sort(key=line_split.index)
            # print("new:", new_line_split)
        else:
            new_line_split = line_split

        if is_ratio:
            NUM_WORD = max(int(len(new_line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH
        # new_line_split = [stemmer.stem(word) for word in new_line_split]
        if len(new_line_split) > NUM_WORD:
            line = " ".join(new_line_split[0:NUM_WORD])
        else:
            line = " ".join(new_line_split)
        # print("new:",line)
        ret.append(line)

    return ret
##### list 分段切分函数：接近等长划分.
def chunkIt(seq, num):
    """
    :param seq: 原始 list 数据
    :param num: 要分chunk是数量.
    :return:
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        # print("add!")
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def clean_en_text_parallel(dat, worker_num=NCPU, partition_num=10, vocab=None):
    sub_data_list = chunkIt(dat, num=partition_num)
    p = Pool(processes=worker_num)
    # mp_pooler = set_mp(processes=4)
    # data = [mp_pooler.apply_async(clean_en_text, args=(dat, vocab))]
    data = p.map(clean_en_text, sub_data_list)
    p.close()

    # 把 list of list of str 结果 flat 回到 list of str
    flat_data = [item for sublist in data for item in sublist]
    # flat_data = [p.get() for p in data][0]
    # print(flat_data[:3])
    return flat_data

def clean_data(data, language, max_length, vocab, rmv_stop_words=True):
    # print(data)
    if language=="EN":
        data = clean_en_text(data, vocab=vocab, rmv_stop_words=rmv_stop_words)
    else:
        data = clean_zh_text(data)
    return data

def clean_en_text_single(line, vocab, ratio=0.1, is_ratio=True, rmv_stop_words=True):
    trantab = str.maketrans(
        dict.fromkeys(string.punctuation + "@!#$%^&*()-<>[]<=>;:?.\/+[\\]^_`{|}~\t\n" + '0123456789'))
    line = line.strip()
    line = line.translate(trantab)
    line_split = line.split()
    line_split = [word.lower() for word in line_split if (len(word) < MAX_EN_CHAR_LENGTH and len(word) > 1)]
    if vocab is not None:
        # print("use tfidf vocab!")
        _line_split = list(set(line_split).intersection(vocab))
        _line_split.sort(key=line_split.index)
        line_split = _line_split

    # fixme: 是否要去除stopwords
    if rmv_stop_words:
        # print("original:",line_split)
        new_line_split = list(set(line_split).difference(set(full_stop_words)))
        new_line_split.sort(key=line_split.index)
        # print("new:", new_line_split)
    else:
        new_line_split = line_split

    if is_ratio:
        NUM_WORD = max(int(len(new_line_split) * ratio), MAX_SEQ_LENGTH)
    else:
        NUM_WORD = MAX_SEQ_LENGTH
    # new_line_split = [stemmer.stem(word) for word in new_line_split]
    if len(new_line_split) > NUM_WORD:
        _line = " ".join(new_line_split[0:NUM_WORD])
    else:
        _line = " ".join(new_line_split)
    return _line

def pad_sequence(data_ids, padding_val, max_length):
    # print(data)
    # max_length = len(max(data_ids, key=len))
    # print("max_length_word_training:", max_length)
    x_ids = sequence.pad_sequences(data_ids, maxlen=max_length, padding='post', value=padding_val)
    return x_ids

