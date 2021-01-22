from data_utils import downcast
import numpy as np
import pandas as pd


# Type: c -> n
def c_values_cnt(ss):
    counts = ss.value_counts()
    ss = ss.map(counts)
    ss = downcast(ss)
    return ss


#  Type: (c,c,...,c) -> c
def c_cats_combine(df, col2max):
    columns = df.columns
    ss = df[columns[0]].astype('float64')
    for col in columns[1:]:
        mx = col2max[col]
        ss *= mx
        ss += df[col]
    downcast(ss, accuracy_loss=False)
    return ss


# Type n,n->n
def n_div_n(ss_1, ss_2):
    new_ss = ss_1 / ss_2
    return downcast(new_ss)


# Type n,n->n
def n_multiply_n(ss_1, ss_2):
    new_ss = ss_1 * ss_2
    return downcast(new_ss)


# Type n,n->n
def n_minus_n(ss_1, ss_2):
    new_ss = ss_1 - ss_2
    return downcast(new_ss)


# Type n,n->n
def n_plus_n(ss_1, ss_2):
    new_ss = ss_1 + ss_2
    return downcast(new_ss)



# Type c,n->n
def groupby_mean(df):
    col = df.columns[0]
    num_col = df.columns[1]
    means = df.groupby(col, sort=False)[num_col].mean()
    ss = df[col].map(means)
    ss = downcast(ss)
    return ss