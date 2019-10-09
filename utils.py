#! -*- coding: utf-8 -*-
import numpy as np


def padding(batch):
    max_len = len(batch[0])
    for seq in batch:
        if len(seq) < max_len:
            seq += [2]*(max_len-len(seq))
    return max_len, batch


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)

    num = float(vector_a*vector_b.T)
    denom = np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    cos = num / denom

    return cos
