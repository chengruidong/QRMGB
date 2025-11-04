"""
evaluation indicators, including mAP.
mAP is the most common metric in supervised cross-modal retrieval, which measures the performance of the retrieval model on each category.
"""
import numpy as np
import scipy.spatial
import torch
from collections import Counter

def fx_calc_map_label(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)

