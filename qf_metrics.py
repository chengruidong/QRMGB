import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calcdist(img, txt):
    '''
    Input img = (batch,dim), txt = (batch,dim)
    Output Euclid Distance Matrix = Tensor(batch,batch), and dist[i,j] = d(img_i,txt_j)
    '''
    dist = img.unsqueeze(1) - txt.unsqueeze(0)
    dist = torch.sum(torch.pow(dist, 2), dim=2)
    return torch.sqrt(dist)

def calcmatch(label):
    '''
    Input label = (batch,)
    Output Match Matrix = Tensor(batch,batch) and match[i,j] == 1 iff. label[i]==label[j]
    '''
    match = label.unsqueeze(1) - label.unsqueeze(0)
    match[match != 0] = 1
    return 1 - match

def detect_false_negatives(dist, match, false_negative_threshold=0.1):
    different_labels = (match == 0)
    close_pairs = (dist < false_negative_threshold)
    false_neg_mask = different_labels & close_pairs
    return false_neg_mask.float()

def Qf_Loss(img, txt, label, margin=0.2, semi_hard=True, false_negative_threshold=10, use_false_neg_detection=True):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss with false negative detection
    '''
    loss = 0
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)

    if use_false_neg_detection:
        false_neg_mask = detect_false_negatives(dist, match, false_negative_threshold)
        match = torch.where(false_neg_mask.bool(), torch.ones_like(match), match)
    
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()
    
    false_neg_count = 0
    for x in positive:
        is_false_neg = False
        if use_false_neg_detection:
            original_match = calcmatch(label)
            if original_match[x[0], x[1]] == 0:
                is_false_neg = True
                false_neg_count += 1
        
        # Semi-Hard Negative Mining
        if semi_hard:
            neg_index = torch.where(match[x[0]] == 0)
            if len(neg_index[0]) > 0:
                neg_dis = dist[x[0]][neg_index]
                tmp = dist[x[0], x[1]] - neg_dis + margin
                tmp = torch.clamp(tmp, 0)
                loss = loss + torch.sum(tmp, dim=-1)
        else:
            # Hard Negative Mining
            negative = calcneg(dist, label, x[0], x[1])
            tmp = dist[x[0], x[1]] - dist[x[0], negative] + margin
            if tmp > 0:
                loss = loss + tmp

    return loss / len(positive) if len(positive) > 0 else torch.tensor(0.0)

def calcneg(dist, label, anchor_idx, positive_idx):
    match = calcmatch(label)
    neg_indices = torch.where(match[anchor_idx] == 0)[0]
    if len(neg_indices) == 0:
        return positive_idx

    neg_dists = dist[anchor_idx, neg_indices]
    hardest_neg_idx = neg_indices[torch.argmin(neg_dists)]
    
    return hardest_neg_idx

def auto_determine_threshold(img,txt,label,percentile=0.8):
    dist= calcdist(img, txt)
    match= calcmatch(label)
    positive_distances=dist[match==1]
    if len(positive_distances) == 0:
        return 0.3
    threshold=np.percentile(positive_distances.detach().cpu().numpy(), percentile)
    return float(threshold)

def Qf_Loss_auto_determine_threshold(img,txt,label,margin=0.2,semi_hard=True,false_negative_threshold=None,auto_threshold_percentile=0.8):
    if false_negative_threshold is None:
        false_negative_threshold = auto_determine_threshold(img,txt,label,margin)
    return Qf_Loss(img,txt,label,margin,semi_hard,false_negative_threshold,True)