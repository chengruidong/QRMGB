import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class dual_softmax_loss(nn.Module):
    def __init__(self, ):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(sim_matrix)
        logpt = F.log_softmax(sim_matrix, dim=-1)  # row softmax and column softmax
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

def log_sum_exp(x):
    '''Utility function for computing log-sun-exp while determining
    This will be used to determine unaveraged confidence loss across all examples in a batch.
    '''
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), -1, keepdim=True)) + x_max

def l2norm(X, dim, eps=1e-8):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

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
    Output Match Matrix =Tensor(batch,batch) and match[i,j] == 1 iff. label[i]==label[j]
    '''
    match = label.unsqueeze(1) - label.unsqueeze(0)
    match[match != 0] = 1
    return 1 - match

def calcneg(dist, label, anchor, positive):
    '''
    Input dist = (batch,batch), label = (batch,), anchor = index, positive = index
    Output chosen negative sample index
    '''
    standard = dist[anchor, positive]
    dist = dist[anchor] - standard
    if max(dist[label != label[anchor]]) >= 0:
        dist[dist < 0] = max(dist) + 2
        dist[label == label[anchor]] = max(dist) + 2
        return int(torch.argmin(dist).cpu())
    else:  # choose argmax
        dist[label == label[anchor]] = min(dist) - 2
        return int(torch.argmax(dist).cpu())

def calcneg_dot(img, txt, match, anchor, positive):
    '''
    Input img = (batch,dim), txt = (batch,dim), match = (batch,batch), anchor = index, positive = index
    Output chosen negative sample index
    '''
    distdot = torch.sum(torch.mul(img.unsqueeze(1), txt.unsqueeze(0)), 2)
    distdot[match == 1] = -66666
    return int(torch.argmax(distdot[anchor]).cpu())


def Triplet(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), triplets = List with shape(pairs,3)
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive_list = np.argwhere(match_n == 1).tolist()
    for positive in positive_list:
        negative = calcneg(dist, label, positive[0], positive[1])
        # negative = calcneg_dot(img, txt, match, anchor, positive)
        triplet_list.append([positive[0], int(positive[1].cuda()), negative])

    return dist, match, triplet_list

def Positive(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), positives = List with shape(pairs,2)
    Remark: return (anchor,positive) without finding triplets
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    sample_list = torch.tensor([x for x in range(batch)]).int().cuda()
    positive_list = [[i, int(j.cpu())] for i in range(batch) for j in sample_list[label == label[i]]]
    return dist, match, positive_list

def Modality_invariant_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate invariant loss between images and texts belonging to the same class
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)
    pos = torch.mul(dist, match)
    loss = torch.sum(pos)

    return loss / batch

def Contrastive_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)
    pos = torch.mul(dist, match)
    neg = margin - torch.mul(dist, 1-match)
    neg = torch.clamp(neg, 0)
    loss = torch.sum(pos) + torch.sum(neg)

    return loss / batch

def GnbLoss(v1,v2,tem):
    v1=F.normalize(v1,dim=1)
    v2=F.normalize(v2,dim=1)
    numerator=torch.exp(torch.diag(torch.inner(v1.float(),v2.float()))/tem)
    numerator=torch.cat((numerator,numerator),0)
    joint_vector=torch.cat((v1,v2),0)
    pairs_product=torch.exp(torch.mm(joint_vector,joint_vector.t())/tem)
    denominator=torch.sum(pairs_product-pairs_product*torch.eye(joint_vector.shape[0]).to(device),0)
    loss=-torch.mean(torch.log(numerator/denominator))
    return loss

def Triplet_Loss(img, txt, label, margin=0.2, semi_hard=True):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    loss = 0
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()
    for x in positive:
        if semi_hard:
            neg_index = torch.where(match[x[0]] == 0)
            neg_dis = dist[x[0]][neg_index]
            tmp = dist[x[0], x[1]] - neg_dis + margin
            tmp = torch.clamp(tmp, 0)
            loss = loss + torch.sum(tmp, dim=-1)
        else:
            negative = calcneg(dist, label, x[0], x[1])
            tmp = dist[x[0], x[1]] - dist[x[0], negative] + margin
            if tmp > 0:
                loss = loss + tmp

    return loss / len(positive)


def Lifted_Loss(img, txt, label, margin=1):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate lifted structured embedding loss
    '''

    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)
        neg_dis_anchor = dist[x[0]][neg_index]
        neg_dis_postive = dist[x[1]][neg_index]
        tmp = dist[x[0], x[1]] + log_sum_exp(margin - neg_dis_postive) + log_sum_exp(margin - neg_dis_anchor)
        loss = loss + tmp

    return loss / (2 * len(positive))

def Npairs(img, txt, label, margin=0.2, alpha=0.1):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter, alpha = parameter
    Calculate N-pairs loss
    '''
    batch = img.shape[0]
    distdot_it = torch.exp(F.linear(img, txt))
    distdot_ti = torch.t(distdot)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)
        tmp_i2t = distdot_it[x[0], x[1]] - log_sum_exp(distdot_it[x[0]][neg_index])
        tmp_t2i = distdot_ti[x[0], x[1]] - log_sum_exp(distdot_ti[x[0]][neg_index])
        loss = loss + (tmp_i2t + tmp_t2i)/2
    loss = -loss / len(positive)
    for x in range(batch):
        loss = loss + alpha * (torch.norm(img[x]) + torch.norm(txt[x])) / batch

    return loss

def Supervised_Contrastive_Loss(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    An unofficial implementation of supervised contrastive loss for multimodal learning
    '''
    loss = 0
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    dist = dist / (torch.sum(dist) / (batch * batch))
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)
        pos_sim = -dist[x[0], x[1]]
        neg_sims = -dist[x[0]][neg_index]
        tmp = pos_sim - log_sum_exp(neg_sims)

        loss = loss + tmp

    loss = -loss / len(positive)

    return loss

def regularization(features, centers, labels):
    distance = (features - centers[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]
    return distance

def PAN(features, centers, labels, add_regularization=False):
    batch = features.shape[0]
    features_square = torch.sum(torch.pow(features, 2), 1, keepdim=True)
    centers_square = torch.sum(torch.pow(torch.t(centers), 2), 0, keepdim=True)
    features_into_centers = 2 * torch.matmul(features, torch.t(centers))
    dist = -(features_square + centers_square - features_into_centers)
    output = F.log_softmax(dist, dim=1)
    dce_loss = F.nll_loss(output, labels)
    loss = dce_loss
    if add_regularization:
        reg_loss = regularization(features, centers, labels)
        loss = dce_loss + reg_loss
    return loss / batch

def Label_Regression_Loss(view1_predict, view2_predict, label_onehot):
    loss = ((view1_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean()
    return loss

def Proxy_NCA(features, label, proxies, mrg=0.1, alpha=1):
    P = torch.t(proxies)
    n_classes = P.shape[0]
    cos = F.linear(features, P)
    loss = 0
    for x in range(features.shape[0]):
        pos = torch.exp(cos[x, label[x]])
        neg = torch.exp(cos[x]).sum(dim=-1)-pos
        loss = loss + torch.log(pos / neg)
    loss = -loss / features.shape[0]
    
    return loss

def Proxy_Anchor(features, label, proxies, mrg=0.1, alpha=1):
    P = torch.t(proxies)
    n_classes = P.shape[0]
    cos = F.linear(features, P)
    P_one_hot = label
    N_one_hot = 1 - P_one_hot
    pos_exp = torch.exp(-alpha * (cos - mrg))
    neg_exp = torch.exp(alpha * (cos + mrg))
    with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
    num_valid_proxies = len(with_pos_proxies)
    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
    N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
    neg_term = torch.log(1 + N_sim_sum).sum() / n_classes
    loss = pos_term + neg_term
    return loss

def AngleLoss2(input, target,gamma=2,it = 0,LambdaMin = 5.0,LambdaMax = 1500.0,lamb = 1500.0):
    it += 1
    cos_theta,phi_theta = input
    target = target.view(-1,1)
    index = cos_theta.data * 0.0
    index.scatter_(1,target.data.view(-1,1),1)
    index = index.byte()
    index = Variable(index)
    lamb = max(LambdaMin,LambdaMax/(1+0.1*it ))
    output = cos_theta * 1.0
    output[index] -= cos_theta[index]*(1.0+0)/(1+lamb)
    output[index] += phi_theta[index]*(1.0+0)/(1+lamb)
    logpt = F.log_softmax(output)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())
    loss = -1 * (1-pt)**gamma * logpt
    loss = loss.mean()
    return loss