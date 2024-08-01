import numpy as np
import torch 
import dgl
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def binary_scores(scores, threshold):
    return np.where(scores>threshold, 1, 0)


def compute_metrics(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    # scores = binary_scores(scores, 0)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores) # , precision_score(labels,scores), f1_score(labels,scores)


def construct_negative_graph(hg, k, etype):
    utype, _, vtype = etype
    src, dst = hg.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, hg.num_nodes(vtype), (len(src)*k, ))
    return dgl.heterograph({etype: (neg_src, neg_dst)}, num_nodes_dict = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes})


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).reshape(-1, 1)
    return F.binary_cross_entropy_with_logits(scores, labels)