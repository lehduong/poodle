import torch
import torch.nn as nn
import copy
from numpy import linalg as LA
from fsl import get_logits

def distance_based_classifier(gallery, query, support_label, test_label, shot, norm_type='L2N', args=None):
    n_tasks = gallery.shape[0]
    feat_dim = gallery.shape[-1]
    n_classes = args.meta_val_way
    gallery = gallery.reshape(-1, feat_dim)
    query = query.reshape(-1, feat_dim)
    support_label = support_label.reshape(-1)
    test_label = test_label.reshape(-1)
    
    # compute normalization
    if norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    instance_gallery = copy.deepcopy(gallery)
    gallery = gallery.reshape(-1, shot, gallery.shape[-1]).mean(1) # (5, 1, D)

    # prepare for training linear
    support = torch.tensor(instance_gallery, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    gallery = torch.tensor(gallery, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    query = torch.tensor(query, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    y_s = torch.tensor(support_label, device='cuda:0').reshape(n_tasks, -1)
    y_q = torch.tensor(test_label, device='cuda:0').reshape(n_tasks, -1)

    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    scale = torch.FloatTensor(1).fill_(10.0).repeat(weights.size(0)).unsqueeze(1).unsqueeze(1).cuda()

    predict = get_logits(query, weights, scale).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 