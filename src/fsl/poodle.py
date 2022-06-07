import torch
import torch.nn as nn
import copy
from numpy import linalg as LA
from torch.nn import functional as F
from fsl import get_logits, get_one_hot
import numpy as np


def poodle(gallery, query, support_label, test_label, shot, train_feature=None, norm_type='L2N', alpha=1, beta=0.5, transductive=True, args=None):
    n_tasks = gallery.shape[0]
    n_query = query.shape[1]
    feat_dim = gallery.shape[-1]
    n_classes = args.meta_val_way
    gallery = gallery.reshape(-1, feat_dim)
    query = query.reshape(-1, feat_dim)
    support_label = support_label.reshape(-1)
    test_label = test_label.reshape(-1)
    
    # compute normalization
    if norm_type == 'L2N':
        if train_feature is not None:
            train_feature = train_feature / LA.norm(train_feature, 2, 1)[:, None]
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    instance_gallery = copy.deepcopy(gallery)
    gallery = gallery.reshape(-1, shot, gallery.shape[-1]).mean(1) # (5, 1, D)
    if train_feature is not None:
        neg_samples = np.random.randint(0, train_feature.shape[0], n_query * n_tasks * n_classes)
        neg_samples = train_feature[neg_samples]

    # prepare for training linear
    support = torch.tensor(instance_gallery, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    gallery = torch.tensor(gallery, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    query = torch.tensor(query, device='cuda:0').reshape(n_tasks, -1, feat_dim)

    if train_feature is not None:
        neg_samples = torch.tensor(neg_samples, device='cuda:0').reshape(n_tasks, -1, feat_dim)

    y_s = torch.tensor(support_label, device='cuda:0').reshape(n_tasks, -1)
    y_q = torch.tensor(test_label, device='cuda:0').reshape(n_tasks, -1)

    # init weight matrix for prototype of each class 
    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    scale = torch.FloatTensor(1).fill_(10.0).repeat(weights.size(0)).unsqueeze(1).unsqueeze(1).cuda()

    weights.requires_grad_()
    scale.requires_grad_()
    optimizer = torch.optim.Adam([weights, scale])

    y_s_one_hot = get_one_hot(y_s)

    if train_feature is None:
        neg_samples = torch.rand_like(query, device=query.device)
    if norm_type == 'L2N':
        neg_samples = F.normalize(neg_samples, p=2, dim=-1)
    
    for i in range(args.number_fc_update):
        logits_n = get_logits(neg_samples, weights, scale)
        logits_s = get_logits(support, weights, scale)
        logits_q = get_logits(query, weights, scale)
        if transductive:
            logits = torch.cat([logits_s, logits_q], 1)
        else:
            logits = logits_s

        # loss 
        ce_loss = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
        pull_loss = - (logits * torch.softmax(logits, -1).detach()).sum(2).mean(1).sum(0)
        push_loss = - (logits_n * torch.softmax(logits_n, -1).detach()).sum(2).mean(1).sum(0)
        poodle_loss = alpha * pull_loss - beta * push_loss
        
        loss = ce_loss + poodle_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predict = get_logits(query, weights, scale).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 
