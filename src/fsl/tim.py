import torch
import torch.nn as nn
import copy
from numpy import linalg as LA
from torch.nn import functional as F
from fsl import get_logits, get_one_hot


def tim(gallery, query, support_label, test_label, shot, norm_type='L2N', alpha=10, beta=1, args=None):
    """_summary_

    Args:
        gallery (numpy.array): support features (N_tasks, N_shot * N_way, dim)
        query (numpy.array): dim features (N_tasks, N_shot * N_way, dim)
        support_label (numpy.array): labels of support data
        test_label (numpy.array): label of query data
        shot (int): number of shot
        norm_type (str, optional): normalization of tran features either ['UN' or 'L2N'], 'UN' for unnormalized and 'L2N' for L2-norm. Defaults to 'L2N'.
        alpha (float, optional): Weight of unconditional entropy. Defaults to 10.
        beta (float, optional): Weight of conditional entropy. Defaults to 1.
        args (dict, optional): Additional argument for few-shot learning evaluation. Defaults to None.

    Returns:
        _type_: _description_
    """
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

    # init weight matrix for prototype of each class
    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    scale = torch.FloatTensor(1).fill_(10.0).repeat(weights.size(0)).unsqueeze(1).unsqueeze(1).cuda()

    weights.requires_grad_()
    scale.requires_grad_()
    optimizer = torch.optim.Adam([weights, scale])

    y_s_one_hot = get_one_hot(y_s)

    for i in range(args.number_fc_update):
        logits_s = get_logits(support, weights, scale)
        logits_q = get_logits(query, weights, scale)

        # entropy
        q_probs = F.softmax(logits_q, dim=2)
        q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
        q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
        ce_loss = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
        
        loss = ce_loss - alpha * q_ent + beta * q_cond_ent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predict = get_logits(query, weights, scale).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 
