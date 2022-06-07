import torch
import copy
from numpy import linalg as LA
from fsl import get_logits, get_one_hot
import torch.nn.functional as F


def kmeans(gallery, query, support_label, test_label, shot, norm_type='L2N', args=None):
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

    # hyperparam
    n_steps = args.number_fc_update

    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    scale = torch.FloatTensor(1).fill_(10.0).repeat(weights.size(0)).unsqueeze(1).unsqueeze(1).cuda()

    weights.requires_grad_()
    scale.requires_grad_()

    predict_old = None
    support_query = torch.cat([support, query], 1)
    for i in range(n_steps):
        logits = get_logits(support_query, weights, scale)

        predict = logits.argmax(-1)
        predict_one_hot = get_one_hot(predict)
        weights = torch.bmm(predict_one_hot.transpose(1,2), support_query)
        weights = weights / predict_one_hot.transpose(1,2).sum(-1, keepdim=True)
        if predict_old is not None:
            if (predict_old == predict).prod() > 0:
                print('Early stop at:', i, 'iterations')
                break
        predict_old = predict

    predict = get_logits(query, weights, scale).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 

def bayes_kmeans(gallery, query, support_label, test_label, shot, norm_type='L2N', args=None):
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

    # hyperparam
    n_steps = args.number_fc_update

    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    scale = torch.FloatTensor(1).fill_(10.0).repeat(weights.size(0)).unsqueeze(1).unsqueeze(1).cuda()

    weights.requires_grad_()
    scale.requires_grad_()

    y_s_one_hot = get_one_hot(y_s)

    predict_old = None
    support_query = torch.cat([support, query], 1)
    for i in range(n_steps):
        logits = get_logits(support_query, weights, scale)

        predict = logits.argmax(-1)
        predict_one_hot = get_one_hot(predict)
        weights = torch.bmm(predict_one_hot.transpose(1,2), support_query)
        weights = weights / predict_one_hot.transpose(1,2).sum(-1, keepdim=True)
        if predict_old is not None:
            if (predict_old == predict).prod() > 0:
                print('Early stop at:', i, 'iterations')
                break
        predict_old = predict

    logits_s = get_logits(support, weights, scale)
    logits_q = get_logits(query, weights, scale)

    prob_s = torch.softmax(logits_s, -1)
    prob_q = torch.softmax(logits_q, -1)

    pw_score = get_logits(query, support, scale)
    pw_score = torch.exp(pw_score)

    prob_q_i_given_k = pw_score.unsqueeze(-1) * prob_s.unsqueeze(1)
    prob_q_i_given_k = (prob_q_i_given_k.unsqueeze(2) * y_s_one_hot.transpose(1,2).unsqueeze(1).unsqueeze(-1)).sum(3)
    prob_q_i_given_k = prob_q_i_given_k / prob_q_i_given_k.sum(2, keepdim=True)

    predict = (prob_q_i_given_k * prob_q.unsqueeze(2)).sum(-1).argmax(-1)
    
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 
    
def soft_kmeans(gallery, query, support_label, test_label, shot, norm_type='L2N', args=None):
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
    gallery = gallery.reshape(-1, shot, gallery.shape[-1]).mean(1)  # (5, 1, D)

    # prepare for training linear
    support = torch.tensor(instance_gallery, device='cuda:0').reshape(
        n_tasks, -1, feat_dim)
    gallery = torch.tensor(gallery, device='cuda:0').reshape(
        n_tasks, -1, feat_dim)
    query = torch.tensor(query, device='cuda:0').reshape(n_tasks, -1, feat_dim)
    y_s = torch.tensor(support_label, device='cuda:0').reshape(n_tasks, -1)
    y_q = torch.tensor(test_label, device='cuda:0').reshape(n_tasks, -1)

    weights = copy.deepcopy(gallery.reshape(-1, n_classes, feat_dim))
    logits_s = get_logits(support, weights, 10)
    logits_q = get_logits(query, weights, 10)

    prob_s = torch.softmax(logits_s, -1)
    prob_q = torch.softmax(logits_q, -1)
    prob = torch.cat([prob_s, prob_q], 1).sum(1).unsqueeze(-1)
    
    weights_new = (support.unsqueeze(2) * prob_s.unsqueeze(-1)
                   ).sum(1) + (query.unsqueeze(2) * prob_q.unsqueeze(-1)).sum(1)
    weights_new = weights_new / prob

    predict = get_logits(query, weights_new, 1).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc

def CAN_T(gallery, query, support_label, test_label, shot, norm_type='L2N', args=None):
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

    weights.requires_grad_()
    scale.requires_grad_()

    y_s_one_hot = get_one_hot(y_s)
    weights = torch.bmm(y_s_one_hot.transpose(1,2), support)
    weights = weights / y_s_one_hot.transpose(1,2).sum(-1, keepdim=True)
    weights = F.normalize(weights, dim=2)

    cls_scores = get_logits(query, weights, scale)

    num_images_per_iter = 30
    num_iter = query.size(1) // num_images_per_iter

    for i in range(num_iter):
        max_scores, preds = torch.max(cls_scores, 2)
        chose_index = torch.argsort(max_scores, descending=True)
        chose_index = chose_index[:, : num_images_per_iter * (i + 1)]
        align_index = torch.arange(n_tasks, device=query.device).unsqueeze(-1)

        chose_index = chose_index + align_index * 75
        query_iter = query.view(-1, feat_dim)[chose_index].contiguous().view(n_tasks, -1, feat_dim)
        preds_iter = preds.view(-1)[chose_index].contiguous().view(n_tasks, -1)
        support_iter = torch.cat((support, query_iter), 1)
        y_s_iter = torch.cat((y_s, preds_iter), 1)

        y_s_one_hot = get_one_hot(y_s_iter)
        weights = torch.bmm(y_s_one_hot.transpose(1,2), support_iter)
        weights = weights / y_s_one_hot.transpose(1,2).sum(-1, keepdim=True)
        weights = F.normalize(weights, dim=2)

        cls_scores = get_logits(query, weights, scale)

    

    predict = get_logits(query, weights, scale).argmax(-1)
    acc = (predict.cpu().numpy() == y_q.cpu().numpy()).mean(-1)
    acc = [_ for _ in acc]
    return acc 
