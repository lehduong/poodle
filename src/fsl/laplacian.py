import numpy as np
import math
import torch
import torch.nn as nn
from numpy import linalg as LA
from scipy import sparse
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F

def laplacian(gallery, query, support_label, test_label, shot, train_mean=None, norm_type='L2N', args=None):
    if norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    eta = gallery.mean(0) - query.mean(0) # shift
    query = query + eta[np.newaxis,:]
    query_aug = np.concatenate((gallery, query),axis=0)
    gallery_ = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)
    gallery_ = torch.from_numpy(gallery_)
    query_aug = torch.from_numpy(query_aug)
    distance = get_metric('cosine')(gallery_, query_aug)
    predict = torch.argmin(distance, dim=1)
    cos_sim = F.cosine_similarity(query_aug[:, None, :], gallery_[None, :, :], dim=2)
    cos_sim = 10 * cos_sim
    W = F.softmax(cos_sim,dim=1)
    gallery_list = [(W[predict==i,i].unsqueeze(1)*query_aug[predict==i]).mean(0,keepdim=True) for i in predict.unique()]
    gallery = torch.cat(gallery_list,dim=0).numpy()

    support_label = support_label[::shot]
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)
    test_label = np.array(test_label)
    # with LapLacianShot
    if args.lshot and args.lmd!=0:
        knn = args.knn
        lmd = args.lmd
        unary = distance.transpose() ** 2
        acc = lshot_prediction(args, knn, lmd, query, unary, support_label, test_label)
    else:
        idx = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN]
        nearest_samples = np.take(support_label, idx)
        out = mode(nearest_samples, axis=0)[0]
        out = out.astype(int)
        acc = (out == test_label).mean()
    return acc

def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]

def lshot_prediction(args, knn, lmd, X, unary, support_label, test_label):
    W = create_affinity(X, knn)
    l = bound_update(args, unary, W, lmd)
    out = np.take(support_label, l)
    acc, _ = (test_label == out).mean()
    return acc

def create_affinity(X, knn):
    N, D = X.shape
    nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
    dist, knnind = nbrs.kneighbors(X)

    row = np.repeat(range(N), knn - 1)
    col = knnind[:, 1:].flatten()
    data = np.ones(X.shape[0] * (knn - 1))
    W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
    return W

def bound_update(args, unary, kernel, bound_lambda, bound_iteration=20, batch=False):
    """
    """
    oldE = float('inf')
    Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel, bound_lambda, batch)
        E_list.append(E)
        if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
            break
        else:
            oldE = E.copy()

    l = np.argmax(Y, axis=1)
    return l

def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    N = Y_in.shape[0]
    size_limit = 150000
    if N > size_limit:
        batch_size = 1280
        Y_out = []
        num_batch = int(math.ceil(1.0 * N / batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            tmp = np.exp(Y_in[start:end, :])
            tmp = tmp / (np.sum(tmp, axis=1)[:, None])
            Y_out.append(tmp)
        del Y_in
        Y_out = np.vstack(Y_out)
    else:
        Y_out = np.exp(Y_in)
        Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out

def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    if batch == False:
        temp = (unary * Y) + (-bound_lambda * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
            E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

    return E