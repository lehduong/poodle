import numpy as np
import torch
import random
from torch.utils.data import Sampler

__all__ = ['CategoriesSampler']


class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield batch


def sample_case(args, ld_dict, shot):
    # Sample meta task
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for i, each_class in enumerate(sample_class):
        total_samples = shot + args.meta_val_query
        if len(ld_dict[each_class]) < total_samples:
            total_samples = len(ld_dict[each_class])

        samples = random.sample(ld_dict[each_class], total_samples)
        train_label += [i] * len(samples[:shot])
        test_label += [i] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label

def nonuniform_sample_case(args, ld_dict, shot):
    # Sample meta task
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []

    # random number of query for each task
    total_query = args.meta_val_way * args.meta_val_query

    # random number of samples for each query
    concentration = np.random.uniform(1, 5)
    concentration = 1
    m = torch.distributions.dirichlet.Dirichlet(torch.tensor([float(concentration)]  * args.meta_val_way))
    dist = m.sample().tolist()
    dist = list(map(lambda x: x/sum(dist), dist)) 
    num_samples = list(map(lambda x: int(total_query * x), dist)) # num samples of each class
    num_samples[-1] = total_query - sum(num_samples[:-1]) # enforce number of query consistent for each task

    for i, each_class in enumerate(sample_class):
        total_samples = shot + num_samples[i]
        if len(ld_dict[each_class]) < total_samples:
            total_samples = len(ld_dict[each_class])

        samples = random.sample(ld_dict[each_class], total_samples)
        train_label += [i] * len(samples[:shot])
        test_label += [i] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label