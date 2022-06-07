import torch
import torch.nn.functional as F

def get_logits(samples, weights, scale):
    """
    inputs:
        samples : torch.Tensor of shape [n_task, shot, feature_dim]
    returns :
        logits : torch.Tensor of shape [n_task, shot, num_class]
    """
    n_tasks = samples.size(0)
    # samples_norm = F.normalize(samples, dim=2)
    # weights_norm = F.normalize(weights, dim=2)
    # logits = scale * torch.bmm(samples_norm, weights_norm.transpose(1,2))
    logits = scale * (samples.matmul(weights.transpose(1, 2)) \
                         - 1 / 2 * (weights**2).sum(2).view(n_tasks, 1, -1) \
                         - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))
    return logits

def get_one_hot(y_s):
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot