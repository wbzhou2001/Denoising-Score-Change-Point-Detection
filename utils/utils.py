import random
import torch
import numpy as np

def reset_seed(seed = 0):
    '''reproducibility'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def LpNorm(x, p, ord):
    if ord is not np.inf:
        loss = np.sum(np.abs(x)**ord**(1/ord) * p)
    else:
        loss = np.sum(np.max(np.abs(x)) * p)
    return loss