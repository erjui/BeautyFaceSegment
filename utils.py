import random
import numpy as np
import torch

def fix_seed(seed):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # torch
    torch.cuda.manual_seed(seed) # cuda
    torch.cuda.manual_seed_all(seed) # cuda multi-GPU
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False # cudnn
