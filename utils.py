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

# color dictionary
color_dict = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32),
}

def labelVisualize(num_class, color_dict, img):
    import numpy as np
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
