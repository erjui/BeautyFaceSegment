import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import cv2

from torchvision.transforms import transforms

normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalize)
])

class MaskDataset(Dataset):
    def __init__(self, img_dir, annt_dir, split='train'):
        self.imgs = sorted(glob(f"{img_dir}/*.jpg"))
        self.annts = sorted(glob(f"{annt_dir}/*.png"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)

        label_path = self.annts[idx]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        img = transform(img)
        label = torch.tensor(label, dtype=torch.int64)

        return img, label

# color dictionary
# TODO : how to deal with background label
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

if __name__ == '__main__':
    print('Dataset Validation 👻')

    num_labels = 19 # background + 18 classes
    img_dir = '/home/seongjae/MyDataset/CelebAMask-HQ/CelebA-HQ-img'    
    annt_dir = '/home/seongjae/MyDataset/CelebAMask-HQ/mask'
    dataset = MaskDataset(img_dir, annt_dir)
    # dataset = Subset(dataset, range(1000))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print(len(dataset))

    # data intergrity check
    # from tqdm import tqdm
    # for _ in tqdm(data_loader):
    #     pass

    # data visualization
    for batch in data_loader:
        x, y = batch

        x = x[0].permute(1, 2, 0).detach().numpy()
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x = (x+1)/2
        y = y[0].detach().numpy()

        print(x.shape)
        print(y.shape)

        import numpy as np
        print(np.max(y))
        y = labelVisualize(num_labels, color_dict, y)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', x)
        cv2.waitKey(0)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', y)
        cv2.waitKey(0)

    # attr
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']