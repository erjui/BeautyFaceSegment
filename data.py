import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

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
        img = Image.open(img_path)
        img.resize((512, 512), Image.BILINEAR)

        label_path = self.annts[idx]
        label = Image.open(label_path)

        img = transform(img)
        label = torch.tensor(label, dtype=torch.int64)

        return img, label

if __name__ == '__main__':
    print('Dataset Validation 👻')

    img_dir = '/home/seongjae/MyDataset/CelebAMask-HQ/CelebA-HQ-img'    
    annt_dir = '/home/seongjae/MyDataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    dataset = MaskDataset(img_dir, annt_dir)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # data intergrity check
    for _ in data_loader:
        pass

    # data visualization
    for batch in data_loader:
        x, y = batch

        x = x[0].detach().numpy()
        y = y[0].detach().numpy()

        print(x.shape)
        print(y.shape)

        break

    # attr
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']