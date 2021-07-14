from glob import glob

from torch.utils.data import Dataset, DataLoader, Subset

def integrate_segments():
    pass

class dataset(Dataset):
    def __init__(self, img_dir, annt_dir, split='train'):
        self.imgs = glob(f"{img_dir}/*.jpg")
        self.annts = glob(f"{annt_dir}/**/*.png")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]


        pass

if __name__ == '__main__':
    print('hello, world')


    img_dir = '/home/seongjae/Downloads/CelebAMask-HQ/CelebA-HQ-img'
    imgs = glob(f"{img_dir}/*.jpg")
    print(len(imgs))
    
    annt = '/home/seongjae/Downloads/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    annts = glob(f"{annt}/**/*.png")
    print(len(annts))

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']