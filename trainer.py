import math
import logging
from torch.utils.data.dataset import random_split

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from utils import fix_seed
fix_seed(0)

class TrainerConfig:
    # model config
    in_channels = 3
    out_channels = 19

    # optimization parameters
    max_epochs = 10
    max_epochs = 800
    batch_size = 64
    # learning_rate = 3e-4
    learning_rate = 1e-3
    weight_decay = 0.1
    num_workers = 4

    # checkpoint settings
    ckpt_path = None

    # logging settings
    print_step = 10

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        # optimizer = raw_model.configure_optimizers(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = MultiStepLR(optimizer, milestones=[1e10], gamma=0.1)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits = model(x)
                    loss = criterion(logits, y)
                    losses.append(loss.item())

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    # report progress
                    print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr_scheduler.get_last_lr()}")
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr_scheduler.get_last_lr()}")

            if epoch % 100 == 0:
                # img = x[0].cpu().detach().permute(1, 2, 0).numpy()
                import torch.nn.functional as F
                out = F.softmax(logits[0], dim=0)
                out = torch.argmax(out, dim=0).cpu().detach().numpy()
                
                import cv2
                from utils import color_dict, labelVisualize
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f'img_{epoch}.jpg', np.uint8((img+1)/2 * 255.0))
                cv2.imwrite(f'out_{epoch}.jpg', (labelVisualize(19, color_dict, out) * 255.0).astype(np.uint8))

            if not is_train:
                valid_loss = float(np.mean(losses))
                logger.info("valid loss: %f", valid_loss)
                return valid_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.valid_dataset is not None:
                valid_loss = run_epoch('valid')

            good_model = self.valid_dataset is None or valid_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = valid_loss
                self.save_checkpoint()

if __name__ == '__main__':
    print('Start training 🚀')

    # config
    c = TrainerConfig()

    # model
    from model import UNet
    model = UNet(c.in_channels, c.out_channels)

    # train & valid data
    from data import MaskDataset
    img_dir = '/home/seongjae/MyDataset/CelabA/CelebAMask-HQ/CelebA-HQ-img'    
    annt_dir = '/home/seongjae/MyDataset/CelabA/CelebAMask-HQ/mask'

    # dataset = MaskDataset(img_dir, annt_dir)
    # train_dataset, valid_dataset = random_split(dataset, [24000, 6000], generator=torch.Generator().manual_seed(0))
    

    # One sample overfitting
    from torch.utils.data import Subset
    dataset = MaskDataset(img_dir, annt_dir)
    dataset = Subset(dataset, range(1))
    trainer = Trainer(model, dataset, None, c)

    # train
    # trainer = Trainer(model, train_dataset, valid_dataset, c)

    trainer.train()
