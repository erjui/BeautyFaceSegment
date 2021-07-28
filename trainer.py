import math
import logging

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
    out_channels = 18

    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.1
    num_workers = 0

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
        lr_scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

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
                    logits = model(x, y)
                    loss = criterion(logits, y)
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad() # optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr_scheduler.get_last_lr()}")

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
    print('hello, world')

    # config
    c = TrainerConfig()

    # model
    from model import UNet
    model = UNet(c.in_channels, c.out_channels)

    # train & valid data
    from data import MaskDataset
    img_dir = '/home/seongjae/Downloads/CelebAMask-HQ/CelebA-HQ-img'    
    annt_dir = '/home/seongjae/Downloads/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    train_dataset = MaskDataset(img_dir, annt_dir, 'train')
    valid_dataset = MaskDataset(img_dir, annt_dir, 'valid')

    # train
    traner = Trainer(model, train_dataset, valid_dataset, c)

