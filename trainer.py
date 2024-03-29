import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import MultiStepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from schedule import GradualWarmupScheduler

class SegmentModel(pl.LightningModule):
    # TODO: use epic optimizer

    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()

        self.model = smp.FPN(
            encoder_name='timm-mobilenetv3_large_100',
            encoder_weights='imagenet',
            in_channels=3,
            classes=19,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        x = torch.argmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=7e-3)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=cosine_scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid_loss",
            }
        }

    # TODO: testing step


if __name__ == '__main__':
    # data
    from data import MaskDataset
    img_dir = '/home/seongjae/MyDataset/CelabA/CelebAMask-HQ/CelebA-HQ-img'    
    annt_dir = '/home/seongjae/MyDataset/CelabA/CelebAMask-HQ/mask'

    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.utils.data import random_split
    from glob import glob

    imgs = sorted(glob(f"{img_dir}/*.jpg"))
    annts = sorted(glob(f"{annt_dir}/*.png"))

    train_imgs, valid_imgs = imgs[:24000], imgs[24000:]
    train_annts, valid_annts = annts[:24000], annts[24000:]

    train_dataset = MaskDataset(train_imgs, train_annts, split='train')
    valid_dataset = MaskDataset(valid_imgs, valid_annts, split='valid')
    # train_dataset, valid_dataset = random_split(dataset, [24000, 6000], generator=torch.Generator().manual_seed(42))
    # dataset = Subset(dataset, range(12))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True, drop_last=True)

    print(f"Total number of train dataset: {len(train_dataset)}")
    print(f"Total number of valid dataset: {len(valid_dataset)}")

    # training
    pl.seed_everything(42)

    lighting_model = SegmentModel()
    callbacks = [
        ModelCheckpoint(monitor="valid_loss", dirpath="ckpt"),
        ModelSummary(),
    ]
    wandb_logger = WandbLogger(project="face-segment")
    trainer = pl.Trainer(
        max_epochs=100,

        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,

        devices="auto",
        accelerator="gpu",
        strategy="dp",
        gpus=-1,
        auto_select_gpus=True, 
        benchmark=True,
        deterministic=False,

        # fast_dev_run=7,
        # overfit_batches=10,
    )
    trainer.fit(
        model=lighting_model,
        train_dataloader=train_loader,
        val_dataloaders=valid_loader,
        # ckpt_path=".chpt", # for resuming training
    )