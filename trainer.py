import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class LitAutoEncoder(pl.LightningModule):
    # TODO: add lr scheduling
    # TODO: add remote logging
    # TODO: prepare dataset
    # TODO: use epic optimizer
    # TODO: save checkpoint
    # TODO: printing messages along trianing

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # TODO: implement training_step
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: implement validation_step
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer