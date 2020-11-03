import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class LitModel(pl.LightningModule):
    def __init__(self, lr: float = 0.0001, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default=os.getcwd())
    return parser.parse_args(args)


def main(args=None):

    # Parse up arguments
    args = parse_args(args)

    # Enforce random seed
    pl.seed_everything(args.seed)

    # Init dataset
    dataset = MNIST(args.data_dir, download=True, transform=transforms.ToTensor())

    # Convert to loader
    train_loader = DataLoader(dataset, batch_size=args.batch_size)

    # init model
    model = LitModel(lr=args.lr)

    # init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # train!
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
