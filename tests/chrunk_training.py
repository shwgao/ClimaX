"""
In this file, I will split the dataset to k parts, and train the model on k parts sequentially.
In each fold, the model has to be trained to converge.
The idea is to validate the number of k how to influence the performance of the model.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, m):
        super().__init__()
        self.batch_size = batch_size
        self.m = m
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(self.mean, self.std)])
        self.train_splits = None  # 初始化 train_splits 属性

    def prepare_data(self):
        CIFAR10(root='.', train=True, download=True)
        CIFAR10(root='.', train=False, download=True)

    def setup(self, stage=None):
        cifar10_full = CIFAR10(root='.', train=True, transform=self.transform, download=True)
        self.train_size = len(cifar10_full) // self.m
        self.train_splits = random_split(cifar10_full, [self.train_size] * self.m)
        self.val_set = CIFAR10(root='.', train=False, transform=self.transform, download=True)

    def train_dataloader(self, index):
        return DataLoader(self.train_splits[index], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)


class ResNet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main(m, k, batch_size, max_epochs, patience, log_dir):
    data_module = CIFAR10DataModule(batch_size=batch_size, m=m)
    data_module.setup()  # 确保 setup 方法被调用，初始化 train_splits

    model = ResNet50Model()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        verbose=False
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        patience=patience,
        verbose=False,
        mode='max'
    )
    logger = TensorBoardLogger(log_dir, name=f"resnet50_m{m}")

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=50
    )

    for i in range(k):
        print(f"Training on split {i + 1}/{k}")
        trainer.fit(model, data_module.train_dataloader(i), data_module.val_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet50 on CIFAR-10 with data splits using PyTorch Lightning.')
    parser.add_argument('--m', type=int, required=True, help='Number of parts to split the dataset into.')
    parser.add_argument('--k', type=int, required=True, help='Number of parts to train the model on.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs to train for each split.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--log_dir', type=str, default='./train_chrunks',
                        help='Directory to save TensorBoard logs.')
    args = parser.parse_args()

    main(args.m, args.m, args.batch_size, args.max_epochs, args.patience, args.log_dir)
