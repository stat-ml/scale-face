"""
level 2:
use tensorboard
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from torch.utils.tensorboard import SummaryWriter


categories = [
    'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle',
    'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster'
]


def make_small(data_dir, small_dir):
    df = pd.read_csv(data_dir / 'Ebay_info.txt', delim_whitespace=True, index_col='image_id')
    # for category in categories:
    #     (small_dir / f'{category}_final').mkdir()

    for path in tqdm(df.path):
        # print(Path(path))
        img = Image.open(data_dir / path)
        img = ImageOps.fit(
            img,
            (32, 32),
            Image.ANTIALIAS
        )
        img.save(small_dir / path)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.layers(x)


class DatasetS(Dataset):
    def __init__(self, base_dir, image_paths, labels):
        self.base_dir = base_dir
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(str(self.base_dir / self.image_paths[idx]), ImageReadMode.RGB)
        image = ((image - 128) / 255).flatten()
        return image, self.labels[idx]


def loader_by_df(df, base_dir, batch_size):
    image_paths = np.array(list(df.path))
    labels = np.array(list(df.super_class_id)) - 1
    dataset = DatasetS(base_dir, image_paths, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    data_dir = Path('/home/kirill/data/stanford/Stanford_Online_Products')
    small_dir = Path('/home/kirill/data/stanford/small')

    df = pd.read_csv(data_dir / 'Ebay_info.txt', delim_whitespace=True, index_col='image_id')
    df = df[df.super_class_id.isin([1, 2])]
    idx = np.random.choice(range(len(df)), 5000)
    df = df.iloc[idx]
    split = 3000
    train_loader = loader_by_df(df.iloc[:split], small_dir, batch_size=50)
    val_loader = loader_by_df(df.iloc[split:], small_dir, batch_size=10000)

    model = Model().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_iter = 0
    writer = SummaryWriter()
    for epoch in tqdm(range(200)):
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), train_iter)
            train_iter += 1

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                preds = model(x)
                loss = criterion(preds, y)
                writer.add_scalar('Loss/val', loss.item(), train_iter)

    writer.close()


if __name__ == '__main__':
    main()
