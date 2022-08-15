"""
level 5:
resnet 9 number of params?
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
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor, ToDevice, ToTorchImage, Cutout, Convert, Squeeze, RandomTranslate, RandomHorizontalFlip
)
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, FloatDecoder, SimpleRGBImageDecoder
from PIL import Image

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


def convert_to_rgb(base_directory):
    subdicrectories = os.listdir(base_directory)
    for directory in subdicrectories:
        print(directory)
        files = os.listdir(base_directory / directory)
        for file in tqdm(files):
            filename = base_directory / directory / file
            img = Image.open(filename)
            if img.mode != 'RGB':
                rgb_img = Image.new("RGB", img.size)
                rgb_img.paste(img)
                rgb_img.save(filename)


def ffcv_loader_by_df(df, base_dir, write_path, random_order=False, batch_size=100, augment=False):
    image_paths = np.array(list(df.path))
    labels = np.array(list(df.super_class_id)) - 1
    dataset = DatasetS(base_dir, image_paths, labels)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(max_resolution=32),
        'label': IntField()
    })
    writer.from_indexed_dataset(dataset)

    image_pipeline = [SimpleRGBImageDecoder()]

    if augment:
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(2, (127, 127, 127)),
        ])
    image_pipeline.extend([
        ToTensor(), ToDevice(0), ToTorchImage(), Convert(torch.float32)
    ])

    # # Imagenet augmentation
    # if augment:
    #     image_pipeline.extend([
    #         #         # transforms.RandomRotation(10),
    #         # transforms.ColorJitter(brightness=.1, hue=.1),
    #         #         transforms.RandomGrayscale()
    #     ])
    image_pipeline.extend([
        transforms.Normalize(127., 50.)
    ])



    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0), Squeeze()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    if random_order:
        order = OrderOption.QUASI_RANDOM
    else:
        order = OrderOption.SEQUENTIAL

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=batch_size, num_workers=12, order=order, pipelines=pipelines)
    return loader


NUM_CLASSES = 4


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)



class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv(in_size, out_size, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ELU()
            )

        self.layers = nn.Sequential(
            conv(3, 64, kernel_size=3, stride=1, padding=1),
            conv(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(
                conv(128, 128, kernel_size=3, stride=1, padding=1),
                conv(128, 128, kernel_size=3, stride=1, padding=1),
            )),
            conv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            Residual(nn.Sequential(
                conv(256, 256, kernel_size=3, stride=1, padding=1),
                conv(256, 256, kernel_size=3, stride=1, padding=1),
            )),
            conv(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
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
        image = Image.open(str(self.base_dir / self.image_paths[idx]))
        return image, self.labels[idx]


def loader_by_df(df, base_dir, batch_size):
    image_paths = np.array(list(df.path))
    labels = np.array(list(df.super_class_id)) - 1
    dataset = DatasetS(base_dir, image_paths, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# import random
# SEED = 42

def main():
    data_dir = Path('/home/kirill/data/stanford/Stanford_Online_Products')
    small_dir = Path('/home/kirill/data/stanford/small2')
    # convert_to_rgb(small_dir)

    # random.seed(SEED)
    # np.random.seed(SEED)

    df = pd.read_csv(data_dir / 'Ebay_info.txt', delim_whitespace=True, index_col='image_id')
    df = df[df.super_class_id.isin(np.arange(NUM_CLASSES)+1)]
    idx = np.random.choice(range(len(df)), 20000, replace=False)
    split = 12000
    df = df.iloc[idx]

    train_loader = ffcv_loader_by_df(
        df.iloc[:split], small_dir, '/tmp/ds_train.beton', random_order=True, batch_size=128,
        augment=True
    )
    val_df = df.iloc[split:]
    val_loader = ffcv_loader_by_df(
        val_df, small_dir, '/tmp/ds_val.beton', random_order=False, batch_size=len(val_df)
    )

    model = Model(NUM_CLASSES).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_iter = 0
    writer = SummaryWriter()
    for epoch in tqdm(range(50)):
        model.train()
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
            model.eval()
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                preds = model(x)
                loss = criterion(preds, y)
                writer.add_scalar('Loss/val', loss.item(), train_iter)
                accuracy = (torch.argmax(preds, dim=-1) == y).to(torch.float).mean()
                writer.add_scalar('Accuracy/val', accuracy.item(), train_iter)

    writer.close()


if __name__ == '__main__':
    main()
