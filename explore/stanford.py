"""
Level 6:
cosine index?
"""
import os
from pathlib import Path
import random

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
from easydict import EasyDict

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor, ToDevice, ToTorchImage, Cutout, Convert, Squeeze, RandomTranslate, RandomHorizontalFlip
)
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, FloatDecoder, SimpleRGBImageDecoder
from PIL import Image

import sys
sys.path.append('.')
from explore.random_model import SimpleCNN, ResNet9

SEED = 42


categories = [
    'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle',
    'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster'
]


def make_small(data_dir, small_dir):
    df = pd.read_csv(data_dir / 'Ebay_info.txt', delim_whitespace=True, index_col='image_id')

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
    labels = np.array(list(df.labels))
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
    loader = Loader(
        write_path, batch_size=batch_size, num_workers=12, order=order, pipelines=pipelines, drop_last=False
    )
    return loader


SPLIT_CLASSES = 4
NUM_CLASSES = 4



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


def remap_labels(labels):
    # made labels from 0 to N-1
    classes = np.unique(labels)
    mapper = {klass: ordered for ordered, klass in enumerate(classes)}
    return np.array([mapper[old_label] for old_label in labels])


def get_loaders(base_dir, super_classes=False, split=(20_000, 12_000), batch_size=128):
    data_dir = base_dir / 'Stanford_Online_Products'
    small_dir = base_dir / 'small'
    df = pd.read_csv(data_dir / 'Ebay_train.txt', delim_whitespace=True, index_col='image_id')
    df = df[df.super_class_id.isin(np.arange(SPLIT_CLASSES) + 1)]
    if super_classes:
        df['labels'] = remap_labels(df.super_class_id)
    else:
        df['labels'] = remap_labels(df.class_id)

    idx = np.random.choice(range(len(df)), split[0], replace=False)
    split = split[1]
    df = df.iloc[idx]

    train_loader = ffcv_loader_by_df(
        df.iloc[:split], small_dir, '/tmp/ds_train.beton', random_order=True, batch_size=batch_size,
        augment=True
    )
    val_df = df.iloc[split:]
    val_loader = ffcv_loader_by_df(
        val_df, small_dir, '/tmp/ds_val.beton', random_order=False, batch_size=batch_size
    )
    return train_loader, val_loader


class CrossEntropyTrainer:
    def __init__(self, model, checkpoint_path, epochs):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs

    def train(self, train_loader, val_loader):
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
        criterion = torch.nn.CrossEntropyLoss()
        train_iter = 0
        writer = SummaryWriter()
        for epoch in tqdm(range(self.epochs)):
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
                epoch_losses = []
                correct = []

                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    preds = model(x)
                    loss = criterion(preds, y)
                    epoch_losses.append(loss.item())
                    correct.extend(list((torch.argmax(preds, dim=-1) == y).detach().cpu()))

                writer.add_scalar('Loss/val', np.mean(epoch_losses), train_iter)
                writer.add_scalar('Accuracy/val', np.mean(correct), train_iter)

        writer.close()
        torch.save(model.state_dict(), self.checkpoint_path)


def knn_eval(model, loader, writer):
    embeddings = []
    labels = []
    # with torch.no_grad()
    #     for x, y in loader:
    #         model(x)
    #         embeddings.append()
    #         pass
    # pass


class TripletsTrainer(CrossEntropyTrainer):
    def train(self, train_loader, val_loader):
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
        from pytorch_metric_learning import miners, losses
        miner = miners.MultiSimilarityMiner()
        criterion = losses.TripletMarginLoss()

        train_iter = 0
        writer = SummaryWriter()

        for epoch in tqdm(range(self.epochs)):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                model(x)
                embeddings = model.features
                hard_pairs = miner(embeddings, y)
                loss = criterion(embeddings, y, hard_pairs)
                loss.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', loss.item(), train_iter)
                train_iter += 1

            with torch.no_grad():
                model.eval()
                epoch_losses = []
                correct = []

                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    model(x)
                    loss = criterion(model.features, y)
                    epoch_losses.append(loss.item())

                    # correct.extend(list((torch.argmax(preds, dim=-1) == y).detach().cpu()))
                writer.add_scalar('Loss/val', np.mean(epoch_losses), train_iter)
            #     writer.add_scalar('Accuracy/val', np.mean(correct), train_iter)

            knn_eval(model, val_loader, writer)

        writer.close()
        torch.save(model.state_dict(), self.checkpoint_path)


def main():
    args = EasyDict({
        'base_dir': '/home/kirill/data/stanford/',
        'method': 'triplets',  # ['classification', 'triplets']
        'super_classes': False,
        'model_label': 'resnet9_triplets.pth'
    })
    base_dir = Path(args.base_dir)
    checkpoint_dir = base_dir / 'models'

    random.seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = get_loaders(
        base_dir, super_classes=args.super_classes, split=(20_000, 12_000), batch_size=256
    )

    if args.super_classes:
        num_classes = 4
    else:
        num_classes = 3580

    model = ResNet9(num_classes).cuda()
    # trainer = CrossEntropyTrainer(model, checkpoint_dir / args.model_label, epochs=5)
    trainer = TripletsTrainer(model, checkpoint_dir / args.model_label, epochs=120)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
