import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor, ToDevice, ToTorchImage, Cutout, Convert, Squeeze, RandomTranslate, RandomHorizontalFlip
)
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, FloatDecoder, SimpleRGBImageDecoder



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
