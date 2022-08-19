"""
Level 2:
rejection
"""

import os
from pathlib import Path
import random

import numpy as np

from easydict import EasyDict


import sys
sys.path.append('.')
from explore.random_model import SimpleCNN, ResNet9
from explore.stanford_trainers import TripletsTrainer, CrossEntropyTrainer, ArcFaceTrainer
from explore.stanford_dataset import get_loaders

SEED = 42


def main():
    args = EasyDict({
        'base_dir': '/home/kirill/data/stanford/',
        'method': 'arcface',  # ['classification', 'triplets']
        'super_classes': False,
        'model_label': 'resnet9_arcface.pth'
    })
    base_dir = Path(args.base_dir)

    random.seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = get_loaders(
        base_dir, super_classes=args.super_classes, split=(20_000, 12_000), batch_size=256
    )

    if args.super_classes:
        num_classes = 4
    else:
        num_classes = 3580

    embedding_size = 128

    model = ResNet9(num_classes).cuda()


    checkpoint_path = base_dir / 'models' / args.model_label
    if args.method == 'classification':
        trainer = CrossEntropyTrainer(model, checkpoint_path, epochs=5)
    elif args.method == 'triplets':
        trainer = TripletsTrainer(model, checkpoint_path, epochs=120)
    else:
        trainer = ArcFaceTrainer(
            model, checkpoint_path, embedding_size=embedding_size, num_classes=num_classes,
            epochs=120
        )
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
