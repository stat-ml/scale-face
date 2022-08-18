"""
Level 8:
arcface
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
    trainer = TripletsTrainer(model, checkpoint_dir / args.model_label, epochs=300)
    # trainer = ArcFaceTrainer(model, checkpoint_dir / args.model_label, epochs=300)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
