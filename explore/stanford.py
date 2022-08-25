"""
level 1
Scale directly
"""
import os
from pathlib import Path
import random

import numpy as np

from easydict import EasyDict

import sys
sys.path.append('.')
from explore.random_model import get_model, get_confidence_model
from explore.stanford_trainers import TripletsTrainer, CrossEntropyTrainer, ArcFaceTrainer
from explore.stanford_dataset import get_loaders
import torch

SEED = 42



def save_model(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/kirill/data/stanford', help='Directory with dataset and models')
    parser.add_argument('--method', default='classification', choices=['classification', 'triplets', 'arcface', 'scale'])
    parser.add_argument('--super_classes', action='store_true', default=False)
    args = parser.parse_args()
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

    embedding_size = 128

    if args.method == 'classification':
        model = get_model('resnet9', num_classes).cuda()
        trainer = CrossEntropyTrainer(model, epochs=5)
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_classification.pth')

    elif args.method == 'triplets':
        model = get_model('resnet9_embeddings', num_classes).cuda()
        trainer = TripletsTrainer(model, epochs=120)
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_triplets.pth')

    elif args.method == 'arcface':
        model = get_model('resnet9_embeddings', num_classes).cuda()
        trainer = ArcFaceTrainer(
            model, embedding_size=embedding_size, num_classes=num_classes,
            epochs=50
        )
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_arcface.pth')

    elif args.method == 'scale':
        model = get_confidence_model('resnet9_scale', num_classes, checkpoint_dir / 'resnet9_arcface.pth')
        trainer = ScaleFaceTrainer(model, embedding_size, num_classes, epochs=10)
    else:
        raise ValueError('Incorrect method')


if __name__ == '__main__':
    main()
