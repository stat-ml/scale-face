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
from explore.random_model import SimpleCNN, ResNet9, ResNet9PFE
from explore.stanford_trainers import TripletsTrainer, CrossEntropyTrainer, ArcFaceTrainer
from explore.stanford_dataset import get_loaders
from face_lib.models.heads import PFEHead
import torch

SEED = 42



def save_model(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)


# def get_model(method, num_classes):
#     base_dir = Path('/home/kirill/data/stanford/')
#     if method == 'pfe':
#         backbone = ResNet9(num_classes)
#         head = PFEHead(128)
#         model = ResNet9PFE(backbone, head).cuda()
#     elif method == 'scale':
#         backbone = ResNet9(num_classes).cuda()
#         pass
#     else:
#         model = ResNet9(num_classes).cuda()
#     return model


def main():
    args = EasyDict({
        'base_dir': '/home/kirill/data/stanford/',
        'method': 'scale',  # ['classification', 'triplets', 'arcface', 'scale']
        'super_classes': False,
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

    embedding_size = 128

    # 'model_label': 'resnet9_arcface.pth'
    if args.method == 'classification':
        model = get_model('resnet9', num_classes)
        trainer = CrossEntropyTrainer(model, epochs=5)
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_classification.pth')
    elif args.method == 'triplets':
        model = get_model('resnet9', num_classes)
        trainer = TripletsTrainer(model, epochs=120)
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_triplets.pth')
    elif args.method == 'argface':
        model = get_model('resnet9', num_classes)
        trainer = ArcFaceTrainer(
            model, embedding_size=embedding_size, num_classes=num_classes,
            epochs=120
        )
        trainer.train(train_loader, val_loader)
        save_model(model, checkpoint_dir / 'resnet9_arcface.pth')
    elif args.method == 'scale':
        backbone = get_confidence_model('resnet9_scale', num_classes, 'resnet9_arcface.pth')



    # elif args.method == 'scale':
    #     trainer = ScaleTrainer()
    # else:
    #     raise ValueError
    # trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
