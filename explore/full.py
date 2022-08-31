import os
from pathlib import Path

import torch


def main():
    base_dir = Path('/home/kirill/data/stanford/')
    print(os.listdir(base_dir / 'models'))
    checkpoint = torch.load(base_dir / 'models/resnet50.pth.tar')
    print(checkpoint)


if __name__ == '__main__':
    main()