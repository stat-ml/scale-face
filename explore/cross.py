# level 1
# Launch a model with a batch of images

from pathlib import Path
import os
import sys

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(".")
# from iresnet import iresnet50
from face_lib.models.iresnet import iresnet50

data_dir = Path("~/data/faces").expanduser()
cplfw_dir = data_dir / 'cplfw'


def parse_pairs():
    pairs_files = data_dir / 'pairs_CPLFW.txt'

    with open(pairs_files, 'r') as f:
        lines = f.readlines()
    lines = [l.split() for l in lines]
    print(*(lines[:10] + lines[-10:]), sep='\n')

    pairs = []
    for i in range(len(lines) // 2):
        pair = lines[2*i][0], lines[2*i+1][0], int(lines[2*i][1])
        pairs.append(pair)

    as_dict = {
        'photo_1': [p[0] for p in pairs],
        'photo_2': [p[1] for p in pairs],
        'label': [p[2] for p in pairs]
    }
    df = pd.DataFrame(as_dict)
    df.to_csv(data_dir / 'pairs.csv', index=False)


def load_model():
    model = iresnet50()
    checkpoint = torch.load(data_dir / 'models/backbone_resnet50.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def main():
    # model = load_model()
    # print(model)
    # pairs = pd.read_csv(data_dir / )
    # print(os.listdir(data_dir))

    df = pd.read_csv(cplfw_dir / 'pairs.csv')
    print(df)


if __name__ == '__main__':
    main()


