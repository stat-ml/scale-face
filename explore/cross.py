# level 3
# got the pos/neg pairs

from pathlib import Path
import os

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_dir = Path("~/data/faces/cplfw").expanduser()


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

parse_pairs()


