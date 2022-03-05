import pickle
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--test_folder', default='/gpfs/gpfs0/k.fedyanin/space/figures/test')
parser.add_argument('--last_timestamp', action="store_true")
args = parser.parse_args()

print(args.test_folder)

folder = Path(args.test_folder)

print(1)
with open(folder / 'scale_features.pickle', 'rb') as f:
    sf = pickle.load(f)

print(2)
with open(folder / 'head_features.pickle', 'rb') as f:
    hf = pickle.load(f)

for i, key in tqdm(enumerate(sf)):
    if not np.all(sf[key] == hf[key]):
        import ipdb; ipdb.set_trace()

import ipdb; ipdb.set_trace()
