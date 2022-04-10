# WTF with the norm
# What is ue values now?
# Check the single eval_reje
#

from argparse import ArgumentParser
import pickle
from pathlib import Path
import numpy as np


parser = ArgumentParser()
parser.add_argument('--cache_path', default='/gpfs/gpfs0/k.fedyanin/space/figures/test')

args = parser.parse_args()


with open(Path(args.cache_path) / f'scale_features.pickle', 'rb') as f:
    feature_dict = pickle.load(f)

uncertainty_dict = {}

for key, value in feature_dict.items():
    uncertainty_dict[key] = np.array([np.linalg.norm(value)])

uncertainty_strategy = 'emb_norm'

with open(Path(args.cache_path) / f'{uncertainty_strategy}_features.pickle', 'wb') as f:
    pickle.dump(feature_dict, f)
with open(Path(args.cache_path) / f'{uncertainty_strategy}_uncertainty.pickle', 'wb') as f:
    pickle.dump(uncertainty_dict, f)
