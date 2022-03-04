import pickle
import numpy as np
from path import Path
from tqdm import tqdm

FILE = '/gpfs/data/gpfs0/k.fedyanin/space/IJB/magface_evaluation/features_old_mtcnn/ir50_features'
FEATURE_PATH = '/gpfs/gpfs0/k.fedyanin/space/figures/test/magface_features.pickle'
UE_PATH = '/gpfs/gpfs0/k.fedyanin/space/figures/test/magface_uncertainty.pickle'


# print(os.listdir(FILE))
with open(FILE, 'r') as f:
    lines = [l.strip().split() for l in f.readlines()]


def map_float(array):
    return np.array([float(el) for el in array])


def short_path(path):
    return "/".join(Path(path).parts()[-2:])


feature_dict = {short_path(l[0]): map_float(l[1:]) for l in tqdm(lines)}
ue_dict = {k: np.linalg.norm(v, keepdims=True) for k, v in tqdm(feature_dict.items())}
feature_dict = {k: v.astype(np.float32) for k, v in tqdm(feature_dict.items())}


with open(FEATURE_PATH, 'wb') as f:
    pickle.dump(feature_dict, f)
with open(UE_PATH, 'wb') as f:
    pickle.dump(ue_dict, f)
