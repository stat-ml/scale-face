import pickle
import numpy as np

FILE = '/gpfs/data/gpfs0/k.fedyanin/space/IJB/magface_evaluation/features_old_mtcnn/ir50_features'
FEATURE_PATH = '/gpfs/gpfs0/k.fedyanin/space/figures/test/magface_features.pickle'
UE_PATH = '/gpfs/gpfs0/k.fedyanin/space/figures/test/magface_uncertainty.pickle'


# print(os.listdir(FILE))
with open(FILE, 'r') as f:
    lines = [l.strip().split() for l in f.readlines()]
lines = lines

def map_float(array):
    return [float(el) for el in array]

feature_dict = {l[0]: map_float(l[1:]) for l in lines}
ue_dict = {k: np.linalg.norm(v, keepdims=True) for k, v in feature_dict.items()}

with open(FEATURE_PATH, 'wb') as f:
    pickle.dump(feature_dict, f)
with open(UE_PATH, 'wb') as f:
    pickle.dump(ue_dict, f)



print(dict)
import ipdb; ipdb.set_trace()
