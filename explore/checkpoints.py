import torch

# path_0 = '/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth'
# path_1 = '/gpfs/gpfs0/k.fedyanin/space/models/pfe/normalized_pfe/sota.pth'
path_0 = '/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth'
path_1 = '/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/classic_normalized_pfe/sota.pth'

model_0 = torch.load(path_0, map_location='cuda:0')['backbone']
model_1 = torch.load(path_1, map_location='cuda:0')['backbone']

print(model_0["conv1.weight"][0, 0])
print(model_1["conv1.weight"][0, 0])


import ipdb; ipdb.set_trace()





