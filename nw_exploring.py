#%%
from pathlib import Path
import os
import sys
import inspect
import torch
import torchvision
import random
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import face_lib.models as mlib
import face_lib.datasets as dlib

#%%
# checkpoint_path = "/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt"
# checkpoint_path = "/trinity/home/r.kail/Faces/face-evaluation/exman/runs/000067-2021-06-07-12-20-37/checkpoints/sota.pth"
# casia_dir = "/gpfs/gpfs0/r.karimov/casia"
# cats_dir = "/gpfs/gpfs0/r.karimov/cats"

checkpoint_path = 'data/sota.pt'
casia_dir = 'data/casia'
cats_dir = 'data/cats'

#%%
class Args:
    def __init__(self):
        self.train_file = os.path.join(casia_dir, "ldmarks.txt")
        self.casia_dir = os.path.join(casia_dir, "data_")
        self.cats_dir = os.path.join(cats_dir)
        self.try_times = 5
        self.is_debug = False
        self.in_size = (112, 112)
        self.num_face_pb = 4
        self.in_feats = 512


args = Args()
#%%
device = 'cuda'
checkpoint = torch.load(checkpoint_path, map_location=device)
#%%
print(checkpoint['backbone'].keys())
model = mlib.iresnet50()
model.load_state_dict(checkpoint["backbone"])
model = model.eval().to(device)
print(model)
#%%
head = mlib.PFEHeadAdjustable(25088, args.in_feats)
head.load_state_dict(checkpoint["head"])
head.eval().to(device)
#%%
face_set = dlib.CASIAWebFace(args)
#%%
# folder = 'data/casia/data_/0000045'
# print(os.listdir(folder))
# file = Path(folder) / '015.jpg'
# image_2 = cv2.imread(str(file))
# plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
# plt.show()
# line = face_set.lines[0]
# info = line.strip().split(" ")
# print(info[0], int(float(info[1])))

#%%
def show(torch_image):
    image = cv2.cvtColor(np.array(torch_image.permute(1, 2, 0)), cv2.COLOR_BGR2RGB)
    image *= 0.5
    image += 0.5
    plt.imshow(image)
    plt.show()


for i, (image, label) in enumerate(face_set):
    print(label)
    print(image.shape)
    show(image)
    if i == 5:
        break

with torch.no_grad():
    pred = model(image.unsqueeze(0))


#%%
with torch.no_grad():
    print(head(**model(face_set[5000][0].unsqueeze(0))))

#%%
image, _ = face_set[6100]
show(image)

#%%
loader = DataLoader(face_set, batch_size=50)

#%%
embeddings = []
labels = []
#%%
from tqdm import tqdm
#%%
for i, (x_batch, y_batch) in enumerate(tqdm(loader)):
    with torch.no_grad():
        embeddings.append(model(x_batch.cuda())['feature'].cpu())
        labels.append(y_batch)

    if i == 500:
        break
#%%
labels_ = torch.cat(labels)
print(labels_.shape)
#%%
embeddings_ = torch.cat(embeddings)
#%%
x_full = embeddings_.numpy()
y_full = labels_.numpy()
relabel = {label: i for i, label in enumerate(np.unique(y_full))}
y_full = np.array(list(map(relabel.get, y_full)))
#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#%%
model = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, stratify=y_full)


#%%
len(y_full)
#%%
int(float(face_set.lines[0].strip().split(" ")[1]))

#%%
import os

#%%
class CasiaWeb(Dataset):
    def __init__(self, basedir, debug=False):
        self.basedir = basedir

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

#%%
pass



