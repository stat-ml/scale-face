#%%
from pathlib import Path
import os
import sys
import inspect
import torch
from tqdm import tqdm
import torchvision
import random
import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import colorize
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import face_lib.models as mlib
import face_lib.datasets as dlib

#%%
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
print(checkpoint['backbone'].keys())
model = mlib.iresnet50()
model.load_state_dict(checkpoint["backbone"])
model = model.eval().to(device)
print(model)
head = mlib.PFEHeadAdjustable(25088, args.in_feats)
head.load_state_dict(checkpoint["head"])
head.eval().to(device)
#%%

class CasiaWeb(Dataset):
    def __init__(self, basedir, debug=False):
        self.debug = debug
        self.basedir = Path(basedir)
        self.records = self._build_list(basedir)
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def _build_list(self, basedir):
        dirs = os.listdir(basedir)
        dirs = sorted(dirs)
        if self.debug:
            dirs = dirs[:20]
        faces = []
        class_ = 0
        for directory in dirs:
            files = os.listdir(self.basedir/ directory)
            files = sorted(files)
            faces.extend([
                (self.basedir / directory / f, class_) for f in files
            ])
            class_ += 1
        return faces

    def __getitem__(self, idx):
        path, class_ = self.records[idx]
        face = Image.open(path).convert('RGB').resize((112, 112))
        face_tensor = self.transforms(face)
        return face_tensor, class_

    def __len__(self):
        return len(self.records)

face_set = CasiaWeb('data/casia/data_', debug=True)

#%%
def show(torch_image):
    image = np.array(torch_image.permute(1, 2, 0))
    image *= 0.5
    image += 0.5
    plt.imshow(image)
    plt.show()


# for i, (image, label) in enumerate(face_set):
#     print(label)
#     print(image.shape)
#     show(image)
#     if i == 5:
#         break
#%%
# model2 = deepcopy(model)
# model2 = mlib.iresnet50()
@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
        # m.weight.fill_(1.0)
        torch.nn.init.normal_(m.weight, 0, 100)
    if type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm1d:
        m.reset_parameters()
# model2.apply(init_weights)

model = model.cuda()
loader = DataLoader(face_set, batch_size=50)
embeddings = []
labels = []
# model2 = model2.cuda()
for i, (x_batch, y_batch) in enumerate(tqdm(loader)):
    with torch.no_grad():
        embeddings.append(model(x_batch.cuda())['feature'].cpu())
        labels.append(y_batch)
labels_ = torch.cat(labels)
print(labels_.shape)
embeddings_ = torch.cat(embeddings)
x_full = embeddings_.numpy()
y_full = labels_.numpy()
print(x_full[:5, :5])
#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, stratify=y_full)

knn.fit(x_train, y_train)
print(np.mean(knn.predict(x_test) == y_test))
#%%
from ncvis import NCVis
reduction = NCVis(distance='cosine')
x_tr = reduction.fit_transform(x_train)

#%%

#%%

n = 20
from_list = matplotlib.colors.LinearSegmentedColormap.from_list
colors = np.concatenate((
    plt.cm.Set1(range(7)),
    plt.cm.Set2(range(7)),
    plt.cm.Set3(range(6)),
))
cm = from_list(None, colors, n)
plt.figure(figsize=(6, 5), dpi=120)
plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_train[:], alpha=0.3, cmap=cm)
plt.title("Face embedidng visualization (NCVis, cosine)")
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.clim(-0.5, n-0.5)
cb = plt.colorbar(ticks=range(0, n), label='Identity')
cb.ax.tick_params(length=0)
plt.show()
print('hi')
#%%
len(x_tr)



#%%%
from nw_uncertainty import NewNW
#%%
nw_classifier = NewNW(bandwidth=np.array([0.4, 0.4]), strategy='isj', tune_bandwidth=True,
                       n_neighbors=20, coeff=1e-10)
nw_classifier.fit(X=x_train, y=y_train)
#%%
uncertainty = nw_classifier.predict_uncertartainty(x_test)
#%%
ue = uncertainty['total']
print(ue.shape)
#%%
preds = knn.predict(x_test)
#%%
error = preds != y_test
print(error)
#%%
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(error, ue)
#%%
plt.figure(dpi=120)
auc = roc_auc_score(error, ue)
plt.title(f'Error prediction ROC-AUC score {auc:.3f}')
plt.plot(fpr, tpr)
plt.show()

#%%
model = model.cpu()
#%%
from copy import deepcopy
#%%

#%%
img, label = face_set[20]

#%%
n = 5
xs = [x_full[y_full == i] for i in range(n)]

mean_correlations = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        mean_correlations[i, j] = np.mean(np.corrcoef(xs[i], xs[j])[:len(xs[i]), len(xs[i]):])

import seaborn as sns
sns.heatmap(mean_correlations, annot=True)
plt.show()
#%%

#%%
# sns.heatmap(
#     np.corrcoef(x_train[:10], x_train[:10])
# )
# plt.show()
print(np.corrcoef(x_train[:10], x_train[:10]).shape)

#%%
# x_0 = x_train[:10]
# print(np.corrcoef(x_0, x_0).shape)
# print(x_0.shape)

np.corrcoef
#%%
# import numpy as np
# rng = np.random.default_rng(seed=42)
# # xarr = rng.random((3, 3))
# xarr = x_train[:3]
# print(xarr)
# print(xarr.shape)
# R1 = np.corrcoef(xarr, xarr)
# print(R1)



#%%
# np.corrcoef(xs[0], xs[1]).shape
#%%
sns.heatmap(
    np.corrcoef(xs[0], xs[1])[:len(xs[0]), len(xs[0]):]
)
plt.show()


#%%
# plt.hist(model2.conv1.weight.detach().cpu().numpy().flatten(), bins=100)
# plt.show()
#%%
# np.corrcoef(x_full[:10])
#%%
# model2.fc.weight
#%%









        
        








