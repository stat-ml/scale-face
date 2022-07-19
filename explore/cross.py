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
from face_lib.utils.imageprocessing import preprocess

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


def load_model(device='cuda'):
    model = iresnet50()
    # checkpoint = torch.load(data_dir / 'models/backbone_resnet50.pth')
    checkpoint = torch.load(data_dir / 'models/scaleface.pth', map_location='cpu')['backbone']
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    return model


def full_path(image_path):
    return str(cplfw_dir / 'aligned images' / image_path)


def precalculate_images(model, image_paths):
    size = (112, 112)
    full_paths = [full_path(path) for path in image_paths]
    batch = preprocess(full_paths, size)
    return batch
    device = 'cuda'
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        predictions = model(batch)['feature'].cpu()
        cached = {path: embeddings for path, embeddings in zip(image_paths, predictions)}
    return cached


def main():
    model = load_model()
    df = pd.read_csv(cplfw_dir / 'pairs.csv')
    cut = 5
    df = pd.concat([df.iloc[:cut], df.iloc[-cut:]])
    lst = np.unique(df.photo_1.to_list() + df.photo_2.to_list())
    cached = precalculate_images(model, lst)
    cached = np.floor((cached + 1) * 128).astype(np.uint8)

    for image in cached:
        print(image.shape)
        plt.imshow(image)
        plt.show()

    # for path, pred in cached.items():
    #     cached[path] = pred / torch.norm(pred)

    # def check(row):
    #     x_0, x_1 = cached[row['photo_1']], cached[row['photo_2']]
    #     return (x_0 * x_1).sum().item()
    #
    # scores = df.apply(check, axis=1)
    # labels = list(df.label)
    # from sklearn.metrics import roc_curve, precision_recall_curve
    #
    # preds = (scores > 0.19).astype(np.uint)
    # print((preds == labels).mean())
    #
    # precisions, recalls, threshs = precision_recall_curve(labels, scores)
    # plt.plot(threshs, precisions[1:])
    # plt.plot(threshs, recalls[1:])
    # plt.show()


if __name__ == '__main__':
    main()
