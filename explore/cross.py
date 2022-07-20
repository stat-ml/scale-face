from pathlib import Path
import os
import sys

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.append(".")
from face_lib.models.iresnet import iresnet50
from face_lib.utils.imageprocessing import preprocess
from face_lib.evaluation.feature_extractors import extract_features_uncertainties_from_list


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


def precalculate_images(model, image_paths, batch_size=20):
    size = (112, 112)
    device = 'cuda'
    cached = {}
    for idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[idx: idx+batch_size]
        full_paths = [full_path(path) for path in batch_paths]
        batch = preprocess(full_paths, size)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            predictions = model(batch)['feature'].cpu()
            cached.update({path: embeddings for path, embeddings in zip(batch_paths, predictions)})
    return cached


def main():
    model = load_model()
    df = pd.read_csv(cplfw_dir / 'pairs.csv')
    cut = 300
    # df = pd.concat([df.iloc[:cut], df.iloc[-cut:]])
    lst = np.unique(df.photo_1.to_list() + df.photo_2.to_list())
    cached = precalculate_images(model, lst, batch_size=100)

    ue = {}
    for path, pred in cached.items():
        ue[path] = torch.norm(pred).item()
        cached[path] = pred / torch.norm(pred)

    def check(row):
        x_0, x_1 = cached[row['photo_1']], cached[row['photo_2']]
        return (x_0 * x_1).sum().item()

    def basic_ue(row):
        x_0, x_1 = ue[row['photo_1']], ue[row['photo_2']]
        # return x_1
        return np.sqrt(x_0*x_1)

    scores = df.apply(check, axis=1)
    ues = np.array(df.apply(basic_ue, axis=1))
    labels = list(df.label)

    print(ues)

    preds = (scores > 0.19).astype(np.uint)
    print((preds == labels).mean())

    correct = np.array((preds == labels))
    idxs = np.argsort(ues)
    splits = np.arange(0, 1, 0.1)
    accuracies = []
    for split in splits:
        i = int(len(correct) * split)
        accuracies.append(correct[idxs[i:]].mean())

    plt.plot(splits, accuracies)
    plt.show()


    # print(len(preds))
    #
    # precisions, recalls, threshs = precision_recall_curve(labels, scores)
    # plt.plot(threshs, precisions[1:])
    # plt.plot(threshs, recalls[1:])
    # plt.show()


if __name__ == '__main__':
    main()
