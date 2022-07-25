"""
Level 1:
predict?

Pipeline should be pipeline?
1. Calculate the mu, sigma for a list of photos
2. Calc the ue for pairs
3. Reject verification
4. Plots
"""
from pathlib import Path
import os
import sys

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from easydict import EasyDict

sys.path.append(".")
from face_lib.models.iresnet import iresnet50
from face_lib.utils.imageprocessing import preprocess
from face_lib.evaluation.feature_extractors import extract_features_uncertainties_from_list
from face_lib.utils.cfg import load_config
from face_lib.evaluation.utils import get_required_models


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


def generate_val_test_split(dataframe):
    lst = np.unique(dataframe.photo_1.to_list() + dataframe.photo_2.to_list())

    def cut_name(name):
        return '_'.join(name.split('_')[:-1])

    names = np.sort(np.unique([cut_name(name) for name in lst]))
    np.random.seed(42)
    np.random.shuffle(names)

    test_identities = names[:2000]
    val_identities = names[2000:]
    print(test_identities)

    def suitable(row, identities):
        return cut_name(row.photo_1) in identities and cut_name(row.photo_2) in identities

    test_df = dataframe[dataframe.apply(lambda row: suitable(row, test_identities), axis=1)]
    val_df = dataframe[dataframe.apply(lambda row: suitable(row, val_identities), axis=1)]

    test_df.to_csv(cplfw_dir / 'pairs_test.csv', index=False)
    val_df.to_csv(cplfw_dir / 'pairs_val.csv', index=False)


def load_model(device='cuda'):
    model = iresnet50()
    uncertainty_type = ['embeddings_norm', 'scale'][1]

    if uncertainty_type == 'embeddings_norm':
        checkpoint = torch.load(data_dir / 'models/backbone_resnet50.pth')
        model.load_state_dict(checkpoint)
        model.eval().cuda()
        return model, None

    elif uncertainty_type == 'scale':
        config_path = "./configs/scale/02_sigm_mul_coef_selection/64.yaml"
        checkpoint_path = str(data_dir / 'models/scaleface.pth')
        # checkpoint = torch.load(data_dir / 'models/scaleface.pth', map_location='cpu')

        model_args = load_config(config_path)
        print(model_args)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = EasyDict({'uncertainty_strategy': 'scale'})

        backbone, head, discriminator, classifier, scale_predictor, uncertainty_model = \
            get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device=device)

        # backbone.features = torch.nn.Sequential()
        backbone.eval()
        print(backbone)

        return backbone, scale_predictor


def full_path(image_path):
    return str(cplfw_dir / 'aligned images' / image_path)


def precalculate_images(model, image_paths, batch_size=20, uncertainty_model=None):
    size = (112, 112)
    device = 'cuda'
    mu = {}
    sigma = {}

    for idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[idx: idx+batch_size]
        full_paths = [full_path(path) for path in batch_paths]
        batch = preprocess(full_paths, size)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            output = model(batch)
            # import ipdb; ipdb.set_trace()
            predictions, bottleneck = output['feature'].cpu(), output['bottleneck_feature']
            # uncertainties = torch.norm(predictions, dim=-1)
            uncertainties = uncertainty_model.mlp(bottleneck).cpu()[:, 0]
            predictions = torch.nn.functional.normalize(predictions, dim=-1)
            mu.update({path: embeddings for path, embeddings in zip(batch_paths, predictions)})
            sigma.update({path: uncertainty.item() for path, uncertainty in zip(batch_paths, uncertainties)})
    return mu, sigma


def naive_ue_calculation(df, model, uncertainty_model):
    lst = np.unique(df.photo_1.to_list() + df.photo_2.to_list())
    cached, ue = precalculate_images(model, lst, batch_size=100, uncertainty_model=uncertainty_model)

    def check(row):
        x_0, x_1 = cached[row['photo_1']], cached[row['photo_2']]
        return (x_0 * x_1).sum().item()

    def basic_ue(row):
        x_0, x_1 = ue[row['photo_1']], ue[row['photo_2']]
        return x_0+x_1

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


def main():
    model, ue_predictor = load_model()
    df = pd.read_csv(cplfw_dir / 'pairs_test.csv')
    cut = 500
    # df = pd.concat([df.iloc[:cut], df.iloc[-cut:]])
    naive_ue_calculation(df, model, ue_predictor)


if __name__ == '__main__':
    main()
