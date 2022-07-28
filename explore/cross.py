"""
Level 4: disentangled
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
from face_lib.evaluation.argument_parser import parse_cli_arguments


def get_pairs(data_directory, short=False):
    pairs = pd.read_csv(data_directory / 'cplfw/pairs_test.csv')
    if short:
        cut = 300
        pairs = pd.concat([pairs.iloc[:cut], pairs.iloc[-cut:]])
    return pairs


class Inferencer:
    def __init__(self, preprocessing, full_model, batch_size=100):
        self.preprocessing = preprocessing
        self.full_model = full_model
        self.batch_size = batch_size

    def __call__(self, image_paths):
        mus = {}
        sigmas = {}
        for idx in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[idx: idx + self.batch_size]
            batch = self.preprocessing(batch_paths)
            batch_mus, batch_sigmas = self.full_model(batch)

            mus.update({
                path: embeddings for path, embeddings in zip(batch_paths, batch_mus)
            })
            sigmas.update({
                path: uncertainty.item() for path, uncertainty in zip(batch_paths, batch_sigmas)
            })
        return mus, sigmas


class EmbeddingNorm:
    """
    Basic uncertainty model with embedding norm as confidence notion
    """
    def __init__(self):
        self.backbone = None

    def from_checkpoint(self, checkpoint_path):
        backbone = iresnet50()
        checkpoint = torch.load(checkpoint_path, map_location='cuda')['backbone']
        backbone.load_state_dict(checkpoint)
        backbone.eval().cuda()
        self.backbone = backbone
        return self

    def __call__(self, batch):
        with torch.no_grad():
            output = self.backbone(batch)
            predictions = output['feature']
            uncertainties = torch.norm(predictions, dim=-1).cpu()
            predictions = torch.nn.functional.normalize(predictions, dim=-1).cpu()
        return predictions, uncertainties


class ScaleFace:
    def __init__(self):
        self.backbone = None
        self.scale = None

    def from_checkpoint(self, checkpoint_path):
        config_path = "./configs/scale/02_sigm_mul_coef_selection/64.yaml"
        model_args = load_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        args = EasyDict({'uncertainty_strategy': 'scale'})

        self.backbone, _, _, _, self.scale, _ = \
            get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device='cuda')

        return self

    def __call__(self, batch):
        with torch.no_grad():
            output = self.backbone(batch)
            predictions, bottleneck = output['feature'].cpu(), output['bottleneck_feature']
            uncertainties = self.scale.mlp(bottleneck).cpu()[:, 0]
            predictions = torch.nn.functional.normalize(predictions, dim=-1)
        return predictions, uncertainties


class PFE:
    def __init__(self):
        self.backbone = None
        self.head = None

    def __call__(self, batch):
        with torch.no_grad():
            predictions = self.backbone(batch)
            uncertainties = -torch.mean(self.head(**predictions)['log_sigma'], dim=-1).cpu()
            predictions = predictions['feature'].cpu()
        return predictions, uncertainties

    def from_checkpoint(self, checkpoint_path):
        config_path = "./configs/models/iresnet_ms1m_pfe_normalized.yaml"
        model_args = load_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        args = EasyDict({'uncertainty_strategy': 'head'})

        self.backbone, self.head, _, _, _, _ = \
            get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device='cuda')

        return self


def load_model(uncertainty_type, checkpoint_path):
    if uncertainty_type == 'embedding_norm':
        model = EmbeddingNorm().from_checkpoint(checkpoint_path)
    elif uncertainty_type == 'scale':
        model = ScaleFace().from_checkpoint(checkpoint_path)
    elif uncertainty_type == 'pfe':
        model = PFE().from_checkpoint(checkpoint_path)
    else:
        raise ValueError()
    return model


class Preprocessor:
    """
    Converts the list of image paths to ready-to-use tensors
    """
    def __init__(self, base_directory, image_size=(112, 112)):
        self.base_directory = Path(base_directory)
        self.image_size = image_size

    def _full_paths(self, paths):
        return [str(self.base_directory/p) for p in paths]

    def __call__(self, image_paths):
        full_paths = self._full_paths(image_paths)
        batch = preprocess(full_paths, self.image_size)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to('cuda')
        return batch


class Scorer:
    def __init__(self):
        pass

    def __call__(self, pairs, mus, sigmas):
        def check(row):
            x_0, x_1 = mus[row['photo_1']], mus[row['photo_2']]
            return (x_0 * x_1).sum().item()

        def basic_ue(row):
            x_0, x_1 = sigmas[row['photo_1']], sigmas[row['photo_2']]
            return x_0+x_1

        similarities = pairs.apply(check, axis=1)
        confidences = np.array(pairs.apply(basic_ue, axis=1))
        labels = list(pairs.label)

        scores = {
            'similarities': similarities,
            'confidences': confidences,
            'labels': labels
        }

        return scores


def plot_rejection(scores):
    for name, method_scores in scores.items():
        similarities = method_scores['similarities']
        confidences = method_scores['confidences']
        labels = method_scores['labels']

        preds = (similarities > 0.19).astype(np.uint)
        print((preds == labels).mean())

        correct = np.array((preds == labels))
        idxs = np.argsort(confidences)
        splits = np.arange(0, 0.9, 0.02)
        accuracies = []
        for split in splits:
            i = int(len(correct) * split)
            accuracies.append(correct[idxs[i:]].mean())

        plt.plot(splits, accuracies, label=name)

    plt.legend()
    plt.show()


def main():
    """
    Main pipeline
    1. Generate mu and sigmas for each individual photo
    2. Use mu and sigmas for pairs to generate the similarity and confidence scores
    3. Calculate the metrics
    """
    # args = parse_cli_arguments()
    configs = {
        'Embedding norm': './configs/cross/play.yaml',
        'PFE': './configs/cross/pfe.yaml',
        'ScaleFace': './configs/cross/scale.yaml',
    }
    scores = {}

    for name, config in configs.items():
        args = load_config(config)
        args.short = False
        data_directory = Path(args.data_directory)

        pairs = get_pairs(data_directory, short=args.short)
        photo_list = np.unique(pairs.photo_1.to_list() + pairs.photo_2.to_list())

        preprocessor = Preprocessor(data_directory / args.images_path)
        checkpoint_path = data_directory / args.checkpoint_path
        model = load_model(args.uncertainty_type, checkpoint_path)

        inferencer = Inferencer(preprocessor, model, 20)
        mus, sigmas = inferencer(photo_list)
        scorer = Scorer()
        scores[name] = scorer(pairs, mus, sigmas)

    plot_rejection(scores)


if __name__ == '__main__':
    main()
