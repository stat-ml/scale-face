"""
What should I've done?
validation value
draw a line
1 million
what is the true value?
"""

from argparse import ArgumentParser
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from face_lib.evaluation.utils import get_required_models, extract_statistics
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import register
from face_lib.evaluation.feature_extractors import extract_features_head, extract_features_scale, get_features_uncertainties_labels
from face_lib.evaluation.distance_uncertainty_funcs import cosine_similarity, pair_MLS_score, pair_sqrt_scale_harmonic_biased_cosine_score
import imageio
from PIL import Image, ImageFilter


print('Imported')


def preprocess(
    images, center_crop_size, mode="RGB", align: tuple = None, *, is_training=False, blur=None
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    image_paths = images
    images = []
    for image_path in image_paths:
        # images.append(imageio.imread(image_path, pilmode=mode))
        image = Image.open(image_path)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur))
        images.append(np.array(image))

    images = np.stack(images, axis=0)

    proc_funcs = [
        ["center_crop", center_crop_size],
        ["standardize", "mean_scale"],
    ]

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert (
            proc_name in register
        ), "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images


def main(args):
    print(args)

    #
    # df = pd.read_csv(args.pairs_table_path, names=['source', 'target', 'label'])
    # df = crop_pairs(df, 500)
    # df.loc[df['label'] == 1, 'target'] = df[df['label'] == 1]['source']
    # image_paths = get_image_paths(df)
    # dataset_path = Path(args.dataset_path)
    # full_paths = [str(dataset_path / p) for p in image_paths]

    backbone, head = get_model(args)
    # mus, sigmas = extract_features(backbone, head, full_paths, blur=blur,
    #                                uncertainty_strategy=args.uncertainty_strategy)
    val_data = get_features_uncertainties_labels(
        backbone, head, args.dataset_path, args.pairs_table_path,
        uncertainty_strategy=args.uncertainty_strategy, batch_size=50, verbose=True,
        scale_predictor=head, precalculated_path=None
    )
    stats = extract_statistics(val_data)
    print(stats)
    return



    blurs = np.arange(0, 20, 1)
    original = []
    original_std = []
    impostors = []
    impostors_std = []

    for i, blur in enumerate(blurs):
        mus, sigmas = extract_features(backbone, head, full_paths, blur=blur, uncertainty_strategy=args.uncertainty_strategy)
        image_mu = {path: mu for path, mu in zip(image_paths, mus)}
        image_sigma = {path: sigma for path, sigma in zip(image_paths, sigmas)}
        if i == 0:
            image_mu_0 = image_mu.copy()
            image_sigma_0 = image_sigma.copy()

        def distance(row):
            if row.label == 1:
                mu_0 = image_mu_0[row['source']][None, :]
                sigma_0 = image_sigma_0[row['source']][None, :]
            else:
                mu_0 = image_mu[row['source']][None, :]
                sigma_0 = image_sigma[row['source']][None, :]
            mu_1 = image_mu[row['target']][None, :]
            sigma_1 = image_sigma[row['target']][None, :]
            # return cosine_similarity(mu_0, mu_1)
            # return pair_MLS_score(mu_0, mu_1, sigma_0, sigma_1)
            return pair_sqrt_scale_harmonic_biased_cosine_score(mu_0, mu_1, sigma_0, sigma_1, 0.26)


        df['distance'] = df.apply(distance, axis=1)
        original.append(df[df.label == 1].distance.mean())
        original_std.append(df[df.label == 1].distance.std())
        impostors.append(df[df.label == 0].distance.mean())
        impostors_std.append(df[df.label == 0].distance.std())

    original = np.array(original)[:, 0]
    original_std = np.array(original_std)
    impostors = np.array(impostors)[:, 0]
    impostors_std = np.array(impostors_std)

    plt.plot(blurs, original)
    plt.fill_between(blurs, original-original_std, original+original_std, alpha=0.2)
    plt.plot(blurs, impostors)
    plt.fill_between(blurs, impostors-impostors_std, impostors+impostors_std, alpha=0.2)
    plt.title('ScaleFace improved')
    plt.show()

    import ipdb; ipdb.set_trace()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pairs_table_path", default="/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/test_pairs_1000_prob_0.5.csv"
    )
    parser.add_argument(
        "--dataset_path", default="/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big"
    )
    # "--checkpoint_path", default = "/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/64/checkpoint.pth"
    parser.add_argument(
        "--checkpoint_path", default="/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/classic_normalized_pfe/sota.pth"
    )
    # "--config_path", default = "./configs/scale/02_sigm_mul_coef_selection/64.yaml"
    parser.add_argument(
        "--config_path", default="./configs/models/iresnet_ms1m_pfe_normalized.yaml"
    )
    parser.add_argument(
        "--discriminator_path", type=str, default=None
    )
    parser.add_argument(
        '--uncertainty_strategy', default='head'
    )
    return parser.parse_args()


def crop_pairs(df, num=20):
    pos = df[df.label == 0].iloc[:num]
    neg = df[df.label == 1].iloc[:num]
    return pd.concat((pos, neg))


def get_model(args):
    device = torch.device("cuda")
    model_args = cfg.load_config(args.config_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    backbone, head, discriminator, classifier, scale_predictor, uncertainty_model = \
        get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device=device)

    if args.uncertainty_strategy == 'head':
        ue = head
    elif args.uncertainty_strategy == 'scale':
        ue = scale_predictor
    else:
        ue = None

    return backbone, ue



def extract_features(backbone, head, image_paths, blur=None, uncertainty_strategy='head'):
    proc_func = lambda images: preprocess(images, [112, 112], is_training=False, blur=blur)

    if uncertainty_strategy == 'head':
        features, uncertainties = extract_features_head(
            backbone,
            head,
            image_paths,
            batch_size=50,
            proc_func=proc_func,
            verbose=True,
            device=torch.device("cuda"),
        )
    elif uncertainty_strategy == 'scale':
        features, uncertainties = extract_features_scale(
            backbone,
            head,
            image_paths,
            batch_size=50,
            proc_func=proc_func,
            verbose=True,
            device=torch.device("cuda"),
        )
    else:
        raise ValueError()
    return features, uncertainties


def get_image_paths(df):
    paths = np.unique(df.source.to_list() + df.target.to_list())
    return [str(path) for path in paths]


if __name__ == '__main__':
    args = parse_args()
    main(args)
