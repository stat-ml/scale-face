import os
import sys
import argparse
import numpy as np
import torch
from path import Path
from tqdm import tqdm
from collections import defaultdict

# path = str(
#     Path(Path(Path(__file__).parent.abspath()).parent.abspath()).parent.abspath()
# )
path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.utils.fusion import extract_features, force_compare
from face_lib.utils.fusion_metrics import (
    pair_euc_score,
    pair_cosine_score,
    pair_MLS_score,
)
from face_lib.utils.utils import harmonic_mean
import face_lib.utils.fusion_metrics as metrics
from face_lib.utils.fusion_metrics import l2_normalize, aggregate_PFE
import matplotlib.pyplot as plt


def calculate_uncertainties(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) + harmonic_mean(sigma_sq_2, axis=1)


def plot_rejected_TAR_FAR(table, rejected_portions, save_fig_path=None):
    fig, ax = plt.subplots()
    for far, tars in table.items():
        ax.plot(rejected_portions, tars, label="TAR@FAR=" + str(far), marker=".")
    fig.legend()
    if save_fig_path:
        fig.savefig(save_fig_path)
    return fig


def eval_reject_verification(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    batch_size=64,
    rejected_portions=None,
    FARs=None):

    if rejected_portions is None:
        rejected_portions = [0., ]
    if FARs is None:
        FARs = [0.]

    pairs, label_vec = [], []
    unique_imgs = set()
    with open(pairs_table_path, "r") as f:
        for line in f.readlines():
            left_path, right_path, label = line.split(",")
            pairs.append((left_path, right_path))
            label_vec.append(int(label))
            unique_imgs.add(left_path)
            unique_imgs.add(right_path)

    image_paths = list(unique_imgs)
    img_to_idx = {img_path: idx for idx, img_path in enumerate(image_paths)}

    proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

    mu, sigma_sq = extract_features(
        backbone,
        head,
        list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
        batch_size,
        proc_func=proc_func,
        verbose=True,
        device=device,
    )

    mu_1 = np.array([mu[img_to_idx[pair[0]]] for pair in pairs])
    mu_2 = np.array([mu[img_to_idx[pair[1]]] for pair in pairs])
    sigma_sq_1 = np.array([sigma_sq[img_to_idx[pair[0]]] for pair in pairs])
    sigma_sq_2 = np.array([sigma_sq[img_to_idx[pair[1]]] for pair in pairs])
    label_vec = np.array(label_vec, dtype=bool)

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :",  label_vec.shape,  label_vec.dtype)

    score_vec = force_compare(pair_cosine_score)(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    uncertainty_vec = calculate_uncertainties(mu_1, mu_2, sigma_sq_1, sigma_sq_2)

    sorted_indices = uncertainty_vec.argsort()
    score_vec = score_vec[sorted_indices]
    label_vec = label_vec[sorted_indices]
    assert score_vec.shape == label_vec.shape

    result_table = defaultdict(list)
    result_fars = defaultdict(list)
    for rejected_portion in tqdm(rejected_portions):
        cur_len = int(score_vec.shape[0] * (1 - rejected_portion))
        tars, fars, thresholds = metrics.ROC(score_vec[:cur_len], label_vec[:cur_len], FARs=FARs)
        for far, tar in zip(FARs, tars):
            result_table[far].append(tar)
        for wanted_far, real_far in zip(FARs, fars):
            result_fars[wanted_far].append(real_far)

    print("Result fars :")
    print(result_fars)
    print("TAR@FARs :")
    print(result_table)

    return result_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="The path to the pre-trained model directory",
        type=str, required=True, )
    parser.add_argument(
        "--dataset_path",
        help="The path to the IJB-C dataset directory",
        type=str, required=True, )
    parser.add_argument(
        "--pairs_table_path",
        help="Path to csv file with pairs names",
        type=str, required=True, )
    parser.add_argument(
        "--batch_size",
        help="Number of images per mini batch",
        type=int, default=64)
    parser.add_argument(
        "--config_path",
        help="The paths to config .yaml file",
        type=str, required=True, )
    parser.add_argument(
        "--rejected_portions",
        help="Portion of rejected pairs of images",
        nargs="+", )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+", )
    parser.add_argument(
        "--figure_path",
        help="The figure will be saved to this path",
        type=str, default=None, )
    parser.add_argument(
        "--device_id",
        help="Gpu id on which the algorithm will be launched",
        type=int, default=0, )
    parser.add_argument(
        "--save_fig_path",
        help="Path to save figure to",
        type=str, default=None, )

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device_id))

    model_args = cfg.load_config(args.config_path)
    backbone = mlib.model_dict[model_args.backbone["name"]](
        **utils.pop_element(model_args.backbone, "name")
    )
    head = mlib.heads[model_args.head.name](
        **utils.pop_element(model_args.head, "name")
    )
    backbone, head = backbone.to(device), head.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["uncertain"])

    rejected_portions = list(map(float, args.rejected_portions))
    FARs = list(map(float, args.FARs))

    result_table = eval_reject_verification(
        backbone,
        head,
        args.dataset_path,
        args.pairs_table_path,
        batch_size=args.batch_size,
        rejected_portions=rejected_portions,
        FARs=FARs)

    if args.save_fig_path:
        plot_rejected_TAR_FAR(result_table, rejected_portions, args.save_fig_path)
