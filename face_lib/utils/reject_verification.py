import os
import sys
import argparse
import numpy as np
import torch
from path import Path

# path = str(
#     Path(Path(Path(__file__).parent.abspath()).parent.abspath()).parent.abspath()
# )
path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.utils.fusion import extract_features
from face_lib.utils.fusion_metrics import (
    pair_euc_score,
    pair_cosine_score,
    pair_MLS_score,
)
from face_lib.utils.fusion_metrics import l2_normalize, aggregate_PFE


def eval_reject_verification(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    batch_size=64,
    rejected_portions=None):

    if rejected_portions is None:
        rejected_portions = [0., ]

    pairs, labels = [], []
    unique_imgs = set()
    with open(pairs_table_path, "r") as f:
        for line in f.readlines():
            left_path, right_path, label = line.split(",")
            pairs.append((left_path, right_path))
            labels.append(int(label))
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
    labels = np.array(labels, dtype=bool)

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :",  labels.shape,  labels.dtype)
    print(labels.dtype)

    # score_vec = compare_func(features1, features2)
    print("Hello world")


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
        "--figure_path",
        help="The figure will be saved to this path",
        type=str, default=None, )
    parser.add_argument(
        "--device_id",
        help="Gpu id on which the algorithm will be launched",
        type=int, default=0, )

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

    eval_reject_verification(
        backbone,
        head,
        args.dataset_path,
        args.pairs_table_path,
        batch_size=args.batch_size,
        rejected_portions=rejected_portions)
