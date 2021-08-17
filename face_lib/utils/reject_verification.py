import os
import sys
import argparse
import numpy as np
import torch
from path import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc
import matplotlib.pyplot as plt

path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.utils.fusion import force_compare
from face_lib.utils.feature_extractors import (
    extract_features_head,
    extract_features_tta,
)
from face_lib.utils.fusion_metrics import (
    pair_euc_score,
    pair_cosine_score,
    pair_MLS_score,
    pair_uncertainty_sum,
    pair_uncertainty_harmonic_sum,
    pair_uncertainty_concatenated_harmonic,
)
import face_lib.utils.fusion_metrics as metrics

name_to_distance_func = {
    "euc": pair_euc_score,
    "cosine": pair_cosine_score,
    "MLS": pair_MLS_score,
}

name_to_uncertainty_func = {
    "mean": pair_uncertainty_sum,
    "harmonic-sum": pair_uncertainty_harmonic_sum,
    "harmonic-harmonic": pair_uncertainty_concatenated_harmonic,
}


def plot_rejected_TAR_FAR(table, rejected_portions, title=None, save_fig_path=None):
    fig, ax = plt.subplots()
    for FAR, TARs in table.items():
        ax.plot(rejected_portions, TARs, label="TAR@FAR=" + str(FAR), marker=" ")
    fig.legend()
    ax.set_xlabel("Rejected portion")
    ax.set_ylabel("TAR")
    if title:
        ax.set_title(title)
    if save_fig_path:
        fig.savefig(save_fig_path)
    return fig


def plot_TAR_FAR_different_methods(
    results, rejected_portions, AUCs, title=None, save_figs_path=None
):
    plots_indices = {
        FAR: idx for idx, FAR in enumerate(next(iter(results.values())).keys())
    }
    fig, axes = plt.subplots(
        ncols=len(plots_indices), nrows=1, figsize=(9 * len(plots_indices), 8)
    )
    for (distance_name, uncertainty_name), table in results.items():
        for FAR, TARs in table.items():
            auc = AUCs[(distance_name, uncertainty_name)][FAR]
            axes[plots_indices[FAR]].plot(
                rejected_portions,
                TARs,
                label=distance_name
                + "_"
                + uncertainty_name
                + "_AUC="
                + str(round(auc, 5)),
                marker=" ",
            )
            axes[plots_indices[FAR]].set_title(f"TAR@FAR={FAR}")
            axes[plots_indices[FAR]].set_xlabel("Rejected portion")
            axes[plots_indices[FAR]].set_ylabel("TAR")
            axes[plots_indices[FAR]].legend()
    if title:
        fig.suptitle(title)
    if save_figs_path:
        fig.savefig(save_figs_path)
    return fig


def get_features_sigmas_labels(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    batch_size=64,
):

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

    mu, sigma_sq = extract_features_head(
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

    return mu_1, mu_2, sigma_sq_1, sigma_sq_2, label_vec


def get_rejected_tar_far(
    mu_1,
    mu_2,
    sigma_sq_1,
    sigma_sq_2,
    label_vec,
    distance_func,
    pair_uncertainty_func,
    FARs,
):

    score_vec = force_compare(distance_func)(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    uncertainty_vec = pair_uncertainty_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)

    sorted_indices = uncertainty_vec.argsort()
    score_vec = score_vec[sorted_indices]
    label_vec = label_vec[sorted_indices]
    assert score_vec.shape == label_vec.shape

    result_table = defaultdict(list)
    result_fars = defaultdict(list)
    for rejected_portion in tqdm(rejected_portions):
        cur_len = int(score_vec.shape[0] * (1 - rejected_portion))
        tars, fars, thresholds = metrics.ROC(
            score_vec[:cur_len], label_vec[:cur_len], FARs=FARs
        )
        for far, tar in zip(FARs, tars):
            result_table[far].append(tar)
        for wanted_far, real_far in zip(FARs, fars):
            result_fars[wanted_far].append(real_far)

    # print("Result fars :")
    # for wanted_far, resulted_fars in result_fars.items():
    #     print(f"\tWanted FAR : {wanted_far} resulted fars : ", end="")
    #     for far in resulted_fars:
    #         print(round(far, 5), end=" ")
    #     print()
    # print("TAR@FARs :")
    # for far, TARs in result_table.items():
    #     print(f"\tFAR : {far} TARS : ", end="")
    #     for tar in TARs:
    #         print(round(tar, 5), end=" ")
    #     print()

    return result_table


def eval_reject_verification(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    batch_size=64,
    rejected_portions=None,
    FARs=None,
    distances_uncertainties=None,
):

    if rejected_portions is None:
        rejected_portions = [
            0.0,
        ]
    if FARs is None:
        FARs = [
            0.0,
        ]

    mu_1, mu_2, sigma_sq_1, sigma_sq_2, label_vec = get_features_sigmas_labels(
        backbone, head, dataset_path, pairs_table_path, batch_size=batch_size
    )

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :", label_vec.shape, label_vec.dtype)

    all_results = OrderedDict()
    for distance_name, uncertainty_name in distances_uncertainties:
        print(f"=== {distance_name} {uncertainty_name} ===")
        result_table = get_rejected_tar_far(
            mu_1,
            mu_2,
            sigma_sq_1,
            sigma_sq_2,
            label_vec,
            distance_func=name_to_distance_func[distance_name],
            pair_uncertainty_func=name_to_uncertainty_func[uncertainty_name],
            FARs=FARs,
        )

        all_results[(distance_name, uncertainty_name)] = result_table

    # Please don't fuck up here. The distance between edge points must be the same in all of the experiments.
    # Otherwise the results are incomparable
    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    for (distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    for (distance_name, uncertainty_name), result_table in all_results.items():
        title = (
            pairs_table_path.split("/")[-1][-4]
            + " "
            + distance_name
            + " "
            + uncertainty_name
        )
        save_to_path = (
            args.save_fig_path + "_" + distance_name + "_" + uncertainty_name + ".jpg"
        )
        if args.save_fig_path:
            plot_rejected_TAR_FAR(result_table, rejected_portions, title, save_to_path)

    plot_TAR_FAR_different_methods(
        all_results,
        rejected_portions,
        res_AUCs,
        title=pairs_table_path.split("/")[-1][:-4],
        save_figs_path=args.save_fig_path + "_" + "all_methods" + ".jpg",
    )
    torch.save(all_results, args.save_fig_path + "_table.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="The path to the pre-trained model directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        help="The path to the IJB-C dataset directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pairs_table_path",
        help="Path to csv file with pairs names",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size", help="Number of images per mini batch", type=int, default=64
    )
    parser.add_argument(
        "--config_path",
        help="The paths to config .yaml file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rejected_portions",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--distance_uncertainty_metrics",
        help="Pairs of distance and uncertainty metrics to evaluate with, separated with '_' (ex. cosine_harmonic)",
        nargs="+",
    )
    parser.add_argument(
        "--figure_path",
        help="The figure will be saved to this path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device_id",
        help="Gpu id on which the algorithm will be launched",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_fig_path",
        help="Path to save figure to",
        type=str,
        default=None,
    )

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
    head.load_state_dict(checkpoint["head"])

    rejected_portions = list(
        map(lambda x: float(x.replace(",", ".")), args.rejected_portions)
    )
    FARs = list(map(float, args.FARs))
    distances_uncertainties = list(
        map(lambda x: x.split("_"), args.distance_uncertainty_metrics)
    )

    eval_reject_verification(
        backbone,
        head,
        args.dataset_path,
        args.pairs_table_path,
        batch_size=args.batch_size,
        rejected_portions=rejected_portions,
        FARs=FARs,
        distances_uncertainties=distances_uncertainties,
    )
