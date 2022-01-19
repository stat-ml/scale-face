import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from path import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc

path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

import face_lib.utils.metrics as metrics
import face_lib.evaluation.plots as plots
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.evaluation import name_to_distance_func, name_to_uncertainty_func
from face_lib.evaluation.argument_parser import parse_args_reject_verification
from face_lib.evaluation.feature_extractors import (
    extract_features_head,
    extract_features_gan,
    extract_features_scale,
)
from face_lib.evaluation.wrappers import (
    classifier_to_distance_wrapper,
    classifier_to_uncertainty_wrapper,
    split_wrapper,
)


def get_features_sigmas_labels(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    uncertainty_strategy="head",
    batch_size=64,
    discriminator=None,
    scale_predictor=None,
    device=torch.device("cuda:0"),
    verbose=False,
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

    if uncertainty_strategy == "head":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        mu, sigma_sq = extract_features_head(
            backbone,
            head,
            list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "GAN":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        if discriminator is None:
            raise RuntimeError("Please determine a discriminator")
        mu, sigma_sq = extract_features_gan(
            backbone,
            discriminator,
            list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "classifier":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        mu, sigma_sq = extract_features_head(
            backbone,
            head,
            list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "scale":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        mu, sigma_sq = extract_features_scale(
            backbone,
            scale_predictor,
            list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    else:
        raise NotImplementedError("Don't know this type of uncertainty strategy")

    mu_1 = np.array([mu[img_to_idx[pair[0]]] for pair in pairs])
    mu_2 = np.array([mu[img_to_idx[pair[1]]] for pair in pairs])
    sigma_sq_1 = np.array([sigma_sq[img_to_idx[pair[0]]] for pair in pairs])
    sigma_sq_2 = np.array([sigma_sq[img_to_idx[pair[1]]] for pair in pairs])
    label_vec = np.array(label_vec, dtype=bool)

    return mu_1, mu_2, sigma_sq_1, sigma_sq_2, label_vec


def eval_reject_verification(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    uncertainty_strategy="head",
    uncertainty_mode="uncertainty",
    batch_size=64,
    distaces_batch_size=None,
    rejected_portions=None,
    FARs=None,
    distances_uncertainties=None,
    discriminator=None,
    classifier=None,
    scale_predictor=None,
    save_fig_path=None,
    verbose=False,
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
        backbone, head, dataset_path, pairs_table_path,
        uncertainty_strategy=uncertainty_strategy, batch_size=batch_size, verbose=verbose,
        discriminator=discriminator, scale_predictor=scale_predictor,
    )

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :", label_vec.shape, label_vec.dtype)

    uncertainty_fig, uncertainty_axes = None, [None] * len(distances_uncertainties)
    if save_fig_path is not None:
        uncertainty_fig, uncertainty_axes = plt.subplots(
            nrows=1, ncols=len(distances_uncertainties),
            figsize=(9 * len(distances_uncertainties), 8))

    all_results = OrderedDict()
    device = next(backbone.parameters()).device

    for (distance_name, uncertainty_name), uncertainty_ax in zip(distances_uncertainties, uncertainty_axes):
        print(f"=== {distance_name} {uncertainty_name} ===")
        if distance_name == "classifier":
            distance_func = classifier_to_distance_wrapper(
                classifier, device=device)
        else:
            distance_func = name_to_distance_func[distance_name]

        if uncertainty_name == "classifier":
            uncertainty_func = classifier_to_uncertainty_wrapper(
                classifier, device=device)
        else:
            uncertainty_func = name_to_uncertainty_func[uncertainty_name]

        if distaces_batch_size:
            distance_func = split_wrapper(distance_func, batch_size=distaces_batch_size)
            uncertainty_func = split_wrapper(uncertainty_func, batch_size=distaces_batch_size)

        result_table = get_rejected_tar_far(
            mu_1,
            mu_2,
            sigma_sq_1,
            sigma_sq_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=uncertainty_mode,
            FARs=FARs,
            uncertainty_ax=uncertainty_ax,
        )

        if save_fig_path is not None:
            uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        all_results[(distance_name, uncertainty_name)] = result_table

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    for (distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    if save_fig_path:
        for (distance_name, uncertainty_name), result_table in all_results.items():
            title = (
                    pairs_table_path.split("/")[-1][-4]
                    + " "
                    + distance_name
                    + " "
                    + uncertainty_name
            )
            save_to_path = (
                os.path.join(save_fig_path, distance_name + "_" + uncertainty_name + ".jpg")
            )
            if save_fig_path:
                plots.plot_rejected_TAR_FAR(result_table, rejected_portions, title, save_to_path)

        plots.plot_TAR_FAR_different_methods(
            all_results,
            rejected_portions,
            res_AUCs,
            title=pairs_table_path.split("/")[-1][:-4],
            save_figs_path=os.path.join(save_fig_path, "all_methods.jpg")
        )

        uncertainty_fig.savefig(os.path.join(save_fig_path, "uncertainty.jpg"), dpi=400)

        torch.save(all_results, os.path.join(save_fig_path, "table.pt"))


def get_rejected_tar_far(
    mu_1,
    mu_2,
    sigma_sq_1,
    sigma_sq_2,
    label_vec,
    distance_func,
    pair_uncertainty_func,
    FARs,
    uncertainty_mode="uncertainty",
    uncertainty_ax=None,
):
    # If something's broken, uncomment the line below

    # score_vec = force_compare(distance_func)(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    score_vec = distance_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    uncertainty_vec = pair_uncertainty_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)

    sorted_indices = uncertainty_vec.argsort()
    score_vec = score_vec[sorted_indices]
    label_vec = label_vec[sorted_indices]
    uncertainty_vec = uncertainty_vec[sorted_indices]
    assert score_vec.shape == label_vec.shape

    if uncertainty_mode == "uncertainty":
        pass
    elif uncertainty_mode == "confidence":
        score_vec, label_vec, uncertainty_vec = score_vec[::-1], label_vec[::-1], uncertainty_vec[::-1]
    else:
        raise RuntimeError("Don't know this type uncertainty mode")

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

    plots.plot_uncertainty_distribution(
        uncertainty_vec, label_vec, ax=uncertainty_ax)

    return result_table


if __name__ == "__main__":
    args = parse_args_reject_verification()

    if os.path.isdir(args.save_fig_path) and not args.save_fig_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_fig_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))
    model_args = cfg.load_config(args.config_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    backbone = mlib.model_dict[model_args.backbone["name"]](
        **utils.pop_element(model_args.backbone, "name")
    )
    backbone.load_state_dict(checkpoint["backbone"])
    backbone = backbone.eval().to(device)

    head = None
    if args.uncertainty_strategy == "head" or (args.uncertainty_strategy == "classifier" and "head" in model_args):
        head = mlib.heads[model_args.head.name](
            **utils.pop_element(model_args.head, "name")
        )
        head.load_state_dict(checkpoint["head"])
        head = head.eval().to(device)

    discriminator = None
    if args.discriminator_path:
        discriminator = mlib.StyleGanDiscriminator()
        discriminator.load_state_dict(torch.load(args.discriminator_path)["d"])
        discriminator.eval().to(device)

    classifier = None
    if args.uncertainty_strategy == "classifier":
        classifier_name = model_args.pair_classifier.pop("name")
        classifier = mlib.pair_classifiers[classifier_name](
            **model_args.pair_classifier,
        )
        classifier.load_state_dict(checkpoint["pair_classifier"])
        classifier = classifier.eval().to(device)

    scale_predictor = None
    if args.uncertainty_strategy == "scale":
        scale_predictor_name = model_args.scale_predictor.pop("name")
        scale_predictor = mlib.scale_predictors[scale_predictor_name](
            **model_args.scale_predictor,
        )
        scale_predictor.load_state_dict(checkpoint["scale_predictor"])
        scale_predcitor = scale_predictor.eval().to(device)

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
        uncertainty_strategy=args.uncertainty_strategy,
        uncertainty_mode=args.uncertainty_mode,
        batch_size=args.batch_size,
        distaces_batch_size=args.distaces_batch_size,
        rejected_portions=rejected_portions,
        FARs=FARs,
        distances_uncertainties=distances_uncertainties,
        discriminator=discriminator,
        classifier=classifier,
        scale_predictor=scale_predictor,
        save_fig_path=args.save_fig_path,
        verbose=args.verbose,
    )
