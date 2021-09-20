import sys
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from pathlib import Path

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)

from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess
from face_lib.utils.imageprocessing import preprocess_tta
from face_lib.utils.feature_extractors import (
    extract_features_head,
    extract_features_tta,
    extract_features_fourier,
    extract_features_grad,
    extract_features_ssim,
)
from face_lib.utils.fusion_metrics import (
    pair_euc_score,
    pair_cosine_score,
    pair_MLS_score,
    l2_normalize,
    aggregate_PFE,
    aggregate_min,
    aggregate_softmax,
)

name_to_distance_func = {
    "euc": pair_euc_score,
    "cosine": pair_cosine_score,
    "MLS": pair_MLS_score,
}


def aggregate_templates(templates, mu, sigma_sq, method):
    sum_fuse_len = 0
    number_of_templates = 0
    for i, t in enumerate(templates):
        if len(t.indices) > 0:
            if method == "random":
                t.feature = l2_normalize(mu[np.random.choice(t.indices)])
                t.sigma_sq = None
            if method == "mean":
                t.feature = l2_normalize(np.mean(mu[t.indices], axis=0))
                t.sigma_sq = None
            if method == "PFE":
                t.mu, t.sigma_sq = aggregate_PFE(
                    mu[t.indices],
                    sigma_sq=sigma_sq[t.indices],
                    normalize=True,
                    concatenate=False,
                )
                t.feature = t.mu
            if method == "min":
                t.mu, t.sigma_sq = aggregate_min(
                    mu[t.indices],
                    sigma_sq=sigma_sq[t.indices],
                    normalize=True,
                    concatenate=False,
                )
                t.feature = t.mu
            if "softmax" in method:
                temperature = float(method.split("-")[1])
                t.feature = aggregate_softmax(
                    mu[t.indices],
                    sigma_sq=sigma_sq[t.indices],
                    temperature=temperature,
                    normalize=True,
                    concatenate=False,
                )
                t.sigma_sq = None
        else:
            t.feature = None
        if i % 1000 == 0:
            sys.stdout.write("Fusing templates {}/{}...\t\r".format(i, len(templates)))

        sum_fuse_len += len(t.indices)
        number_of_templates += int(len(t.indices) > 0)
    print("")
    print("Mean aggregated size : ", sum_fuse_len / number_of_templates)


def force_compare(compare_func, verbose=False):
    def compare(t1, t2, s1, s2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = -9999
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None], s1[i], s2[i])
            if verbose and i % 1000 == 0:
                sys.stdout.write("Matching pair {}/{}...\t\r".format(i, len(t1)))
        if verbose:
            print("")
        return score_vec

    return compare


def eval_fusion_ijb(
    backbone,
    head,
    dataset_path,
    protocol_path,
    batch_size=64,
    protocol="ijbc",
    uncertainty_strategy="head",
    fusion_distance_methods=None,
    FARs=None,
    device=torch.device("cpu"),
    verbose=False,
):
    if uncertainty_strategy == "TTA":
        proc_func = lambda images: preprocess_tta(images, [112, 112], is_training=False)
    elif uncertainty_strategy == "head":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
    elif uncertainty_strategy == "ssim":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
    elif uncertainty_strategy == "fourier":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
    elif uncertainty_strategy == "grad":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
    else:
        raise AssertionError("Unknown type of uncertainty calculating strategy")

    testset = IJBDataset(dataset_path)
    if protocol == "ijba":
        tester = IJBATest(testset["abspath"].values)
        tester.init_proto(protocol_path)
    elif protocol == "ijbc":
        tester = IJBCTest(testset["abspath"].values)
        tester.init_proto(protocol_path)
    else:
        raise ValueError('Unkown protocol. Only accept "ijba" or "ijbc".')

    backbone = backbone.eval().to(device)

    if uncertainty_strategy == "head":
        head = head.eval().to(device)
        mu, sigma_sq = extract_features_head(
            backbone,
            head,
            tester.image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "TTA":
        mu, sigma_sq = extract_features_tta(
            backbone,
            tester.image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "fourier":
        mu, sigma_sq = extract_features_fourier(
            backbone,
            tester.image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "grad":
        mu, sigma_sq = extract_features_grad(
            backbone,
            tester.image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "ssim":
        mu, sigma_sq = extract_features_ssim(
            backbone,
            tester.image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    else:
        raise AssertionError("Don't know this uncertainty calculation strategy")

    print(f"mu : {mu.shape} sigma : {sigma_sq.shape}")

    result = defaultdict(dict)
    for (fusion_name, distance_name) in fusion_distance_methods:
        print(f"==== fuse : {fusion_name} distance : {distance_name} ====")
        aggregate_templates(tester.verification_templates, mu, sigma_sq, fusion_name)
        TARs, stds, res_FARs = tester.test_verification(
            force_compare(name_to_distance_func[distance_name]), FARs=FARs
        )
        for FAR, std, TAR in zip(FARs, stds, TARs):
            result[(fusion_name, distance_name)][FAR] = TAR
            if verbose:
                print("TAR: {:.5} +- {:.5} FAR: {:.5}".format(TAR, std, FAR))

    return result


def dump_fusion_ijb(
    backbone,
    head,
    dataset_path,
    protocol_path,
    batch_size=64,
    protocol="ijbc",
    uncertainty_strategy="head",
    fusion_distance_methods=None,
    FARs=None,
    device=torch.device("cpu"),
    verbose=False,
    save_to=None,
):
    results = eval_fusion_ijb(
        backbone,
        head,
        dataset_path,
        protocol_path,
        batch_size=batch_size,
        protocol=protocol,
        uncertainty_strategy=uncertainty_strategy,
        fusion_distance_methods=fusion_distance_methods,
        FARs=FARs,
        device=device,
        verbose=verbose,
    )

    if save_to is not None:
        torch.save(results, save_to + ".pt")
        table = pd.DataFrame(results)
        table.to_pickle(save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="The path to the pre-trained model directory",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        help="The path to the IJB-A dataset directory",
        type=str,
        default="data/ijba_mtcnncaffe_aligned",
    )
    parser.add_argument(
        "--protocol_path",
        help="The path to the IJB-A protocol directory",
        type=str,
        default="proto/IJB-A",
    )
    parser.add_argument(
        "--protocol", help="The dataset to test", type=str, default="ijbc"
    )
    parser.add_argument(
        "--config_path", help="The paths to config .yaml file", type=str, default=None
    )
    parser.add_argument(
        "--uncertainty_strategy",
        help="Strategy to get uncertainty (ex. head or TTA)",
        type=str,
        default="head",
    )
    parser.add_argument(
        "--fusion_distance_methods",
        help="Pairs of distance metric and fusion distance to evaluate with, separated with '_' (ex. mean_cosine)",
        nargs="+",
    )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--device_id",
        help="Device on which the algorithm will be ran",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of images per mini batch",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--save_table_path",
        help="Path where the resulted table will be dumped",
        type=str,
        default="/gpfs/gpfs0/r.kail/tables/result.pkl",
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

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    backbone, head = backbone.eval().to(device), head.eval().to(device)

    fusion_distance_methods = list(
        map(lambda x: x.split("_"), args.fusion_distance_methods)
    )
    FARs = list(map(float, args.FARs))

    dump_fusion_ijb(
        backbone,
        head,
        args.dataset_path,
        args.protocol_path,
        batch_size=64,
        protocol="ijbc",
        uncertainty_strategy=args.uncertainty_strategy,
        fusion_distance_methods=fusion_distance_methods,
        FARs=FARs,
        device=device,
        verbose=args.verbose,
        save_to=args.save_table_path,
    )
