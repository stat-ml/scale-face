# make the short self-contained setup
# new IJB-C
# Should we update the mu?
# Cleaned the match from missing
#
#
#


import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc
from pathlib import Path

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)

from face_lib.datasets import IJBDataset, IJBATest, IJBCTest
from face_lib.utils import cfg
import face_lib.evaluation.plots as plots
from face_lib.evaluation.utils import get_required_models, get_distance_uncertainty_funcs
from face_lib.evaluation.feature_extractors import get_features_uncertainties_labels
from face_lib.evaluation.feature_extractors import extract_features_uncertainties_from_list
from face_lib.evaluation.reject_verification import get_rejected_tar_far

from face_lib.evaluation import name_to_distance_func, l2_normalize
from face_lib.evaluation.aggregation import aggregate_PFE, aggregate_min, aggregate_softmax
from face_lib.evaluation.argument_parser import parse_args_template_reject_verification
print('imported')


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
                # FIXME: This is not right, better think how to it right
                t.sigma_sq = np.mean(sigma_sq[t.indices], axis=0)
            if method == "PFE":
                t.mu, t.sigma_sq = aggregate_PFE(
                    mu[t.indices],
                    sigma_sq=sigma_sq[t.indices],
                    normalize=True,
                    concatenate=False,
                )
                t.feature = t.mu
                # TODO: update the mu?
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


def eval_template_reject_verification(
    backbone,
    dataset_path,
    protocol="ijbc",
    protocol_path=".",
    uncertainty_strategy="head",
    uncertainty_mode="uncertainty",
    batch_size=64,
    distaces_batch_size=None,
    rejected_portions=None,
    FARs=None,
    fusions_distances_uncertainties=None,
    head=None,
    discriminator=None,
    classifier=None,
    scale_predictor=None,
    save_fig_path=None,
    device=torch.device("cpu"),
    verbose=False,
    uncertainty_model=None
):

    # Setup the plots
    if rejected_portions is None:
        rejected_portions = [0.0,]
    if FARs is None:
        FARs = [0.0,]

    all_results = OrderedDict()
    n_figures = len(fusions_distances_uncertainties)
    distance_fig, distance_axes = None, [None] * n_figures
    uncertainty_fig, uncertainty_axes = None, [None] * n_figures
    if save_fig_path is not None:
        distance_fig, distance_axes = plt.subplots(
            nrows=1, ncols=n_figures,
            figsize=(9 * n_figures, 8))
        uncertainty_fig, uncertainty_axes = plt.subplots(
            nrows=1, ncols=n_figures,
            figsize=(9 * n_figures, 8))

    # Setup the data
    if protocol != "ijbc":
        raise ValueError('Unkown protocol. Only accept "ijbc" at the moment.')

    testset = IJBDataset(dataset_path)
    image_paths = testset["abspath"].values
    tester = IJBCTest(image_paths)
    tester.init_proto(protocol_path)


    # print(f"{tester.image_paths.dtype=} {tester.image_paths.shape=} {tester.image_paths[0]=}")
    # mu_1, mu_2, uncertainty_1, uncertainty_2, label_vec = get_features_uncertainties_labels(
    #     backbone, head, dataset_path, tester.image_paths,
    #     uncertainty_strategy=uncertainty_strategy, batch_size=batch_size, verbose=verbose,
    #     discriminator=discriminator, scale_predictor=scale_predictor,
    # )

    # returns features and uncertainties for a list of images
    features, uncertainties = extract_features_uncertainties_from_list(
        backbone,
        head,
        image_paths=tester.image_paths,
        uncertainty_strategy=uncertainty_strategy,
        batch_size=batch_size,
        discriminator=discriminator,
        scale_predictor=scale_predictor,
        uncertainty_model=uncertainty_model,
        device=device,
        verbose=verbose,
    )

    prev_fusion_name = None
    for (fusion_name, distance_name, uncertainty_name), distance_ax, uncertainty_ax in \
            zip(fusions_distances_uncertainties, distance_axes, uncertainty_axes):
        print(f"==={fusion_name} {distance_name} {uncertainty_name} ===")

        distance_func, uncertainty_func = get_distance_uncertainty_funcs(
            distance_name=distance_name,
            uncertainty_name=uncertainty_name,
            classifier=classifier,
            device=device,
            distaces_batch_size=distaces_batch_size,
        )

        if fusion_name != prev_fusion_name:
            aggregate_templates(
                tester.verification_templates, features, uncertainties, fusion_name)

        feat_1, feat_2, unc_1, unc_2, label_vec = \
            tester.get_features_uncertainties_labels(verbose=verbose)

        result_table = get_rejected_tar_far(
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=uncertainty_mode,
            FARs=FARs,
            distance_ax=distance_ax,
            uncertainty_ax=uncertainty_ax,
            rejected_portions=rejected_portions
        )

        if save_fig_path is not None:
            distance_ax.set_title(f"{distance_name} {uncertainty_name}")
            uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        all_results[(fusion_name, distance_name, uncertainty_name)] = result_table
        prev_fusion_name = fusion_name

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    for (fusion_name, distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(fusion_name, distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    if save_fig_path:
        for (fusion_name, distance_name, uncertainty_name), result_table in all_results.items():
            title = "Template" + distance_name + " " + uncertainty_name
            save_to_path = (
                os.path.join(save_fig_path, fusion_name + '_' + distance_name + "_" + uncertainty_name + ".jpg")
            )
            if save_fig_path:
                plots.plot_rejected_TAR_FAR(result_table, rejected_portions, title, save_to_path)

        plots.plot_TAR_FAR_different_methods(
            all_results,
            rejected_portions,
            res_AUCs,
            title="Template reject verification",
            save_figs_path=os.path.join(save_fig_path, "all_methods.jpg")
        )

        distance_fig.savefig(os.path.join(save_fig_path, "distance_dist.jpg"), dpi=400)
        uncertainty_fig.savefig(os.path.join(save_fig_path, "uncertainry_dist.jpg"), dpi=400)

        torch.save(all_results, os.path.join(save_fig_path, "table.pt"))


def main():
    args = parse_args_template_reject_verification()
    print(args)

    if os.path.isdir(args.save_fig_path) and not args.save_fig_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_fig_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))
    model_args = cfg.load_config(args.config_path)
    print(model_args)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    backbone, head, discriminator, classifier, scale_predictor, uncertainty_model = \
        get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device=device)

    rejected_portions = list(
        map(lambda x: float(x.replace(",", ".")), args.rejected_portions)
    )
    FARs = list(map(float, args.FARs))
    fusions_distances_uncertainties = list(
        map(lambda x: x.split("_"), args.fusion_distance_uncertainty_metrics)
    )

    eval_template_reject_verification(
        backbone,
        dataset_path=args.dataset_path,
        protocol=args.protocol,
        protocol_path=args.protocol_path,
        uncertainty_strategy=args.uncertainty_strategy,
        uncertainty_mode=args.uncertainty_mode,
        batch_size=args.batch_size,
        distaces_batch_size=args.distaces_batch_size,
        rejected_portions=rejected_portions,
        FARs=FARs,
        fusions_distances_uncertainties=fusions_distances_uncertainties,
        head=head,
        discriminator=discriminator,
        classifier=classifier,
        scale_predictor=scale_predictor,
        save_fig_path=args.save_fig_path,
        device=device,
        verbose=args.verbose,
        uncertainty_model=uncertainty_model
    )


if __name__ == "__main__":
    main()
