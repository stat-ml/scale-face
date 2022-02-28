import os
import sys
import numpy as np
import torch
from pathlib import Path

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)

from face_lib.utils import cfg
import face_lib.evaluation.plots as plots
from face_lib.evaluation.utils import get_required_models
from face_lib.evaluation.feature_extractors import extract_features_uncertainties_from_list
from face_lib.evaluation.argument_parser import parse_args_dataset_distribution


def get_uncertainties_IJBC(
    backbone,
    dataset_path=None,
    image_paths_table=None,
    uncertainty_strategy="scale",
    batch_size=64,
    head=None,
    discriminator=None,
    scale_predictor=None,
    device=torch.device("cpu"),
    verbose=False):

    with open(image_paths_table, "r") as f:
        relative_paths = f.readlines()
    relative_paths = list(map(lambda x: x.strip(), relative_paths))

    features, uncertainties = extract_features_uncertainties_from_list(
        backbone,
        head,
        image_paths=list(map(lambda x: os.path.join(dataset_path, x), relative_paths)),
        uncertainty_strategy=uncertainty_strategy,
        batch_size=batch_size,
        discriminator=discriminator,
        scale_predictor=scale_predictor,
        uncertainty_model=uncertainty_model,
        device=device,
        verbose=verbose,
    )
    return uncertainties.squeeze(axis=1)


def get_uncertainties_LFW():
    pass


def get_uncertainties_MS1MV2():
    pass


def draw_figures(
    backbone,
    dataset_name="IJBC",
    dataset_path=None,
    image_paths_table=None,
    uncertainty_strategy="scale",
    batch_size=64,
    head=None,
    discriminator=None,
    scale_predictor=None,
    save_fig_path=None,
    device=torch.device("cpu"),
    verbose=False,
):
    if dataset_name == "IJBC":
        uncertainties = get_uncertainties_IJBC(
            backbone=backbone,
            dataset_path=dataset_path,
            image_paths_table=image_paths_table,
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            head=head,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            device=device,
            verbose=verbose,)
    else:
        raise KeyError("Don't know this type of dataset")

    print(type(uncertainties))
    print(f"{uncertainties.shape=} {uncertainties.dtype}")

    np.save(os.path.join(save_fig_path, "uncertainties.npy"), uncertainties)

    if save_fig_path:
        plots.plot_uncertainty_distribution(
            uncertainties, os.path.join(save_fig_path, "uncertainty_distribution.pdf"),
            n_bins=50, fig_name="Uncertainties distribution",
            xlabel_name="Uncertainty", ylabel_name="Probability",
        )


if __name__ == "__main__":
    args = parse_args_dataset_distribution()

    if os.path.isdir(args.save_fig_path) and not args.save_fig_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_fig_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))
    model_args = cfg.load_config(args.config_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    backbone, head, discriminator, classifier, scale_predictor, uncertainty_model = \
        get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device=device)

    draw_figures(
        backbone,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        image_paths_table=args.image_paths_table,
        uncertainty_strategy=args.uncertainty_strategy,
        batch_size=args.batch_size,
        head=head,
        discriminator=discriminator,
        scale_predictor=scale_predictor,
        save_fig_path=args.save_fig_path,
        device=device,
        verbose=args.verbose,
    )
