import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.dataset_lfw import Dataset as LFW_Dataset
from utils.imageprocessing import preprocess


__all__ = ["visualize_ambiguity_dilemma_lfw", "visualize_low_high_similarity_pairs"]

# TODO: remove from here and move to the utils submodule
# after that we can expose this
def _gaussian_blur(image: np.array, k: int):
    kernel = np.ones((k, k), np.float32) / (k ** 2)
    dst = cv2.filter2D(image, -1, kernel)
    return dst


def visualizer_decorator(function):
    def wrap_function(*args, **kwargs):
        fig = function(*args, **kwargs)
        if "save_fig" in kwargs and kwargs["save_fig"] is True:
            assert "save_path" in kwargs, "You need to provide the image path"
            fig.savefig(kwargs["save_path"])
        if "board" in kwargs and kwargs["board"] is True:
            # return np.ndarray if tensorboard flag is given
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data
        return fig

    return wrap_function


@visualizer_decorator
def visualize_ambiguity_dilemma_lfw(
    pfe_backbone: torch.nn.Module,
    pfe_head: torch.nn.Module,
    criterion: torch.nn.Module,
    lfw_path: str,
    *,
    number_of_iters: int = 10,
    in_size: tuple = (112, 96),
    device=None,
    **kwargs,  # TODO: can we get rid of this dict?
):

    if device is None:
        device = "cpu"

    paths_full = LFW_Dataset(lfw_path)["abspath"]
    linspace_size = 20
    kernel_values = np.linspace(1, min(in_size) - 5, linspace_size).astype(np.int)

    determinist_distances = {
        "original_vs_distorted": np.zeros((number_of_iters, 20)),
        "impostor": np.zeros((number_of_iters, 20)),
    }

    pfe_distances = {
        "original_vs_distorted": np.zeros((number_of_iters, 20)),
        "impostor": np.zeros((number_of_iters, 20)),
    }

    for i in range(number_of_iters):
        inds = np.random.choice(len(paths_full), 2)
        paths = [paths_full[inds[0]], paths_full[inds[1]]]
        images = (
            torch.from_numpy(preprocess(paths, in_size, False))
            .permute(0, 3, 1, 2)
            .to(torch.float32)
        )
        image1, image2 = images

        for idx in range(linspace_size):
            image1_d, image2_d = (
                _gaussian_blur(image1.numpy(), kernel_values[idx]),
                _gaussian_blur(image2.numpy(), kernel_values[idx]),
            )

            image_batch1 = torch.cat(
                (image1[None], torch.from_numpy(image1_d[None])), dim=0
            )
            image_batch2 = torch.cat(
                (torch.from_numpy(image1_d[None]), torch.from_numpy(image2_d[None])),
                dim=0,
            )

            feature1, sig_feat1 = pfe_backbone(image_batch1.to(device))
            feature2, sig_feat2 = pfe_backbone(image_batch2.to(device))

            log_sig_sq1 = pfe_head(sig_feat1)
            log_sig_sq2 = pfe_head(sig_feat2)

            dist1 = torch.norm(feature1[0] - feature1[1]).item()
            dist2 = torch.norm(feature2[0] - feature2[1]).item()

            determinist_distances["original_vs_distorted"][i][idx] = dist1
            determinist_distances["impostor"][i][idx] = dist2

            dist1 = criterion(feature2, log_sig_sq1, torch.ones(2).to(device)).item()
            dist2 = criterion(feature2, log_sig_sq2, torch.ones(2).to(device)).item()

            pfe_distances["original_vs_distorted"][i][idx] = dist1
            pfe_distances["impostor"][i][idx] = dist2

    # TODO: less verbose here
    all_size = linspace_size * number_of_iters
    data1 = {
        "od": ["Original vs distorted"] * all_size + ["Impostor"] * all_size,
        "kernel_size": list(
            np.array(
                [
                    np.linspace(1, min(in_size) - 5, linspace_size).astype(np.int32)
                    for i in range(number_of_iters)
                ]
            ).flatten()
        )
        + list(
            np.array(
                [
                    np.linspace(1, min(in_size) - 5, linspace_size).astype(np.int32)
                    for i in range(number_of_iters)
                ]
            ).flatten()
        ),
        "values": list(determinist_distances["original_vs_distorted"].flatten())
        + list(determinist_distances["impostor"].flatten()),
    }
    data2 = {
        "od": ["Original vs distorted"] * all_size + ["Impostor"] * all_size,
        "kernel_size": list(
            np.array(
                [
                    np.linspace(1, min(in_size) - 5, linspace_size).astype(np.int32)
                    for i in range(number_of_iters)
                ]
            ).flatten()
        )
        + list(
            np.array(
                [
                    np.linspace(1, min(in_size) - 5, linspace_size).astype(np.int32)
                    for i in range(number_of_iters)
                ]
            ).flatten()
        ),
        "values": list(pfe_distances["original_vs_distorted"].flatten())
        + list(pfe_distances["impostor"].flatten()),
    }

    pd_data_deterministic = pd.DataFrame.from_dict(data1)
    pd_data_pfe = pd.DataFrame.from_dict(data2)

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x="kernel_size", y="values", hue="od", data=pd_data_deterministic, ax=ax[0]
    )
    sns.lineplot(x="kernel_size", y="values", hue="od", data=pd_data_pfe, ax=ax[1])

    ax[0].set_title("Deterministic features")
    ax[1].set_title("MLS features")

    return fig


@visualizer_decorator
def visualize_low_high_similarity_pairs(
    pfe_backbone: torch.nn.Module,
    pfe_head: torch.nn.Module,
    criterion: torch.nn.Module,
    lfw_path: str,
    *,
    number_of_iters: int = 10,
    in_size: tuple = (112, 96),
    device=None,
    **kwargs,  # TODO: can we get rid of this dict?
):
    raise NotImplementedError
