import numpy as np
import matplotlib.pyplot as plt

datasaets_paths = {
    "IJBC": "/beegfs/home/r.kail/faces/figures/22_dataset_distribution/IJBC/sigm_64/uncertainties.npy",
    "LFW": "/beegfs/home/r.kail/faces/figures/22_dataset_distribution/LFW/sigm_64/uncertainties.npy",
    "MS1MV2": "/beegfs/home/r.kail/faces/figures/22_dataset_distribution/MS1MV2/sigm_64/uncertainties.npy",
}

datasets_uncertainties = {name: np.load(path) for name, path in datasaets_paths.items()}


def plot_distributions(
    datasets_uncertainties, save_fig_path, n_bins=50,
    fig_name="", xlabel_name="", ylabel_name="",
):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["b", "g", "y", "r"]

    for (name, values), color in zip(datasets_uncertainties.items(), colors):
        ax.hist(values, bins=n_bins, density=True, color=color, alpha=0.3, label=name)

    ax.set_title(fig_name)
    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)
    ax.legend()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=400)

plot_distributions(
    datasets_uncertainties=datasets_uncertainties,
    save_fig_path="/beegfs/home/r.kail/faces/figures/22_dataset_distribution/merged/merged.pdf",
    n_bins=50,
    fig_name="Datasets' uncertainty distributions",
    xlabel_name="Uncertainty",
    ylabel_name="probability density",
)