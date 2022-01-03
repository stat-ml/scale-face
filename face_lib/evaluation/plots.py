import matplotlib.pyplot as plt


def plot_uncertainty_distribution(
    uncertainties, labels,
    fig_name="", n_bins=40, ax=None,
):
    if ax is None:
        return None

    ax.hist(uncertainties[labels], bins=n_bins, color="r", label="positive", alpha=0.3)
    ax.hist(uncertainties[~labels], bins=n_bins, color="b", label="negative", alpha=0.3)

    ax.set_xlabel("Uncertainty")
    ax.set_title(fig_name)
    ax.legend()


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
        fig.savefig(save_fig_path, dpi=400)
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
        fig.savefig(save_figs_path, dpi=400)
    return fig