import sys
import time
import torch
import numpy as np
from tqdm import tqdm

def extract_features_head(
    backbone,
    head,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
    sigma_sq = []
    start_time = time.time()
    for start_idx in tqdm(range(0, num_images, batch_size)):
        if verbose:
            elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - start_time)
            )
            sys.stdout.write(
                "# of images: %d Current image: %d Elapsed time: %s \t\r"
                % (num_images, start_idx, elapsed_time)
            )
        end_idx = min(num_images, start_idx + batch_size)
        images_batch = images[start_idx:end_idx]

        batch = proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = backbone(batch)
        output.update(head(**output))
        mu.append(np.array(output["feature"].detach().cpu()))
        sigma_sq.append(np.array(output["log_sigma"].exp().detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq


def extract_features_tta(
    backbone,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
    sigma_sq = []
    start_time = time.time()
    for start_idx in tqdm(range(0, num_images, batch_size)):
        if verbose:
            elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - start_time)
            )
            sys.stdout.write(
                "# of images: %d Current image: %d Elapsed time: %s \t\r"
                % (num_images, start_idx, elapsed_time)
            )
        end_idx = min(num_images, start_idx + batch_size)
        images_batch = images[start_idx:end_idx]

        # batch = proc_func(images_batch)  # imagesprocessing.py -> preprocess

        batch_tta = proc_func(images_batch)
        embeds_augments = []

        for ind, batch in enumerate(batch_tta):
            if ind == 0:
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                output = backbone(batch)
                mu.append(np.array(output["feature"].detach().cpu()))
            else:
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                output = backbone(batch)
                embeds_augments.append(np.array(output["feature"].detach().cpu()))

        sigma_sq.append(np.mean(np.var(embeds_augments, axis=0), axis=1).reshape(-1, 1))
        print("len mu: ", len(mu))
        print("len siqma: ", len(sigma_sq))
        print("shape mu", mu[0].shape)
        print("shape sigma", sigma_sq[0].shape)

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq

