import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

from face_lib.utils.imageprocessing import (
    preprocess,
    preprocess_tta,
    preprocess_gan,
    preprocess_magface,
    preprocess_blurred)
from face_lib.evaluation.utils import get_precalculated_embeddings


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


def extract_features_backbone(
    backbone,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
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
        mu.append(np.array(output["feature"].detach().cpu()))

    mu = np.concatenate(mu, axis=0)

    if verbose:
        print("")
    return mu, None


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


def extract_features_ssim(
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

        batch = proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = backbone(batch)
        mu.append(output["feature"].detach().cpu().numpy())

        img_random = np.random.rand(*images_batch[0].shape)
        sigma_sq_temp = []
        for i in range(images_batch.shape[0]):
            ssim_noise = ssim(
                images_batch[i],
                img_random,
                data_range=img_random.max() - img_random.min(),
                multichannel=True,
            )
            sigma_sq_temp.append(ssim_noise)
        sigma_sq_temp = np.concatenate(sigma_sq_temp, axis=0)

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq


def extract_features_grad(
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

        batch = proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = backbone(batch)
        mu.append(output["feature"].detach().cpu().numpy())

        img_random = np.random.rand(*images_batch[0].shape)
        sigma_sq_temp = []
        for i in range(images_batch.shape[0]):

            sx = ndimage.sobel(images_batch[0], axis=0, mode="constant")
            sy = ndimage.sobel(images_batch[0], axis=1, mode="constant")
            sobel = np.hypot(sx, sy)
            grad_noise = np.var(sobel)
            sigma_sq_temp.append(grad_noise)
        sigma_sq_temp = np.concatenate(sigma_sq_temp, axis=0)

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq


def extract_features_fourier(
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

        batch = proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = backbone(batch)
        mu.append(output["feature"].detach().cpu().numpy())

        img_random = np.random.rand(*images_batch[0].shape)
        sigma_sq_temp = []
        box_size = 20
        for i in range(images_batch.shape[0]):
            img = images_batch[i][0]
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            value = np.min(
                magnitude_spectrum.mean()
                / (
                    magnitude_spectrum[
                        magnitude_spectrum.shape[0] // 2
                        - box_size : magnitude_spectrum.shape[0] // 2
                        + box_size,
                        magnitude_spectrum.shape[1] // 2
                        - box_size : magnitude_spectrum.shape[1] // 2
                        + box_size,
                    ].mean()
                ),
                1.0,
            )

            sigma_sq_temp.append(value)
        sigma_sq_temp = np.concatenate(sigma_sq_temp, axis=0)

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq


def extract_features_gan(
    backbone,
    discriminator,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
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
        mu.append(np.array(output["feature"].detach().cpu()))

    mu = np.concatenate(mu, axis=0)

    if verbose:
        print("")

    uncertainties = []
    uncertainty_proc_func = lambda images: preprocess_gan(images, resize_size=[256, 256], is_training=False)
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

        batch = uncertainty_proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = discriminator(batch)
        uncertainties.append(np.array(output.detach().cpu()))

    uncertainties = np.concatenate(uncertainties, axis=0)
    if verbose:
        print("")

    print("Mu :", mu.shape, "Uncertainty :", uncertainties.shape)

    return mu, uncertainties


def extract_features_scale(
    backbone,
    scale_predictor,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
    uncertainty = []
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
        output.update(scale_predictor(**output))
        mu.append(np.array(output["feature"].detach().cpu()))
        uncertainty.append(np.array(output["scale"].detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    uncertainty = np.concatenate(uncertainty, axis=0)

    if verbose:
        print("")
    return mu, uncertainty


def extract_features_emb_norm(
    backbone,
    images,
    batch_size,
    proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):
    num_images = len(images)
    mu = []
    uncertainty = []
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

        mu.append(np.array(output["feature"].detach().cpu()))
        cur_uncertainty = torch.linalg.norm(output["feature"], dim=1, keepdims=True)
        uncertainty.append(np.array(cur_uncertainty.detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    uncertainty = np.concatenate(uncertainty, axis=0)

    if verbose:
        print("")
    return mu, uncertainty


def extract_features_backbone_uncertainty(
    backbone,
    uncertainty_model,
    images,
    batch_size,
    backbone_proc_func=None,
    uncertainty_proc_func=None,
    verbose=False,
    device=torch.device("cpu"),
):

    num_images = len(images)
    mu = []
    uncertainty = []
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

        batch = backbone_proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = backbone(batch)
        mu.append(np.array(output["feature"].detach().cpu()))

        batch = uncertainty_proc_func(images_batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        output = uncertainty_model(batch)
        cur_uncertainty = torch.linalg.norm(output["feature"], dim=1, keepdims=True)
        uncertainty.append(np.array(cur_uncertainty.detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    uncertainty = np.concatenate(uncertainty, axis=0)

    if verbose:
        print("")
    return mu, uncertainty


def extract_features_uncertainties_from_list(
    backbone,
    head,
    image_paths,
    uncertainty_strategy="head",
    batch_size=64,
    discriminator=None,
    scale_predictor=None,
    uncertainty_model=None,
    blur_intensity=None,
    device=torch.device("cuda:0"),
    verbose=False,
):
    if uncertainty_strategy == "backbone+magface":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_backbone(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )

    elif uncertainty_strategy == "perfect":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_backbone(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )

    elif uncertainty_strategy == "head":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_head(
            backbone,
            head,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )

    elif uncertainty_strategy == "TTA":
        proc_func = lambda images: preprocess_tta(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_tta(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "fourier":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_fourier(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "grad":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_grad(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "ssim":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        features, uncertainties = extract_features_ssim(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=True,
            device=device,
        )
    elif uncertainty_strategy == "GAN":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        if discriminator is None:
            raise RuntimeError("Please determine a discriminator")
        features, uncertainties = extract_features_gan(
            backbone,
            discriminator,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "classifier":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        features, uncertainties = extract_features_head(
            backbone,
            head,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "scale":
        assert scale_predictor is not None
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        features, uncertainties = extract_features_scale(
            backbone,
            scale_predictor,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "blurred_scale":
        assert scale_predictor is not None
        assert blur_intensity is not None
        proc_func = lambda images: preprocess_blurred(
            images, [112, 112], is_training=False, blur_intensity=blur_intensity)

        features, uncertainties = extract_features_scale(
            backbone,
            scale_predictor,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "emb_norm":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        features, uncertainties = extract_features_emb_norm(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "magface":
        proc_func = lambda images: preprocess_magface(images, [112, 112], is_training=False)

        features, uncertainties = extract_features_emb_norm(
            backbone,
            image_paths,
            batch_size,
            proc_func=proc_func,
            verbose=verbose,
            device=device,
        )
    elif uncertainty_strategy == "backbone+uncertainty_model":
        backbone_proc_func = lambda images: preprocess(images, [112, 112], is_training=False)
        uncertainty_proc_func = lambda images: preprocess_magface(images, [112, 112], is_training=False)

        features, uncertainties = extract_features_backbone_uncertainty(
            backbone,
            uncertainty_model,
            image_paths,
            batch_size,
            backbone_proc_func=backbone_proc_func,
            uncertainty_proc_func=uncertainty_proc_func,
            verbose=verbose,
            device=device,
        )
    else:
        raise NotImplementedError("Don't know this type of uncertainty strategy")

    return features, uncertainties


def extract_uncertainties_from_dataset(
    backbone,
    scale_predictor,
    dataset,
    batch_size,
    verbose=False,
    device=torch.device("cpu"),
):

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )

    uncertainties = []

    with torch.no_grad():
        for batch, labels in tqdm(dataloader):
            batch = batch.to(device)
            output = backbone(batch)
            output.update(scale_predictor(**output))
            # mu.append(np.array(output["feature"].detach().cpu()))
            uncertainties.append(np.array(output["scale"].detach().cpu()))

    # mu = np.concatenate(mu, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)

    if verbose:
        print("")
    # return mu, uncertainties
    return uncertainties


def extract_pairs_info(pairs_table_path):
    pairs, labels = [], []
    unique_imgs = set()
    with open(pairs_table_path, "r") as f:
        for line in f.readlines():
            left_path, right_path, label = line.split(",")
            pairs.append((left_path, right_path))
            labels.append(int(label))
            unique_imgs.add(left_path)
            unique_imgs.add(right_path)
    return pairs, labels, unique_imgs


def get_features_uncertainties_labels(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    uncertainty_strategy="head",
    batch_size=64,
    discriminator=None,
    scale_predictor=None,
    uncertainty_model=None,
    precalculated_path=None,
    device=torch.device("cuda:0"),
    verbose=False,
):

    pairs, label_vec, unique_imgs = extract_pairs_info(pairs_table_path)

    if uncertainty_strategy == "magface_precalculated":
        features, img_to_idx = get_precalculated_embeddings(precalculated_path, verbose=verbose)
        # TODO: Fair calculation of uncertainty
        uncertainties = np.linalg.norm(features, axis=1, keepdims=True)

    elif uncertainty_strategy == "backbone+magface":
        features, img_to_idx = get_precalculated_embeddings(precalculated_path, verbose=verbose)
        uncertainties = np.linalg.norm(features, axis=1, keepdims=True)

        unc_1 = np.array([uncertainties[img_to_idx[pair[0]]] for pair in pairs])
        unc_2 = np.array([uncertainties[img_to_idx[pair[1]]] for pair in pairs])

        image_paths = list(unique_imgs)
        img_to_idx = {img_path: idx for idx, img_path in enumerate(image_paths)}
        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=verbose,
        )
        assert uncertainties is None
        mu_1 = np.array([features[img_to_idx[pair[0]]] for pair in pairs])
        mu_2 = np.array([features[img_to_idx[pair[1]]] for pair in pairs])
        label_vec = np.array(label_vec, dtype=bool)

        return mu_1, mu_2, unc_1, unc_2, label_vec

    elif uncertainty_strategy == "perfect":
        image_paths = list(unique_imgs)
        img_to_idx = {img_path: idx for idx, img_path in enumerate(image_paths)}

        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=verbose,
        )
        assert uncertainties is None
        mu_1 = np.array([features[img_to_idx[pair[0]]] for pair in pairs])
        mu_2 = np.array([features[img_to_idx[pair[1]]] for pair in pairs])

        label_vec = np.array(label_vec, dtype=bool)
        unc_1 = label_vec.astype(float)
        unc_2 = label_vec.astype(float)

        return mu_1, mu_2, unc_1, unc_2, label_vec

    else:
        image_paths = list(unique_imgs)
        img_to_idx = {img_path: idx for idx, img_path in enumerate(image_paths)}

        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=list(map(lambda x: os.path.join(dataset_path, x), image_paths)),
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=verbose,
        )

    mu_1 = np.array([features[img_to_idx[pair[0]]] for pair in pairs])
    mu_2 = np.array([features[img_to_idx[pair[1]]] for pair in pairs])
    unc_1 = np.array([uncertainties[img_to_idx[pair[0]]] for pair in pairs])
    unc_2 = np.array([uncertainties[img_to_idx[pair[1]]] for pair in pairs])
    label_vec = np.array(label_vec, dtype=bool)

    return mu_1, mu_2, unc_1, unc_2, label_vec
