import os
import sys
import argparse
import numpy as np
import torch
from path import Path
from tqdm import tqdm
import time
import torch.nn.functional as F

path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

from face_lib import models as mlib, utils
from face_lib.utils import cfg
from face_lib.utils.imageprocessing import preprocess

from scipy.spatial import distance


def classifier_to_distance(classifier, mu_1, mu_2, device=torch.device("cpu")):
    inputs = torch.cat((torch.from_numpy(mu_1), torch.from_numpy(mu_2)), dim=1)
    probes = F.softmax(classifier(feature=inputs.to(device), dim=1)["pair_classifiers_output"], dim=-1)
    probes = probes.cpu().detach().numpy()
    return probes[:, 1]


def cosine_distance(mu_1, mu_2):
    cos_dist = []
    for m1, m2 in zip(mu_1, mu_2):
        cos_dist.append(pair_cosine_score(m1, m2))

    return cos_dist


def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis, keepdims=True))
    return x


def pair_cosine_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    x1, x2 = np.array(x1), np.array(x2)
    x1, x2 = l2_normalize(x1, axis=0), l2_normalize(x2, axis=0)
    dist = np.sum(x1 * x2, axis=0)
    return dist


def pair_cosine_score_big(x1, x2, sigma_sq1=None, sigma_sq2=None):
    x1, x2 = np.array(x1), np.array(x2)
    x1, x2 = l2_normalize(x1, axis=1), l2_normalize(x2, axis=1)
    dist = np.sum(x1 * x2, axis=1)
    return dist


def extract_distance_classifier(
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


        # mu = np.array(output["feature"].detach().cpu())
        # print("LEN_MU:", len(mu))
        # print(len(mu[0]))

        mu.append(np.array(output["feature"].detach().cpu()))
        sigma_sq.append(np.array(output["log_sigma"].exp().detach().cpu()))

    mu = np.concatenate(mu, axis=0)
    sigma_sq = np.concatenate(sigma_sq, axis=0)

    if verbose:
        print("")
    return mu, sigma_sq


def eval_pairs_distances(
        backbone,
        head,
        dataset_path,
        pairs_table_path,
        pairs_distance_strategy="classifier",
        batch_size=64,
        classifier=None,
        save_results_path=None,
):
    mu_1, mu_2, sigma_sq_1, sigma_sq_2, label_vec = get_pairs_distances(
        backbone, head, dataset_path, pairs_table_path,
        pairs_distance_strategy=pairs_distance_strategy, batch_size=batch_size,
        classifier=classifier
    )

    #score_vec = classifier_to_distance(classifier, mu_1, mu_2, device=device)
    score_vec = cosine_distance(mu_1, mu_2)

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :", label_vec.shape, label_vec.dtype)

    true_text_file = open(os.path.join(args.save_results_path, "true_dist.txt"), "w")
    false_text_file = open(os.path.join(args.save_results_path, "false_dist.txt"), "w")

    for l, d in zip(label_vec, score_vec):
        print(f"Label: {l} Distance: {d}")
        if l == True:
            true_text_file.write(str(d))
            true_text_file.write("\n")
        else:
            false_text_file.write(str(d))
            false_text_file.write("\n")

    true_text_file.close()
    false_text_file.close()


def get_pairs_distances(
        backbone,
        head,
        dataset_path,
        pairs_table_path,
        pairs_distance_strategy="classifier",
        batch_size=64,
        classifier=None,
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

    if pairs_distance_strategy == "classifier":
        proc_func = lambda images: preprocess(images, [112, 112], is_training=False)

        mu, sigma_sq = extract_distance_classifier(
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
        "--pairs_distance_strategy",
        help="Strategy to get distance between pairs",
        type=str,
        default="cosine",
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
        "--save_results_path",
        help="Path to save results",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if os.path.isdir(args.save_results_path) and not args.save_results_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_results_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))
    model_args = cfg.load_config(args.config_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    backbone = mlib.model_dict[model_args.backbone["name"]](
        **utils.pop_element(model_args.backbone, "name")
    )
    backbone.load_state_dict(checkpoint["backbone"])
    backbone = backbone.to(device).eval()

    head = None
    if args.pairs_distance_strategy == "head" or (args.pairs_distance_strategy == "classifier" and "head" in model_args):
        head = mlib.heads[model_args.head.name](
            **utils.pop_element(model_args.head, "name")
        )
        head.load_state_dict(checkpoint["head"])
        head = head.to(device).eval()


    classifier = None

    print("+"*10, args.pairs_distance_strategy, "+"*10)

    if args.pairs_distance_strategy == "classifier":
        classifier_name = model_args.pair_classifier.pop("name")
        classifier = mlib.pair_classifiers[classifier_name](
            **model_args.pair_classifier,
        )
        classifier.load_state_dict(checkpoint["pair_classifier"])
        classifier = classifier.eval().to(device)


    eval_pairs_distances(
        backbone,
        head,
        args.dataset_path,
        args.pairs_table_path,
        pairs_distance_strategy=args.pairs_distance_strategy,
        batch_size=args.batch_size,
        classifier=classifier,
        save_results_path=args.save_results_path,
    )