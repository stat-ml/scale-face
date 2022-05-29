import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from face_lib import models as mlib, utils
from face_lib.evaluation import name_to_distance_func, name_to_uncertainty_func
from face_lib.evaluation.distance_uncertainty_funcs import pair_cosine_score
from face_lib.evaluation.wrappers import (
    classifier_to_distance_wrapper,
    classifier_to_uncertainty_wrapper,
    split_wrapper,
)


def get_required_models(
    checkpoint,
    args,
    model_args,
    device=torch.device("cpu")

):
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
    if "discriminator_path" in args and args.discriminator_path:
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

    if args.uncertainty_strategy in ["scale", "scale_finetuned", "blurred_scale"]:

        scale_predictor_name = model_args.scale_predictor.pop("name")
        scale_predictor = mlib.scale_predictors[scale_predictor_name](
            **model_args.scale_predictor,
        )
        scale_predictor.load_state_dict(checkpoint["scale_predictor"])
        scale_predictor = scale_predictor.eval().to(device)

    uncertainty_model = None
    if args.uncertainty_strategy == "backbone+uncertainty_model":
        uncertainty_model_name = model_args.uncertainty_model.pop("name")
        uncertainty_model = mlib.model_dict[uncertainty_model_name](
            **model_args.uncertainty_model,
        )
        uncertainty_model.load_state_dict(checkpoint["uncertainty_model"])
        uncertainty_model = uncertainty_model.eval().to(device)

    return backbone, head, discriminator, classifier, scale_predictor, uncertainty_model


def get_distance_uncertainty_funcs(
        distance_name, uncertainty_name,
        classifier=None, device=torch.device("cpu"),
        distaces_batch_size=None, val_statistics=None):

    assert uncertainty_name != "classifier" or classifier is not None

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

    if distance_name in [
        "biased-cosine", "scale-mul-biased-cosine", "scale-harmonic-biased-cosine",
        "scale-sqrt-mul-biased-cosine", "scale-sqrt-harmonic-biased-cosine",
        "pfe-mul-biased-cosine", "pfe-harmonic-biased-cosine",
        "pfe-sqrt-mul-biased-cosine", "pfe-sqrt-harmonic-biased-cosine"]:

        assert val_statistics is not None

        print(f"Updating [{distance_name}]")
        # print(f"{distance_func(np.ones((3, 4)), np.ones((3, 4)), np.ones((3, 1)), np.ones((3, 1)))=}")

        new_distance_func = lambda x1, x2, unc1, unc2: \
            distance_func(x1, x2, unc1, unc2, bias=val_statistics["mean_cos"])  # TODO: Fix it
        new_uncertainty_func = uncertainty_func
    else:
        new_distance_func = distance_func
        new_uncertainty_func = uncertainty_func

    if distaces_batch_size:
        new_distance_func = split_wrapper(new_distance_func, batch_size=distaces_batch_size)
        new_uncertainty_func = split_wrapper(new_uncertainty_func, batch_size=distaces_batch_size)

    return new_distance_func, new_uncertainty_func


def get_precalculated_embeddings(precalculated_path, verbose=False):
    img_to_idx = {}
    emb_matrix = []

    with open(precalculated_path, "r") as f:
        lines_iterator = enumerate(f)
        if verbose:
            lines_iterator = tqdm(lines_iterator)

        for idx, line in lines_iterator:
            split = line.split(" ")
            path = "/".join(split[0].split("/")[-2:])

            img_to_idx[path] = idx
            emb_matrix.append(np.array(list(map(float, split[1:-1]))))

    return np.stack(emb_matrix, axis=0), img_to_idx


def find_best_f1_threshold(similarities, labels):
    precision, recall, thresholds = precision_recall_curve(labels, similarities)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    return max_f1_thresh


def find_gaussian_optimal_threshold(similarities, labels):
    positives = similarities[labels]
    negatives = similarities[~labels]

    pos_mean, pos_var = positives.mean(), positives.var()
    neg_mean, neg_var = negatives.mean(), negatives.var()

    p_0 = 1 / (2 * pos_var) - 1 / (2 * neg_var)
    p_1 = - (pos_mean / pos_var - neg_mean / neg_var)
    p_2 = (pos_mean ** 2) / (2 * pos_var) - (neg_mean ** 2) / (2 * neg_var) + 0.5 * np.log(neg_var / pos_var)

    roots = np.roots(np.array((p_0, p_1, p_2), dtype=float))
    assert len(roots) == 2
    print(roots)

    if roots[0] > 0. and roots[0] < 1.:
        return roots[0]
    elif roots[1] > 0 and roots[1] < 1:
        return roots[1]
    else:
        raise AssertionError("Had not found acceptable root")


def get_mean_classes_centres(similarities, label_vec):
    return 0.5 * (similarities[label_vec].mean(axis=0) + similarities[~label_vec].mean(axis=0))


def extract_statistics(data):
    x1, x2, unc1, unc2, label_vec = data
    cosines = pair_cosine_score(x1, x2, unc1=None, unc2=None)

    mean_cosine = get_mean_classes_centres(cosines, label_vec)
    best_f1_thres = find_best_f1_threshold(cosines, label_vec)
    best_gauss_thres = find_gaussian_optimal_threshold(cosines, label_vec)

    print(f"Mean thres : {mean_cosine} best_f1 : {best_f1_thres} optimal_gauss : {best_gauss_thres}")

    return {
        "mean_cos": mean_cosine,
        "best_f1_thres": best_f1_thres,
        "optimal_gauss": best_gauss_thres,
    }