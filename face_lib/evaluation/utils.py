import torch
import numpy as np
from tqdm import tqdm

from face_lib import models as mlib, utils
from face_lib.evaluation import name_to_distance_func, name_to_uncertainty_func
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
    if args.discriminator_path:
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
    if args.uncertainty_strategy == "scale" or args.uncertainty_strategy == "blurred_scale":
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
        classifier=None, device=torch.device("cpu"), distaces_batch_size=None):

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

    if distaces_batch_size:
        distance_func = split_wrapper(distance_func, batch_size=distaces_batch_size)
        uncertainty_func = split_wrapper(uncertainty_func, batch_size=distaces_batch_size)

    return distance_func, uncertainty_func


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
