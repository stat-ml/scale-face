import torch

from face_lib import models as mlib, utils


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
    if args.uncertainty_strategy == "scale":
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

