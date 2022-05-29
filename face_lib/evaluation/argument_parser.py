import argparse
from face_lib.utils import cfg


uncertainty_methods = [
    "head", "GAN", "classifier", "scale", "blurred_scale", "emb_norm",
    "magface", "backbone+uncertainty_model", "magface_precalculated",
    "backbone+magface"]

uncertainty_modes = ["uncertainty", "confidence"]
known_datasets = ["ijba", "ijbc"]
distribution_datasets = ["IJBC", "LFW", "MS1MV2"]


# Fusion
required_fusion_parameters = [
    "checkpoint_path",
    "dataset_path",
    "protocol_path",
    "fusion_distance_methods",
    "FARs",
    "save_table_path",
]

default_fusion_parameters = {
    "config_path": None,
    "uncertainty_strategy": "head",
    "protocol": "ijbc",
    "device_id": 0,
    "batch_size": 64,
    "verbose": False,
}

choice_fusion_parameters = {
    "uncertainty_strategy": uncertainty_methods
}


# Reject_verification
required_reject_verification_parameters = [
    "checkpoint_path",
    "dataset_path",
    "pairs_table_path",
    "config_path",
    "uncertainty_strategy",
    "distance_uncertainty_metrics",
]

default_reject_verification_parameters = {
    "batch_size": 16,
    "distances_batch_size": None,
    "uncertainty_mode": "uncertainty",
    "rejected_portions": [0.0, 0.5, 250],
    "FARs": [0.0001, 0.001, 0.01],
    "precalculated_path": None,
    "figure_path": None,
    "save_fig_path": None,
    "device_id": 0,
    "verbose": False,
    "val_pairs_table_path": None,
    "discriminator_path": None,
}

choice_reject_verification_parameters = {
    "uncertainty_strategy": uncertainty_methods,
    "uncertainty_mode": uncertainty_modes,
}


# Template reject verification
required_template_reject_verification_parameters = [
    "checkpoint_path",
    "dataset_path",
    "protocol_path",
    "config_path",
    "fusion_distance_uncertainty_metrics",
]

default_template_reject_verification_parameters = {
    "protocol": "ijbc",
    "batch_size": 64,
    "uncertainty_strategy": "head",
    "uncertainty_mode": "uncertainty",
    "rejected_portions": [0.0, 0.5, 250],
    "FARs": [0.0001, 0.001, 0.01],
    "precalculated_path": None,
    "distaces_batch_size": None,
    "device_id": 0,
    "cached_embeddings": False,
    "equal_uncertainty_enroll": False,
    "verbose": False,
    "discriminator_path": None,
    "save_fig_path": None,
}

choice_template_reject_verification_parameters = {
    "protocol": known_datasets,
    "uncertainty_strategy": uncertainty_methods,
    "uncertainty_mode": uncertainty_modes,
}


# Dataset distribution
required_dataset_distribution_parameters = [
    "checkpoint_path",
    "dataset_path",
    "config_path",
]

default_dataset_distribution_parameters = {
    "image_paths_table": None,
    "dataset_name": "ijbc",
    "batch_size": 64,
    "uncertainty_strategy": "head",
    "discriminator_path": None,
    "blur_intensity": None,
    "device_id": 0,
    "save_fig_path": None,
    "verbose": False,
}

choice_dataset_distribution_parameters = {
    "dataset_name": distribution_datasets,
    "uncertainty_strategy": uncertainty_methods
}


def parse_cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to .yaml file with the configuration for the test",
    )
    args = parser.parse_args()
    return cfg.load_config(args.config_path)


def verify_arguments(args, required_params, default_params, choice_parameters=None):
    for param_name in required_params:
        if not param_name in args:
            raise ValueError(f"Parameter {param_name} is required")

    for param_name, default_value in default_params.items():
        if not param_name in args:
            args[param_name] = default_value

    if choice_parameters is not None:
        for param_name, choices in choice_parameters.items():
            if args[param_name] not in choices:
                raise ValueError(f"Parameter {param_name} should be chosen from {choices}, you've chosen {args[param_name]}")

    return args


def verify_arguments_fusion(args):
    return verify_arguments(
        args,
        required_params=required_fusion_parameters,
        default_params=default_fusion_parameters,
        choice_parameters=choice_fusion_parameters,)


def verify_arguments_reject_verification(args):
    return verify_arguments(
        args,
        required_params=required_reject_verification_parameters,
        default_params=default_reject_verification_parameters,
        choice_parameters=choice_reject_verification_parameters,)


def verify_arguments_template_reject_verification(args):
    return verify_arguments(
        args,
        required_params=required_template_reject_verification_parameters,
        default_params=default_template_reject_verification_parameters,
        choice_parameters=choice_template_reject_verification_parameters,)


def verify_arguments_dataset_distribution(args):
    return verify_arguments(
        args,
        required_params=required_dataset_distribution_parameters,
        default_params=default_dataset_distribution_parameters,
        choice_parameters=choice_dataset_distribution_parameters,)

