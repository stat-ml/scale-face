import argparse

uncertainty_methods = ["head", "GAN", "classifier", "scale", "emb_norm", "magface", "backbone+uncertainty_model"]
uncertainty_modes = ["uncertainty", "confidence"]
known_datasets = ["ijba", "ijbc"]


def parse_args_reject_verification():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="The path to the pre-trained model directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--discriminator_path",
        help="If you use GAN score to sort pairs, pah to weights of discriminator are determined here",
        type=str,
        default=None,
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
        "--batch_size",
        help="Number of images per mini batch",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--distaces_batch_size",
        help="Number of embeddings in batch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="The paths to config .yaml file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--uncertainty_strategy",
        help="Strategy to get uncertainty (ex. head/GAN/classifier, emb_norm)",
        type=str,
        default="head",
        choices=uncertainty_methods,
    )
    parser.add_argument(
        "--uncertainty_mode",
        help="Defines whether pairs with biggest or smallest uncertainty will be rejected",
        type=str,
        default="uncertainty",
        choices=uncertainty_modes,
    )
    parser.add_argument(
        "--rejected_portions",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--distance_uncertainty_metrics",
        help="Pairs of distance and uncertainty metrics to evaluate with, separated with '_' (ex. cosine_harmonic)",
        nargs="+",
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
        "--save_fig_path",
        help="Path to save figure to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="Dump verbose information",
        action="store_true",
    )

    return parser.parse_args()


def parse_args_fusion():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="The path to the pre-trained model directory",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        help="The path to the IJB-A dataset directory",
        type=str,
        default="data/ijba_mtcnncaffe_aligned",
    )
    parser.add_argument(
        "--protocol_path",
        help="The path to the IJB-A protocol directory",
        type=str,
        default="proto/IJB-A",
    )
    parser.add_argument(
        "--protocol",
        help="The dataset to test",
        type=str,
        default="ijbc",
        choices=known_datasets
    )
    parser.add_argument(
        "--config_path",
        help="The paths to config .yaml file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--uncertainty_strategy",
        help="Strategy to get uncertainty (ex. head or TTA)",
        type=str,
        default="head",
    )
    parser.add_argument(
        "--fusion_distance_methods",
        help="Pairs of distance metric and fusion distance to evaluate with, separated with '_' (ex. mean_cosine)",
        nargs="+",
    )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--device_id",
        help="Device on which the algorithm will be ran",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of images per mini batch",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--save_table_path",
        help="Path where the resulted table will be dumped",
        type=str,
        default="/gpfs/gpfs0/r.kail/tables/result.pkl",
    )
    return parser.parse_args()


def parse_args_template_reject_verification():
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
        "--protocol",
        help="The dataset to test",
        type=str,
        default="ijbc",
        choices=known_datasets,
    )
    parser.add_argument(
        "--protocol_path",
        help="Path to csv file with pairs names",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_path",
        help="The paths to config .yaml file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of images per mini batch",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--uncertainty_strategy",
        help="Strategy to get uncertainty (ex. head/GAN/classifier, emb_norm)",
        type=str,
        default="head",
        choices=uncertainty_methods,
    )
    parser.add_argument(
        "--uncertainty_mode",
        help="Defines whether pairs with biggest or smallest uncertainty will be rejected",
        type=str,
        default="uncertainty",
        choices=uncertainty_modes,
    )
    parser.add_argument(
        "--FARs",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--rejected_portions",
        help="Portion of rejected pairs of images",
        nargs="+",
    )
    parser.add_argument(
        "--distaces_batch_size",
        help="Number of embeddings in batch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fusion_distance_uncertainty_metrics",
        help="Pairs of distance and uncertainty metrics to evaluate with, separated with '_' (ex. cosine_harmonic)",
        nargs="+",
    )
    parser.add_argument(
        "--device_id",
        help="Gpu id on which the algorithm will be launched",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_fig_path",
        help="Path to save figure to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="Dump verbose information",
        action="store_true",
    )
    parser.add_argument(
        "--discriminator_path",
        help="If you use GAN score to sort pairs, pah to weights of discriminator are determined here",
        type=str,
        default=None,
    )

    return parser.parse_args()
