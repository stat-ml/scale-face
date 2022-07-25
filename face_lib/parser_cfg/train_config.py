import argparse
import exman


def training_args():

    # TODO: don't like the path manipulation here
    parser = exman.ExParser(
        root=exman.simpleroot(__file__).parent.parent.parent / "exman"
    )

    parser.add_argument(
        "--model-config", type=str, default=None, help="Model configuration"
    )

    parser.add_argument("--resume", type=str, default=None)  # checkpoint

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    args = parser.parse_args()

    return args


def parse_args_scale():

    parser = argparse.ArgumentParser(description="Parsing arguments for scale")

    parser.add_argument(
        "--model-config", type=str, default=None, help="Model configuration"
    )

    parser.add_argument("--resume", type=str, default=None)  # checkpoint
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--tmp", action="store_true")

    args = parser.parse_args()
    return args