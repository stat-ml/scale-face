import exman
import os.path as osp

def training_args():

    # TODO: don't like the path manipulation here
    parser = exman.ExParser(root=exman.simpleroot(__file__).parent.parent / "exman")

    parser.add_argument(
        "--env-config", type=str, default=None, help="Environment configuration"
    )

    parser.add_argument(
        "--model-config", type=str, default=None, help="Model configuration"
    )

    parser.add_argument(
        "--optimizer-config", type=str, default=None, help="Optimizer configuration"
    )

    parser.add_argument(
        "--dataset-config", type=str, default=None, help="Dataset configuration"
    )

    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--resume", type=str, default=None
    )  # checkpoint

    parser.add_argument("--freeze-backbone", type=bool, default=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-classes-batch", type=int, default=64)

    parser.add_argument(
        "--print_freq", type=int, default=20
    )  # v0 : <64， 166> | <128, 83>
    parser.add_argument("--save_freq", type=int, default=1)  # TODO

    args = parser.parse_args()

    return args
