import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt1",
        help="The path to the pre-trained model directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt2",
        help="The path to the pre-trained model directory",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main(args):
    ckpt1 = torch.load(args.ckpt1, map_location=torch.device("cpu"))["backbone"]
    ckpt2 = torch.load(args.ckpt2, map_location=torch.device("cpu"))["backbone"]

    print(f"{len(ckpt1)=} {len(ckpt2)=}")

    same_backbones = True
    for (name1, p1), (name2, p2) in zip(ckpt1.items(), ckpt2.items()):
        if name1 != name2:
            print(f"Found different names")
            return

        if p1.dtype == torch.float:
            print(f"[{p1.numel()}]{torch.linalg.norm(p1 - p2)}")

        if not torch.allclose(p1, p2):
            same_backbones = False

    if same_backbones:
        print(f"Backbones are the same")
    else:
        print("Backbones are NOT the same")

if __name__ == "__main__":
    main(parse_args())