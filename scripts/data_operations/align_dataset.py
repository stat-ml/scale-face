import sys

sys.path.append("../")
import argparse
from face_lib.utils import align_dataset_from_list


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str, help="A list file of image paths and landmarks."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory with aligned face thumbnails.",
        default=None,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="The prefix of the image files in the input_file.",
        default=None,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        help="Image size (height, width) in pixels.",
        default=[112, 112],
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Scale the face size in the target image.",
        default=1.0,
    )
    parser.add_argument(
        "--dir_depth",
        type=int,
        help="When writing into new directory, how many layers of the dir tree should be kept.",
        default=2,
    )
    parser.add_argument(
        "--transpose_input",
        action="store_true",
        help="Set true if the input landmarks is in the format x1 x2 ... y1 y2 ...",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the aligned images."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    align_dataset_from_list(**vars(args))
