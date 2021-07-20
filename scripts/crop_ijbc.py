import os
import sys
import argparse
import numpy as np
from scipy import misc
import cv2
from tqdm import tqdm
import pickle

from joblib import Parallel, delayed
from collections import defaultdict

square_crop = True
padding_ratio = 0.0  # Add padding to bounding boxes by a ratio
target_size = (256, 256)  # If not None, resize image after processing


def square_bbox(bbox):
    """
    Output a square-like bounding box. But because all the numbers are float,
    it is not guaranteed to really be a square.
    """
    x, y, w, h = tuple(bbox)
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    _w = _h = max(w, h)
    _x = cx - 0.5 * _w
    _y = cy - 0.5 * _h
    return (_x, _y, _w, _h)


def pad_bbox(bbox, padding_ratio):
    x, y, w, h = tuple(bbox)
    pad_x = padding_ratio * w
    pad_y = padding_ratio * h
    return (x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def crop(image, bbox):
    rint = lambda a: int(round(a))
    x, y, w, h = tuple(map(rint, bbox))
    safe_pad = max(0, -x, -y, x + w - image.shape[1], y + h - image.shape[0])
    img = np.zeros(
        (image.shape[0] + 2 * safe_pad, image.shape[1] + 2 * safe_pad, image.shape[2])
    )
    img[
        safe_pad : safe_pad + image.shape[0], safe_pad : safe_pad + image.shape[1], :
    ] = image
    img = img[safe_pad + y : safe_pad + y + h, safe_pad + x : safe_pad + x + w, :]
    return img


def create_dict(args):
    lines = open(os.path.join(args.meta_path, "ijbc_face_tid_mid.txt"), "r").readlines()
    lines_dict = [_.split()[:2] for _ in lines]
    lines_dict = {a[:-4]: b for a, b in lines_dict}
    return lines, lines_dict


def main(args):
    with open(os.path.join(args.meta_path, "ijbc_metadata_modified.csv"), "r") as fr:
        lines = fr.readlines()

    # create mapping from mid to filename and crop box
    mid_map = defaultdict(list)
    mid_map_index = defaultdict(int)
    for item in lines:
        parts = item.split(",")
        mid_map[parts[2]].append([parts[1], parts[3 : 3 + 4]])

    # Some files have different extensions in the meta file,
    # record their oroginal name for reading
    files_img = os.listdir(args.prefix + "/img/")
    files_frames = os.listdir(args.prefix + "/frames/")
    dict_path = {}
    for img in files_img:
        basename = os.path.splitext(img)[0]
        dict_path["img/" + basename] = args.prefix + "/img/" + img
    for img in files_frames:
        basename = os.path.splitext(img)[0]
        dict_path["frames/" + basename] = args.prefix + "/frames/" + img

    tid_mid_list, dict_ = create_dict(args)

    count_success = 0
    count_fail = 0
    dict_name = {}
    broken_lines = []
    for i, item in tqdm(enumerate(tid_mid_list[args.start_index :])):
        parts = item[:-1].split(" ")
        map_elem = mid_map[parts[2]][mid_map_index[parts[2]]]
        mid_map_index[parts[2]] += 1
        mid_map_index[parts[2]] %= len(mid_map[parts[2]])

        impath = os.path.join(args.prefix, map_elem[0])
        imname = parts[0]

        img = cv2.imread(impath, flags=1)

        if img is None:
            # if img is None then create empty image
            img = np.zeros((256, 256, 3), dtype=np.float64)
            impath_new = os.path.join(args.save_prefix, imname)
            cv2.imwrite(impath_new, img)
            broken_lines.append(i + args.start_index)
        else:
            try:
                if img.ndim == 0:
                    print("Invalid image: %s" % impath)
                    raise
                else:
                    bbox = tuple(map(float, map_elem[1]))
                    if square_crop:
                        bbox = square_bbox(bbox)
                    bbox = pad_bbox(bbox, padding_ratio)
                    img = crop(img, bbox)

                    impath_new = os.path.join(args.save_prefix, imname)
                    if os.path.isdir(os.path.dirname(impath_new)) == False:
                        os.makedirs(os.path.dirname(impath_new))
                    if target_size:
                        img = cv2.resize(img, target_size)
                    cv2.imwrite(impath_new, img)
            except:
                broken_lines.append(i + args.start_index)

    with open(f"broken_lines_{args.start_index}.pkl", "wb") as f:
        pickle.dump(broken_lines, f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-path",
        type=str,
        default="/gpfs/gpfs0/r.karimov/ijbc_meta/",
        help="Path to metadata files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Path to the folder containing the original images of IJB-C.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="./loose_crop",
        help="Directory for output images.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index of the image to be pre-processed. Could be used with parallel utility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
