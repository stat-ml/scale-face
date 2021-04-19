import os
import sys
import argparse
import numpy as np
import cv2
import joblib

from tqdm import tqdm

square_crop = True  # Take the max of (w,h) for a square bounding box
padding_ratio = 0.0  # Add padding to bounding boxes by a ratio
target_size = (256, 256)  # If not None, resize image after processing


def square_bbox(bbox):
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


def main(args):
    with open(args.meta_file, "r") as fr:
        lines = fr.readlines()

    files_img = os.listdir(args.prefix + "/img/")
    files_frames = os.listdir(args.prefix + "/frames/")
    dict_path = {}
    for img in files_img:
        basename = os.path.splitext(img)[0]
        dict_path["img/" + basename] = args.prefix + "/img/" + img
    for img in files_frames:
        basename = os.path.splitext(img)[0]
        dict_path["frames/" + basename] = args.prefix + "/frames/" + img

    count_success = 0
    count_fail = 0
    dict_name = {}

    def _f(i, line):
        if i > 0:
            parts = line.split(",")
            label = parts[0]
            impath = os.path.join(args.prefix, parts[1])
            imname = os.path.join(label, parts[1].replace("/", "_"))
            # Check name duplication
            if imname in dict_name:
                print(
                    "image %s at line %d collision with  line %d"
                    % (imname, i, dict_name[imname])
                )
            dict_name[imname] = i

            # Check extention difference
            if not os.path.isfile(impath):
                basename = os.path.splitext(parts[1])[0]
                if basename in dict_path:
                    impath = dict_path[basename]
                else:
                    print("%s not found in the input directory, skipped" % (impath))
                    return
            if impath[-4:] == ".mp4":
                return 0
            img = cv2.imread(impath, flags=1)

            if img.ndim == 0:
                print("Invalid image: %s" % impath)
                return 0
            else:
                bbox = tuple(map(float, parts[3:7]))
                if square_crop:
                    bbox = square_bbox(bbox)
                bbox = pad_bbox(bbox, padding_ratio)
                try:
                    img = crop(img, bbox)
                except:
                    return 0

                impath_new = os.path.join(args.save_prefix, imname)
                if not os.path.isdir(os.path.dirname(impath_new)):
                    os.makedirs(os.path.dirname(impath_new))
                if target_size:
                    img = cv2.resize(img, target_size)
                cv2.imwrite(impath_new, img)
                return 1

    count = joblib.Parallel(n_jobs=1)(
        joblib.delayed(_f)(i, line) for i, line in tqdm(enumerate(lines[141296:]))
    )
    count = np.array(count)

    print(
        "%d images cropped, %d images failed"
        % (len(count[count == 1]), len(count[count == 0]))
    )
    print("%d image names created" % len(dict_name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_file", type=str, help="Path to metadata file.")
    parser.add_argument(
        "prefix",
        type=str,
        help="Path to the folder containing the original images of IJB-A.",
    )
    parser.add_argument("save_prefix", type=str, help="Directory for output images.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
