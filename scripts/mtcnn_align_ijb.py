import os
import glob
import argparse
import pickle
from pathlib import PosixPath
from mtcnn import MTCNN
import numpy as np
import cv2
import sys

sys.path.append("../")
from face_lib.utils import align_image
import matplotlib.pyplot as plt

from tqdm import tqdm
import joblib
import multiprocessing as mp

CPU_COUNT = mp.cpu_count()


def transform_key_dict_to_list(keyps: dict):
    res_ls = []
    key_order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    for key_name in key_order:
        res_ls.extend(list(keyps[key_name]))
    res_ls = [float(_) for _ in res_ls]
    return res_ls


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to cropped ijb images")
parser.add_argument("--save-prefix", type=str, help="Path to cropped ijb images")
parser.add_argument("--cpus", type=int, default=None, help="Path to cropped ijb images")
args = parser.parse_args()

detector = MTCNN()

all_files = []
exts = ["jpg", "png"]
for ext in exts:
    all_files.extend(glob.glob(str(PosixPath(args.path) / "*" / f"*.{ext}")))
all_files = sorted(all_files)

failed_examples = []

os.makedirs(args.save_prefix, exist_ok=True)


def _f(item):
    img = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if len(faces) < 1:
        return 1
    keyps = faces[0]["keypoints"]
    keyps_list = transform_key_dict_to_list(keyps)
    keyps_list_np = np.array(keyps_list).reshape((5, 2)).T
    img_aligned = align_image(
        img, keyps_list_np, [112, 112], scale=1.0, transpose_input=True
    )

    im_dir = os.path.join(args.save_prefix, "/".join(item.split("/")[-2:-1]))

    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)

    out_impath = "/".join(item.split("/")[-2:])
    plt.imsave(os.path.join(args.save_prefix, out_impath), img_aligned)
    return 0


failed_examples = joblib.Parallel(
    n_jobs=args.cpus or CPU_COUNT, backend="multiprocessing"
)((joblib.delayed(_f)(item) for i, item in tqdm(enumerate(all_files))))
failed_examples = np.array(failed_examples)

with open("failed_ijbc_indices_pickled", "wb") as f:
    pickle.dump(failed_examples, f)
