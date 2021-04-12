# TODO: joblib parallelize this
import numpy as np
import imageio
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, Tuple, List, Optional

from face_lib.utils import get_similarity_transform_for_cv2

REF_PTS_ST = np.array(
    [
        [-1.58083929e-01, -3.84258929e-02],
        [1.56533929e-01, -4.01660714e-02],
        [2.25000000e-04, 1.40505357e-01],
        [-1.29024107e-01, 3.24691964e-01],
        [1.31516964e-01, 3.23250893e-01],
    ]
)


def align(src_img, src_pts, ref_pts, image_size, scale=1.0, transpose_input=False):
    w, h = image_size = tuple(image_size)
    scale_ = max(w, h) * scale
    cx_ref = cy_ref = 0.0
    offset_x = 0.5 * w - cx_ref * scale_
    offset_y = 0.5 * h - cy_ref * scale_
    s = np.array(src_pts).astype(np.float32).reshape([-1, 2])
    r = np.array(ref_pts).astype(np.float32) * scale_ + np.array([[offset_x, offset_y]])
    if transpose_input:
        s = s.reshape([2, -1]).T
    tfm = get_similarity_transform_for_cv2(s, r)
    import pdb

    pdb.set_trace()
    dst_img = cv2.warpAffine(src_img, tfm, image_size)
    return dst_img


def align_dataset_from_list(
    input_file: str,
    prefix,
    transpose_input,
    image_size: Union[List, Tuple],
    scale=1.0,
    *,
    ref_pts=REF_PTS_ST,
    output_dir: str = Optional[None],
    progress_desc: str = "",
    visualize: bool = False,
    dir_depth: int = 2,
):
    lines = open(input_file, "r").readlines()
    for i, line in tqdm(
        enumerate(lines),
        desc=progress_desc if progress_desc else "Aligning the dataset progress",
    ):
        line = line.strip()
        items = line.split()
        img_path = items[0]
        src_pts = [float(item) for item in items[1:]]

        if prefix:
            img_path = os.path.join(prefix, img_path)
        img = imageio.imread(img_path)
        print("\n", img_path)
        img_new = align(img, src_pts, ref_pts, image_size, scale, transpose_input)

        if visualize:
            # break the iteration if visualize flag is True
            # TODO: better debug logic here
            plt.imshow(img_new)
            plt.show()
            return

        if output_dir:
            file_name = os.path.basename(img_path)
            sub_dir = [d for d in img_path.split("/") if d != ""]
            sub_dir = "/".join(sub_dir[-dir_depth:-1])
            dir_path = os.path.join(output_dir, sub_dir)

            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            img_path_new = os.path.join(dir_path, file_name)
            imageio.imsave(img_path_new, img_new)
