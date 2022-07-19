import os

import numpy as np
from mtcnn import MTCNN
from PIL import Image
import cv2
from skimage import transform

src = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)
src[:, 0] += 8.0

face_parts = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


def align_images(image_paths):
    from tqdm import tqdm
    detector = MTCNN()
    for path in tqdm(image_paths):
        full = str(cplfw_dir / 'aligned images' / path)
        aligned = aligned_image(full, detector)
        new_path = str(cplfw_dir / 'mtcnn_images' / path)
        cv2.imwrite(
            new_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
        )




def aligned_image(image_path, detector=None):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detection = detector.detect_faces(image)
    image_aligned = None
    if len(detection) > 0:
        keypoints = detection[0]["keypoints"]
        ref_points = np.array(
            [keypoints[point_name] for point_name in face_parts]
        )

        tform = transform.SimilarityTransform()
        tform.estimate(ref_points, src)
        M = tform.params[0:2, :]

        image_aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
    else:
        image_aligned = image[56:168, 56:168]
    return image_aligned
