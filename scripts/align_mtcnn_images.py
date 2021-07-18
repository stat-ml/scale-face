import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from skimage import transform as trans

src = np.array([
             [30.2946, 51.6963],
             [65.5318, 51.5014],
             [48.0252, 71.7366],
             [33.5493, 92.3655],
             [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0  # this is for 112 x 112 pictures, comment this if you have 112 x 96

face_parts = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

# This is the second step of data preperoccessing in ijbc. The images should already be cropped


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default='/gpfs/gpfs0/r.kail/IJB/IJBC_cropped_debug',
        help='the dir your dataset of face which need to crop')
    parser.add_argument(
        '--output_path',
        type=str,
        default='/gpfs/gpfs0/r.kail/IJB/IJBC_aligned_debug',
        help='the dir the cropped faces of your dataset where to save')
    parser.add_argument(
        '--gpu',
        default=-1,
        type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument(
        '--face_size',
        type=str,
        default='224',
        help='the size of the face to save, the size x%2==0, and width equal height')
    args = parser.parse_args()
    return args


def crop_align_face(args):
    input_dir = args.input_path
    output_dir = args.output_path
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    detector = MTCNN()

    number_of_aligned_images = 0
    number_of_images = 0
    sum_confidence = 0

    for root, dirs, files in tqdm(os.walk(input_dir)):
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        for file_name in files:
            # not crop the file end with bmp
            if file_name.split('.')[-1] == 'bmp':
                continue
            file_path = os.path.join(root, file_name)

            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            detection = detector.detect_faces(img)
            if len(detection) > 0:

                keypoints = detection[0]["keypoints"]
                ref_points = np.array([keypoints[point_name] for point_name in face_parts])

                tform = trans.SimilarityTransform()
                tform.estimate(ref_points, src)
                M = tform.params[0:2, :]

                img_aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)

                file_path_save = os.path.join(output_root, file_name)
                cv2.imwrite(file_path_save, cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR))

                number_of_aligned_images += 1
                sum_confidence += detection[0]["confidence"]

            number_of_images += 1

    print(f"{number_of_aligned_images} of {number_of_images} were aligned")
    print(f"Mean confidence : {sum_confidence / number_of_aligned_images}")


if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)
