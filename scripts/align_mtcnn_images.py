import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
# from align_mtcnn.mtcnn_detector import MtcnnDetector
from mtcnn import MTCNN
import cv2
import numpy as np
from skimage import transform as trans

src = np.array([
             [30.2946, 51.6963],
             [65.5318, 51.5014],
             [48.0252, 71.7366],
             [33.5493, 92.3655],
             [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0  # this is for 112 x 112 pictures, comment this if you have 112 x 96

face_parts = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='F:\images_5', help='the dir your dataset of face which need to crop')
    parser.add_argument('--output_path', type=str, default='F:\images_5_face', help='the dir the cropped faces of your dataset where to save')
    # parser.add_argument('--face-num', '-face_num', type=int, default=1, help='the max faces to crop in each image')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=str, default='224', help='the size of the face to save, the size x%2==0, and width equal height')
    args = parser.parse_args()
    return args

def crop_align_face(args):
    input_dir = args.input_path
    output_dir = args.output_path
    # face_num = args.face_num
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # if args.gpu == -1:
    #     ctx = mx.cpu()
    # else:
    #     ctx = mx.gpu(args.gpu)

    # mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')

    # mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

    detector = MTCNN()

    number_of_aligned_images = 0
    number_of_images = 0
    sum_confidence = 0

    for root, dirs, files in tqdm(os.walk(input_dir)):
        # print(root)
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        for file_name in files:
            '''
            the specific request of the datasets in file name
            if you not need, please comment out
            '''
            # not crop the file end with bmp
            if file_name.split('.')[-1] == 'bmp':
                continue
            file_path = os.path.join(root, file_name)
            face_img = cv2.imread(file_path)

            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            # detection = detector.detect_faces(img)[0]

            detection = detector.detect_faces(img)
            # detection = detector.detect_faces(img)
            if len(detection) > 0:

                keypoints = detection[0]["keypoints"]
                ref_points = np.array([keypoints[point_name] for point_name in face_parts])

                # print(ref_points)

                tform = trans.SimilarityTransform()
                tform.estimate(ref_points, src)
                M = tform.params[0:2, :]

                img_aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)

                file_path_save = os.path.join(output_root, file_name)
                cv2.imwrite(file_path_save, cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR))

                number_of_aligned_images += 1
                sum_confidence += detection[0]["confidence"]

            number_of_images += 1

            # ret = mtcnn.detect_face(face_img)
            # if ret is None:
            #     print('%s do not find face'%file_path)
            #     count_no_find_face += 1
            #     continue
            # bbox, points = ret
            # if bbox.shape[0] == 0:
            #     print('%s do not find face'%file_path)
            #     count_no_find_face += 1
            #     continue
            # # print(bbox, points)
            # for i in range(bbox.shape[0]):
            #     bbox_ = bbox[i, 0:4]
            #     points_ = points[i, :].reshape((2, 5)).T
            #     face = mtcnn.preprocess(face_img, bbox_, points_, image_size=args.face_size)
            #     face_name = '%s_%d.jpg'%(file_name.split('.')[0], i)
            #     file_path_save = os.path.join(output_root, face_name)
            #     cv2.imwrite(file_path_save, face)
            #     # cv2.imshow('face', face)
            #     # cv2.waitKey(0)
            # count_crop_images += 1
    # print('%d images crop successful!' % count_crop_images)
    # print('%d images do not crop successful!' % count_no_find_face)
    print(f"{number_of_aligned_images} of {number_of_images} were aligned")
    print(f"Mean confidence : {sum_confidence / number_of_aligned_images}")

if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)
