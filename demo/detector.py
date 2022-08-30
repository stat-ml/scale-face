"""
Lvl 10
align
"""

import cv2
from time import sleep, time

import numpy as np
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from skimage import transform as trans


src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
src[:, 0] += 8.0  # this is for 112 x 112 pictures, comment this if you have 112 x 96

face_parts = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")



def detect_face(detector, frame):
    box, confidence, landmarks = detector.detect(frame, landmarks=True)
    if landmarks is not None:
        confidence = confidence[0]
        landmarks = landmarks[0]
        box = np.round(box[0]).astype(np.int)
        print(confidence)
    return box, landmarks

def draw_box(image, b):
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (36, 255, 12), 2)
    return image
    pass

def align_face(full_image, landmarks):
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]
    face_aligned = cv2.warpAffine(full_image, M, (112, 112), borderValue=0)
    return face_aligned


def process_image(detector, frame):
    try:
        box, landmarks = detect_face(detector, frame)
        if box is not None:
            face_aligned = align_face(frame, landmarks)
            print(face_aligned.shape)

            frame = draw_box(frame, box)
            cv2.imshow('frame', frame)
    except IndexError:
        print('wtf')
    pass



def main():
    print('boo')

    vid = cv2.VideoCapture(0)

    detector = MTCNN(keep_all=True)
    frame_rate = 10
    prev = 0

    while (True):
        time_elapsed = time() - prev
        # Capture the video frame by frame
        ret, frame = vid.read()

        if time_elapsed > 1. / frame_rate:
            prev = time()
            process_image(detector, frame)
            print()
        else:
            print('.', end='', flush=True)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()