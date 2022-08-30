"""
Lvl 11
draw green bar
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

face_parts = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


def draw_box(image, b):
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (36, 255, 12), 2)
    return image

def align_face(full_image, landmarks):
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]
    face_aligned = cv2.warpAffine(full_image, M, (112, 112), borderValue=0)
    return face_aligned



class Detector:
    def __init__(self):
        self.base = MTCNN(keep_all=True)

    def detect(self, frame):
        box, confidence, landmarks = self.base.detect(frame, landmarks=True)
        if landmarks is not None:
            confidence = confidence[0]
            landmarks = landmarks[0]
            box = np.round(box[0]).astype(np.int)
            print(confidence)
        return box, confidence, landmarks


class Barista:
    def __init__(self):
        self.confidence = 0

    def __call__(self, image, confidence):
        confidence = (confidence - 0.95) * 20
        confidence = max(0, confidence)
        alpha = 0.2
        self.confidence = (1-alpha)*self.confidence + alpha*confidence
        print(self.confidence)

        bar_top = (600, int(470 - 400 * self.confidence))
        color = self._color(self.confidence)
        image = cv2.rectangle(image, bar_top, (620, 470), color, 20)
        return image

    def _color(self, confidence):
        """confidence should b"""
        assert confidence >= 0
        assert  confidence < 1
        green = 20 + 235 * confidence
        red = 255 - 235 * confidence
        return (36, green, red)




class ImageProcessor:
    def __init__(self):
        self.detector = Detector()
        self.barista = Barista()

    def __call__(self, frame):
        try:
            box, confidence, landmarks = self.detector.detect(frame)
            if box is not None:
                face_aligned = align_face(frame, landmarks)
                print(face_aligned.shape)

                frame = draw_box(frame, box)
                frame = self.barista(frame, confidence)
                cv2.imshow('frame', frame)
                # cv2.imshow('frame', face_aligned)
        except IndexError:
            print('wtf')


def main():
    print('boo')

    vid = cv2.VideoCapture(0)

    processor = ImageProcessor()
    frame_rate = 10
    prev = 0

    while (True):
        time_elapsed = time() - prev
        # Capture the video frame by frame
        ret, frame = vid.read()

        if time_elapsed > 1. / frame_rate:
            prev = time()
            processor(frame)
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