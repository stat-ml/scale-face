import cv2
from time import sleep, time
import sys

import numpy as np
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from skimage import transform as trans

sys.path.append(".")
from explore.cross import ScaleFace, preprocess
import torch


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
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (36, 255, 12), 1)
    return image

def align_face(full_image, landmarks):
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]
    face_aligned = cv2.warpAffine(full_image, M, (112, 112), borderValue=0)
    return face_aligned



class Detector:
    def __init__(self):
        self.base = MTCNN(keep_all=True, thresholds=[0.4, 0.4, 0.4])

    def detect(self, frame):
        box, confidence, landmarks = self.base.detect(frame, landmarks=True)
        if landmarks is not None:
            confidence = confidence[0]
            landmarks = landmarks[0]
            box = np.round(box[0]).astype(int)
        return box, confidence, landmarks


class Barista:
    def __init__(self, regime=(0.95, 1), alpha=0.2):
        self.confidence = 0
        self.regime = regime
        self.alpha = alpha

    def _update_confidence(self, confidence):
        start = self.regime[0]
        spread = self.regime[1] - self.regime[0]
        confidence = (confidence - start) / spread
        confidence = min(max(0, confidence), 1)
        alpha = 0.2
        self.confidence = (1-alpha)*self.confidence + alpha*confidence

    def __call__(self, image, confidence):
        self._update_confidence(confidence)
        print(self.confidence, 'res')

        bar_top = (600, int(470 - 470 * self.confidence))
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


class ScaleConfidence:
    def __init__(self):
        self.model = ScaleFace()
        self.model.from_checkpoint('/home/kirill/data/faces/models/scaleface.pth')

    def __call__(self, frame):
        frame = frame[None, :, :, :]
        batch = preprocess(frame, (112, 112))
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to('cuda')
        res = self.model(batch)
        return res[1].item()


class ImageProcessor:
    def __init__(self):
        self.detector = Detector()
        self.barista = Barista((0, 14), alpha=0.25)
        self.conman = ScaleConfidence()
        self.prev_confidence = 0

    def __call__(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            box, confidence, landmarks = self.detector.detect(frame_rgb)
            if box is not None:
                face_aligned = align_face(frame_rgb, landmarks)
                scale_confidence = self.conman(face_aligned)
                print(scale_confidence)
                self.prev_confidence = scale_confidence

                frame = draw_box(frame, box)
                frame = self.barista(frame, scale_confidence)
            else:
                frame = self.barista(frame, self.prev_confidence)

            cv2.imshow('frame', frame)
            return frame
                # cv2.imshow('frame', face_aligned)
        except IndexError:
            print('wtf')


import imageio
def main():
    print('boo')

    vid = cv2.VideoCapture(0)

    processor = ImageProcessor()
    frame_rate = 10
    prev = 0

    frames = []
    while (True):
        time_elapsed = time() - prev
        # Capture the video frame by frame
        ret, frame = vid.read()

        if time_elapsed > 1. / frame_rate:
            prev = time()
            frames.append(processor(frame))
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

    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames if frame is not None]
    imageio.mimsave(f'tmp/scale_demo_{time()}.gif', frames, fps=int(frame_rate*0.7))


if __name__ == '__main__':
    main()