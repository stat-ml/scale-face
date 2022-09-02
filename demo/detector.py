import cv2
from time import time
import sys

import numpy as np
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
from skimage import transform as trans

sys.path.append(".")
from face_lib.evaluation.cross import ScaleFace, preprocess
import torch
import imageio
import pyshine as ps


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
    def __init__(self, fullscreen=False):
        self.detector = Detector()
        self.barista = Barista((0, 14), alpha=0.25)
        self.conman = ScaleConfidence()
        self.prev_confidence = 0

        if fullscreen:
            cv2.namedWindow("ScaleFace", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("ScaleFace", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            frame = self.barista(frame, self.prev_confidence)
            frame = ps.putBText(
                frame, 'Face recognition confidence', text_offset_x=20,
                text_offset_y=20, vspace=10, hspace=10, font_scale=0.5,
                background_RGB=(240, 240, 240), text_RGB=(50, 50, 50), thickness=1
            )
            # cv2.rectangle(frame, (40, 10), (550, 70), (50, 50, 50), -1)
            # frame = cv2.putText(
            #     frame, 'Face recognition confidence', org=(50, 40),
            #     fontFace=2, fontScale=1,
            #     color=(200, 200, 200), thickness=2
            # )

            cv2.imshow('ScaleFace', frame)
            return frame
                # cv2.imshow('frame', face_aligned)
        except IndexError:
            print('wtf')

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--gif', default=False, action='store_true', help='record a gif in tmp folder'
    )
    args = parser.parse_args()

    vid = cv2.VideoCapture(0)

    processor = ImageProcessor(fullscreen=True)
    frame_rate = 10
    prev = 0

    frames = []

    while (True):
        time_elapsed = time() - prev
        # Capture the video frame by frame
        ret, frame = vid.read()

        if time_elapsed > 1. / frame_rate:
            prev = time()
            modified_frame = processor(frame)
            if args.gif:
                frames.append(modified_frame)
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

    if args.gif:
        frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames if frame is not None
        ]
        imageio.mimsave(f'tmp/scale_demo_{time()}.gif', frames, fps=int(frame_rate*0.7))


if __name__ == '__main__':
    main()