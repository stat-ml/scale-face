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
from argparse import ArgumentParser


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
    # color = (36, 255, 12)
    color = (240, 240, 240)
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 1)
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


lineType = cv2.LINE_AA

def putBText(img,text,text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(1,1,1),font = cv2.FONT_HERSHEY_DUPLEX,thickness = 2,alpha=0.6,gamma=0):
    """"
    copied from pyshine library

    Inputs:
    img: cv2 image img
    text_offset_x, text_offset_x: X,Y location of text start
    vspace, hspace: Vertical and Horizontal space between text and box boundries
    font_scale: Font size
    background_RGB: Background R,G,B color
    text_RGB: Text R,G,B color
    font: Font Style e.g. cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_COMPLEX
          cv2.FONT_HERSHEY_TRIPLEX, etc
    thickness: Thickness of the text font
    alpha: Opacity 0~1 of the box around text
    gamma: 0 by default

    Output:
    img: CV2 image with text and background
	"""
    R,G,B = background_RGB[0],background_RGB[1],background_RGB[2]
    text_R,text_G,text_B = text_RGB[0],text_RGB[1],text_RGB[2]
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    x, y, w, h = text_offset_x, text_offset_y, text_width , text_height
    crop = img[y-vspace:y+h+vspace, x-hspace:x+w+hspace]
    white_rect = np.ones(crop.shape, dtype=np.uint8)
    b,g,r = cv2.split(white_rect)
    rect_changed = cv2.merge((B*b,G*g,R*r))
    res = cv2.addWeighted(crop, alpha, rect_changed, 1-alpha, gamma)
    img[y-vspace:y+vspace+h, x-hspace:x+w+hspace] = res
    cv2.putText(
        img, text, (x, (y+h)), font, fontScale=font_scale, color=(text_B,text_G,text_R ), thickness=thickness,
        lineType=cv2.LINE_AA
    )
    return img



class ImageProcessor:
    def __init__(self, full_screen=False):
        self.detector = Detector()
        self.barista = Barista((0, 14), alpha=0.25)
        self.conman = ScaleConfidence()
        self.prev_confidence = 0

        if full_screen:
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
            frame = putBText(
                frame, 'Face recognition confidence', text_offset_x=20,
                text_offset_y=20, vspace=10, hspace=10, font_scale=1,
                background_RGB=(240, 240, 240), text_RGB=(50, 50, 50), thickness=1,
                alpha=0.3
            )

            frame = putBText(
                frame, 'Try hide part of the face, put your mask on,', text_offset_x=20,
                text_offset_y=420, vspace=10, hspace=10, font_scale=0.7,
                background_RGB=(240, 240, 240), text_RGB=(50, 50, 50), thickness=1,
                alpha=0.3
            )

            frame = putBText(
                frame, 'turn sideways or make faces', text_offset_x=20,
                text_offset_y=456, vspace=10, hspace=10, font_scale=0.7,
                background_RGB=(240, 240, 240), text_RGB=(50, 50, 50), thickness=1,
                alpha=0.3
            )


            cv2.imshow('ScaleFace', frame)
            return frame
        except IndexError:
            print('wtf')


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--gif', default=False, action='store_true', help='record a gif in tmp folder'
    )
    parser.add_argument('--frame_rate', type=float, default=5.)
    parser.add_argument('--full_screen', default=False, action='store_true')
    args = parser.parse_args()

    vid = cv2.VideoCapture(0)

    processor = ImageProcessor(full_screen=args.full_screen)
    prev = 0

    frames = []

    while (True):
        time_elapsed = time() - prev
        # Capture the video frame by frame
        ret, frame = vid.read()

        if time_elapsed > 1. / args.frame_rate:
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
        imageio.mimsave(f'tmp/scale_demo_{time()}.gif', frames, fps=int(args.frame_rate*0.7))


if __name__ == '__main__':
    main()