import glob
import argparse
from pathlib import PosixPath
from mtcnn import MTCNN
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to cropped ijb images")
args = parser.parse_args()


detector = MTCNN()

all_files = []
exts = ["jpg", "png"]
for ext in exts:
    all_files.extend(glob.glob(str(PosixPath(args.path)/"*"/f"*.{ext}")))

all_faces = []

for item in all_files:
    print(item)
    img = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    all_faces.append(faces)

import pickle
with open("all_faces.pkl", "wb") as f:
    pickle.dump(all_faces, f)
