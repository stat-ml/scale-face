import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_files = glob.glob(f"{path}/*/*.png")
        self.all_files.extend(glob.glob(f"{path}/*/*.jpg"))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_raw = cv2.cvtColor(
            cv2.imread(self.all_files[idx]),
            cv2.IMREAD_COLOR,
        )
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img
