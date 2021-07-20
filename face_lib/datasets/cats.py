import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset


class CatsDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.file_list = []
        for cat_folder_num in range(7):
            cat_folder = os.path.join(self.args.cats_dir, "CAT_0" + str(cat_folder_num))
            for root, dirs, files in os.walk(args.cats_dir):
                for file in files:
                    if file.endswith(".jpg"):
                        self.file_list.append(os.path.join(root, file))

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.args.in_size)

        img = self.transforms(img)
        return img, 0
