import os
import torch
import imageio
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class DimensionAdder:
    def __init__(self):
        pass

    def __call__(self, image):
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        if image.shape[2] == 1:
            return np.concatenate([image] * 3, axis=2)
        else:
            return image


class ProductsDataset(Dataset):
    def __init__(self, root_dir: str, local_rank: int):
        super(ProductsDataset, self).__init__()

        self.root_dir = root_dir
        self.local_rank = local_rank

        file_path = os.path.join(self.root_dir, "Ebay_info.txt")
        self.index = pd.read_csv(file_path, sep=" ", header=0, index_col=0)

        self.transform = transforms.Compose(
            [DimensionAdder(),
             transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Resize((112, 112)),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

    def __getitem__(self, index):
        label = self.index["class_id"].iloc[index]
        label = torch.tensor(label, dtype=torch.long)

        file_path = os.path.join(self.root_dir, self.index["path"].iloc[index])
        img = imageio.imread(file_path)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.index)