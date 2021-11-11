import numbers
import os
import random

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .ms1m_pfe import MS1MDatasetPFE


class MS1MDatasetRandomPairs(MS1MDatasetPFE):
    def __init__(self, root_dir, in_size, p_same=0.5, hor_flip_prob=0.5, **kwargs):
        super(MS1MDatasetPFE, self).__init__()

        self.p_same = p_same    # probability of pick the pair of same faces

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(hor_flip_prob),
                transforms.Resize(size=in_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        print("Root dir : ", root_dir)

        if os.path.exists(root_dir + "/identities_indices.pt"):
            self.class_to_first_idx = torch.load(
                os.path.join(root_dir, "identities_indices.pt")
            )

    def __getitem__(self, idx):
        left_idx = self.class_to_first_idx[idx]
        right_idx = self.class_to_first_idx[idx + 1]

        first_face_img_idx = random.choice(range(left_idx, right_idx))

        is_face_same = int(np.random.random() < self.p_same)
        if(is_face_same == 1):
            list_range_idx = list(range(left_idx, right_idx))
            list_range_idx.remove(first_face_img_idx)
            second_face_img_idx = random.choice(list_range_idx)
        else:
            list_range_first_class_idx = list(range(0, len(self.class_to_first_idx) - 1))
            list_range_first_class_idx.remove(idx)
            different_face_class_idx = random.choice(list_range_first_class_idx)
            second_left_idx = self.class_to_first_idx[different_face_class_idx]
            second_right_idx = self.class_to_first_idx[different_face_class_idx + 1]
            second_face_img_idx = random.choice(range(second_left_idx, second_right_idx))

        first_face_img, first_face_identity = self.__get_pic_by_idx__(first_face_img_idx)
        second_face_img, second_face_identity = self.__get_pic_by_idx__(second_face_img_idx)
        label = is_face_same

        return first_face_img, second_face_img, torch.tensor(label, dtype=torch.long)   # long for CroosEntropy, float for BCE?