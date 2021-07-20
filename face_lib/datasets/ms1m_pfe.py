import numbers
import os
import random

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def ms1m_collate_fn(batch):

    imgs, gtys = [], []
    for pid_imgs, gty in batch:
        imgs.extend(pid_imgs)
        gtys.extend([gty] * len(pid_imgs))

    return torch.stack(imgs, dim=0), torch.tensor(gtys).long()


class MS1MDatasetPFE(Dataset):
    def __init__(self, root_dir, num_face_pb, local_rank, **kwargs):
        super(MS1MDatasetPFE, self).__init__()

        self.num_face_pb = num_face_pb

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
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
            self.class_to_first_idx = torch.load(root_dir + "identities_indices.pt")

    def __get_pic_by_idx__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __getitem__(self, idx):
        left_idx = self.class_to_first_idx[idx]
        right_idx = self.class_to_first_idx[idx + 1]

        if right_idx - left_idx >= self.num_face_pb:
            indices = random.sample(range(left_idx, right_idx), k=self.num_face_pb)
        else:
            indices = list(range(left_idx, right_idx))

        imgs = []
        for pic_idx in indices:
            img, identity = self.__get_pic_by_idx__(pic_idx)
            assert identity == idx
            imgs.append(img)

        return imgs, idx

    def __len__(self):
        return len(self.class_to_first_idx) - 1

    def get_same_class_idx(idx):
        pass
