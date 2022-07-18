import numbers
import os

# import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.imgidx)


class MXFaceDatasetDistorted(MXFaceDataset):
    def __init__(self, *args, **kwargs):
        super(MXFaceDatasetDistorted, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.RandomAdjustSharpness(sharpness_factor=4, p=0.2),
             transforms.RandomEqualize(p=0.2),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.RandomApply(
                 [transforms.RandomChoice([
                     transforms.GaussianBlur(kernel_size=7, sigma=3.),
                     transforms.GaussianBlur(kernel_size=7, sigma=5.),
                     transforms.GaussianBlur(kernel_size=7, sigma=7.),
                     transforms.GaussianBlur(kernel_size=7, sigma=9.),
                 ])], p=0.3),
             transforms.RandomPerspective(distortion_scale=0.25, p=0.15)
        ])


class MXFaceDatasetGauss(MXFaceDataset):
    def __init__(self, *args, **kwargs):
        super(MXFaceDatasetGauss, self).__init__(*args, **kwargs)
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             transforms.RandomApply(
                 [transforms.RandomChoice([
                     transforms.GaussianBlur(kernel_size=7, sigma=0.5),
                     transforms.GaussianBlur(kernel_size=7, sigma=1.),
                     transforms.GaussianBlur(kernel_size=7, sigma=3.),
                     transforms.GaussianBlur(kernel_size=7, sigma=5.),
                     transforms.GaussianBlur(kernel_size=7, sigma=7.),
                     transforms.GaussianBlur(kernel_size=7, sigma=9.),
                 ])], p=0.7),
        ])


class SyntheticDataset(Dataset):
    def __init__(self, local_rank):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000
