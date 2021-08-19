import torch

from face_lib.utils import Dataset
from face_lib.utils import imageprocessing


def inference_example(model):
    ims = [

        "/trinity/home/r.karimov/face-evaluation/inference/rasul1.jpg",
        "/trinity/home/r.karimov/face-evaluation/inference/rasul2.jpg",
        "/trinity/home/r.karimov/face-evaluation/inference/rasul3.jpg",
        "/trinity/home/r.karimov/face-evaluation/inference/rasul4.jpg",
        "/gpfs/gpfs0/k.fedyanin/space/lfw/data_aligned_mtcnn/Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
        "/gpfs/gpfs0/k.fedyanin/space/lfw/data_aligned_mtcnn/Abbas_Kiarostami/Abbas_Kiarostami_0001.jpg",
        "/gpfs/gpfs0/k.fedyanin/space/lfw/data_aligned_mtcnn/Zdravko_Mucic/Zdravko_Mucic_0001.jpg",
    ]
    ims = imageprocessing.preprocess(ims, [112, 96])
    ims = torch.from_numpy(ims).permute(0, 3, 1, 2).cuda()
    feature = model(ims)["feature"]
    import pdb

    pdb.set_trace()
    f1, f2, f3, f4, f5, f6, f7 = feature
    func = lambda f1, f2: f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    import pdb

    pdb.set_trace()
