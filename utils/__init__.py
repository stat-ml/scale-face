import torch

# FIXME fix this ASAP
from . import *
from .matlab_cp2tform import get_similarity_transform_for_cv2
from .align_datasets import align_dataset_from_list
from .face_metrics import accuracy_lfw_6000_pairs, FACE_METRICS

optimizers_map = {"sgd": torch.optim.SGD}


def pop_element(obj: dict, key: str):
    obj.pop(key)
    return obj
