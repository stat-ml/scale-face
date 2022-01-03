# FIXME fix this ASAP
from . import *
from .matlab_cp2tform import get_similarity_transform_for_cv2
from .align_datasets import align_dataset_from_list
from .face_metrics import accuracy_lfw_6000_pairs, accuracy_lfw_6000_pairs_binary_classification, FACE_METRICS
from .visualize_plots import (
    visualize_ambiguity_dilemma_lfw,
    visualize_in_out_class_distribution,
)
from .dataset import Dataset, MXFaceDataset, DataLoaderX

from .utils_callback import (
    CallBackVerification,
    CallBackLogging,
    CallBackModelCheckpoint,
)
from .utils_logging import AverageMeter
from .utils_amp import MaxClipGradScaler
from .utils_inference import inference_example

import torch
optimizers_map = {"sgd": torch.optim.SGD}
scheduler_map = {"multistep_lr": torch.optim.lr_scheduler.MultiStepLR}

def pop_element(obj: dict, key: str):
    obj.pop(key)
    return obj
