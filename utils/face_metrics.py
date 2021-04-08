from itertools import product

from utils.dataset_lfw import Dataset as LFW_Dataset
from utils.dataset import Dataset
from utils.imageprocessing import preprocess_paralleld


def tpr_fpr_lfw(distractor_path, lfw_path, *, in_size=(112, 96)):
    # load LFW dataset (distractor)
    paths_full = LFW_Dataset(lfw_path)["abspath"]
    # load query dataset
    trainset = Dataset(distractor_path)
    proc_func = lambda images: preprocess_paralleld(images, in_size, True, n_jobs=1)
    images_path, labels = trainset.get_all_data(proc_func=proc_func)


def tpr_fpr_megaface():
    raise NotImplementedError
