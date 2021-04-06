import torch
from utils.dataset import Dataset
from utils.imageprocessing import preprocess

def evaluate_lfw_paired(model, batches=100, path="e/list_casi_mtcnncaffe_aligned_nooverlap.txt"):
    trainset = Dataset(path)
    batch_format = {
        "size": 256,
        "num_classes": 64,
    }
    proc_func = lambda images: preprocess(images, True)
    trainset.start_batch_queue(batch_format, proc_func=proc_func)
    for ids in range(batches):
        batch_casia = trainset.pop_batch_queue()
        feature, sig_feat = model["backbone"](img)
        log_sig_sq = model["uncertain"](sig_feat)
        loss = model["criterion"](feature, log_sig_sq, gty)
