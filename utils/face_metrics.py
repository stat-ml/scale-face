from typing import List
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import Dataset
from utils.imageprocessing import preprocess

_neg_inf = -1e6


def _collect_outputs(model, set, device, debug=False):
    features, log_sig_sq, gtys, angles_xs = [], [], [], []
    while True:
        try:
            batch = set.pop_batch_queue()
        except:
            break
        img = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).to(device)
        gtys.append(torch.from_numpy(batch["label"]))

        feature, sig_feat, angle_x = model["backbone"](img)
        log_sig_sq.append(model["uncertain"](sig_feat).detach().cpu())
        features.append(feature.detach().cpu())
        # angles_xs.append(angle_x.cpu())

        if debug is True:
            break
    features, log_sig_sq, gtys, angles_xs = (
        torch.cat(features),
        torch.cat(log_sig_sq),
        torch.cat(gtys),
        torch.cat(angles_xs),
    )
    return features, log_sig_sq, gtys, angles_xs


def _calculate_tpr(threshold_value, features_query, features_distractor, gtys_query):
    features_query_mat = torch.norm(
        features_query[:, None] - features_query[None], p=2, dim=-1
    )

    non_diag_mask = (1 - torch.eye(features_query.size(0))).long()
    gty_mask = (torch.eq(gtys_query[:, None], gtys_query[None, :])).int()
    pos_mask = (non_diag_mask * gty_mask) > 0

    R = pos_mask.sum()

    # we mark all the pairs that are not from the same identity
    # by making them negative so we won't choose them later as TPR
    features_query_mat[~pos_mask] = _neg_inf

    feature_pairs = torch.norm(
        features_query[:, None] - features_distractor[None], dim=-1
    )
    feature_pairs_max = feature_pairs.max(dim=-1)[0]
    final_mask = torch.bitwise_and(
        features_query_mat > feature_pairs_max, features_query_mat > threshold_value
    )
    return final_mask.sum() / R


def tpr_pfr(
    model: dict,
    query_path: str,
    distractor_path: str,
    FPRs=[0.1],
    *,
    in_size: tuple = (112, 96),
    batch_size: int = 128,
    device=None,
    debug=False,
) -> List[torch.Tensor]:
    if device is None:
        device = torch.device("cpu")
    # load LFW dataset (distractor)
    query_set = Dataset(query_path)
    # load query dataset
    distractor_set = Dataset(distractor_path)

    proc_func = lambda images: preprocess(images, in_size, True)
    query_set.start_sequential_batch_queue(batch_size, proc_func=proc_func)
    distractor_set.start_sequential_batch_queue(batch_size, proc_func=proc_func)

    features_query, log_sig_sq_query, gtys_query, angles_xs_query = _collect_outputs(
        model, query_set, device, debug=debug
    )
    (
        features_distractor,
        log_sig_sq_distractor,
        gtys_distractor,
        angles_xs_distractor,
    ) = _collect_outputs(model, distractor_set, device, debug=debug)
    from model import AngleLoss

    features_distractor = features_distractor[:50]

    feature_pairs = torch.norm(
        features_query[None] - features_distractor[:, None], dim=-1
    ).view(-1)
    feature_pairs, sorted_inds = torch.sort(feature_pairs, descending=True)
    len_features = feature_pairs.size(0)
    TPRs = []
    for fpr_value in FPRs:
        threshold_value = feature_pairs[int(fpr_value * len_features)]
        TPRs.append(
            _calculate_tpr(
                threshold_value, features_query, features_distractor, gtys_query
            )
        )
    print(TPRs)
    return TPRs


def accuracy_lfw_6000_pairs(model: nn.Model, device=None):
    """
        #TODO: need to understand this protocol
        #TODO: any paper to link
        This is the implementation of accuracy on 6000 pairs
    """

    if device is None:
        device = torch.device("cpu")

    def KFold(n=10, n_folds=10, shuffle=False):
        folds = []
        base = list(range(n))
        for i in range(n_folds):
            test = base[i * n // n_folds: (i + 1) * n // n_folds]
            train = list(set(base) - set(test))
            folds.append([train, test])
        return folds

    def eval_acc(threshold, diff):
        y_true = []
        y_predict = []
        for d in diff:
            same = 1 if float(d[2]) > threshold else 0
            y_predict.append(same)
            y_true.append(int(d[3]))
        y_true = np.array(y_true)
        y_predict = np.array(y_predict)
        accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
        return accuracy

    def find_best_threshold(thresholds, predicts):
        best_threshold = best_acc = 0
        for threshold in thresholds:
            accuracy = eval_acc(threshold, predicts)
            if accuracy >= best_acc:
                best_acc = accuracy
                best_threshold = threshold
        return best_threshold

    lfw_prefix = "/gpfs/gpfs0/r.karimov/lfw/data_"

    predicts = []

    from utils.imageprocessing import preprocess

    proc_func = lambda images: preprocess(images, [112, 96], is_training=False)
    lfw_set = Dataset("/gpfs/gpfs0/r.karimov/lfw/data_", preprocess_func=proc_func)

    with open("/gpfs/gpfs0/r.karimov/lfw/pairs_val_6000.txt") as f:
        pairs_lines = f.readlines()[1:]

    for i in tqdm(range(6000)):
        p = pairs_lines[i].replace("\n", "").split("\t")

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))

        try:
            img1 = lfw_set.get_item_by_the_path(name1)
            img2 = lfw_set.get_item_by_the_path(name2)
        except:
            # FIXME: mtcncaffe and spherenet alignments are not the same
            continue

        img_batch = torch.from_numpy(np.concatenate((img1[None], img1[None]), axis=0)).permute(0, 3, 1, 2).to(device)
        output = model(img_batch)
        f1, f2 = output
        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        predicts.append("{}\t{}\t{}\t{}\n".format(name1, name2, cosdistance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=10, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)

    predicts = np.array(list(map(lambda line: line.strip("\n").split(), predicts)))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    return np.mean(accuracy)

