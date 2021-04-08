from typing import List
import torch
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
        #angles_xs.append(angle_x.cpu())

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
