import numpy as np
from warnings import warn


def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    """
    Find thresholds given FARs
    but the real FARs using these thresholds could be different
    the exact FARs need to recomputed using calcROC
    """
    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def ROC(score_vec: np.ndarray, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    """
    Compute Receiver operating characteristic (ROC) with a score and label vector.
    """
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool

    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)

    assert len(thresholds.shape) == 1
    if np.size(thresholds) > 10000:
        warn(
            "number of thresholds (%d) very large, computation may take a long time!"
            % np.size(thresholds)
        )

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i, threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = np.mean(accept[label_vec])
        FARs[i] = np.mean(accept[~label_vec])
        if get_false_indices:
            false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
            false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds
