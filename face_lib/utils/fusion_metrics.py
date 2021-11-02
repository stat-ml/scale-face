import torch
import torch.nn.functional as F
import numpy as np
from warnings import warn

from .utils import harmonic_mean


def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis, keepdims=True))
    return x


def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x
    attention = 1.0 / sigma_sq
    attention = attention / np.sum(attention, axis=0, keepdims=True)

    mu_new = np.sum(mu * attention, axis=0)
    sigma_sq_new = np.min(sigma_sq, axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new


def aggregate_min(x, sigma_sq, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x

    best_features_indices = sigma_sq.argmin(axis=0)
    mu_new = mu[best_features_indices, range(x.shape[1])]
    sigma_sq_new = sigma_sq[best_features_indices, range(x.shape[1])]
    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new


def aggregate_softmax(x, sigma_sq, temperature=1.0, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:, :D], x[:, D:]
    else:
        mu = x

    weights = np.exp(sigma_sq * temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    mu_new = (mu * weights).sum(axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    return mu_new


def pair_euc_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def pair_cosine_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    x1, x2 = np.array(x1), np.array(x2)
    x1, x2 = l2_normalize(x1, axis=1), l2_normalize(x2, axis=1)
    dist = np.sum(x1 * x2, axis=1)
    return dist


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert (
            sigma_sq2 is None
        ), "either pass in concated features, or mu, sigma_sq for both!"
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:, :D], x1[:, D:]
        mu2, sigma_sq2 = x2[:, :D], x2[:, D:]
    else:
        mu1, mu2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = l2_normalize(mu1, axis=1), l2_normalize(mu2, axis=1)
        # mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(
        np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1
    )
    return -dist


def pair_uncertainty_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return sigma_sq_1.sum(axis=1) + sigma_sq_2.sum(axis=1)


def pair_uncertainty_harmonic_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) + harmonic_mean(sigma_sq_2, axis=1)


def pair_uncertainty_concatenated_harmonic(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(
        np.concatenate(
            (
                sigma_sq_1,
                sigma_sq_2,
            ),
            axis=1,
        ),
        axis=1,
    )


def pair_uncertainty_cosine_analytic(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return (sigma_sq_1 * sigma_sq_2 + (mu_1 ** 2) * sigma_sq_2 + (mu_2 ** 2) * sigma_sq_1).sum(axis=1)


def classifier_to_distance_wrapper(classifier, device=torch.device("cpu")):
    def wrapped_classifier(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
        inputs = torch.cat((torch.from_numpy(mu_1), torch.from_numpy(mu_2)), dim=1)
        probes = F.softmax(classifier(feature=inputs.to(device), dim=1)["pair_classifiers_output"], dim=-1)
        probes = probes.cpu().detach().numpy()
        return probes[:, 1]  # TODO : It is supposed to be 1 - probes[:, 1] why ???
    return wrapped_classifier


def classifier_to_uncertainty_wrapper(classifier, device=torch.device("cpu")):
    def wrapped_classifier(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
        inputs = torch.cat((torch.from_numpy(mu_1), torch.from_numpy(mu_2)), dim=1)
        probes = F.softmax(classifier(feature=inputs.to(device))["pair_classifiers_output"], dim=-1)
        probes = probes.cpu().detach().numpy()
        return 1 - probes.max(axis=1) #  TODO : fix this, it is not supposed to be this way
    return wrapped_classifier


def split_wrapper(distance_func, batch_size=64):
    def wrapped_distance(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
        distances = []
        for mu_1_batch, mu_2_batch, sigma_sq_1_batch, sigma_sq_2_batch in zip(
            np.array_split(mu_1, len(mu_1) // batch_size + 1),
            np.array_split(mu_2, len(mu_2) // batch_size + 1),
            np.array_split(sigma_sq_1, len(sigma_sq_1) // batch_size + 1),
            np.array_split(sigma_sq_2, len(sigma_sq_2) // batch_size + 1),
        ):
            distances.append(distance_func(
                mu_1_batch, mu_2_batch, sigma_sq_1_batch, sigma_sq_2_batch))

        return np.concatenate(distances)
    return wrapped_distance


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
