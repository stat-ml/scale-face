import numpy as np

from .distance_uncertainty_funcs import l2_normalize


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

