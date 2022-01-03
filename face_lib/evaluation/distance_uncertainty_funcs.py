import numpy as np


def harmonic_mean(x, axis: int = -1):
    x_sum = ((x ** (-1)).mean(axis=axis)) ** (-1)
    return x_sum


def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis, keepdims=True))
    return x


def pair_euc_score(x1, x2, unc1=None, unc2=None):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def pair_cosine_score(x1, x2, unc1=None, unc2=None):
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


def pair_uncertainty_mul(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return sigma_sq_1.prod(axis=1) * sigma_sq_2.prod(axis=1)


def pair_uncertainty_harmonic_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) + harmonic_mean(sigma_sq_2, axis=1)


def pair_uncertainty_harmonic_mul(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) * harmonic_mean(sigma_sq_2, axis=1)

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
