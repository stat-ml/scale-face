import numpy as np
import torch
import torch.nn.functional as F


def classifier_to_distance_wrapper(classifier, device=torch.device("cpu")):
    def wrapped_classifier(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
        inputs = torch.cat((torch.from_numpy(mu_1), torch.from_numpy(mu_2)), dim=1)
        probes = F.softmax(classifier(feature=inputs.to(device), dim=1)["pair_classifiers_output"], dim=-1)
        probes = probes.cpu().detach().numpy()
        return probes[:, 1]
    return wrapped_classifier


def classifier_to_uncertainty_wrapper(classifier, device=torch.device("cpu")):
    def wrapped_classifier(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
        inputs = torch.cat((torch.from_numpy(mu_1), torch.from_numpy(mu_2)), dim=1)
        probes = F.softmax(classifier(feature=inputs.to(device))["pair_classifiers_output"], dim=-1)
        probes = probes.cpu().detach().numpy()
        return 1 - probes.max(axis=1)
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
