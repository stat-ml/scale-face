import torch


def harmonic_mean(x: torch.Tensor, dim: int = -1):
    x_sum = ((x ** (-1)).sum(dim=dim) / x.shape[-1]) ** -1
    return x_sum
