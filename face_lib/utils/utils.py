def harmonic_mean(x, axis: int = -1):
    x_sum = ((x ** (-1)).mean(axis=axis)) ** (-1)
    return x_sum
