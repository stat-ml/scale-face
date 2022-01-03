from .distance_uncertainty_funcs import (
    l2_normalize,
    pair_euc_score,
    pair_cosine_score,
    pair_MLS_score,
    pair_uncertainty_sum,
    pair_uncertainty_mul,
    pair_uncertainty_harmonic_sum,
    pair_uncertainty_harmonic_mul,
    pair_uncertainty_concatenated_harmonic,
    pair_uncertainty_cosine_analytic,
)

name_to_distance_func = {
    "euc": pair_euc_score,
    "cosine": pair_cosine_score,
    "MLS": pair_MLS_score,
}

name_to_uncertainty_func = {
    "mean": pair_uncertainty_sum,
    "mul": pair_uncertainty_mul,
    "harmonic-sum": pair_uncertainty_harmonic_sum,
    "harmonic-mul": pair_uncertainty_harmonic_mul,
    "harmonic-harmonic": pair_uncertainty_concatenated_harmonic,
    "cosine-analytic": pair_uncertainty_cosine_analytic,
}