import numpy as np


def weighted_quantiles(values, quantiles, sample_weight):
    """Compute the weighted quantile of a 1D numpy array."""
    sorter = np.argsort(values)
    sorted_values = np.array(values)[sorter]
    sorted_weights = np.array(sample_weight)[sorter]
    w_quantiles = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    w_quantiles /= np.sum(sorted_weights)
    return np.interp(quantiles, w_quantiles, sorted_values)
