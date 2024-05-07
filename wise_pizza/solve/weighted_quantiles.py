import numpy as np


def weighted_quantiles(values, quantiles, sample_weight):
    """Compute the weighted quantile of a 1D numpy array."""
    values_ = np.array(values)
    sample_weight_ = np.array(sample_weight)
    nice = ~np.isnan(values) & ~np.isnan(sample_weight)
    if np.any(~nice):
        raise ValueError("Data contains NaNs")
    sorter = np.argsort(values_)
    sorted_values = values_[sorter]
    sorted_weights = sample_weight_[sorter]
    w_quantiles = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    w_quantiles /= np.sum(sorted_weights)

    try:
        return np.interp(quantiles, w_quantiles, sorted_values)
    except Exception as e:
        print(e)
        raise e
