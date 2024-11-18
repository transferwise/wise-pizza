from typing import List

import numpy as np
import pandas as pd


from .weighted_quantiles import weighted_quantiles


def target_encode(df: pd.DataFrame, dim: str) -> dict:
    df = df[[dim, "totals", "weights"]]
    agg = df.groupby(dim, as_index=False).sum()
    agg["__avg"] = agg["totals"] / agg["weights"]
    agg["__avg"] = agg["__avg"].fillna(agg["__avg"].mean())
    enc_map = {k: v for k, v in zip(agg[dim], agg["__avg"])}

    if np.isnan(np.array(list(enc_map.values()))).any():
        raise ValueError("NaNs in encoded values")
    return enc_map


def target_encoding_partitions(df: pd.DataFrame, dim: str, num_bins: int):
    enc_map = target_encode(df, dim)
    df[dim + "_encoded"] = df[dim].apply(lambda x: enc_map[x])
    if np.any(np.isnan(df[dim + "_encoded"])):  # pragma: no cover
        raise ValueError("NaNs in encoded values")
    # Get split candidates for brute force search
    deciles = np.array([q / num_bins for q in range(1, num_bins)])

    splits = weighted_quantiles(df[dim + "_encoded"], deciles, df["weights"])

    partitions = []
    for split in np.unique(splits):
        left = df[df[dim + "_encoded"] < split]
        right = df[df[dim + "_encoded"] >= split]
        if len(left) == 0 or len(right) == 0:
            continue
        dim_values1 = [k for k, v in enc_map.items() if v < split]
        dim_values2 = [k for k, v in enc_map.items() if v >= split]
        partitions.append((dim_values1, dim_values2))

    return partitions


def kmeans_partition(df: pd.DataFrame, dim: str, groupby_dims: List[str]):
    assert len(df[dim].unique()) >= 3
    # Get split candidates
    agg_df = df.groupby([dim] + groupby_dims, as_index=False).sum()
    agg_df["__avg"] = agg_df["totals"] / agg_df["weights"]
    pivot_df = agg_df.pivot(
        index=groupby_dims, columns=dim, values="__avg"
    ).reset_index()
    value_cols = [c for c in pivot_df.columns if c not in groupby_dims]

    if len(groupby_dims) == 2:
        nice_mats = {}
        for chunk in ["Average", "Weights"]:
            this_df = pivot_df[pivot_df["chunk"] == chunk]
            nice_values = fill_gaps(this_df[value_cols].values)
            if chunk == "Weights":
                nice_values = (
                    np.mean(nice_mats["Average"])
                    * nice_values
                    / np.sum(nice_values, axis=0, keepdims=True)
                )
            nice_mats[chunk] = nice_values
        joint_mat = np.concatenate([nice_mats["Average"], nice_mats["Weights"]], axis=0)
    else:
        joint_mat = fill_gaps(pivot_df[value_cols].values)

    weights = pivot_df[value_cols].T.sum(axis=1)
    vector_dict = {}
    for i, c in enumerate(value_cols):
        vector_dict[c] = (weights.loc[c], joint_mat[:, i])

    cluster1, cluster2 = weighted_kmeans_two_clusters(vector_dict)
    if cluster1 is None:
        return []
    else:
        return [(cluster1, cluster2)]


def weighted_kmeans_two_clusters(data_dict, tol=1e-4, max_iter=100, max_retries=10):
    keys = list(data_dict.keys())
    weights = np.array([data_dict[key][0] for key in keys])
    data = np.array([data_dict[key][1] for key in keys])

    rng = np.random.default_rng()

    for retry in range(max_retries):
        # Initialize centroids by randomly choosing two data points
        centroids = data[rng.choice(len(data), size=2, replace=False)]

        for iteration in range(max_iter):
            # Compute weighted distances to each centroid
            distances = np.array(
                [np.linalg.norm(data - centroid, axis=1) for centroid in centroids]
            )

            # Assign points to the closest centroid
            labels = np.argmin(distances, axis=0)

            # Check if any cluster is empty
            if not np.any(labels == 0) or not np.any(labels == 1):
                # If a cluster is empty, reinitialize centroids and restart
                print(
                    f"Empty cluster detected on retry {retry + 1}, reinitializing centroids."
                )
                break

            # Update centroids with weighted averages
            new_centroids = np.array(
                [
                    np.average(data[labels == i], axis=0, weights=weights[labels == i])
                    for i in range(2)
                ]
            )

            # Check for convergence
            if np.linalg.norm(new_centroids - centroids) < tol:
                # Successful clustering with no empty clusters
                centroids = new_centroids
                return (
                    [keys[i] for i in range(len(keys)) if labels[i] == 0],
                    [keys[i] for i in range(len(keys)) if labels[i] == 1],
                )

            centroids = new_centroids

    return None, None


def fill_gaps(x: np.ndarray, num_iter=50):
    nans = np.isnan(x)
    # calculate the marginal, fill the gaps, use that to interpolate individual columns

    est = x
    for _ in range(num_iter):
        marg = np.nanmean(est, axis=1)
        nice_marg = interpolate_and_extrapolate(marg)
        tile_marg = np.tile(nice_marg, (x.shape[1], 1)).T
        tile_marg[nans] = np.nan
        reg = np.nanmedian(x) * 1e-6
        coeffs = (np.nansum(x * tile_marg, axis=0) + reg) / (
            np.nansum(tile_marg * tile_marg, axis=0) + reg
        )
        interp = coeffs[None, :] * nice_marg[:, None]
        est[nans] = interp[nans]
    return x


def interpolate_and_extrapolate(arr: np.ndarray) -> np.ndarray:
    # Check if input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    # Find indices of valid (non-NaN) and NaN values
    nans = np.isnan(arr)
    not_nans = ~nans

    # If there are no NaNs, return the array as is
    if not nans.any():
        return arr

    # Perform linear interpolation for NaNs within valid values
    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), arr[not_nans])

    # Perform constant extrapolation for edges
    if nans[0]:  # If the first values are NaNs, fill with the first non-NaN value
        arr[: np.flatnonzero(not_nans)[0]] = arr[not_nans][0]
    if nans[-1]:  # If the last values are NaNs, fill with the last non-NaN value
        arr[np.flatnonzero(not_nans)[-1] + 1 :] = arr[not_nans][-1]

    return arr
