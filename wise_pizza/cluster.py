import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score


def guided_kmeans(X: np.ndarray, power_transform: bool = True) -> np.ndarray:
    """
    Cluster segment averages to calculate aggregated segments
    @param X: Segment mean minus global mean, for each dimension value
    @param power_transform: Do we power transform before clustering
    @return: cluster labels and the transformed values
    """
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X = X.values

    if power_transform:
        if len(X[X > 0] > 1):
            X[X > 0] = PowerTransformer(standardize=False).fit_transform(X[X > 0].reshape(-1, 1)).reshape(-1)
        if len(X[X < 0] > 1):
            X[X < 0] = -PowerTransformer(standardize=False).fit_transform(-X[X < 0].reshape(-1, 1)).reshape(-1)

    best_score = -1
    best_labels = None
    # If we allow 2 clusters, it almost always just splits positive vs negative - boring!
    for n_clusters in range(3, int(len(X) / 2) + 1):
        cluster_labels = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10).fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        # print(n_clusters, score)
        if score > best_score:
            best_score = score
            best_labels = cluster_labels
            best_n = n_clusters

    # print(best_n)
    return best_labels, X


def to_matrix(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((len(labels), len(labels.unique())))
    for i in labels.unique():
        out[labels == i, i] = 1.0
    return out
