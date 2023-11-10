from scipy.sparse import csc_matrix, hstack
import numpy as np


class HeuristicSelector:
    def __init__(self, weights:np.ndarray, totals:np.ndarray, max_cols: int = 300):
        self.max_cols = max_cols
        self.col_defs = []
        self.weights = weights
        self.totals = totals
        self.X = None

    def __call__(self, X, col_defs):
        self.col_defs += col_defs
        self.X = X if self.X is None else hstack([self.X, X])
        # TODO: make this stateful and incremental
        if self.X.shape[1] > self.max_cols:
            chunk_size = int(self.max_cols / 2)

            # TODO: filter by t-values instead of absolute discrepancies
            seg_wgt = np.abs(self.X).T @ self.weights
            seg_avg = (self.X.T @ self.totals) / seg_wgt
            avg = self.totals.sum() / self.weights.sum()

            inds = []

            delta = seg_avg - avg
            unusual = np.argsort(np.abs(delta))
            inds += list(unusual[-chunk_size:])

            delta2 = delta * np.sqrt(seg_wgt)
            unusual2 = np.argsort(np.abs(delta2))
            inds += list(unusual2[-chunk_size:])

            delta3 = delta * seg_wgt
            unusual3 = np.argsort(np.abs(delta3))
            inds += list(unusual3[-chunk_size:])

            best = np.array(list(set(inds)))

            self.X = self.X[:, best]
            self.col_defs = [self.col_defs[i] for i in best]
            # end naive pre-filter

        return self.X, self.col_defs

