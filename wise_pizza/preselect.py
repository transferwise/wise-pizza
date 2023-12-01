import pandas as pd
from scipy.sparse import csc_matrix, hstack
from typing import Optional
import numpy as np


class HeuristicSelector:
    def __init__(self, weights:np.ndarray, totals:np.ndarray, max_cols: int = 300, time_basis: Optional[pd.DataFrame]=None):
        self.max_cols = max_cols
        self.col_defs = []
        self.weights = weights
        self.totals = totals
        self.time_basis = time_basis
        self.X = None

    def __call__(self, nextX, col_defs):
        assert len(col_defs) == nextX.shape[1]

        self.col_defs += col_defs
        self.X = nextX if self.X is None else hstack([self.X, nextX])
        assert len(self.col_defs) == self.X.shape[1]

        w = self.weights.reshape(-1,1)
        w=w*w
        X = self.X.toarray()
        WX = w * X
        y = self.totals.reshape(-1, 1)
        if self.X.shape[1] > self.max_cols:
            chunk_size = int(2*self.max_cols / 3)
            # Do a weighted regression **on each col of X separately**
            XtWy = WX.T.dot(y).T
            XtWX = (WX * X).sum(axis=0, keepdims=True)
            coeffs = XtWy/XtWX
            err = coeffs*X - y
            sigmasq = (err*w*err).sum(axis=0, keepdims=True)
            stds = np.sqrt(sigmasq/XtWX)

            # One way of choosing "good" candidates is individually good t-values
            tvalues = coeffs/stds
            # Another is the total absolute difference in totals that this regressor would predict alone
            impact = np.abs(WX).sum(axis=0, keepdims=True)*coeffs

            inds = []

            unusual = np.argsort(np.abs(tvalues.reshape(-1)))
            inds += list(unusual[-chunk_size:])

            unusual2 = np.argsort(np.abs(impact.reshape(-1)))
            inds += list(unusual2[-chunk_size:])

            best = np.array(list(set(inds)))

            self.X = self.X[:, best]
            self.col_defs = [self.col_defs[i] for i in best]
            assert len(self.col_defs) == self.X.shape[1]
            # end naive pre-filter

        return self.X, self.col_defs

