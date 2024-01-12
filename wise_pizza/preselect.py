import pandas as pd
from scipy.sparse import csc_matrix, hstack
from typing import Optional
import numpy as np
# import numba

class HeuristicSelector:
    def __init__(
        self,
        weights: np.ndarray,
        totals: np.ndarray,
        max_cols: int = 300,
        time_basis: Optional[pd.DataFrame] = None,
        verbose: bool = False,
    ):
        self.max_cols = max_cols
        self.col_defs = []
        self.weights = weights.reshape(-1, 1)
        self.totals = totals.reshape(-1, 1)
        self.time_basis = time_basis
        self.tvalues = np.array([])
        self.impact = np.array([])
        self.X = None
        self.verbose = verbose

    def __call__(self, nextX, col_defs):
        if self.verbose:
            print(f"Preprocessor received X: {nextX.shape[0]}x{nextX.shape[1]}")
        assert len(col_defs) == nextX.shape[1]

        tvalues, impact = get_metrics_new(nextX, self.totals, self.weights)

        self.tvalues = np.concatenate([self.tvalues, tvalues])
        self.impact = np.concatenate([self.impact, impact])
        self.col_defs += col_defs
        self.X = nextX if self.X is None else hstack([self.X, nextX])

        assert len(self.col_defs) == self.X.shape[1]

        if self.X.shape[1] > self.max_cols:
            chunk_size = int(2 * self.max_cols / 3)

            inds = []

            unusual = np.argsort(np.abs(self.tvalues))
            inds += list(unusual[-chunk_size:])

            unusual2 = np.argsort(np.abs(self.impact))
            inds += list(unusual2[-chunk_size:])

            best = np.array(list(set(inds)))

            self.tvalues = np.array(self.tvalues[best])
            self.impact = np.array(self.impact[best])
            self.X = csc_matrix(self.X[:, best])
            self.col_defs = [self.col_defs[i] for i in best]
            assert len(self.col_defs) == self.X.shape[1]
            del nextX, col_defs
        if self.verbose:
            print(f"Done preprocessing that one!")
        return self.X, self.col_defs



def get_metrics_new(X, y, w):
    # y is totals, so we're approximating y by WX, unweighted
    X = X.toarray()
    WX = w * X

    # Do a weighted regression **on each col of X separately**
    XtWy = WX.T.dot(y).T
    XtWX = (WX * WX).sum(axis=0, keepdims=True)
    coeffs = XtWy / XtWX
    sigmasq=get_sigmasq(coeffs, WX, y)
    stds = np.sqrt(sigmasq / XtWX)

    # One way of choosing "good" candidates is individually good t-values
    tvalues = coeffs / stds
    # Another is the total absolute difference in totals that this regressor would predict alone
    impact = np.abs(WX).sum(axis=0, keepdims=True) * coeffs
    return tvalues.reshape(-1), impact.reshape(-1)

# This function for some reason is by far the slowest part of the code
# JIT didn't seem to help as all the calc is in the numpy binaries anyway
# @numba.jit(nopython=True)  # Enable the JIT compiler with nopython mode for best performance
def get_sigmasq(coeffs: np.ndarray, WX: np.ndarray, y: np.ndarray) -> np.ndarray:
    err = coeffs * WX - y
    sigmasq = (err * err).sum(axis=0)
    return sigmasq.reshape(1,-1)

def get_metrics(X, y, w):
    w2 = w * w
    X = X.toarray()
    W2X = w2 * X
    # Do a weighted regression **on each col of X separately**
    XtWy = W2X.T.dot(y).T
    XtWX = (W2X * X).sum(axis=0, keepdims=True)
    coeffs = XtWy / XtWX
    err = coeffs * X - y
    sigmasq = (err * w * err).sum(axis=0, keepdims=True)
    stds = np.sqrt(sigmasq / XtWX)

    # One way of choosing "good" candidates is individually good t-values
    tvalues = coeffs / stds
    # Another is the total absolute difference in totals that this regressor would predict alone
    impact = np.abs(W2X).sum(axis=0, keepdims=True) * coeffs
    return tvalues.reshape(-1), impact.reshape(-1)