from typing import List
from abc import ABC, abstractmethod

import numpy as np


class Fitter(ABC):
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X)

    def error(self, X, y, sample_weight=None):
        # Error is chosen so that it's minimized by the weighted mean of y
        err = y - self.predict(X)
        errsq = err**2
        if sample_weight is not None:
            errsq *= sample_weight
        return np.nansum(errsq)


class AverageFitter(Fitter):
    def __init__(self):
        self.avg = None

    def fit(self, X, y, sample_weight=None):
        y = np.array(y)
        sample_weight = np.array(sample_weight)
        if sample_weight is None:
            self.avg = np.nanmean(y)
        else:
            self.avg = np.nansum(y * sample_weight) / np.nansum(sample_weight)

    def predict(self, X):
        return np.full(X.shape[0], self.avg)
