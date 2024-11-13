from typing import List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


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
        # debug_plot(X, y, self.predict(X), sample_weight)
        err = y - self.predict(X)
        errsq = err**2
        if sample_weight is not None:
            errsq *= sample_weight
        return np.nansum(errsq)


def debug_plot(X, y, y_pred, w):
    X["y_totals"] = y * w
    X["y_pred"] = y_pred * w
    X["weights"] = w
    X_agg = X[["y_totals", "y_pred", "weights", "__time"]].groupby("__time").sum()
    import matplotlib.pyplot as plt

    plt.plot(X_agg["y_totals"] / X_agg["weights"], label="y_totals")
    plt.plot(X_agg["y_pred"] / X_agg["weights"], label="y_pred")
    plt.legend()
    plt.show()
    print("yay!")


class TimeFitterModel(ABC):
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X)


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


class TimeFitter(Fitter):
    def __init__(
        self, dims: List[str], time_col: str, time_fitter_model: TimeFitterModel
    ):
        self.dims = dims
        self.time_col = time_col
        self.time_fitter = time_fitter_model
        self.time_df = None

    def fit(self, X, y, sample_weight=None):
        X = X.copy()
        X["weights"] = sample_weight
        X["totals"] = y * sample_weight
        self.time_df = (
            X[["weights", "totals", self.time_col]]
            .groupby(self.time_col, as_index=False)
            .sum()
        )
        self.time_df["avg_profile"] = self.time_df["totals"] / self.time_df["weights"]
        self.time_fitter.fit(
            self.time_df[[self.time_col]],
            self.time_df["avg_profile"],
            self.time_df["weights"],
        )

    def predict(self, X):
        # predict straight away on the big table, it's row-wise anyway
        return self.time_fitter.predict(X[[self.time_col] + self.dims])


class TimeFitterLinearModel(TimeFitterModel):
    def __init__(self, basis: pd.DataFrame, time_col: str):
        self.basis = basis
        self.time_col = time_col
        self.reg = None

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        assert self.time_col in X.columns
        X["target"] = y
        X["weights"] = sample_weight
        this_basis = pd.merge(
            X[[self.time_col, "target", "weights"]],
            self.basis,
            left_on=self.time_col,
            right_index=True,
        )
        self.reg = Lasso(alpha=1e-6).fit(
            this_basis[self.basis.columns],
            this_basis["target"],
            sample_weight=None if sample_weight is None else this_basis["weights"],
        )
        ## testing code begins
        # self.prediction = self.reg.predict(this_basis[self.basis.columns])
        # test = pd.DataFrame(
        #     {
        #         "time": this_basis[self.time_col],
        #         "target": this_basis["target"],
        #         "prediction": self.prediction,
        #     }
        # )
        ## testing code ends
        print("yay!")

    def predict(self, X: pd.DataFrame):
        assert self.time_col in X.columns
        this_basis = pd.merge(
            X,
            self.basis,
            left_on=self.time_col,
            right_index=True,
        )
        this_basis = this_basis.sort_values(list(X.columns))
        return self.reg.predict(this_basis[self.basis.columns])
