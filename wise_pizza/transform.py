from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class TransformWithWeights(ABC):
    @abstractmethod
    def transform_mean(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform_mean(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform_weight(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform_weight(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    def transform_totals_weights(self, t: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = t / w
        t_mean = self.transform_mean(mean)
        t_w = self.transform_weight(w, mean)
        return t_mean * t_w, t_w

    def inverse_transform_totals_weights(self, t_total: np.ndarray, t_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t_mean = t_total / t_w
        mean = self.inverse_transform_mean(t_mean)
        w = self.inverse_transform_weight(t_w, t_mean)
        return mean * w, w


class IdentityTransform(TransformWithWeights):
    def transform_mean(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform_mean(self, x: np.ndarray) -> np.ndarray:
        return x

    def transform_weight(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w

    def inverse_transform_weight(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w


class LogTransform(TransformWithWeights):
    def __init__(self, offset: float, weight_pow_sc: float = 0.1):
        self.offset = offset
        self.weight_pow_sc = weight_pow_sc

    def transform_mean(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.offset + x)

    def inverse_transform_mean(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.exp(x) - self.offset)

    def transform_weight(self, w: np.ndarray, t_mean: np.ndarray) -> np.ndarray:
        # pure math would give weight_pow_sc = 1, but then
        # there's too much information from actuals being leaked into the weights,
        # so the
        return w * np.power(self.offset + t_mean, self.weight_pow_sc)

    def inverse_transform_weight(self, t_w: np.ndarray, t_mean: np.ndarray) -> np.ndarray:
        return t_w / np.power(self.offset + self.inverse_transform_mean(t_mean), self.weight_pow_sc)
