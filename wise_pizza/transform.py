from abc import ABC, abstractmethod

import numpy as np


class TransformWithWeights(ABC):
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass


class IdentityTransform(TransformWithWeights):
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w

    def inverse_weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w

class LogTransform(TransformWithWeights):
    def __init__(self, offset: float, weight_pow_sc: float=0.1):
        self.offset = offset
        self.weight_pow_sc = weight_pow_sc

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.offset + x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.exp(x)-self.offset)

    def weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        # pure math would give weight_pow_sc = 1, but then
        # there's too much information from actuals being leaked into the weights,
        # so the
        return w * np.power(self.offset+x, self.weight_pow_sc)

    def inverse_weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w / np.power(self.offset+self.inverse_transform(x), self.weight_pow_sc)