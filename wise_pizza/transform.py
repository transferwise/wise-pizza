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
    def __init__(self, offset: float):
        self.offset = offset

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.offset + x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.exp(x)-self.offset)

    def weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w*(self.offset+x)

    def inverse_weight_transform(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        return w/(self.offset+self.inverse_transform(x))