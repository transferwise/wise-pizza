from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


class SliceFinderPlottingInterface(ABC):
    @property
    @abstractmethod
    def actual_totals(self):
        pass

    @property
    @abstractmethod
    def predicted_totals(self):
        pass

    @property
    @abstractmethod
    def weights(self):
        pass

    @property
    @abstractmethod
    def time(self):
        pass

    @property
    @abstractmethod
    def segments(self):
        pass

    @abstractmethod
    def segment_impact_on_totals(self, s: Dict) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def total_name(self):
        pass

    @property
    @abstractmethod
    def size_name(self):
        pass

    @property
    @abstractmethod
    def predict(
        self,
        steps: Optional[int] = None,
        basis: Optional[pd.DataFrame] = None,
        weight_df: Optional[pd.DataFrame] = None,
    ):
        pass
