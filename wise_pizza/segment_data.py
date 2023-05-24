from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class SegmentData:
    data: pd.DataFrame
    dimensions: List[str]
    segment_total: str
    segment_size: Optional[str] = None
    segment_std: Optional[str] = None

    def mean(self):
        total = self.data[self.segment_total].sum()
        if self.segment_size is None:
            weight = len(self.data)
        else:
            weight = self.data[self.segment_size].sum()

        return total / weight
