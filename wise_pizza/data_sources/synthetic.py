from typing import Union, Dict

import pandas as pd
import numpy as np

from wise_pizza.segment_data import SegmentData

np.random.seed(42)


def synthetic_data(num_dims: int = 5, dim_values: int = 5, init_len=10000):
    np.random.seed(42)
    cols = {}
    for dim in range(num_dims):
        cols[f"dim{dim}"] = np.random.choice(dim_values, size=init_len)

    cols["totals"] = np.random.lognormal(0, 1, size=init_len)
    dims = [k for k in cols.keys() if "dim" in k]

    df = pd.DataFrame(cols).groupby(dims, as_index=False).sum()

    df.loc[(df["dim0"] == 0) & (df["dim1"] == 1), "totals"] += 100
    df.loc[(df["dim1"] == 0) & (df["dim2"] == 1), "totals"] += 300

    return SegmentData(data=df, dimensions=dims, segment_total="totals")
