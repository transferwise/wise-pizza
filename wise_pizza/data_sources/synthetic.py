from typing import Union, Dict

import pandas as pd
import numpy as np

from wise_pizza.segment_data import SegmentData

np.random.seed(42)


def synthetic_data(num_dims: int = 5, dim_values: int = 5, init_len=10000) -> SegmentData:
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


def synthetic_ts_data(num_dims: int = 5, dim_values: int = 5, init_len=10000, ts_len: int = 12):
    pre_data = synthetic_data(num_dims, dim_values, init_len)
    months = pd.date_range(start="2023-01-01", periods=ts_len, freq="MS")
    pre_data.data["TIME"] = np.random.choice(list(months), len(pre_data.data))
    pre_data.time_col = "TIME"
    # TODO: insert time patterns to be detected

    return pre_data
