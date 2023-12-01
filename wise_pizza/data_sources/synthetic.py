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

    # deduplicate dimension values
    df = pd.DataFrame(cols).groupby(dims, as_index=False).sum().reset_index(drop=True)
    return SegmentData(data=df, dimensions=dims, segment_total="totals")


def synthetic_ts_data(num_dims: int = 5, dim_values: int = 5, init_len=10000, ts_len: int = 12):
    pre_data = synthetic_data(num_dims, dim_values, int(init_len/ts_len))
    small_df = pre_data.data
    dfs = []
    months = np.array(pd.date_range(start="2023-01-01", periods=ts_len, freq="MS"))

    for m in months:
        this_df = small_df.copy()
        this_df["TIME"] = m
        this_df["totals"] = np.random.lognormal(0, 1, size=len(this_df))
        dfs.append(this_df)

    df = pd.concat(dfs)
    pre_data.time_col = "TIME"

    pre_data.data = df.sort_values(pre_data.dimensions + [pre_data.time_col]).reset_index(drop=True)
    return pre_data
