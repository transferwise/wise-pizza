from typing import Union, Dict

import pandas as pd
import numpy as np

from wise_pizza.segment_data import SegmentData
from wise_pizza.time import create_time_basis

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
    pre_data = synthetic_data(num_dims, dim_values, int(init_len/ts_len))
    small_df = pre_data.data
    dfs = []
    months = np.array(pd.date_range(start="2023-01-01", periods=ts_len, freq="MS"))

    for m in months:
        this_df = small_df.copy()
        this_df["TIME"] = m
        this_df["totals"] = np.random.lognormal(0, 1, size=len(this_df))
        dfs.append(this_df)

    df =  pd.concat(dfs)
    pre_data.time_col = "TIME"

    # Add some big trends to the data\
    # TODO: insert trend break patterns too
    basis = create_time_basis(months, baseline_dims=1)
    joined = pd.merge(df, basis, left_on="TIME", right_index=True)

    loc1 = (df["dim0"] == 0) & (df["dim1"] == 1)
    loc2 = (df["dim1"] == 0) & (df["dim2"] == 1)

    df.loc[loc1, "totals"] += 100 * joined.loc[loc1, "Slope"]
    df.loc[loc2, "totals"] += 300 * joined.loc[loc2, "Slope"]

    pre_data.data = df.sort_values(pre_data.dimensions + [pre_data.time_col])
    return pre_data
