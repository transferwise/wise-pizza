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
    pre_data = synthetic_data(num_dims, dim_values, init_len)
    df = pre_data.data
    months = np.array(pd.date_range(start="2023-01-01", periods=ts_len, freq="MS"))
    df["TIME"] = np.random.choice(list(months), len(pre_data.data))
    pre_data.time_col = "TIME"

    basis = create_time_basis(months, baseline_dims=1)

    joined = pd.merge(df, basis, left_on="TIME", right_index=True)
    loc1 = (df["dim0"] == 0) & (df["dim1"] == 1)
    loc2 = (df["dim1"] == 0) & (df["dim2"] == 1)

    # TODO: insert trend break patterns too
    df.loc[loc1, "totals"] += 100 * joined.loc[loc1, "Slope"]
    df.loc[loc2, "totals"] += 300 * joined.loc[loc2, "Slope"]

    pre_data.df = df.sort_values(pre_data.dimensions + [pre_data.time_col])
    return pre_data
