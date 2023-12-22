from typing import Union, List

import numpy as np
import pandas as pd


def create_time_basis(
    time_values: Union[pd.DataFrame, np.ndarray], include_breaks: bool = False, baseline_dims: int = 1
):
    if baseline_dims != 1:
        raise NotImplementedError
    if isinstance(time_values, pd.DataFrame):
        time_values = time_values.values

    t = np.sort(np.unique(time_values))
    const = np.ones(len(t))
    linear = np.cumsum(const)
    linear -= linear.mean()  # now orthogonal to const
    col_names = ["Flat", "Slope"]

    dummies = [const/1e5, linear]

    if include_breaks:
        for i in range(1, len(t)):
            dummy = np.ones(len(t))
            dummy[:i] = 0
            dummies.append(dummy - dummy.mean())
            col_names.append(f"{t[i].astype('datetime64[M]').astype(str)}_step")
            cum_dummy = np.cumsum(dummy)
            dummies.append(cum_dummy - cum_dummy.mean())
            col_names.append(f"{t[i].astype('datetime64[M]').astype(str)}_dtrend")

    dummies = np.stack(dummies)
    out = pd.DataFrame(index=t, columns=col_names, data=dummies.T)
    return out


def average_over_time(
    df: pd.DataFrame, dims: List[str], total_name: str, size_name: str, time_name: str) -> pd.DataFrame:
    avgs = df[dims + [total_name, size_name]].groupby(dims, as_index=False).sum()

    avgs["avg"] = avgs[total_name] / avgs[size_name]
    joined = pd.merge(df, avgs[dims + ["avg"]], on=dims)
    joined["total_adjustment"] = joined[size_name] * joined["avg"]
    out = joined[dims + [total_name, size_name, time_name, "total_adjustment"]]
    tmp = out[dims + [total_name, "total_adjustment"]].groupby(dims).sum()
    assert (tmp[total_name]-tmp["total_adjustment"]).abs().sum() < 1e-6 * df[total_name].abs().max()
    return out
