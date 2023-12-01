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
    col_names = ["Intercept", "Slope"]

    dummies = [const, linear]

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


def strip_out_baseline(
    df: pd.DataFrame, dims: List[str], total_name: str, size_name: str, time_name: str,  basis: pd.DataFrame, vectors_to_strip: int = 1
) -> pd.DataFrame:
    if vectors_to_strip != 1:
        raise NotImplementedError
        # TODO: implement stripping out larger baselines, eg intercept + trend

    avgs = df[dims + [total_name, size_name]].groupby(dims, as_index=False).sum()
    avgs["avg"] = avgs[total_name] / avgs[size_name]
    joined = pd.merge(df, avgs[dims + ["avg"]], on=dims)
    joined[total_name] -= joined[size_name] * joined["avg"]
    out = joined[dims + [total_name, size_name, time_name]]
    tmp = out.groupby(dims).sum()
    assert tmp[total_name].abs().sum() < 1e-6 * df[total_name].abs().max()
    return out
