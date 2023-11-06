from typing import Union

import numpy as np
import pandas as pd


def create_time_basis(time_values: Union[pd.DataFrame, np.ndarray], baseline_dims):
    if isinstance(time_values, pd.DataFrame):
        time_values = time_values.values

    t = np.sort(np.unique(time_values))
    const = np.ones(len(t))
    linear = np.cumsum(const)
    linear -= linear.mean()  # now orthogonal to const
    basis = np.stack([const, linear])
    col_names = ["Intercept", "Slope"]

    dummies = [
        const,
    ]
    for i in range(1, len(t)):
        dummy = np.ones(len(t))
        dummy[:i] = 0
        dummies.append(dummy)
        col_names.append(f"{t[i]}_step")
        cum_dummy = np.cumsum(dummy)
        dummies.append(cum_dummy)
        col_names.append(f"{t[i]}_dtrend")

    dummies = np.stack(dummies)
    # TODO: make the baseline_dims: vectors orthogonal to the first baseline_dims ones
    out = pd.DataFrame(index=t, columns=col_names, data=np.concatenate([basis, dummies]))
    return out


def strip_out_baseline(df: pd.DataFrame, basis: pd.DataFrame, vectors_to_strip: int = 1) -> pd.DataFrame:
    # TODO: implement
    return df
