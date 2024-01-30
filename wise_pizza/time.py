from typing import Union, List, Tuple
import datetime

import numpy as np
import pandas as pd


def create_time_basis(
    time_values: Union[pd.DataFrame, np.ndarray],
    include_breaks: int = 0,
    baseline_dims: int = 1,
):
    if baseline_dims != 1:
        raise NotImplementedError
    if isinstance(time_values, pd.DataFrame):
        time_values = time_values.values

    t = np.sort(np.unique(time_values))
    const = np.ones(len(t))
    linear = np.cumsum(const)
    linear -= linear.mean()  # now orthogonal to const
    col_names = ["Slope"]

    dummies = [linear]

    if include_breaks:
        for i in range(1, len(t)):
            dummy = np.ones(len(t))
            dummy[:i] = 0
            dummies.append(dummy - dummy.mean())
            # TODO: force date format
            col_names.append(f"{t[i]}_step")
            cum_dummy = np.cumsum(dummy)
            dummies.append(cum_dummy - cum_dummy.mean())
            col_names.append(f"{t[i]}_dtrend")

    dummies = np.stack(dummies)
    out = pd.DataFrame(index=t, columns=col_names, data=dummies.T)
    return out


def extend_dataframe(df: pd.DataFrame, N: int) -> pd.DataFrame:
    df_extended = df.copy()

    # Try to infer the frequency from the original index
    freq = pd.infer_freq(df.index)

    # Check the type of the original index
    index_type = df.index[0].__class__

    for _ in range(N):
        diff = df_extended.iloc[-1] - df_extended.iloc[-2]
        new_row = df_extended.iloc[-1] + diff

        # If the frequency could not be inferred, use the difference of the last two index values
        if freq is None:
            offset = df_extended.index[-1] - df_extended.index[-2]
            new_date = df_extended.index[-1] + offset
        else:
            new_date = df_extended.index[-1] + pd.tseries.frequencies.to_offset(freq)

        # If the original index was of type date, convert the new date to date
        if index_type == datetime.date:
            new_date = new_date.date()

        df_extended.loc[new_date] = new_row

    return df_extended


def add_average_over_time(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: str,
    time_name: str,
    cartesian: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    avgs = df[dims + [total_name, size_name]].groupby(dims, as_index=False).sum()
    avgs["avg"] = avgs[total_name] / avgs[size_name]
    if cartesian:
        # make sure that the cartesian product of dimension combinations x time is present,
        # without changing the totals
        times = df[[time_name]].groupby(time_name, as_index=False).sum()
        times["key"] = 1
        avgs["key"] = 1
        cartesian_df = pd.merge(avgs, times, on="key").drop(columns=["key"])
        joined = pd.merge(
            df,
            cartesian_df[dims + [time_name]],
            on=dims + [time_name],
            how="right",
        )
        joined[size_name] = joined[size_name].fillna(
            np.nanmean(joined[size_name].values)
        )
        joined[total_name] = joined[total_name].fillna(0.0)
        df = joined

    avgs = df[dims + [total_name, size_name]].groupby(dims, as_index=False).sum()
    avgs["avg"] = avgs[total_name] / avgs[size_name]
    joined = pd.merge(df, avgs[dims + ["avg"]], on=dims)

    joined["total_adjustment"] = joined[size_name] * joined["avg"]
    out = joined[dims + [total_name, size_name, time_name, "total_adjustment"]]
    tmp = out[dims + [total_name, "total_adjustment"]].groupby(dims).sum()
    assert (tmp[total_name] - tmp["total_adjustment"]).abs().sum() < 1e-6 * df[
        total_name
    ].abs().max()
    return out, avgs[dims + ["avg"]]
