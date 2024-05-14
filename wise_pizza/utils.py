import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from wise_pizza.segment_data import SegmentData


def diff_dataset(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    dims: List[str],
    totals: str,
    weights: Optional[str] = None,
    split_deltas: bool = False,
    return_multiple: bool = False,
):
    """
    Create a single dataset that breaks down the change in totals
    to pct change in average and segment size
    @param df_pre: first dataset
    @param df_post: second dataset
    @param dims: List of discrete dimensions
    @param totals: Name of column that contains totals per segment
    @param weights: Name of column containing segment sizes
    @param split_deltas: for calculating differences in sizes and totals
    @param return_multiple: to return multiple prepared datasets
    @return: prepared dataset
    """

    if weights is None:
        split_deltas = False
        weights = "dummy_weight"
        df_pre[weights] = 1.0
        df_post[weights] = 1.0

    avg = "average"

    cols = dims + [totals, weights]

    df_pre = df_pre[cols].groupby(dims, as_index=False).sum()
    df_post = df_post[cols].groupby(dims, as_index=False).sum()

    # create some ratios for pro-rating baselimes
    totals_ratio = df_post[totals].sum() / df_pre[totals].sum()
    weight_ratio = df_post[weights].sum() / df_pre[weights].sum()
    avg_ratio = totals_ratio / weight_ratio

    combined = pd.merge(df_pre, df_post, on=dims, how="outer")

    def rel_error(x, y):
        return abs(x - y) / (abs(x) + abs(y) + 1.0)

    eps = 1e-6
    # use for testing
    # assert rel_error(combined[weights + "_x"].sum(), df_pre[weights].sum()) < eps
    # assert rel_error(combined[weights + "_y"].sum(), df_post[weights].sum()) < eps
    # assert rel_error(combined[totals + "_x"].sum(), df_pre[totals].sum()) < eps
    # assert rel_error(combined[totals + "_y"].sum(), df_post[totals].sum()) < eps

    # clean up the NaNs
    combined[avg + "_x"] = (combined[totals + "_x"] / combined[weights + "_x"]).fillna(
        df_pre[totals].sum() / df_pre[weights].sum()
    )
    combined[avg + "_y"] = (combined[totals + "_y"] / combined[weights + "_y"]).fillna(
        df_post[totals].sum() / df_post[weights].sum()
    )
    combined[weights + "_x"] = combined[weights + "_x"].fillna(0.0)
    combined[weights + "_y"] = combined[weights + "_y"].fillna(0.0)
    combined[totals + "_x"] = combined[totals + "_x"].fillna(0.0)
    combined[totals + "_y"] = combined[totals + "_y"].fillna(0.0)

    combined[weights] = 0.5 * (combined[weights + "_y"] + combined[weights + "_x"])

    # # Baseline prediction for next period is just pro-rated from the totals
    # # Why did I want that in the first place? :)
    # combined["baseline_" + weights] = combined[weights + "_x"] * weight_ratio
    #
    # combined["baseline_" + avg] = combined[avg + "_x"] * avg_ratio
    # combined["baseline_" + totals] = (
    #     combined["baseline_" + weights] * combined["baseline_" + avg]
    # )
    #
    # assert (
    #     rel_error(combined["baseline_" + weights].sum(), combined[weights + "_y"].sum())
    #     < eps
    # )
    #
    # assert (
    #     rel_error(combined["baseline_" + totals].sum(), combined[totals + "_y"].sum())
    #     < eps
    # )

    if split_deltas:  # TODO: add multiplier-based baselines for the diffs
        combined["dweights"] = combined[weights + "_y"] - combined[weights + "_x"]
        combined["davg"] = combined[avg + "_y"] - combined[avg + "_x"]

        # a1 * w1 - a0 * w0 = a1 ( w1 - w0) + w0 ( a1 - a0)
        W0 = combined[weights + "_x"].sum()
        A1 = combined[totals + "_y"].sum()

        # scale the deltas so w and avg deltas are on same scale
        change_from_w = combined["dweights"] * A1
        w_weights = combined[avg + "_y"] / A1

        change_from_avg = combined["davg"] * W0
        avg_weights = combined[weights + "_x"] / W0

        re_change = change_from_w * w_weights + change_from_avg * avg_weights
        combined["dtotals"] = combined[totals + "_y"] - combined[totals + "_x"]

        # assert (combined["dtotals"] - re_change).abs().max() < 0.1

        combined["Change in totals"] = change_from_w * w_weights
        combined[weights] = 1.0  # w_weights

        c2 = combined.copy()
        c2["Change in totals"] = change_from_avg * avg_weights
        c2[weights] = 1.0  # avg_weights

        if return_multiple:
            sd_size = SegmentData(
                combined.rename(
                    columns={"Change in totals": "Change from segment size"}
                ),
                dimensions=dims,
                segment_total="Change from segment size",
                segment_size=weights,
            )
            sd_avg = SegmentData(
                c2.rename(columns={"Change in totals": "Change from segment average"}),
                dimensions=dims,
                segment_total="Change from segment average",
                segment_size=weights,
            )
            return sd_size, sd_avg

        else:
            combined["Change from"] = "Segment size"
            c2["Change from"] = "Segment average"

            df = pd.concat([combined, c2])[
                dims + [weights, "Change in totals", "Change from"]
            ]
            df_change_in_totals = np.array(df["Change in totals"], dtype=np.longdouble)
            combined_dtotals = np.array(combined["dtotals"], dtype=np.longdouble)
            df_change_in_totals_sum = np.nansum(df_change_in_totals)
            combined_dtotals_sum = np.nansum(combined_dtotals)
            # if combined_dtotals_sum > 1e-31:
            #     assert rel_error(df_change_in_totals_sum, combined_dtotals_sum) < eps

            return SegmentData(
                df,
                dimensions=dims + ["Change from"],
                segment_total="Change in totals",
                segment_size=weights,
            )

    else:
        combined["Change in totals"] = combined[totals + "_y"] - combined[totals + "_x"]

        # TODO: why does taking prev period as baseline make the fit so slow?
        # baseline hypothesis is proportional increase from first period
        combined[weights] = 1.0  # combined[totals + "_x"]
        combined[weights] = np.maximum(1.0, combined[weights])
        cols = (
            dims
            + ["Change in totals", totals + "_x", totals + "_y"]
            + [c for c in combined.columns if "baseline" in c]
        )

        return SegmentData(
            combined[cols + [weights]],
            dimensions=dims,
            segment_total="Change in totals",
            segment_size=weights,
        )


def prepare_df(
    df: pd.DataFrame,
    dims: str,
    size_name: Optional[str] = None,
    total_name: str = "VOLUME",
    time_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Takes a pandas dataframe and checks for missing values.
    If a column is numeric and contains missing values, replace them with zeros.
    If a column is categorical and contains missing values, replace them with the column name followed by "_unknown"
    Returns a new pandas dataframe.
    @param df: initial dataset
    @param dims: List of discrete dimensions
    @param size_name: Name of column containing segment sizes
    @param total_name: Name of column that contains totals per segment
    @return: prepared dataset
    """

    new_df = df.copy()  # create a copy of the dataframe to avoid modifying the original

    new_df[total_name] = new_df[total_name].apply(float)

    # replace NaN values in numeric columns with zeros
    if size_name is not None:
        new_df[size_name] = new_df[size_name].apply(float)
        new_df[[size_name, total_name]] = new_df[[size_name, total_name]].fillna(0)
        # If weights are zero, totals must be zero in the same row
        new_df.loc[new_df[size_name] == 0, total_name] = 0
    else:
        new_df[[total_name]] = new_df[[total_name]].fillna(0)

    # replace NaN values in categorical columns with the column name + "_unknown"
    object_columns = list(new_df[dims].select_dtypes("object").columns)
    new_df[object_columns] = new_df[object_columns].fillna(
        new_df[object_columns].apply(lambda x: x.name + "_unknown")
    )
    new_df[object_columns] = new_df[object_columns].astype(str)

    # Groupby all relevant dims to decrease the dataframe size, if possible
    group_dims = dims if time_name is None else dims + [time_name]

    if size_name is not None:
        new_df = (
            new_df.groupby(by=group_dims, observed=True)[[total_name, size_name]]
            .sum()
            .reset_index()
        )
    else:
        new_df = (
            new_df.groupby(by=group_dims, observed=True)[[total_name]]
            .sum()
            .reset_index()
        )

    return new_df


#
# def prepare_time_df(
#     df: pd.DataFrame,
#     dims: str,
#     total_name: str = "VOLUME",
#     time_name: str = "TIME",
#     size_name: str = None,
# ) -> pd.DataFrame:
#     """
#     Takes a pandas dataframe and checks for missing values.
#     If a column is numeric and contains missing values, replace them with zeros.
#     If a column is categorical and contains missing values, replace them with the column name followed by "_unknown".
#     Returns a new pandas dataframe.
#     @param df: initial dataset
#     @param dims: List of discrete dimensions
#     @param size_name: Name of column containing segment sizes
#     @param total_name: Name of column that contains totals per segment
#     @return: prepared dataset
#     """
#
#     # do the standard cleaning
#     new_df = prepare_df(df, dims, total_name, size_name)
#
#     new_df[total_name] = new_df[total_name].apply(float)
#     if
#
#     ptotals = (
#         new_df[dims + [time_name, total_name]].pivot(columns=time_name, index=dims, values=[total_name]).fillna(0.0)
#     )
#     if size_name is not None:
#         new_df[size_name] = new_df[size_name].apply(float)
#
#         psize = (
#             new_df[dims + [time_name, size_name]]
#             .pivot(columns=time_name, index=dims, values=[size_name])
#             .apply(float)
#             .fillna(0.0)
#         )
#         ptotals.values[psize.values == 0.0] = 0.0
#
#     # replace NaN values in numeric columns with zeros,
#     # make sure that zero size implies zero total
#     if size_name is not None:
#         new_df[size_name] = new_df[size_name].apply(float)
#         new_df[[size_name, total_name]] = new_df[[size_name, total_name]].fillna(0)
#         new_df.loc[new_df[size_name] == 0, total_name] = 0
#     else:
#         new_df[[total_name]] = new_df[[total_name]].fillna(0)
#
#     # replace NaN values in categorical columns with the column name + "_unknown"
#     object_columns = list(new_df[dims].select_dtypes("object").columns)
#     new_df[object_columns] = new_df[object_columns].fillna(new_df[object_columns].apply(lambda x: x.name + "_unknown"))
#
#     if size_name is not None:
#         new_df = new_df.groupby(by=dims, observed=True)[[total_name, size_name]].sum().reset_index()
#     else:
#         new_df = new_df.groupby(by=dims, observed=True)[[total_name]].sum().reset_index()
#
#     new_df[object_columns] = new_df[object_columns].astype(str)
#
#     return new_df
def almost_equals(x1, x2, eps: float = 1e-6) -> bool:
    return np.sum(np.abs(x1 - x2)) / np.mean(np.abs(x1 + x2)) < eps


def clean_up_min_max(min_nonzeros: int = None, max_nonzeros: int = None):
    if min_nonzeros is not None:
        logging.warning(
            "min_segments parameter is deprecated, please use max_segments instead."
        )
    if max_nonzeros is None:
        if min_nonzeros is None:
            max_nonzeros = 5
            min_nonzeros = 5
        else:
            max_nonzeros = min_nonzeros
    else:
        if min_nonzeros is None:
            min_nonzeros = max_nonzeros

    assert min_nonzeros <= max_nonzeros
    return min_nonzeros, max_nonzeros
