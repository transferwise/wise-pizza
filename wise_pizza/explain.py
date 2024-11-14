import copy
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.ma.extras import average


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from wise_pizza.plotting import (
    plot_segments,
    plot_split_segments,
    plot_waterfall,
)
from wise_pizza.plotting_time import plot_time, plot_ts_pair
from wise_pizza.plotting_time_tree import plot_time_from_tree
from wise_pizza.slicer import SliceFinder, SlicerPair
from wise_pizza.slicer_facades import TransformedSliceFinder
from wise_pizza.utils import diff_dataset, prepare_df, almost_equals
from wise_pizza.time import (
    create_time_basis,
    add_average_over_time,
    extend_dataframe,
    prune_time_basis,
)
from wise_pizza.transform import IdentityTransform, LogTransform


def explain_changes_in_average(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: str,
    min_segments: Optional[int] = None,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "lasso",
    how: str = "totals",
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = False,
    verbose: int = 0,
):
    """
    Find segments most useful in explaining the difference between the averages of the two datasets
    @param df1: First dataset
    @param df2: Second dataset
    @param dims: List of discrete dimensions
    @param total_name: Name of column that contains totals per segment
    @param size_name: Name of column containing segment sizes
    @param min_segments: Minimum number of segments to find
    @param max_segments: Maximum number of segments to find, defaults to min_segments
    @param min_depth: Minimum number of dimension to constrain in segment definition
    @param max_depth: Maximum number of dimension to constrain in segment definition
    @param solver: "lasso" for most unusual, possibly overlapping segments;
                   "tree" to divide the whole dataset into non-overlapping segments,
                          as homogenous as possible.
    @param how: "totals" to only decompose segment totals (ignoring size vs average contribution)
            "split_fits" to separately decompose contribution of size changes and average changes
            "extra_dim" to treat size vs average change contribution as an additional dimension
            "force_dim" like extra_dim, but each segment must contain a Change_from constraint
    @param force_add_up: Force the contributions of chosen segments to add up
    to the difference between dataset totals
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
    @param cluster_values: In addition to single-value slices, consider slices that consist of a
    group of segments from the same dimension with similar naive averages
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @return: A fitted object
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # replace NaN values in numeric columns with zeros
    # replace NaN values in categorical columns with the column name + "_unknown"
    df1 = prepare_df(df1, dims, size_name, total_name)
    df2 = prepare_df(df2, dims, size_name, total_name)

    # rescale sizes and totals, preserving averages (= total/size)
    df1["Norm_weight"] = df1[size_name] / df1[size_name].sum()
    df2["Norm_weight"] = df2[size_name] / df2[size_name].sum()

    df1["Norm_totals"] = df1[total_name] / df1[size_name].sum()
    df2["Norm_totals"] = df2[total_name] / df2[size_name].sum()

    # subtract the initial average from both totals
    avg1 = df1["Norm_totals"].sum()
    df1["Adj_totals"] = df1["Norm_totals"] - avg1 * df1["Norm_weight"]
    df2["Adj_totals"] = df2["Norm_totals"] - avg1 * df2["Norm_weight"]

    # call explain_changes
    sf = explain_changes_in_totals(
        df1,
        df2,
        dims,
        total_name="Adj_totals",
        size_name="Norm_weight",
        min_segments=min_segments,
        max_segments=max_segments,
        min_depth=min_depth,
        max_depth=max_depth,
        solver=solver,
        how=how,
        force_add_up=force_add_up,
        constrain_signs=constrain_signs,
        cluster_values=cluster_values,
        verbose=verbose,
    )

    if hasattr(sf, "pre_total"):
        sf.pre_total = avg1
        sf.post_total += avg1
        sfs = None
    # Want to put the subtracted avg1 back in, something like
    # for s in sf.segments:
    #     s["naive_avg"] += average
    #     s["total"] += average * s["seg_size"]
    # print(average)
    # sf.reg.intercept_ = average

    # And might want to relabel some plots?
    sf.task = "changes in average"
    return sf


def explain_changes_in_totals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: str,
    min_segments: Optional[int] = None,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "lasso",
    how: str = "totals",
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = False,
    verbose: int = 0,
):
    """
    Find segments most useful in explaining the difference between the totals of the two datasets
    @param df1: First dataset
    @param df2: Second dataset
    @param dims: List of discrete dimensions
    @param total_name: Name of column that contains totals per segment
    @param size_name: Name of column containing segment sizes
    @param min_segments: Minimum number of segments to find
    @param max_segments: Maximum number of segments to find, defaults to min_segments
    @param min_depth: Minimum number of dimension to constrain in segment definition
    @param max_depth: Maximum number of dimension to constrain in segment definition
    @param solver: "lasso" for most unusual, possibly overlapping segments;
                   "tree" to divide the whole dataset into non-overlapping segments,
                          as homogenous as possible.
    @param how: "totals" to only decompose segment totals (ignoring size vs average contribution)
            "split_fits" to separately decompose contribution of size changes and average changes
            "extra_dim" to treat size vs average change contribution as an additional dimension
            "force_dim" like extra_dim, but each segment must contain a Change_from constraint
    @param force_add_up: Force the contributions of chosen segments to add up
    to the difference between dataset totals
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
    @param cluster_values: In addition to single-value slices, consider slices that consist of a
    group of segments from the same dimension with similar naive averages
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @return: A fitted object
    """

    assert how in ["totals", "extra_dim", "split_fits", "force_dim"]
    split_deltas = not (how == "totals")
    return_multiple = how == "split_fits"
    final_size = df2[size_name].sum()

    df1 = df1.copy()
    df2 = df2.copy()

    # replace NaN values in numeric columns with zeros
    # replace NaN values in categorical columns with the column name + "_unknown"
    df1 = prepare_df(df1, dims, size_name, total_name)
    df2 = prepare_df(df2, dims, size_name, total_name)

    my_diff = diff_dataset(
        df1,
        df2,
        dims,
        total_name,
        size_name,
        split_deltas=split_deltas,
        return_multiple=return_multiple,
    )

    if how == "split_fits":
        df_size, df_avg = my_diff
        sf_size = explain_levels(
            df=df_size.data,
            dims=dims,
            total_name=df_size.segment_total,
            size_name=df_size.segment_size,
            min_depth=min_depth,
            max_depth=max_depth,
            min_segments=min_segments,
            solver=solver,
            force_add_up=force_add_up,
            constrain_signs=constrain_signs,
            cluster_values=cluster_values,
            verbose=verbose,
        )

        sf_avg = explain_levels(
            df=df_avg.data,
            dims=dims,
            total_name=df_avg.segment_total,
            size_name=df_avg.segment_size,
            min_depth=min_depth,
            max_depth=max_depth,
            min_segments=min_segments,
            solver=solver,
            force_add_up=force_add_up,
            constrain_signs=constrain_signs,
            cluster_values=cluster_values,
            verbose=verbose,
        )

        sf_size.final_size = final_size
        sf_avg.final_size = final_size
        sp = SlicerPair(sf_size, sf_avg)
        sp.plot = lambda plot_is_static=False, width=2000, height=500, cluster_key_width=180, cluster_value_width=318, return_fig=False: plot_split_segments(
            sp.s1,
            sp.s2,
            plot_is_static=plot_is_static,
            width=width,
            height=height,
            cluster_key_width=cluster_key_width,
            cluster_value_width=cluster_value_width,
            return_fig=return_fig,
        )
        return sp

    else:
        sf = SliceFinder()

        sf.fit(
            my_diff.data[my_diff.dimensions],
            my_diff.data[my_diff.segment_total],
            weights=my_diff.data[my_diff.segment_size],
            min_depth=min_depth,
            max_depth=max_depth,
            min_segments=min_segments,
            max_segments=max_segments,
            solver=solver,
            force_dim="Change from" if how == "force_dim" else None,
            force_add_up=force_add_up,
            constrain_signs=constrain_signs,
            cluster_values=cluster_values,
            verbose=verbose,
        )

        sf.pre_total = df1[total_name].sum()
        sf.post_total = df2[total_name].sum()

        sf.plot = lambda plot_is_static=False, width=1000, height=1000, cluster_key_width=180, cluster_value_width=318, return_fig=False: plot_waterfall(
            sf,
            plot_is_static=plot_is_static,
            width=width,
            height=height,
            cluster_key_width=cluster_key_width,
            cluster_value_width=cluster_value_width,
            return_fig=return_fig,
        )
        sf.task = "changes in totals"
        return sf


def explain_levels(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: Optional[str] = None,
    min_segments: int = None,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver="lasso",
    verbose=0,
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = False,
):
    """
    Find segments whose average is most different from the global one
    @param df: Dataset
    @param dims: List of discrete dimensions
    @param total_name: Name of column that contains totals per segment
    @param size_name: Name of column containing segment sizes
    @param min_segments: Minimum number of segments to find
    @param max_segments: Maximum number of segments to find, defaults to min_segments
    @param min_depth: Minimum number of dimension to constrain in segment definition
    @param max_depth: Maximum number of dimension to constrain in segment definition
    @param solver: "lasso" for most unusual, possibly overlapping segments;
                   "tree" to divide the whole dataset into non-overlapping segments,
                          as homogenous as possible.
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @param force_add_up: Force the contributions of chosen segments to add up to zero
    @param constrain_signs: Whether to constrain weights of segments to have the same sign as naive segment averages
    @param cluster_values: In addition to single-value slices, consider slices that consist of a
    group of segments from the same dimension with similar naive averages
    @return: A fitted object
    """
    df = copy.copy(df)

    # replace NaN values in numeric columns with zeros
    # replace NaN values in categorical columns with the column name + "_unknown"
    df = prepare_df(df, dims, size_name, total_name)

    if size_name is None:
        size_name = "size"
        df[size_name] = 1.0

    # we want to look for deviations from average value
    average = df[total_name].sum() / df[size_name].sum()
    df["_target"] = df[total_name] - df[size_name] * average

    sf = SliceFinder()
    sf.fit(
        df[dims],
        df["_target"],
        weights=None if size_name is None else df[size_name],
        min_segments=min_segments,
        max_segments=max_segments,
        min_depth=min_depth,
        max_depth=max_depth,
        solver=solver,
        verbose=verbose,
        force_add_up=force_add_up,
        constrain_signs=constrain_signs,
        cluster_values=cluster_values,
    )

    for s in sf.segments:
        s["naive_avg"] += average
        s["total"] += average * s["seg_size"]
    # print(average)
    sf.reg.intercept_ = average
    sf.plot = lambda plot_is_static=False, width=2000, height=500, return_fig=False, cluster_key_width=180, cluster_value_width=318: plot_segments(
        sf,
        plot_is_static=plot_is_static,
        width=width,
        height=height,
        return_fig=return_fig,
        cluster_key_width=cluster_key_width,
        cluster_value_width=cluster_value_width,
    )
    sf.task = "levels"
    return sf


def explain_timeseries(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    time_name: str,
    size_name: Optional[str] = None,
    min_segments: int = None,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "omp",
    verbose: bool = False,
    constrain_signs: bool = False,
    cluster_values: bool = False,
    time_basis: Optional[pd.DataFrame] = None,
    fit_log_space: bool = False,
    fit_sizes: Optional[bool] = None,
    num_breaks: int = 2,
    log_space_weight_sc: float = 0.5,
):
    df = copy.copy(df)

    # replace NaN values in numeric columns with zeros
    # replace NaN values in categorical columns with the column name + "_unknown"
    # Group by dims + [time_name]
    df = prepare_df(
        df, dims, total_name=total_name, size_name=size_name, time_name=time_name
    )
    df = df.sort_values(by=dims + [time_name])

    if time_basis is None:
        time_basis = create_time_basis(df[time_name].unique())
        time_basis = prune_time_basis(time_basis, num_breaks=num_breaks, solver=solver)

    if size_name is None:
        size_name = "size"
        df[size_name] = 1.0
        if fit_sizes == True:
            raise ValueError("fit_sizes should be None or False if size_name is None")
        fit_sizes = False
    else:
        if fit_sizes is None:
            fit_sizes = True

    # Transform logic begins (unused for now)
    if fit_log_space:
        tf = LogTransform(
            offset=1,
            weight_pow_sc=log_space_weight_sc,
        )
    else:
        tf = IdentityTransform()

    size_name_orig = size_name + "_orig"
    total_name_orig = total_name + "_orig"

    df2 = df.rename(columns={size_name: size_name_orig, total_name: total_name_orig})
    t, w = tf.transform_totals_weights(
        df2[total_name_orig].values, df2[size_name_orig].values
    )
    df2[total_name] = pd.Series(data=t, index=df2.index)
    df2[size_name] = pd.Series(data=w, index=df2.index)
    # Transform logic ends
    if fit_sizes:
        # block-matrix df2 with itself, for the weights
        re_df = df2.copy()
        df2["chunk"] = "Average"

        # Normalize so the new chunk has same total weight as the original
        re_df[size_name] = df2[total_name].sum() / df2[size_name].sum()
        re_df[total_name] = df2[size_name]
        re_df["chunk"] = "Weights"

        # Block-matrix basis with itself
        re_basis = time_basis.copy().rename(
            {c: c + "_w" for c in time_basis.columns}, axis=1
        )
        time_basis["chunk"] = "Average"
        re_basis["chunk"] = "Weights"

        time_basis = (
            pd.concat([time_basis, re_basis], axis=0)
            .fillna(0.0)
            .reset_index()
            .rename(columns={"index": "__time"})
        )
        print("yay!")
        groupby_dims = ["chunk", "__time"]
    else:
        groupby_dims = ["__time"]

    df2["_target"] = df2[total_name]
    df2["total_adjustment"] = 0.0
    avg_df = 0.0
    average = 0.0

    sf = SliceFinder()
    sf.global_average = average
    sf.total_name = total_name
    sf.size_name = size_name
    sf.time_name = time_name
    sf.y_adj = df2["total_adjustment"].values
    sf.avg_df = avg_df
    sf.time_values = df2[time_name].unique()
    sf.fit(
        df2[dims],
        df2["_target"],
        time_col=df2[time_name],
        time_basis=time_basis,
        weights=df2[size_name],
        min_segments=min_segments,
        max_segments=max_segments,
        min_depth=min_depth,
        max_depth=max_depth,
        solver=solver,
        verbose=verbose,
    )

    # TODO: insert back the normalized bits?
    for s in sf.segments:
        segment_def = s["segment"]
        this_vec = (
            sf.X[:, s["index"]]
            .toarray()
            .reshape(
                -1,
            )
        )
        if "coef" in s:
            s["seg_total_vec"] = this_vec * s["coef"] * sf.weights

        if len(segment_def) > 1:
            elems = np.unique(s["dummy"].astype(float))
            assert len(elems) == 2
            assert 1.0 in elems
            assert 0.0 in elems

        s["naive_avg"] += average
        s["total"] += average * s["seg_size"]

    if solver == "tree":
        sf.segments = sorted(sf.segments, key=lambda x: x["total"], reverse=True)

    if solver == "tree":
        plot_fun = plot_time_from_tree
    else:
        plot_fun = plot_time

    sf.plot = lambda plot_is_static=False, width=1200, height=2000, return_fig=False, average_name=None: plot_fun(
        sf,
        plot_is_static=plot_is_static,
        width=width,
        height=height,
        return_fig=return_fig,
        average_name=average_name,
    )
    sf.task = "time"
    return sf
