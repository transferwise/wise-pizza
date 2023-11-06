import copy
import warnings
from typing import List, Optional

import pandas as pd

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from wise_pizza.plotting import plot_segments, plot_split_segments, plot_waterfall
from wise_pizza.slicer import SliceFinder, SlicerPair
from wise_pizza.utils import diff_dataset, prepare_df
from wise_pizza.time import create_time_basis, strip_out_baseline


def explain_changes_in_average(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: str,
    min_segments: int = 5,
    max_segments: Optional[int] = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "lasso",
    how: str = "totals",
    force_add_up: bool = False,
    constrain_signs: bool = True,
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
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param how: "totals" to only decompose segment totals (ignoring size vs average contribution)
            "split_fits" to separately decompose contribution of size changes and average changes
            "extra_dim" to treat size vs average change contribution as an additional dimension
            "force_dim" like extra_dim, but each segment must contain a Change_from constraint
    @param force_add_up: Force the contributions of chosen segments to add up
    to the difference between dataset totals
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
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
    min_segments: int = 5,
    max_segments: Optional[int] = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "lasso",
    how: str = "totals",
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = True,
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
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param how: "totals" to only decompose segment totals (ignoring size vs average contribution)
            "split_fits" to separately decompose contribution of size changes and average changes
            "extra_dim" to treat size vs average change contribution as an additional dimension
            "force_dim" like extra_dim, but each segment must contain a Change_from constraint
    @param force_add_up: Force the contributions of chosen segments to add up
    to the difference between dataset totals
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
    @param cluster_values In addition to single-value slices, consider slices that consist of a
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
        sp.plot = lambda plot_is_static=False, width=2000, height=500: plot_split_segments(
            sp.s1,
            sp.s2,
            plot_is_static=plot_is_static,
            width=width,
            height=height,
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

        sf.plot = lambda plot_is_static=False, width=1000, height=1000: plot_waterfall(
            sf, plot_is_static=plot_is_static, width=width, height=height
        )
        sf.task = "changes in totals"
        return sf


def explain_levels(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    size_name: Optional[str] = None,
    min_segments: int = 10,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver="lasso",
    verbose=0,
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = True,
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
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @param force_add_up: Force the contributions of chosen segments to add up to zero
    @param constrain_signs: Whether to constrain weights of segments to have the same sign as naive segment averages
    @param cluster_values In addition to single-value slices, consider slices that consist of a
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
    sf.plot = lambda plot_is_static=False, width=2000, height=500, return_fig=False: plot_segments(
        sf, plot_is_static=plot_is_static, width=width, height=height, return_fig=return_fig
    )
    sf.task = "levels"
    return sf


def explain_timeseries(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    time_name: str,
    size_name: Optional[str] = None,
    min_segments: int = 10,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver="lasso",
    verbose=0,
    force_add_up: bool = False,
    constrain_signs: bool = True,
    cluster_values: bool = True,
):
    """
    Find segments whose average is most different from the global one
    @param df: Dataset
    @param dims: List of discrete dimensions
    @param total_name: Name of column that contains totals per segment
    @param size_name: Name of column containing segment sizes
    @param time_name: Name of column containing the time dimension
    @param min_segments: Minimum number of segments to find
    @param max_segments: Maximum number of segments to find, defaults to min_segments
    @param min_depth: Minimum number of dimension to constrain in segment definition
    @param max_depth: Maximum number of dimension to constrain in segment definition
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @param force_add_up: Force the contributions of chosen segments to add up to zero
    @param constrain_signs: Whether to constrain weights of segments to have the same sign as naive segment averages
    @param cluster_values In addition to single-value slices, consider slices that consist of a
    group of segments from the same dimension with similar naive averages
    @return: A fitted object
    """
    df = copy.copy(df)

    # replace NaN values in numeric columns with zeros
    # replace NaN values in categorical columns with the column name + "_unknown"
    # Group by dims + [time_name]
    df = prepare_df(df, dims, total_name=total_name, size_name=size_name, time_name=time_name)

    if size_name is None:
        size_name = "size"
        df[size_name] = 1.0

    # strip out constants and possibly linear trends for each dimension combination
    baseline_dims = 1
    time_basis = create_time_basis(df[time_name].unique(), baseline_dims=baseline_dims)
    df = strip_out_baseline(df, basis=time_basis, strip_trends=False)

    # we want to look for deviations from average value
    average = df[total_name].sum() / df[size_name].sum()
    df["_target"] = df[total_name] - df[size_name] * average

    sf = SliceFinder()
    sf.fit(
        df[dims],
        df["_target"],
        time_col=df[time_name],
        time_basis=time_basis,
        weights=df[size_name],
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

    # TODO: insert back the normalized bits?
    for s in sf.segments:
        s["naive_avg"] += average
        s["total"] += average * s["seg_size"]
    # print(average)
    sf.reg.intercept_ = average
    sf.plot = lambda plot_is_static=False, width=2000, height=500, return_fig=False: plot_segments(
        sf, plot_is_static=plot_is_static, width=width, height=height, return_fig=return_fig
    )
    sf.task = "levels"
    return sf
