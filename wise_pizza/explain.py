import copy
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from wise_pizza.plotting import plot_segments, plot_split_segments, plot_waterfall, plot_time, plot_ts_pair
from wise_pizza.slicer import SliceFinder, SlicerPair, TransformedSliceFinder
from wise_pizza.utils import diff_dataset, prepare_df
from wise_pizza.time import create_time_basis, average_over_time


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
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
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
    min_segments: int = 5,
    max_segments: Optional[int] = None,
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
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
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
            cluster_values=cluster_values,
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
            cluster_values=cluster_values,
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
    min_segments: int = 10,
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
        cluster_values=cluster_values,
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
    min_segments: int = 5,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "omp",
    verbose: bool = False,
    cluster_values: bool = False,
    time_basis: Optional[pd.DataFrame] = None,
    fit_log_space: bool=False
):
    sf_totals = _explain_timeseries(
        df=df,
        dims=dims,
        total_name=total_name,
        time_name=time_name,
        size_name=size_name,
        min_segments=min_segments,
        max_segments=max_segments,
        min_depth=min_depth,
        max_depth=max_depth,
        solver=solver,
        verbose=verbose,
        cluster_values=cluster_values,
        time_basis=time_basis,
    )

    if not size_name:
        return sf_totals

    if fit_log_space:
        # transform
        offset = 1
        # transform = lambda x: x + offset
        # inverse_transform = lambda x: np.maximum(0.0, x-offset)
        transform = lambda x: np.log(offset + x )
        inverse_transform = lambda x: np.maximum(0.0, np.exp(x) - offset)
        weight_transform = lambda w, a: w*(offset+a)
        size_name_orig = size_name + "_orig"
        df2 = df.rename(columns={size_name: size_name_orig })
        df2[size_name] = pd.Series(data=transform(df2[size_name_orig].values), index=df2.index)
        df2["resc_wgt"] = weight_transform(1.0, df2[size_name_orig].values)
    else:
        df2 = df

    sf_wgt = _explain_timeseries(
        df=df2,
        dims=dims,
        total_name=size_name,
        size_name="resc_wgt",
        time_name=time_name,
        min_segments=min_segments,
        max_segments=max_segments,
        min_depth=min_depth,
        max_depth=max_depth,
        solver=solver,
        verbose=verbose,
        cluster_values=cluster_values,
        time_basis=time_basis,
    )
    if fit_log_space:
        sf1 = TransformedSliceFinder(sf_wgt, offset=offset, inverse_transform=inverse_transform)
    else:
        sf1 = sf_wgt

    out = SlicerPair(sf1, sf_totals)
    out.plot = lambda width=600, height=1200, average_name=None, use_fitted_weights=False: plot_ts_pair(
        out, width=width, height=height, average_name=average_name, use_fitted_weights=use_fitted_weights
    )
    out.task = "time with weights"
    return out


def _explain_timeseries(
    df: pd.DataFrame,
    dims: List[str],
    total_name: str,
    time_name: str,
    size_name: Optional[str] = None,
    min_segments: int = 5,
    max_segments: int = None,
    min_depth: int = 1,
    max_depth: int = 2,
    solver: str = "omp",
    verbose: bool = False,
    force_add_up: bool = False,
    constrain_signs: bool = False,
    cluster_values: bool = False,
    time_basis: Optional[pd.DataFrame] = None,
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
    if time_basis is None:
        time_basis = create_time_basis(df[time_name].unique(), baseline_dims=baseline_dims, include_breaks=True)
        dtrend_cols = [t for t in time_basis.columns if "dtrend" in t]
        chosen_cols = []
        num_breaks = 2
        for i in range(1, num_breaks + 1):
            chosen_cols.append(dtrend_cols[int(i * len(dtrend_cols) / (num_breaks + 1))])
        pre_basis = time_basis[list(time_basis.columns[:2]) + chosen_cols].copy()
        # TODO: fix this bug
        for c in chosen_cols:
            pre_basis[c + "_a"] = pre_basis["Slope"] - pre_basis[c]

        print("yay!")

    df = average_over_time(df, dims=dims, total_name=total_name, size_name=size_name, time_name=time_name)

    # This block is pointless as we just normalized each sub-segment to zero average across time
    average = df[total_name].sum() / df[size_name].sum()
    df["_target"] = df[total_name] - df["total_adjustment"]

    sf = SliceFinder()
    sf.global_average = average
    sf.total_name = total_name
    sf.size_name = size_name
    sf.y_adj = df["total_adjustment"].values
    sf.fit(
        df[dims],
        df["_target"],
        time_col=df[time_name],
        time_basis=pre_basis,
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
        segment_def = s["segment"]
        assert "time" in segment_def, "Each segment should have a time profile!"
        this_vec = (
            sf.X[:, s["index"]]
            .toarray()
            .reshape(
                -1,
            )
        )
        time_mult = (
            sf.time_basis[segment_def["time"]]
            .toarray()
            .reshape(
                -1,
            )
        )
        dummy = (this_vec / time_mult).astype(int).astype(np.float64)
        s["dummy"] = dummy
        s["seg_avg"] = this_vec * s["coef"]
        if len(segment_def) > 1:
            elems = np.unique(dummy)
            assert len(elems) == 2
            assert 1.0 in elems
            assert 0.0 in elems

        s["naive_avg"] += average
        s["total"] += average * s["seg_size"]
    # print(average)
    # sf.reg.intercept_ += average
    sf.plot = lambda plot_is_static=False, width=1200, height=2000, return_fig=False, average_name=None: plot_time(
        sf,
        # plot_is_static=plot_is_static,
        width=width,
        height=height,
        # return_fig=return_fig,
        average_name=average_name,
    )
    sf.task = "time"
    return sf
