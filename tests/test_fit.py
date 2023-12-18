import itertools
import os
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from wise_pizza.data_sources.synthetic import synthetic_data, synthetic_ts_data
from wise_pizza.explain import explain_changes_in_average, explain_changes_in_totals, explain_levels, explain_timeseries
from wise_pizza.segment_data import SegmentData
from wise_pizza.solver import solve_lasso, solve_lp
from wise_pizza.time import create_time_basis
from wise_pizza.plotting import plot_time

np.random.seed(42)

dims = [
    "TYPE",
    "REGION",
    "FIRST_PRODUCT",
    "CURRENCY",
]
totals = "VOLUME"
size = "TRANSACTION_COUNT"

# possible values for explain methods
# Too long, delete some values for quick starts, e.g. by deleting the parameters in nan_percent, size_one_percent
deltas_test_values = [
    ("totals", "split_fits", "force_dim", "extra_dim"),  # how
    ("lp", "lasso"),  # solver
    (True,),  # plot_is_static
    (explain_changes_in_average, explain_changes_in_totals),  # function
    (0.0, 90.0),  # nan_percent
    (0.0, 90.0),  # size_one_percent
]
# possible variants for explain methods
deltas_test_cases = list(itertools.product(*deltas_test_values))

# possible values for explain_levels
levels_test_values = [
    ("lp", "lasso"),  # solver
    (0.0, 90.0),  # nan_percent
    (0.0, 90.0),  # size_one_percent
]

# possible variants for explain_levels
levels_test_cases = list(itertools.product(*levels_test_values))


def values_to_nan(df: pd.DataFrame, nan_percent: float = 0.0):
    """
     Randomly change values to NaN in a pandas dataframe for testing.
    Parameters:
        - df: a pandas dataframe
        - percent: the percentage of values to replace with NaN, as a float between 0 and 100
    Returns:
        A new pandas dataframe with NaN values randomly changed.
    """
    df_copy = df.copy()  # create a copy of the original dataframe

    # calculate the number of values to replace with NaN
    num_values = int((nan_percent / 100) * len(df))
    np.random.seed(42)
    # get a random selection of indices to replace with NaN
    random_indices = df_copy.sample(num_values).index

    # replace values with NaN at the random indices
    df_copy.loc[random_indices, df_copy.columns] = np.nan

    return df_copy


def size_to_one(
    df: pd.DataFrame,
    percent: float = 0.0,
    size_name: str = "TRANSACTION_COUNT",
    totals_name: str = "VOLUME",
):
    """
     Randomly change sizes to one in a pandas dataframe for testing.
    Parameters:
        - df: a pandas dataframe
        - percent: the percentage of values to replace with one, as a float between 0 and 100
    Returns:
        A new pandas dataframe with changed sizes.
    """
    df_copy = df.copy()  # create a copy of the original dataframe

    # calculate the number of values to replace with NaN
    num_values = int((percent / 100) * len(df))
    np.random.seed(42)
    # get a random selection of indices to replace with NaN
    random_indices = df_copy.sample(num_values).index

    # replace values with NaN at the random indices
    df_copy.loc[random_indices, size_name] = 1
    df_copy.loc[random_indices, totals_name] = 1

    return df_copy


def monthly_driver_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data", "synth_data.csv"))
    return SegmentData(data=df, dimensions=dims, segment_total=totals, segment_size=size)


def test_categorical():
    all_data = monthly_driver_data()
    df = all_data.data
    last_month = df.COHORT_MONTH.max()

    data = df[df.COHORT_MONTH == last_month]

    sf = explain_levels(
        data,
        dims=all_data.dimensions,
        total_name=all_data.segment_total,
        size_name=all_data.segment_size,
        max_depth=2,
    )
    print("***")
    for s in sf.segments:
        print(s)
    print(sf.summary())
    print(sf.relevant_cluster_names)
    print("yay!")


@pytest.mark.parametrize("nan_percent", [0.0, 1.0])
def test_synthetic_template(nan_percent: float):
    all_data = synthetic_data(init_len=1000)
    data = all_data.data

    data.loc[(data["dim0"] == 0) & (data["dim1"] == 1), "totals"] += 100
    data.loc[(data["dim1"] == 0) & (data["dim2"] == 1), "totals"] += 300

    if nan_percent > 0:
        data = values_to_nan(data, nan_percent)
    sf = explain_levels(
        data,
        dims=all_data.dimensions,
        total_name=all_data.segment_total,
        size_name=all_data.segment_size,
        max_depth=2,
        min_segments=5,
        verbose=1,
        solver="lp",
    )
    print("***")
    for s in sf.segments:
        print(s)

    assert abs(sf.segments[0]["coef"] - 300) < 2
    assert abs(sf.segments[1]["coef"] - 100) < 2

    # sf.plot()
    print("yay!")


@pytest.mark.parametrize("nan_percent", [0.0, 1.0])
def test_synthetic_ts_template(nan_percent: float):
    all_data = synthetic_ts_data(init_len=10000)

    # Add some big trends to the data
    # TODO: insert trend break patterns too
    months = np.array(sorted(all_data.data[all_data.time_col].unique()))
    basis = create_time_basis(months, baseline_dims=1)
    joined = pd.merge(all_data.data, basis, left_on="TIME", right_index=True)
    df = joined.drop(columns=basis.columns)

    loc1 = (df["dim0"] == 0) & (df["dim1"] == 1)
    loc2 = (df["dim1"] == 0) & (df["dim2"] == 1)

    df.loc[loc1, "totals"] += 100 * joined.loc[loc1, "Slope"]
    df.loc[loc2, "totals"] += 300 * joined.loc[loc2, "Slope"]

    if nan_percent > 0:
        df = values_to_nan(df, nan_percent)
    sf = explain_timeseries(
        df,
        dims=all_data.dimensions,
        total_name=all_data.segment_total,
        time_name=all_data.time_col,
        size_name=all_data.segment_size,
        max_depth=2,
        min_segments=5,
        verbose=True,
    )
    print("***")
    for s in sf.segments:
        print(s)

    plot_time(sf)

    assert abs(sf.segments[0]["coef"] - 300) < 2
    assert abs(sf.segments[1]["coef"] - 100) < 2


    # sf.plot()
    print("yay!")


@pytest.mark.parametrize(
    "how, solver, plot_is_static, function, nan_percent, size_one_percent",
    deltas_test_cases,
)
def test_deltas(
    how: str,
    solver: str,
    plot_is_static: bool,
    function: Callable,
    nan_percent: float,
    size_one_percent: float,
):
    all_data = monthly_driver_data()
    df = all_data.data

    months = sorted(df.COHORT_MONTH.unique())

    data = df[df.COHORT_MONTH == months[-1]]
    pre_data = df[df.COHORT_MONTH == months[-2]]
    np.random.seed(42)
    pre_data = values_to_nan(pre_data, nan_percent)
    data = values_to_nan(data, nan_percent)
    pre_data = size_to_one(pre_data, size_one_percent)
    data = size_to_one(data, size_one_percent)

    sf = function(
        pre_data,
        data,
        all_data.dimensions,
        all_data.segment_total,
        all_data.segment_size,
        how=how,
        max_depth=1,
        min_segments=10,
        solver=solver,
    )
    # sf.plot(plot_is_static=plot_is_static)
    print("yay!")


@pytest.mark.parametrize("solver, nan_percent, size_one_percent", levels_test_cases)
def test_explain_levels(solver: str, nan_percent: float, size_one_percent: float):
    all_data = monthly_driver_data()
    df = all_data.data

    months = sorted(df.COHORT_MONTH.unique())

    data = df[df.COHORT_MONTH == months[-1]]
    data = values_to_nan(data, nan_percent)
    data = size_to_one(data, size_one_percent)

    sf = explain_levels(
        df=data,
        dims=all_data.dimensions,
        total_name=all_data.segment_total,
        size_name=all_data.segment_size,
        max_depth=1,
        min_segments=10,
        solver=solver,
    )
    print(sf.summary())
    print("yay!")


def test_solve_lasso():
    X = pd.DataFrame(columns=["vol", "rev", "count"])
    X["vol"] = [-1, -2, -3]
    X["rev"] = [-4, -5, -6]
    X["count"] = [7, 8, 9]
    y = pd.DataFrame(columns=["target"])
    y["target"] = [10, 11, 12]
    lasso = solve_lasso(
        X,
        y,
        alpha=0.1,
        constrain_signs=True,
        drop_last_row=True,
    )
    assert len(lasso.coef_) == 3
    print("COEFS: ", lasso.coef_)


def test_solve_lp():
    X = pd.DataFrame(columns=["vol", "rev", "count"])
    X["vol"] = [-1, -2, -3]
    X["rev"] = [-4, -5, -6]
    X["count"] = [7, 8, 9]
    y = pd.DataFrame(columns=["target"])
    y["target"] = [10, 11, 12]
    X = np.copy(X)
    y = np.copy(y)
    lp = solve_lp(
        X,
        y,
        alpha=0.1,
        constrain_signs=True,
        drop_last_row=True,
    )
    assert len(lp.coef_) == 3
    print("COEFS: ", lp.coef_)
