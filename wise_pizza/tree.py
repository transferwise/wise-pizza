from typing import Optional

import numpy as np
import pandas as pd


def tree_solver(
    dim_df: pd.DataFrame,
    weights: np.ndarray,
    totals: np.ndarray,
    time_basis: np.ndarray,
    max_depth: int = 3,
    num_leaves: Optional[int] = None,
):
    # TODO: fill in
    # Build a tree in the following fashion:
    # 1. Start with a single node containing the whole dataset
    # 2. At each node, find the best split by looping over all dimensions, for each dimension
    # solving the problem of which values to take in the left and right subtrees,
    # by running a regression of totals/weights on time basis in both subsets separately
    # and optimizing the total squared error.
    # the best combination of (node, dimension) is the next one due to be split
    # If expanding the best node would exceed maximum depth:
    # If num_leaves is None: stop
    # If it's not, expand the best node that would not exceed max_depth, until num_leaves is reached

    return X, reg, col_defs
