import itertools
from typing import Optional, List, Dict

import numpy as np
import scipy
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csc_matrix, hstack


def join_to_sparse(dim_df: pd.DataFrame, dim_name: str, verbose=0):
    values = sorted(dim_df[dim_name].unique())

    # create an "eye" dataframe
    ext_df = pd.DataFrame(data=np.eye(len(values)), columns=values)
    ext_df[dim_name] = values

    join_df = pd.merge(dim_df, ext_df, on=[dim_name])
    join_df = join_df.sort_values(list(dim_df.columns))
    vals = csc_matrix(join_df[values].values)
    if verbose > 0:
        print(values, vals.shape)
    return vals, values


def segment_defs(dim_df: pd.DataFrame, used_dims, verbose=0) -> List[Dict[str, str]]:
    col_defs = []
    this_df = dim_df[used_dims].drop_duplicates().reset_index(drop=True)
    # create an "eye" dataframe on the unique reduced dimensions
    for i, vals in enumerate(this_df.itertuples(index=False)):
        col_defs.append(dict(zip(used_dims, vals)))

    if verbose > 0:
        print(used_dims, len(col_defs))
    return col_defs


def construct_dummies(
    segment_defs: List[Dict[str, str]], cache: Dict[str, Dict[str, np.ndarray]]
) -> scipy.sparse.csc_matrix:
    dummies = []
    for sgdf in segment_defs:
        tmp = None
        for k, v in sgdf.items():
            if tmp is None:
                tmp = cache[k][v]
            else:
                tmp = tmp.multiply(cache[k][v])
        dummies.append(tmp)
    return hstack(dummies)


# This approach was way slower than the join one; keeping it here for reference :)
# def join_to_sparse(dim_df, this_df, chunk_size=100, verbose=0):
#     mats = []
#     tuples = []
#     col_defs = []
#
#     these_dims = list(this_df.columns)
#     # create an "eye" dataframe on the unique reduced dimensions
#     for i, vals in enumerate(this_df.itertuples(index=False)):
#         tuples.append(vals)
#         col_defs.append({k: v for k, v in zip(these_dims, vals)})
#
#     # join it against the real thing, one chunk at a time
#     for i in range(0, len(tuples), chunk_size):
#         this_df = dim_df.copy()
#         these_cols = []
#         for tpl in tuples[i : min(i + chunk_size, len(tuples))]:
#             col_name = "_".join(map(str, tpl))
#             these_cols.append(col_name)
#             this_df[col_name] = 0.0
#
#             for i, (col, value) in enumerate(zip(these_dims, tpl)):
#                 if i == 0:
#                     filter = this_df[col] == value
#                 else:
#                     filter = filter & (this_df[col] == value)
#
#             this_df[filter] = 1.0
#
#         vals = csc_matrix(this_df[these_cols].values)
#         del this_df
#
#         mats.append(vals)
#         # print(vals.shape)
#     if len(mats) > 1:
#         out = hstack(mats)
#     else:
#         out = mats[0]
#     if verbose > 0:
#         print(these_dims, out.shape)
#     return out, col_defs


def sparse_dummy_matrix(
    dim_df: pd.DataFrame,
    min_depth: int = 1,
    max_depth: int = 2,
    verbose=0,
    force_dim: Optional[str] = None,
):
    # generate a sparse dummy matrix based on all the combinations
    # TODO: do a  nested sparse regression fit to form groups of dim values, pos, neg, null
    # TODO: first calculate the matrix size, scale down max_depth if matrix too big
    if force_dim is None:
        dims = list(dim_df.columns)
    else:
        assert force_dim in dim_df.columns
        dims = [c for c in dim_df.columns if c != force_dim]

    # drop dimensions with only one value, for clarity
    dims = [d for d in dims if len(dim_df[d].unique()) > 1]

    defs = []
    mats = []
    dims_range_min = min(len(dims), max(1, min_depth))
    dims_range_max = min(len(dims) + 1, max_depth + 1)
    dims_range = range(dims_range_min, dims_range_max)

    # first pass: generate single-dim dummies
    dummy_cache = {}
    for d in dim_df.columns:
        this_mat, these_defs = join_to_sparse(dim_df, d, verbose=verbose)
        dummy_cache[d] = {this_def: this_mat[:, i : i + 1] for i, this_def in enumerate(these_defs)}

    for num_dims in tqdm(dims_range) if verbose else dims_range:
        for these_dims in itertools.combinations(dims, num_dims):
            if num_dims == 1 and these_dims[0] == "Change from":
                continue
            if force_dim is None:
                used_dims = list(these_dims)
            else:
                used_dims = [force_dim] + list(these_dims)

            these_defs = segment_defs(dim_df, used_dims, verbose=verbose)
            this_mat = construct_dummies(these_defs, dummy_cache)
            mats.append(this_mat)
            defs += these_defs
    mat = hstack(mats)
    return mat, defs
