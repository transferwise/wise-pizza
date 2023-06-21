import itertools
from typing import Optional

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csc_matrix, hstack


def join_to_sparse(dim_df: pd.DataFrame, this_df: pd.DataFrame, verbose=0):
    col_names = []
    col_defs = []
    ext_df = this_df.copy()
    mat = np.eye(len(this_df))
    these_dims = list(this_df.columns)
    # create an "eye" dataframe on the unique reduced dimensions
    for i, vals in enumerate(this_df.itertuples(index=False)):
        col_name = "_".join(map(str, vals))
        col_names.append(col_name)
        col_defs.append(dict(zip(these_dims, vals)))
        ext_df[col_name] = mat[i]
    dim_df_columns = list(dim_df.columns)
    join_df = pd.merge(dim_df, ext_df, on=these_dims)
    join_df = join_df.sort_values(dim_df_columns)
    vals = csc_matrix(join_df[col_names].values)
    if verbose > 0:
        print(these_dims, vals.shape)
    return vals, col_defs


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
    dims = [d for d in dims  if len(dim_df[d].unique()) > 1]

    defs = []
    mats = []
    dims_range_min = min(len(dims), max(1, min_depth))
    dims_range_max = min(len(dims) + 1, max_depth + 1)
    dims_range = range(dims_range_min, dims_range_max)
    for num_dims in tqdm(dims_range) if verbose else dims_range:
        for these_dims in itertools.combinations(dims, num_dims):
            if num_dims == 1 and these_dims[0] == "Change from":
                continue
            if force_dim is None:
                used_dims = list(these_dims)
            else:
                used_dims = [force_dim] + list(these_dims)
            this_df = dim_df[used_dims].drop_duplicates().reset_index(drop=True)
            this_mat, these_defs = join_to_sparse(dim_df, this_df, verbose=verbose)
            mats.append(this_mat)
            defs += these_defs
    mat = hstack(mats)
    return mat, defs
