import copy
import itertools
from typing import Optional, List, Dict, Sequence
from collections import defaultdict

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
    clusters: Optional[Dict[str, Sequence[str]]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    time_basis: Optional[pd.DataFrame] = None,
    max_out_size: int = 100000
):
    # generate a sparse dummy matrix based on all the combinations
    if force_dim is None:
        dims = list(dim_df.columns)
    else:
        assert force_dim in dim_df.columns
        dims = [c for c in dim_df.columns if c != force_dim]

    if clusters is None:
        clusters = defaultdict(list)

    # drop dimensions with only one value, for clarity
    dims = [d for d in dims if len(dim_df[d].unique()) > 1]


    dims_range_min = min(len(dims), max(1, min_depth))
    dims_range_max = min(len(dims) + 1, max_depth + 1)
    dims_range = range(dims_range_min, dims_range_max)

    # first pass: generate single-dim dummies
    dummy_cache = {}
    for d in dim_df.columns:
        this_mat, these_defs = join_to_sparse(dim_df, d, verbose=verbose)
        dummy_cache[d] = {this_def: this_mat[:, i : i + 1] for i, this_def in enumerate(these_defs)}

    dims_dict = {dim: list(dim_df[dim].unique()) + list(clusters[dim]) for dim in dim_df.columns}


    defs = []
    mats = []
    # Go over all possible depths
    for num_dims in tqdm(dims_range) if verbose else dims_range:
        # for each depth, sample the possible dimension combinations
        for these_dims in itertools.combinations(dims, num_dims):
            if num_dims == 1 and these_dims[0] == "Change from":
                continue
            if force_dim is None:
                used_dims = list(these_dims)
            else:
                used_dims = [force_dim] + list(these_dims)

            segment_constraints = segment_defs_new(dims_dict, used_dims)
            this_mat, these_defs = construct_dummies_new(used_dims, segment_constraints, dummy_cache, cluster_names)

            if time_basis is None:
                mats.append(this_mat)
                defs += these_defs
            else:
                for b_name, b_mat in time_basis.items():
                    re_defs = copy.deepcopy(these_defs)
                    for d in re_defs:
                        d["time"] = b_name

                    m = csc_matrix(np.ones(shape=(1, this_mat.shape[1])))
                    re_basis = b_mat @ m
                    re_mat = this_mat.multiply(re_basis)

                    mats.append(re_mat)
                    defs.append(re_defs)


            if len(defs) >= max_out_size:
                mat = hstack(mats)
                yield mat, defs
                defs =[]
                mats = []
    # mop up
    if len(defs):
        mat = hstack(mats)
        yield mat, defs


def segment_defs_new(dims_dict: Dict[str, Sequence[str]], used_dims: List[str]) -> List[Dict[str, str]]:
    # Look at all possible combinations of dimension values for the chosen dimensions
    if len(used_dims) == 1:
        return np.array(dims_dict[used_dims[0]]).reshape(-1, 1)
    else:
        tmp = segment_defs_new(dims_dict, used_dims[:-1])
        this_dim_values = np.array(dims_dict[used_dims[-1]])
        repeated_values = np.tile(this_dim_values.reshape(-1, 1), len(tmp)).reshape(-1, 1)
        pre_out = np.tile(tmp, (len(this_dim_values), 1))
        out = np.concatenate([pre_out, repeated_values], axis=1)
        return out


def construct_dummies_new(
    used_dims: List[str],
    segment_defs: np.ndarray,
    cache: Dict[str, Dict[str, np.ndarray]],
    cluster_names: Optional[Dict[str, str]] = None,
) -> scipy.sparse.csc_matrix:
    dummies = []
    segments = []
    for sgdf in segment_defs:
        tmp = None
        for i, d in enumerate(used_dims):
            if isinstance(sgdf[i], str) and sgdf[i] not in cache[d]:  # a group of multiple values from that dim
                sub_values = cluster_names[sgdf[i]].split("@@")
                this_dummy = 0
                for val in sub_values:
                    this_dummy += cache[d][val]

            else:
                this_dummy = cache[d][sgdf[i]]

            if tmp is None:
                tmp = this_dummy
            else:
                tmp = tmp.multiply(this_dummy)
        if tmp.sum() > 0:
            dummies.append(tmp)
            segments.append(dict(zip(used_dims, sgdf)))
    return hstack(dummies), segments
