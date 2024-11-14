import copy
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


from .fitter import AverageFitter, Fitter, TimeFitterModel, TimeFitter
from .partition import target_encoding_partitions, kmeans_partition
from wise_pizza.cluster import nice_cluster_names


def tree_solver(
    dim_df: pd.DataFrame,
    dims: List[str],
    fitter: Fitter,
    max_depth: Optional[int] = None,
    num_leaves: Optional[int] = None,
):
    """
    Partition the data into segments using a greedy binary tree approach
    :param dim_df: DataFrame with dimensions, totals, and similar data, sorted by time and dims
    :param dims: List of dimensions the dataset is segmented by
    :param fitter: A model to fit on the chunks
    :param max_depth: max depth of the tree
    :param num_leaves: num leaves to generate
    :return: Segment description, column definitions, and cluster names
    """

    df = dim_df.copy().reset_index(drop=True)
    df["__avg"] = df["totals"] / df["weights"]
    df["__avg"] = df["__avg"].fillna(df["__avg"].mean())

    root = ModelNode(
        df=df,
        fitter=fitter,
        dims=dims,
        time_col=None if isinstance(fitter, AverageFitter) else "__time",
        max_depth=max_depth,
    )

    build_tree(root=root, num_leaves=num_leaves, max_depth=max_depth)

    leaves = get_leaves(root)

    col_defs, cluster_names = nice_cluster_names([leaf.dim_split for leaf in leaves])

    for l, leaf in enumerate(leaves):
        leaf.df["Segment_id"] = l

    # The convention in the calling code is first dims then time
    re_df = pd.concat([leaf.df for leaf in leaves]).sort_values(
        dims + fitter.groupby_dims
    )

    if len(fitter.groupby_dims) == 2:  # Time series with weights
        re_df_w = re_df[re_df["chunk"] == "Weights"].copy()
        re_df = re_df[re_df["chunk"] == "Average"]
        w_total_prediction = (re_df_w["prediction"] * re_df_w["weights"]).values
    else:
        w_total_prediction = None

    X = pd.get_dummies(re_df["Segment_id"]).values

    return (
        csc_matrix(X),
        col_defs,
        cluster_names,
        re_df["prediction"].values,
        w_total_prediction,
    )


def error(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y) ** 2)


class ModelNode:
    def __init__(
        self,
        df: pd.DataFrame,
        fitter: Fitter,
        dims: List[str],
        time_col: str = None,
        max_depth: Optional[int] = None,
        dim_split: Optional[Dict[str, List]] = None,
    ):
        self.df = df.copy().sort_values(dims + fitter.groupby_dims)
        self.fitter = fitter
        self.dims = dims
        self.time_col = time_col
        self.max_depth = max_depth
        self._best_submodels = None
        self._error_improvement = float("-inf")
        self.children = None
        self.dim_split = dim_split or {}
        self.model = None
        # For dimension splitting candidates, hardwired for now
        self.num_bins = 10

    @property
    def depth(self):
        return len(self.dim_split)

    @property
    def error(self):
        this_X = self.df[self.dims + self.fitter.groupby_dims]
        if self.model is None:
            self.model = copy.deepcopy(self.fitter)
            self.model.fit(
                X=this_X,
                y=self.df["__avg"],
                sample_weight=self.df["weights"],
            )
        self.df["prediction"] = self.model.predict(this_X)
        return self.model.error(
            X=this_X,
            y=self.df["__avg"],
            sample_weight=self.df["weights"],
        )

    @property
    def error_improvement(self):
        if self.max_depth is None:
            self.max_depth = float("inf")
        if self._best_submodels is None:
            best_error = float("inf")

            if self.depth > self.max_depth:
                raise ValueError("Max depth exceeded")
            elif self.depth == self.max_depth:
                iter_dims = list(self.dim_split.keys())
            else:
                iter_dims = self.dims

            for dim in iter_dims:
                if len(self.df[dim].unique()) == 1:
                    continue

                elif len(self.df[dim].unique()) == 2:
                    vals = self.df[dim].unique()
                    partitions = [([vals[0]], [vals[1]])]
                else:
                    if isinstance(self.fitter, AverageFitter):
                        partitions = target_encoding_partitions(
                            self.df, dim, self.num_bins
                        )
                    else:
                        partitions = kmeans_partition(
                            self.df, dim, self.fitter.groupby_dims
                        )

                for dim_values1, dim_values2 in partitions:
                    left = self.df[self.df[dim].isin(dim_values1)]
                    right = self.df[self.df[dim].isin(dim_values2)]
                    left_candidate = ModelNode(
                        df=left,
                        fitter=self.fitter,
                        dims=self.dims,
                        time_col=self.time_col,
                        dim_split={**self.dim_split, **{dim: dim_values1}},
                        max_depth=self.max_depth,
                    )
                    right_candidate = ModelNode(
                        df=right,
                        fitter=self.fitter,
                        dims=self.dims,
                        time_col=self.time_col,
                        dim_split={**self.dim_split, **{dim: dim_values2}},
                        max_depth=self.max_depth,
                    )

                    err = left_candidate.error + right_candidate.error
                    if err < best_error:
                        best_error = err
                        self._error_improvement = self.error - best_error
                        self._best_submodels = (left_candidate, right_candidate)

        return self._error_improvement


def get_best_subtree_result(
    node: ModelNode, max_depth: Optional[int] = 1000
) -> ModelNode:
    if node.children is None or node.depth >= max_depth:
        return node
    else:
        node1 = get_best_subtree_result(node.children[0])
        node2 = get_best_subtree_result(node.children[1])
        improvement1 = node1.error_improvement
        improvement2 = node2.error_improvement
        if improvement1 > improvement2:
            return node1
        else:
            return node2


def build_tree(root: ModelNode, num_leaves: int, max_depth: Optional[int] = 1000):
    for i in range(num_leaves - 1):
        print(f"Adding node {i+1}...")
        best_node = get_best_subtree_result(root, max_depth)
        if best_node.error_improvement > 0:
            best_node.children = best_node._best_submodels
            print("Done!")
        else:
            print("No more improvement, stopping")
            break


def get_leaves(node: ModelNode) -> List[ModelNode]:
    if node.children is None:
        return [node]
    else:
        return get_leaves(node.children[0]) + get_leaves(node.children[1])
