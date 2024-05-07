import copy
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from .weighted_quantiles import weighted_quantiles
from .fitter import AverageFitter, Fitter
from wise_pizza.cluster import nice_cluster_names


def tree_solver(
    dim_df: pd.DataFrame,
    dims: List[str],
    time_basis: Optional[pd.DataFrame] = None,
    max_depth: int = 3,
    num_leaves: Optional[int] = None,
):
    if time_basis is None:
        fitter = AverageFitter()
    else:
        raise NotImplementedError("Time fitter not yet implemented")
        # fitter = TimeFitter(dims, list(time_basis.columns))

    df = dim_df.copy().reset_index(drop=True)
    df["__avg"] = df["totals"] / df["weights"]
    df["__avg"] = df["__avg"].fillna(df["__avg"].mean())

    root = ModelNode(df=df, fitter=fitter, dims=dims)

    build_tree(root=root, num_leaves=num_leaves, max_depth=max_depth)

    leaves = get_leaves(root)

    col_defs, cluster_names = nice_cluster_names([leaf.dim_split for leaf in leaves])

    for l, leaf in enumerate(leaves):
        leaf.df["Segment_id"] = l

    re_df = pd.concat([leaf.df for leaf in leaves]).sort_values(dims)
    X = pd.get_dummies(re_df["Segment_id"]).values

    return csc_matrix(X), col_defs, cluster_names


def error(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y) ** 2)


def target_encode(df: pd.DataFrame, dim: str) -> dict:
    df = df[[dim, "totals", "weights"]]
    agg = df.groupby(dim, as_index=False).sum()
    agg["__avg"] = agg["totals"] / agg["weights"]
    agg["__avg"] = agg["__avg"].fillna(agg["__avg"].mean())
    enc_map = {k: v for k, v in zip(agg[dim], agg["__avg"])}

    if np.isnan(np.array(list(enc_map.values()))).any():
        raise ValueError("NaNs in encoded values")
    return enc_map


class ModelNode:
    def __init__(
        self,
        df: pd.DataFrame,
        fitter: Fitter,
        dims: List[str],
        dim_split: Optional[Dict[str, List]] = None,
        depth: int = 0,
    ):
        self.df = df
        self.fitter = fitter
        self.dims = dims
        self._best_submodels = None
        self._error_improvement = float("-inf")
        self.children = None
        self.dim_split = dim_split or {}
        self.depth = depth
        self.model = None

    @property
    def error(self):
        if self.model is None:
            self.model = copy.deepcopy(self.fitter)
            self.model.fit(
                X=self.df[self.dims],
                y=self.df["totals"],
                sample_weight=self.df["weights"],
            )
        return self.model.error(
            self.df[self.dims], self.df["__avg"], self.df["weights"]
        )

    @property
    def error_improvement(self):
        if self._best_submodels is None:
            best_error = float("inf")
            for dim in self.dims:
                if len(self.df[dim].unique()) == 1:
                    continue
                enc_map = target_encode(self.df, dim)
                self.df[dim + "_encoded"] = self.df[dim].apply(lambda x: enc_map[x])
                if np.any(np.isnan(self.df[dim + "_encoded"])):  # pragma: no cover
                    raise ValueError("NaNs in encoded values")
                # Get split candidates for brute force search
                deciles = np.array([q / 10.0 for q in range(1, 10)])

                splits = weighted_quantiles(
                    self.df[dim + "_encoded"], deciles, self.df["weights"]
                )

                for split in np.unique(splits):
                    left = self.df[self.df[dim + "_encoded"] < split]
                    right = self.df[self.df[dim + "_encoded"] >= split]
                    if len(left) == 0 or len(right) == 0:
                        continue
                    dim_values1 = [k for k, v in enc_map.items() if v < split]
                    dim_values2 = [k for k, v in enc_map.items() if v >= split]
                    left_candidate = ModelNode(
                        df=left,
                        fitter=self.fitter,
                        dims=self.dims,
                        dim_split={**self.dim_split, **{dim: dim_values1}},
                        depth=self.depth + 1,
                    )
                    right_candidate = ModelNode(
                        df=right,
                        fitter=self.fitter,
                        dims=self.dims,
                        dim_split={**self.dim_split, **{dim: dim_values2}},
                        depth=self.depth + 1,
                    )

                    err = left_candidate.error + right_candidate.error
                    if err < best_error:
                        best_error = err
                        self._error_improvement = self.error - best_error
                        self._best_submodels = (left_candidate, right_candidate)

        return self._error_improvement


def mod_improvement(improvement: float, depth: int, max_depth: int) -> float:
    if depth < max_depth:
        return improvement
    else:
        return float("-inf")


def get_best_subtree_result(
    node: ModelNode, max_depth: Optional[int] = 1000
) -> ModelNode:
    if node.children is None or node.depth >= max_depth:
        return node
    else:
        node1 = get_best_subtree_result(node.children[0])
        node2 = get_best_subtree_result(node.children[1])
        improvement1 = mod_improvement(node1.error_improvement, node1.depth, max_depth)
        improvement2 = mod_improvement(node2.error_improvement, node2.depth, max_depth)
        if improvement1 > improvement2:
            return node1
        else:
            return node2


def build_tree(root: ModelNode, num_leaves: int, max_depth: Optional[int] = 1000):
    for _ in range(num_leaves - 1):
        best_node = get_best_subtree_result(root, max_depth)
        if best_node.error_improvement > 0:
            best_node.children = best_node._best_submodels
        else:
            break


def get_leaves(node: ModelNode) -> List[ModelNode]:
    if node.children is None:
        return [node]
    else:
        return get_leaves(node.children[0]) + get_leaves(node.children[1])
