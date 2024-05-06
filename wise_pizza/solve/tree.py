import copy
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import category_encoders as ce

from .weighted_quantiles import weighted_quantiles


def tree_solver(
    dim_df: pd.DataFrame,
    weights: np.ndarray,
    totals: np.ndarray,
    time_basis: Optional[np.ndarray] = None,
    max_depth: int = 3,
    num_leaves: Optional[int] = None,
):
    if time_basis is None:
        fitter = AverageFitter()
    else:
        fitter = TimeFitter()

    df = dim_df.copy().reset_index(drop=True)
    df["__weight"] = weights
    df["__total"] = totals
    df["__avg"] = totals / weights
    df["__avg"] = df["__avg"].fillna(df["__avg"].nanmean())
    for i, vec in enumerate(time_basis.T):
        df[f"__time_{i}"] = vec

    root = ModelNode(df=df, fitter=fitter, dims=dim_df.columns)

    build_tree(root=root, num_leaves=num_leaves, max_depth=max_depth)
    segments = []
    col_defs = []
    for seg in get_leaves(root):
        segments.append(tidy_segment(seg))

    return col_defs


def error(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y) ** 2)


def encode_map(X, y) -> Dict:
    encoder = ce.TargetEncoder()
    encoder.fit(X, y)
    return encoder.mapping


class ModelNode:
    def __init__(
        self,
        df: pd.DataFrame,
        fitter: "Fitter",
        dims: List[str],
        dim_split: Optional[Dict[str, List]] = None,
        depth: int = 0,
    ):
        self.df = df
        self.fitter = fitter
        self.dims = dims
        self._best_submodels = None
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
                y=self.df["__total"],
                sample_weight=self.df["__weight"],
            )
        return self.model.error(self.df)

    @property
    def error_improvement(self):
        if self._best_submodels is None:
            best_error = float("inf")
            for dim in self.dims:
                enc_map = encode_map(self.df[dim], self.df["__avg"])
                self.df[dim + "_encoded"] = self.df[dim].map(encode_map)

                # Get split candidates for brute force search
                deciles = np.array([q / 10.0 for q in range(1, 10)])
                splits = weighted_quantiles(
                    self.df[dim + "_encoded"], deciles, self.df["__weight"]
                )

                for split in splits(self.df[dim + "_encoded"], self.df["__weight"]):
                    left = self.df[self.df[dim + "_encoded"] < split]
                    right = self.df[self.df[dim + "_encoded"] >= split]
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
    # TODO: modify this to also accept max_depth
    for _ in range(num_leaves):
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


class Model:
    def error(self, df: pd.DataFrame) -> float:
        return error(self.predict(df), df[self.target_name])
