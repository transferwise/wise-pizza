from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
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
    root = ModelNode(
        df=dim_df,
        model=Model(weights, totals, time_basis),
        fitter=Fitter(),
        dims=dim_df.columns,
        target_name="target",
    )

    build_tree(root=root, num_leaves=num_leaves, max_depth=max_depth)
    segments = get_leaves(root)

    return X, reg, col_defs


def error(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y) ** 2)


@dataclass
class ModelNode:
    df: pd.DataFrame
    model: "Model"
    fitter: "Fitter"
    dims: List[str]
    target_name: str
    _best_submodels: Tuple["ModelNode", "ModelNode"] | None = None
    children: Tuple["ModelNode", "ModelNode"] | None = None
    dim_split: Dict[str, List[str]] | None = None
    depth: int = 0

    @property
    def error(self):
        return self.model.error(self.df)

    @property
    def error_improvement(self):
        if self._best_submodels is None:
            best_error = float("inf")
            for dim in self.dims:
                encode_map = target_encode(self.df[dim], self.df[self.target_name])
                self.df[dim + "_encoded"] = self.df[dim].map(encode_map)
                for split in splits(self.df[dim + "_encoded"]):
                    left = self.df[self.df[dim + "_encoded"] < split]
                    right = self.df[self.df[dim + "_encoded"] >= split]
                    left_model = self.fitter.fit(left)
                    right_model = self.fitter.fit(right)
                    error = left_model.error(left) + right_model.error(right)
                    if error < best_error:
                        best_error = error
                        dim_values1 = [k for k, v in encode_map.items() if v < split]
                        dim_values2 = [k for k, v in encode_map.items() if v >= split]
                        left_node = ModelNode(
                            df=left,
                            model=left_model,
                            fitter=self.fitter,
                            dims=self.dims,
                            target_name=self.target_name,
                            dim_split={**self.dim_split, **{dim: dim_values1}},
                            depth=self.depth + 1,
                        )
                        right_node = ModelNode(
                            df=right,
                            model=right_model,
                            fitter=self.fitter,
                            dims=self.dims,
                            target_name=self.target_name,
                            dim_split={**self.dim_split, **{dim: dim_values2}},
                            depth=self.depth + 1,
                        )
                        self._error_improvement = self.error - best_error
                        self._best_submodels = (left_node, right_node)

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
