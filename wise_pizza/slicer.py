import json
from typing import Optional, Union, List, Dict, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags

from wise_pizza.find_alpha import clean_up_min_max, find_alpha
from wise_pizza.make_matrix import sparse_dummy_matrix
from wise_pizza.cluster import guided_kmeans
from wise_pizza.preselect import HeuristicSelector


def _summary(obj) -> str:
    out = {"task": obj.task, "segments": obj.segments}
    return json.dumps(out)


class SliceFinder:
    """
    SliceFinder class to find unusual slices
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.min_depth = kwargs.get("min_depth", None)
        self.max_depth = kwargs.get("max_depth", None)
        self.dims = None
        self.X = None
        self.col_defs = None
        self.reg = None
        self.nonzeros = None
        self.weights = None
        self.verbose = 0
        self.task = ""

    def _init_mat(
        self,
        dim_df: pd.DataFrame,
        min_depth: int,
        max_depth: int,
        max_cols: int = 300,
        force_dim: Optional[str] = None,
        clusters: Optional[Dict[str, Sequence[str]]] = None,
        time_basis: Optional[pd.DataFrame] = None,
    ):
        """
        Function to initialize sparse matrix
        @param dim_df: Dataset with dimensions
        @param min_depth: Minimum number of dimension to constrain in segment definition
        @param max_depth: Maximum number of dimension to constrain in segment definition
        @param max_cols: Maxumum number of segments to consider
        @param force_dim: To add dim
        @param clusters: groups of same-dimension values to be considered as candidate segments
        @param time_basis: the set of time profiles to scale the candidate segments by
        @return:
        """
        sel = HeuristicSelector(
            max_cols=max_cols, weights=self.weights, totals=self.totals
        )

        basis_iter = sparse_dummy_matrix(
            dim_df,
            min_depth=min_depth,
            max_depth=max_depth,
            verbose=self.verbose,
            force_dim=force_dim,
            clusters=clusters,
            cluster_names=self.cluster_names,
            time_basis=time_basis,
        )

        # do pre-filter recursively
        for this_X, these_col_defs in basis_iter:
            out = sel(this_X, these_col_defs)

        return out

    def fit(
        self,
        dim_df: pd.DataFrame,
        totals: pd.Series,
        weights: pd.Series = None,
        time_col: pd.Series = None,
        time_basis: pd.DataFrame = None,
        min_segments: int = 10,
        max_segments: int = None,
        min_depth: int = 1,
        max_depth: int = 3,
        solver: str = "lp",
        verbose: Union[bool, None] = None,
        force_dim: Optional[str] = None,
        force_add_up: bool = False,
        constrain_signs: bool = True,
        cluster_values: bool = True,
    ):
        """
        Function to fit slicer and find segments
        @param dim_df: Dataset with dimensions
        @param totals: Column with totals
        @param weights: Column with sizes
        @param min_segments: Minimum number of segments to find
        @param max_segments: Maximum number of segments to find, defaults to min_segments
        @param min_depth: Minimum number of dimension to constrain in segment definition
        @param max_depth: Maximum number of dimension to constrain in segment definition
        @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
        @param verbose: If set to a truish value, lots of debug info is printed to console
        @param force_dim: To add dim
        @param force_add_up: To force add up
        @param constrain_signs: To constrain signs
        @param cluster_values In addition to single-value slices, consider slices that consist of a
        group of segments from the same dimension with similar naive averages

        """
        min_segments, max_segments = clean_up_min_max(min_segments, max_segments)
        if verbose is not None:
            self.verbose = verbose

        totals = np.array(totals).astype(np.float64)

        if weights is None:
            weights = np.ones_like(totals)
        else:
            weights = np.array(weights).astype(np.float64)

        assert min(weights) >= 0
        assert np.sum(np.abs(totals[weights == 0])) == 0

        dims = list(dim_df.columns)
        # sort the dataframe by dimension values,
        # making sure the other vectors stay aligned
        # Why do we do this?
        dim_df = dim_df.reset_index(drop=True)
        dim_df["totals"] = totals
        dim_df["weights"] = weights
        if time_col is not None:
            dim_df["__time"] = time_col
            sort_dims = dims + ["__time"]
        else:
            sort_dims = dims

        dim_df = dim_df.sort_values(sort_dims)
        dim_df = dim_df[dim_df["weights"] != 0]

        # Transform the time basis from table by date to matrices by dataset row
        if time_col is not None:
            tall_basis = pd.merge(
                dim_df[["__time"]], time_basis, left_on="__time", right_index=True
            ).drop(columns=["__time"])
            self.time_basis = {}
            for c in tall_basis.columns:
                self.time_basis[c] = csc_matrix(tall_basis[c].values.reshape((-1,1)))
        else:
            self.time_basis = None

        self.weights = dim_df["weights"].values
        self.totals = dim_df["totals"].values

        # While we still have weights and totals as part of the dataframe, let's produce clusters
        # of dimension values with similar outcomes
        clusters = defaultdict(list)
        self.cluster_names = {}
        if cluster_values:
            for dim in dims:
                if (
                    len(dim_df[dim].unique()) >= 6
                ):  # otherwise what's the point in clustering?
                    grouped_df = (
                        dim_df[[dim, "totals", "weights"]]
                        .groupby(dim, as_index=False)
                        .sum()
                    )
                    grouped_df["avg"] = grouped_df["totals"] / grouped_df["weights"]
                    grouped_df["cluster"], _ = guided_kmeans(grouped_df["avg"])
                    pre_clusters = (
                        grouped_df[["cluster", dim]]
                        .groupby("cluster")
                        .agg({dim: lambda x: "@@".join(x)})
                        .values
                    )
                    # filter out clusters with only one element
                    these_clusters = [c for c in pre_clusters.reshape(-1) if "@@" in c]
                    # create short cluster names
                    for i, c in enumerate(these_clusters):
                        self.cluster_names[f"{dim}_cluster_{i+1}"] = c
                    clusters[dim] = [
                        c for c in self.cluster_names.keys() if c.startswith(dim)
                    ]

        dim_df = dim_df[dims] # if time_col is None else dims + ["__time"]]

        # lazy calculation of the dummy matrix (calculation can be very slow)
        if (
            list(dim_df.columns) != self.dims
            or max_depth != self.max_depth
            or self.X is not None
            and len(dim_df) != self.X.shape[1]
        ):
            self.X, self.col_defs = self._init_mat(
                dim_df,
                min_depth,
                max_depth,
                force_dim=force_dim,
                clusters=clusters,
                time_basis=self.time_basis,
            )
            self.min_depth = min_depth
            self.max_depth = max_depth
            self.dims = list(dim_df.columns)

        Xw = csc_matrix(diags(self.weights) @ self.X)

        self.reg, self.nonzeros = find_alpha(
            Xw,
            self.totals,
            max_nonzeros=max_segments,
            solver=solver,
            min_nonzeros=min_segments,
            verbose=self.verbose,
            adding_up_regularizer=force_add_up,
            constrain_signs=constrain_signs,
        )

        self.segments = [{"segment": self.col_defs[i]} for i in self.nonzeros]
        wgts = np.array(Xw[:, self.nonzeros].sum(axis=0))[0]

        for i, s in enumerate(self.segments):
            s["coef"] = self.reg.coef_[i]
            s["impact"] = s["coef"] * wgts[i]
            s["avg_impact"] = s["impact"] / sum(wgts)
            s["total"] = (self.totals * self.X[:, self.nonzeros[i]]).sum()
            s["seg_size"] = wgts[i]
            s["naive_avg"] = s["total"] / wgts[i]

        self.segments = self.order_segments(self.segments)

        # In some cases (mostly in a/b exps we have a situation where there is no any diff in totals/sizes)
        if len(self.segments) == 0:
            self.segments.append(
                {
                    "segment": {"No unusual segments": "No unusual segments"},
                    "coef": 0,
                    "impact": 0,
                    "avg_impact": 0,
                    "total": 0,
                    "seg_size": 0,
                    "naive_avg": 0,
                }
            )

    @staticmethod
    def order_segments(segments: List[Dict[str, any]]):
        pos_seg = [s for s in segments if s["impact"] > 0]
        neg_seg = [s for s in segments if s["impact"] < 0]

        return sorted(pos_seg, key=lambda x: abs(x["impact"]), reverse=True) + sorted(
            neg_seg, key=lambda x: abs(x["impact"]), reverse=True
        )

    @staticmethod
    def segment_to_str(segment: Dict[str, any]):
        s = {
            k: v
            for k, v in segment.items()
            if k not in ["coef", "impact", "avg_impact"]
        }
        return str(s)

    @property
    def segment_labels(self):
        return [self.segment_to_str(s["segment"]) for s in self.segments]

    def summary(self):
        return _summary(self)

    @property
    def relevant_cluster_names(self):
        relevant_clusters = {}
        for s in self.segments:
            for c in s["segment"].values():
                if c in self.cluster_names:
                    relevant_clusters[c] = self.cluster_names[c].replace("@@", ", ")
        return relevant_clusters


class SlicerPair:
    def __init__(self, s1: SliceFinder, s2: SliceFinder):
        self.s1 = s1
        self.s2 = s2
        self.task = ""

    def summary(self):
        return _summary(self)
