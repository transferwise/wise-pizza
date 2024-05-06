import copy
import json
import warnings
from typing import Optional, Union, List, Dict, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags

from wise_pizza.solve.find_alpha import clean_up_min_max, find_alpha
from wise_pizza.make_matrix import sparse_dummy_matrix
from wise_pizza.cluster import guided_kmeans
from wise_pizza.preselect import HeuristicSelector
from wise_pizza.time import extend_dataframe
from wise_pizza.slicer_facades import SliceFinderPredictFacade
from wise_pizza.solve.tree import tree_solver


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
            max_cols=max_cols,
            weights=self.weights,
            totals=self.totals,
            time_basis=time_basis,
            verbose=self.verbose,
        )

        # This returns the candidate vectors in batches
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
            if this_X is not None:
                X_out, col_defs_out = sel(this_X, these_col_defs)

        if self.verbose:
            print("Preselection done!")
        return X_out, col_defs_out

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
        @param solver: Valid values are "lasso" (default), "tree" (for non-overlapping segments), "omp", or "lp"
        @param verbose: If set to a truish value, lots of debug info is printed to console
        @param force_dim: To add dim
        @param force_add_up: To force add up
        @param constrain_signs: To constrain signs
        @param cluster_values In addition to single-value slices, consider slices that consist of a
        group of segments from the same dimension with similar naive averages

        """

        assert solver.lower() in ["lasso", "tree", "omp", "lp"]
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

        # Cast all dimension values to strings
        dim_df = dim_df.astype(str)

        dims = list(dim_df.columns)
        # sort the dataframe by dimension values,
        # making sure the other vectors stay aligned
        dim_df = dim_df.reset_index(drop=True)
        dim_df["totals"] = totals
        dim_df["weights"] = weights

        if time_col is not None:
            dim_df["__time"] = time_col
            dim_df = pd.merge(dim_df, time_basis, left_on="__time", right_index=True)
            sort_dims = dims + ["__time"]
        else:
            sort_dims = dims

        dim_df = dim_df.sort_values(sort_dims)
        dim_df = dim_df[dim_df["weights"] > 0]

        # Transform the time basis from table by date to matrices by dataset row
        if time_col is not None:
            self.basis_df = time_basis
            self.time_basis = {}
            for c in time_basis.columns:
                this_ts = dim_df[c].values.reshape((-1, 1))
                max_val = np.abs(this_ts).max()
                # take all the values a nudge away from zero so we can divide by them later
                this_ts[np.abs(this_ts) < 1e-6 * max_val] = 1e-6 * max_val
                self.time_basis[c] = csc_matrix(this_ts)
            self.time = dim_df["__time"].values
        else:
            self.time_basis = None

        self.weights = dim_df["weights"].values
        self.totals = dim_df["totals"].values

        # While we still have weights and totals as part of the dataframe, let's produce clusters
        # of dimension values with similar outcomes
        clusters = defaultdict(list)
        self.cluster_names = {}

        if solver == "tree":
            if cluster_values:
                warnings.warn(
                    "Ignoring cluster_values argument as it's irrelevant for tree solver"
                )
                cluster_values = False

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

        dim_df = dim_df[dims]  # if time_col is None else dims + ["__time"]]
        self.dim_df = dim_df

        if solver == "tree":
            self.X, self.reg, self.col_defs = tree_solver(
                self.dim_df, self.weights, self.totals, self.time_basis
            )
            self.nonzeros = np.array(range(self.X.shape[0])) == 1.0
            Xw = csc_matrix(diags(self.weights) @ self.X)
        else:

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
                assert len(self.col_defs) == self.X.shape[1]
                self.min_depth = min_depth
                self.max_depth = max_depth
                self.dims = list(dim_df.columns)

            Xw = csc_matrix(diags(self.weights) @ self.X)

            if self.verbose:
                print("Starting solve!")
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

        if self.verbose:
            print("Solver done!!")

        self.segments = [
            {"segment": self.col_defs[i], "index": int(i)} for i in self.nonzeros
        ]

        wgts = np.array((np.abs(Xw[:, self.nonzeros]) > 0).sum(axis=0))[0]

        for i, s in enumerate(self.segments):
            segment_def = s["segment"]
            this_vec = (
                self.X[:, s["index"]]
                .toarray()
                .reshape(
                    -1,
                )
            )
            if "time" in segment_def:
                # Divide out the time profile mult - we've made sure it's always nonzero
                time_mult = (
                    self.time_basis[segment_def["time"]]
                    .toarray()
                    .reshape(
                        -1,
                    )
                )
                dummy = (this_vec / time_mult).astype(int).astype(np.float64)
            else:
                dummy = this_vec

            this_wgts = self.weights * dummy
            wgt = this_wgts.sum()
            # assert wgt == wgts[i]
            s["orig_i"] = i
            s["coef"] = self.reg.coef_[i]
            s["impact"] = np.abs(s["coef"]) * (np.abs(this_vec) * self.weights).sum()
            s["avg_impact"] = s["impact"] / sum(self.weights)
            s["total"] = (self.totals * dummy).sum()
            s["seg_size"] = wgt
            s["naive_avg"] = s["total"] / wgt

        if time_basis is not None:  # it's a time series product
            # Do we need this bit at all?
            predict = self.reg.predict(self.X[:, self.nonzeros]).reshape(
                -1,
            )
            davg = (predict * self.weights).sum() / self.weights.sum()
            self.reg.intercept_ = -davg

            # And this is the version to use later in TS plotting
            self.predict_totals = self.reg.predict(Xw[:, self.nonzeros]).reshape(
                -1,
            )

            # self.enrich_segments_with_timeless_reg(self.segments, self.y_adj)

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

    # def enrich_segments_with_timeless_reg(self, segments, y):
    #     pre_X = [s["dummy"] for s in segments]
    #     X = np.concatenate()
    #     pass

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

    def segment_impact_on_totals(self, s: Dict) -> np.ndarray:
        return s["seg_total_vec"]

    @property
    def actual_totals(self):
        return self.totals + self.y_adj

    @property
    def predicted_totals(self):
        return self.predict_totals + self.y_adj

    def predict(
        self,
        steps: Optional[int] = None,
        basis: Optional[pd.DataFrame] = None,
        weight_df: Optional[pd.DataFrame] = None,
    ):
        """
        Predict the totals using the given basis
        :param basis: Time profiles going into the future, time as index
        :param weight_df: dataframe with all dimensions and time, plus weight column
        :return:
        """
        if basis is None:
            if weight_df is None:
                if steps is None:
                    steps = 6
            else:
                steps = len(weight_df[self.time_name].unique())
                warnings.warn(
                    "Ignoring steps argument, using weight_df to determine forecast horizon"
                )
            basis = extend_dataframe(self.basis_df, steps)
        else:
            if steps is not None:
                raise ValueError("Can't specify both basis and steps")

        last_ts = self.time.max()
        new_basis = basis[basis.index > last_ts]

        dims = [c for c in self.dim_df.columns if c != "__time"]
        if weight_df is None:
            pre_dim_df = self.dim_df[dims].drop_duplicates()
            pre_dim_df[self.size_name] = 1

            # Do a Cartesian join of the time basis and the dimensions
            b = new_basis.reset_index().rename(columns={"index": self.time_name})
            b["key"] = 1
            pre_dim_df["key"] = 1

            # Perform the merge operation
            new_dim_df = pd.merge(b, pre_dim_df, on="key").drop("key", axis=1)
        else:
            # This branch is as yet untested
            assert self.time_name in weight_df.columns
            for d in dims:
                assert d in weight_df.columns

            new_dim_df = pd.merge(
                new_basis, weight_df, left_index=True, right_on=self.time_name
            )

        # Join the (timeless) averages to these future rows
        new_dim_df = pd.merge(new_dim_df, self.avg_df, on=dims, how="left").rename(
            columns={"avg": "avg_future"}
        )
        # TODO: replace with a simple regression for more plausible baselines
        global_avg = (
            new_dim_df[self.total_name].sum() / new_dim_df[self.size_name].sum()
        )
        new_dim_df["avg_future"] = new_dim_df["avg_future"].fillna(global_avg)

        # Construct the dummies for predicting

        segments = copy.deepcopy(self.segments)
        new_X = np.zeros((len(new_dim_df), len(segments)))

        new_totals = np.zeros(len(new_dim_df))
        for s in segments:
            dummy, Xi = make_dummy(s["segment"], new_dim_df)
            new_X[:, s["orig_i"]] = Xi
            s["dummy"] = np.concatenate([s["dummy"], dummy], axis=0)
            future_impact = new_dim_df[self.size_name].values * Xi * s["coef"]
            new_totals += future_impact
            s["seg_total_vec"] = np.concatenate([s["seg_total_vec"], future_impact])

        # Evaluate the regression
        new_avg = self.reg.predict(new_X)

        # Add in the constant averages and multiply by the weights
        new_totals = (new_avg + new_dim_df["avg_future"].values) * new_dim_df[
            self.size_name
        ].values

        # Return the dataframe with totals and weights
        new_dim_df[self.total_name] = pd.Series(data=new_totals, index=new_dim_df.index)

        out = SliceFinderPredictFacade(self, new_dim_df, segments)
        return out


def make_dummy(segment_def: Dict[str, str], dim_df: pd.DataFrame) -> np.ndarray:
    """
    Function to make dummy vector from segment definition
    @param segment_def: Segment definition
    @param dim_df: Dataset with dimensions
    @return: Dummy vector
    """
    dummy = np.ones((len(dim_df)))
    for k, v in segment_def.items():
        if k != "time":
            dummy = dummy * (dim_df[k] == v).values

    if "time" in segment_def:
        Xi = dummy * dim_df[segment_def["time"]].values
    else:
        raise ValueError("Segments for time series prediction must contain time!")

    assert np.abs(dummy).sum() > 0
    return dummy, Xi


class SlicerPair:
    def __init__(self, s1: SliceFinder, s2: SliceFinder):
        self.s1 = s1
        self.s2 = s2
        self.task = ""

    def summary(self):
        return _summary(self)
