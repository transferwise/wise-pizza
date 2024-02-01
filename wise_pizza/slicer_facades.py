import copy
from typing import Optional, Dict

import numpy as np
import pandas as pd

from wise_pizza.slicer_plotting import SliceFinderPlottingInterface
from wise_pizza.transform import TransformWithWeights, IdentityTransform
from wise_pizza.plotting_time import plot_time


class NotFittedError(Exception):
    """Exception class to raise if predict is called before fit."""

    pass


class SliceFinderPredictFacade(SliceFinderPlottingInterface):
    def __init__(self, sf: "SliceFinder", predict_df: pd.DataFrame, segments: Dict):
        self.sf = sf
        self.df = predict_df
        self._segments = segments

    def plot(
        self,
        plot_is_static=False,
        width=1200,
        height=2000,
        return_fig=False,
        average_name=None,
    ):
        plot_time(
            self,
            # plot_is_static=plot_is_static,
            width=width,
            height=height,
            # return_fig=return_fig,
            average_name=average_name,
        )

    @property
    def actual_totals(self):
        return np.concatenate(
            [self.sf.actual_totals, np.nan * np.zeros((len(self.df)))]
        )

    @property
    def predicted_totals(self):
        return np.concatenate(
            [self.sf.predicted_totals, self.df[self.sf.total_name].values]
        )

    @property
    def actual_avg(self):
        return self.actual_totals / self.weights

    @property
    def predicted_avg(self):
        return self.predicted_totals / self.weights

    @property
    def weights(self):
        return np.concatenate([self.sf.weights, self.df[self.sf.size_name].values])

    @property
    def segments(self):
        return self._segments

    # @property
    # def y_adj(self):
    #     return np.concatenate(
    #         [
    #             self.sf.y_adj,
    #             self.df["avg_future"].values * self.df[self.sf.size_name].values,
    #         ]
    #     )

    @property
    def time(self):
        # This is the column with the time period for each dataset row
        return np.concatenate([self.sf.time, self.df[self.sf.time_name].values])

    @property
    def total_name(self):
        return self.sf.total_name

    @property
    def size_name(self):
        return self.sf.size_name

    def segment_impact_on_totals(self, s: Dict) -> np.ndarray:
        return s["seg_total_vec"]

    def predict(
        self,
        steps: Optional[int] = None,
        basis: Optional[pd.DataFrame] = None,
        weight_df: Optional[pd.DataFrame] = None,
    ):
        raise NotImplementedError("Can't predict on a prediction")

    @property
    def predicted_df(self):
        return self.df


class TransformedSliceFinder(SliceFinderPlottingInterface):
    def __init__(
        self,
        sf: SliceFinderPlottingInterface,
        transformer: Optional[TransformWithWeights] = None,
    ):
        # For now, just use log(1+x) as transform, assume sf was fitted on transformed data
        self.sf = sf
        if transformer is None:
            self.tf = IdentityTransform()
        else:
            self.tf = transformer

        trans_avg = sf.actual_totals / sf.weights  # averages in the transformed space
        trans_pred_avg = sf.predicted_totals / sf.weights

        self.actual_avg = self.tf.inverse_transform_mean(trans_avg)  # a_i
        self.predicted_avg = self.tf.inverse_transform_mean(trans_pred_avg)

        # When we do prediction, the corresponding actuals are nans, so need to use predictions
        # to fill in the gaps for inverse-transforming weights
        patched_avg = copy.deepcopy(trans_avg)
        patched_avg[np.isnan(patched_avg)] = trans_pred_avg[np.isnan(patched_avg)]
        self._weights = self.tf.inverse_transform_weight(sf.weights, patched_avg)

        # probably because of some convexity effect of the exp,
        # predictions end up a touch too high on average post-inverse transform
        # So let's introduce a scaling factor to fix that
        # Can't use self.predicted_totals here, because it needs self.pred_scaler
        predicted_totals = self.predicted_avg * self.weights
        # TODO: debug this!
        # assert not np.isnan(np.sum(predicted_totals))

        nice = ~np.isnan(self.actual_totals * predicted_totals)
        self.pred_scaler = np.sum(self.actual_totals[nice]) / np.sum(
            predicted_totals[nice]
        )

        # Now let's make sure single-segment impacts add up to total impact
        self.segment_mult = 1.0

        # Try to make individual segment  impacts add up to total regression post-transform
        # Didn't really make much difference
        # sum_marginals = 0
        # base, _ = self.tf.inverse_transform_totals_weights(self.sf.y_adj, self.sf.weights)
        # pt, _ = self.tf.inverse_transform_totals_weights(self.sf.predicted_totals, self.sf.weights)
        # total_diff = self.pred_scaler*(pt-base)
        #
        # for s in sf.segments:
        #     sum_marginals += self.segment_impact_on_totals(s)
        #
        # self.segment_mult = np.median(np.abs(total_diff/sum_marginals))

    @property
    def actual_totals(self):
        return self.actual_avg * self.weights

    @property
    def predicted_totals(self):
        return self.pred_scaler * self.predicted_avg * self.weights

    @property
    def weights(self):
        return self._weights

    @property
    def segments(self):
        return self.sf.segments

    # @property
    # def y_adj(self):
    #     return self.sf.y_adj

    @property
    def time(self):
        return self.sf.time

    @property
    def total_name(self):
        return self.sf.total_name

    @property
    def size_name(self):
        return self.sf.size_name

    def segment_impact_on_totals(self, s: Dict) -> np.ndarray:
        # Calculate transformed prediction with and without the segment
        totals_without_segment = (
            self.sf.predicted_totals - self.sf.segment_impact_on_totals(s)
        )

        # Transform back and subtract
        pt, _ = self.tf.inverse_transform_totals_weights(
            self.sf.predicted_totals, self.sf.weights
        )
        dpt, _ = self.tf.inverse_transform_totals_weights(
            totals_without_segment, self.sf.weights
        )

        return self.pred_scaler * self.segment_mult * (pt - dpt)

    def predict(
        self,
        steps: Optional[int] = None,
        basis: Optional[pd.DataFrame] = None,
        weight_df: Optional[pd.DataFrame] = None,
    ):
        # Just predict in the transformed space, then return same wrapper
        inner_predict = self.sf.predict(steps=steps, basis=basis, weight_df=weight_df)
        out = TransformedSliceFinder(inner_predict, self.tf)
        out.__class__.plot = SliceFinderPredictFacade.plot
        return out

    @property
    def predicted_df(self):
        if isinstance(self.sf, SliceFinderPredictFacade):
            return self.sf.df
        else:
            raise NotFittedError("Call predict method first to create the prediction")

    # TODO: factor this out
    def plot(
        self,
        plot_is_static=False,
        width=1200,
        height=2000,
        return_fig=False,
        average_name=None,
    ):
        plot_time(
            self,
            # plot_is_static=plot_is_static,
            width=width,
            height=height,
            # return_fig=return_fig,
            average_name=average_name,
        )
