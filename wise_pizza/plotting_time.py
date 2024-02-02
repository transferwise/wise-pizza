import copy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from IPython.display import Image, display
import plotly.io as pio
from plotly.io import to_image
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from wise_pizza.slicer_plotting import SliceFinderPlottingInterface


@dataclass
class PlotData:
    df: pd.DataFrame
    nonflat_segments: List[Dict[str, Any]]
    global_time_label: str
    total_name: str
    average_name: str
    sub_titles: List[str]


def plot_time(
    sf: SliceFinderPlottingInterface,
    width: int = 1000,
    height: int = 1000,
    average_name: Optional[str] = None,
    plot_is_static: bool = False,
    return_fig: bool = False,
):
    plot_data = preprocess_for_ts_plot(sf, average_name)
    num_rows = len(plot_data.nonflat_segments) + 1
    fig = make_subplots(
        rows=num_rows,
        cols=3,
        subplot_titles=sum(plot_data.sub_titles, []),
        specs=[[{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}]]
        * num_rows,
    )

    plot_single_ts(plot_data, fig, col_nums=(3, 2))
    plot_weights(plot_data, fig, col_num=1)

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 10

    fig.update_layout(
        title_text=f"Actuals vs explanation by segment",
        showlegend=True,
        width=width,
        height=height,
    )
    if plot_is_static:
        image_bytes = to_image(fig, format="png", scale=2)
        return Image(image_bytes, height=height, width=width)
    else:
        if return_fig:
            return fig
        else:
            fig.show()


def plot_ts_pair(
    sf1: SliceFinderPlottingInterface,
    sf2: SliceFinderPlottingInterface,
    width,
    height,
    average_name: str = None,
    plot_is_static: bool = False,
    return_fig: bool = False,
    use_fitted_weights: bool = False,
):
    # if use_fitted_weights:
    #     sf = copy.deepcopy(sf)
    #     sf.s2.totals = (sf.s2.totals/sf.s2.weights)*sf.s1.totals
    #     sf.s2.weights = sf.s1.totals

    wgt_plot_data = preprocess_for_ts_plot(sf1, average_name)  # average name correct?
    totals_plot_data = preprocess_for_ts_plot(sf2, average_name)
    num_rows = max(
        len(wgt_plot_data.nonflat_segments) + 1,
        len(totals_plot_data.nonflat_segments) + 1,
    )
    subplot_titles = []
    for i in range(num_rows):
        if i < len(wgt_plot_data.sub_titles):
            # Totals from weights regression
            subplot_titles.append(wgt_plot_data.sub_titles[i][2])
        else:
            subplot_titles.append("")
        if i < len(totals_plot_data.sub_titles):
            subplot_titles.append(totals_plot_data.sub_titles[i][1])
            subplot_titles.append(totals_plot_data.sub_titles[i][2])
        else:
            subplot_titles.append("")
            subplot_titles.append("")

    fig = make_subplots(
        rows=num_rows,
        cols=3,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": True}] * 3] * num_rows,
    )
    plot_single_ts(wgt_plot_data, fig, col_nums=(1, None), showlegend=False)  # 1, None
    plot_single_ts(totals_plot_data, fig, col_nums=(3, 2))  # 3,2

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 10

    fig.update_layout(
        title_text=f"Actuals vs explanation by segment",
        showlegend=True,
        width=width,
        height=height,
    )

    if plot_is_static:
        image_bytes = to_image(fig, format="png", scale=2)
        return Image(image_bytes, height=height, width=width)
    else:
        if return_fig:
            return fig
        else:
            fig.show()


def plot_single_ts(
    plotdata: PlotData, fig, showlegend: bool = True, col_nums: Tuple[int, int] = (1, 2)
):
    for i, s in enumerate(plotdata.nonflat_segments):
        agg_df = plotdata.df[s["dummy"] == 1.0].groupby("time", as_index=False).sum()
        # Create subplots
        simple_ts_plot(
            fig,
            agg_df["time"],
            agg_df["totals"],
            agg_df["weights"],
            reg_seg=agg_df[s["plot_segment"]],
            reg_totals=agg_df["Regr totals"],
            row_num=i + 2,
            showlegend=False,
            col_nums=col_nums,
        )

    # Show the actuals for stuff not in segments
    outside = np.abs(sum([s["dummy"] for s in plotdata.nonflat_segments])) < 1e-8

    left = plotdata.df[outside].groupby("time", as_index=False).sum()
    all_data = plotdata.df.groupby("time", as_index=False).sum()

    simple_ts_plot(
        fig,
        all_data["time"],
        all_data["totals"],
        all_data["weights"],
        reg_seg=all_data["reg_time_profile"],
        reg_totals=all_data["Regr totals"],
        leftover_totals=left["totals"],
        leftover_avgs=left["totals"] / left["weights"],
        row_num=1,
        showlegend=showlegend,
        col_nums=col_nums,
    )


def plot_weights(plotdata: PlotData, fig, col_num: int = 1):
    for i, s in enumerate(plotdata.nonflat_segments):
        agg_df = plotdata.df[s["dummy"] == 1.0].groupby("time", as_index=False).sum()
        zeros = np.zeros(len(agg_df))
        # Create subplots
        simple_ts_plot(
            fig,
            agg_df["time"],
            agg_df["weights"],
            np.ones(len(agg_df)),
            reg_seg=None,
            reg_totals=None,
            row_num=i + 2,
            showlegend=False,
            col_nums=(col_num, None),
        )

    # Show the actuals for stuff not in segments
    outside = np.abs(sum([s["dummy"] for s in plotdata.nonflat_segments])) < 1e-8

    left = plotdata.df[outside].groupby("time", as_index=False).sum()
    all_data = plotdata.df.groupby("time", as_index=False).sum()

    simple_ts_plot(
        fig,
        all_data["time"],
        all_data["weights"],
        np.ones(len(all_data)),
        reg_seg=None,
        reg_totals=None,
        leftover_totals=left["weights"],
        leftover_avgs=left["totals"] / left["weights"],
        row_num=1,
        showlegend=False,
        col_nums=(col_num, None),
    )


def same_apart_from_time(s1, s2) -> bool:
    return np.sum(np.abs(s1["dummy"] - s2["dummy"])) < 0.5


def preprocess_for_ts_plot(
    sf: SliceFinderPlottingInterface, average_name: Optional[str] = None
) -> PlotData:
    if average_name is None:
        average_name = "Averages"

    df = pd.DataFrame(
        {
            "totals": sf.actual_totals,
            "Regr totals": sf.predicted_totals,
            "weights": sf.weights,
            "time": sf.time,
        }
    )
    df["reg_time_profile"] = 0.0

    # do a pass over the segments, sorting them into time-only and rest
    global_reg = 0.0
    global_time_profile_names = []
    nonflat_segments = []

    # adj_avg = sf.y_adj.sum() / sf.weights.sum()

    # rel_adj = sf.y_adj - adj_avg * sf.weights

    for i, s in enumerate(sf.segments):
        # Get the segment definition
        segment_def = s["segment"]
        seg_impact = sf.segment_impact_on_totals(s)

        if len(segment_def) > 1:
            almost_duplicate = False
            for s2 in nonflat_segments:
                if same_apart_from_time(s, s2):
                    almost_duplicate = True
                    df[s2["plot_segment"]] += seg_impact
                    s2["segment"]["time"] = (
                        s2["segment"]["time"] + "," + s["segment"]["time"]
                    )
            if not almost_duplicate:
                # offset the segment by actual averages' difference from the global average
                df[f"Seg {i + 1}"] = seg_impact  # + s["dummy"]*rel_adj
                s["plot_segment"] = f"Seg {i+1}"
                nonflat_segments.append(copy.deepcopy(s))

        elif len(segment_def) == 1:
            # Accumulate all pure time profiles into one
            # TODO: this de-duping is almost the same as above, merge!
            df["reg_time_profile"] += seg_impact
            global_time_profile_names.append(segment_def["time"])

        # Do the difference between segment average and global average, for display

    # now create the plots
    if len(global_time_profile_names):
        global_time_label = ", time:" + ",".join(global_time_profile_names)
    else:
        global_time_label = ""

    seg_names = ["All" + global_time_label] + [
        drop_time(s["segment"]) for s in nonflat_segments
    ]
    sub_titles = [
        [
            f"{sf.size_name} for <br>" + f"{s}",
            f"{average_name} for <br>" + f"{s}",
            f"{sf.total_name} for <br>" + f"{s}",
        ]
        if s != "All" and s != global_time_label
        else [
            f"{sf.size_name} for " + "<br>".join([key for key in s]),
            f"{average_name} for " + "<br>".join([key for key in s]),
            f"{sf.total_name} for " + "<br>".join([key for key in s]),
        ]
        for s in seg_names
    ]
    # sub_titles = sum(sub_titles, start=[])

    plot_data = PlotData(
        df, nonflat_segments, global_time_label, sf.total_name, average_name, sub_titles
    )
    return plot_data


def drop_time(s: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(s)
    s.pop("time")
    return s


def naive_dummy(dim_df, seg_def):
    dummy = np.ones(len(dim_df))
    for col, val in seg_def.items():
        if col == "time":
            continue
        dummy[dim_df[col].values != val] = 0
    return dummy


def simple_ts_plot(
    fig,
    time,
    totals,
    weights,
    reg_totals=None,
    leftover_totals=None,
    leftover_avgs=None,
    reg_seg=None,
    row_num=1,
    showlegend: bool = False,
    col_nums: Tuple[int, int] = (1, 2),
):
    for col in col_nums:
        if col == col_nums[0]:
            mult = 1.0
        else:
            mult = 1 / weights

        if col is None:
            continue

        fig.add_trace(
            go.Bar(
                x=time,
                y=totals * mult,
                name=f"Actuals",
                marker=dict(color="#ffc091"),
                showlegend=showlegend and col == col_nums[0],
            ),
            row=row_num,
            col=col,
        )
        if reg_totals is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=reg_totals * mult,
                    mode="lines",
                    name=f"Regression",
                    line=dict(color="#a0e1e1"),
                    showlegend=showlegend and col == col_nums[0],
                ),
                row=row_num,
                col=col,
            )
        if reg_seg is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=reg_seg * mult,
                    mode="lines",
                    name=f"Segment's reg contribution (Right axis)",
                    line=dict(color="#9fe870"),
                    showlegend=showlegend and col == col_nums[0],
                ),
                row=row_num,
                col=col,
                secondary_y=True,
            )
        if leftover_totals is not None:
            fig.add_trace(
                go.Bar(
                    x=time,
                    y=leftover_totals if col == 1 else leftover_avgs,
                    name=f"Leftover actuals",
                    marker=dict(color="#ff685f"),
                    showlegend=showlegend and col == col_nums[0],
                ),
                row=row_num,
                col=col,
            )
