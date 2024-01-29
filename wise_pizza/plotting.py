import copy
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio
from plotly.io import to_image
from plotly.subplots import make_subplots


from wise_pizza.slicer import SliceFinder, SlicerPair

pio.templates.default = "plotly_white"

import numpy as np
import pandas as pd
from IPython.display import Image, display


def plot_split_segments(
    sf_size: SliceFinder,
    sf_avg: SliceFinder,
    plot_is_static: bool = False,
    width: int = 2000,
    height: int = 500,
    cluster_values: bool = False,
    cluster_key_width: int = 180,
    cluster_value_width: int = 318,
    return_fig: bool = False,
):
    """
    Plot split segments for explain_changes: split_fits
    @param sf_size: SliceFinder from sizes
    @param sf_avg: SliceFinder from averages
    @param denominator: denominator of the value
    @param plot_is_static: static (True) or dynamic (False) plotly result
    @param width: parameter to modify the final width of the plot
    @param height: parameter to modify the final height of the plot
    """
    size_data = pd.DataFrame(sf_size.segments, index=np.array(sf_size.segment_labels))
    avg_data = pd.DataFrame(sf_avg.segments, index=np.array(sf_avg.segment_labels))

    size_average = sf_size.reg.intercept_
    avg_average = sf_avg.reg.intercept_

    trace1 = go.Bar(
        x=size_data["impact"],
        y=size_data.index,
        orientation="h",
        name="Impact of size chgs",
        marker_color="#a0e1e1",
    )

    trace2 = go.Bar(
        x=size_data["naive_avg"],
        y=size_data.index,
        orientation="h",
        name="Diff in sizes",
        marker_color="#ff685f",
    )

    trace3 = go.Bar(
        x=avg_data["impact"],
        y=avg_data.index,
        orientation="h",
        name="Impact of average chgs",
        marker_color="#9fe870",
    )

    trace4 = go.Bar(
        x=avg_data["naive_avg"],
        y=avg_data.index,
        orientation="h",
        name="Diff in averages",
        marker_color="#ffc091",
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        subplot_titles=[
            "Impact of size chgs on overall total",
            "Segment sizes",
            "Impact of average chgs on overall total",
            "Segment averages",
        ],
    )

    fig.add_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 2)
    fig.add_trace(trace3, 2, 1)
    fig.add_trace(trace4, 2, 2)

    fig.update_layout(
        height=height + len(size_data.index) * 30,
        width=width + len(size_data.index) * 30,
    )

    fig.add_vline(
        x=size_average,
        line_width=3,
        line_dash="dash",
        line_color="red",
        col=2,
        row=1,
        annotation_text="Global average",
    )

    fig.add_vline(
        x=avg_average,
        line_width=3,
        line_dash="dash",
        line_color="red",
        col=2,
        row=2,
        annotation_text="Global average",
    )

    for i in range(1, 3):
        fig.update_yaxes(autorange="reversed", row=i)

    if cluster_values:
        data_dict = sf_size.relevant_cluster_names
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        key_column_width = cluster_key_width  # Adjust the multiplier as needed
        value_column_width = cluster_value_width  # Adjust the multiplier as needed

        # Create a table trace with specified column widths
        table_trace = go.Table(
            header=dict(values=["Cluster", "Segments"]),
            cells=dict(values=[keys, values]),
            columnwidth=[key_column_width, value_column_width],
        )

        # Create a layout
        layout = go.Layout(
            title="Relevant cluster names", title_x=0
        )  # Center the title

        # Create a figure
        fig2 = go.Figure(data=[table_trace], layout=layout)

    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)

        if cluster_values:
            display(
                Image(
                    image_bytes,
                    height=height + len(size_data.index) * 30,
                    width=width + len(size_data.index) * 30,
                )
            )
            fig2.show()

        else:
            # Display the static image in the Jupyter notebook
            return Image(
                image_bytes,
                height=height + len(size_data.index) * 30,
                width=width + len(size_data.index) * 30,
            )
    else:
        if return_fig:
            if cluster_values:
                return [fig, fig2]
            else:
                return fig
        fig.show()
        if cluster_values:
            fig2.show()


def plot_segments(
    sf: SliceFinder,
    plot_is_static: bool = False,
    width: int = 2000,
    height: int = 500,
    return_fig: bool = False,
    cluster_values: bool = False,
    cluster_key_width: int = 180,
    cluster_value_width: int = 318,
):
    """
    Plot segments for explain_levels
    @param sf: SliceFinder
    @param plot_is_static: static (True) or dynamic (False) plotly result
    @param width: parameter to modify the final width of the plot
    @param height: parameter to modify the final height of the plot
    """

    seg_data = pd.DataFrame(sf.segments)
    seg_data["avg"] = seg_data["coef"] + sf.reg.intercept_
    seg_data.index = np.array(sf.segment_labels)

    average = sf.reg.intercept_

    trace1 = go.Bar(
        x=seg_data["impact"],
        y=sf.segment_labels,
        orientation="h",
        name="total",
        marker_color="#a0e1e1",
    )

    trace2 = go.Bar(
        x=seg_data["naive_avg"],
        y=sf.segment_labels,
        orientation="h",
        name="averages",
        marker_color="#ff685f",
    )

    trace3 = go.Bar(
        x=seg_data["seg_size"],
        y=sf.segment_labels,
        orientation="h",
        name="sizes",
        marker_color="#9fe870",
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[
            "Impact on overall total",
            "Segment averages",
            "Segment sizes",
        ],
    )

    fig.add_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 2)
    fig.add_trace(trace3, 1, 3)

    fig.update_layout(
        height=height + len(sf.segment_labels) * 30,
        width=width + len(sf.segment_labels) * 30,
        yaxis=dict(autorange="reversed"),
    )
    fig.add_vline(
        x=average,
        line_width=3,
        line_dash="dash",
        line_color="red",
        col=2,
        annotation_text="Global average",
    )

    if cluster_values:
        data_dict = sf.relevant_cluster_names
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        key_column_width = cluster_key_width  # Adjust the multiplier as needed
        value_column_width = cluster_value_width  # Adjust the multiplier as needed

        # Create a table trace with specified column widths
        table_trace = go.Table(
            header=dict(values=["Cluster", "Segments"]),
            cells=dict(values=[keys, values]),
            columnwidth=[key_column_width, value_column_width],
        )

        # Create a layout
        layout = go.Layout(
            title="Relevant cluster names", title_x=0
        )  # Center the title

        # Create a figure
        fig2 = go.Figure(data=[table_trace], layout=layout)

    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)

        # Display the static image in the Jupyter notebook
        if cluster_values:
            image_bytes2 = to_image(fig2, format="png", scale=2)
            display(
                Image(
                    image_bytes,
                    height=height + len(sf.segment_labels) * 30,
                    width=width + len(sf.segment_labels) * 30,
                )
            )
            display(Image(image_bytes2, height=height, width=width))
        else:
            return Image(
                image_bytes,
                height=height + len(sf.segment_labels) * 30,
                width=width + len(sf.segment_labels) * 30,
            )
    else:
        if return_fig:
            if cluster_values:
                return [fig, fig2]
            else:
                return fig
        else:
            fig.show()
            if cluster_values:
                fig2.show()


def waterfall_args(sf: SliceFinder):
    """
    Waterfall plot arguments
    @param sf: SliceFinder
    """
    segs = [s["impact"] for s in sf.segments]
    other = sf.post_total - (sf.pre_total + sum(segs))

    def rel_error(x, y):
        return abs(x / y - 1.0)

    assert rel_error(sf.pre_total + sum(segs) + other, sf.post_total) < 1e-3

    return {
        "orientation": "v",
        "measure": ["absolute"] + ["relative"] * (len(sf.segments) + 1) + ["total"],
        "x": ["Old total"] + sf.segment_labels + ["Other", "New total"],
        "textposition": "outside",
        # text = ["+60", "+80", "", "-40", "-20", "Total"],
        "y": [sf.pre_total] + segs + [other, sf.post_total],
        "connector": {"line": {"color": "rgb(63, 63, 63)"}},
        "increasing": {"marker": {"color": "#9fe870"}},
        "decreasing": {"marker": {"color": "#ff685f"}},
        "totals": {"marker": {"color": "#a0e1e1"}},
    }


def waterfall_layout_args(sf: SliceFinder, width: int = 1000, height: int = 1000):
    """
    Waterfall plot layout arguments
    @param sf: SliceFinder
    @param width: parameter to modify the final width of the plot
    @param height: parameter to modify the final height of the plot
    """
    range1 = sum([s["impact"] for s in sf.segments if s["impact"] > 0])
    range2 = abs(sum([s["impact"] for s in sf.segments if s["impact"] < 0]))
    range = max(range1, range2)
    tickangle = 30
    tickfront_size = 8
    return {
        "yaxis_range": [
            min(
                sf.pre_total - range / 4,
                sf.pre_total + range1 - range2 - 0.2 * range,
            ),
            sf.pre_total + range1 + 0.2 * range,
        ],
        "xaxis": {
            "tickangle": tickangle,  # set x-axis tick angle to 30 degrees (horizontal orientation)
            "tickfont": {"size": tickfront_size},  # set font size of x-axis tick labels
        },
        "width": width,
        "height": height,
    }


def plot_waterfall(
    sf: SliceFinder,
    plot_is_static: bool = False,
    width: int = 1000,
    height: int = 1000,
    cluster_values: bool = False,
    cluster_key_width: int = 180,
    cluster_value_width: int = 318,
    return_fig: bool = False,
):
    """
    Plot waterfall and Bar for explain_changes
    @param sf: SliceFinder
    @param plot_is_static: static (True) or dynamic (False) plotly result
    @param width: parameter to modify the final width of the plot
    @param height: parameter to modify the final height of the plot
    """
    assert hasattr(sf, "pre_total"), "Please call fit_change first"
    data = pd.DataFrame(sf.segments, index=np.array(sf.segment_labels))
    trace1 = go.Waterfall(name="Segments waterfall", **waterfall_args(sf))

    fig = go.Figure()

    fig.add_trace(trace1)
    fig.update_layout(title="Segments contributing most to the change")

    fig.update_layout(
        title="Segments contributing most to the change",
        #         showlegend = True,
        **waterfall_layout_args(sf, width, height),
    )

    if cluster_values:
        data_dict = sf.relevant_cluster_names
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        key_column_width = cluster_key_width  # Adjust the multiplier as needed
        value_column_width = cluster_value_width  # Adjust the multiplier as needed

        # Create a table trace with specified column widths
        table_trace = go.Table(
            header=dict(values=["Cluster", "Segments"]),
            cells=dict(values=[keys, values]),
            columnwidth=[key_column_width, value_column_width],
        )

        # Create a layout
        layout = go.Layout(
            title="Relevant cluster names", title_x=0
        )  # Center the title

        # Create a figure
        fig2 = go.Figure(data=[table_trace], layout=layout)

    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)
        if cluster_values:
            display(Image(image_bytes, height=height, width=width))
            fig2.show()
        else:
            # Display the static image in the Jupyter notebook
            display(Image(image_bytes, width=width, height=height))
    else:
        if return_fig:
            if cluster_values:
                return [fig, fig2]
            else:
                return fig
        fig.show()
        if cluster_values:
            fig2.show()


@dataclass
class PlotData:
    df: pd.DataFrame
    nonflat_segments: List[Dict[str, Any]]
    global_time_label: str
    total_name: str
    average_name: str
    sub_titles: List[str]


def plot_time(
    sf: SliceFinder,
    width: int = 1000,
    height: int = 1000,
    average_name: Optional[str] = None,
):
    plot_data = preprocess_for_ts_plot(sf, average_name)
    num_rows = len(plot_data.nonflat_segments) + 1
    fig = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=plot_data.sub_titles,
        specs=[[{"secondary_y": True}] * 2] * num_rows,
    )

    plot_single_ts(plot_data, fig, col_nums=(1, 2))

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 10

    fig.update_layout(
        title_text=f"Actuals vs explanation by segment",
        showlegend=True,
        width=width,
        height=height,
    )
    fig.show()


def plot_ts_pair(
    sf: SlicerPair,
    width,
    height,
    average_name: str = None,
    use_fitted_weights: bool = False,
):
    # if use_fitted_weights:
    #     sf = copy.deepcopy(sf)
    #     sf.s2.totals = (sf.s2.totals/sf.s2.weights)*sf.s1.totals
    #     sf.s2.weights = sf.s1.totals

    wgt_plot_data = preprocess_for_ts_plot(sf.s1, average_name)  # average name correct?
    totals_plot_data = preprocess_for_ts_plot(sf.s2, average_name)
    num_rows = max(
        len(wgt_plot_data.nonflat_segments) + 1,
        len(totals_plot_data.nonflat_segments) + 1,
    )
    subplot_titles = []
    for i in range(num_rows):
        if 2 * i < len(wgt_plot_data.sub_titles):
            subplot_titles.append(wgt_plot_data.sub_titles[2 * i])
        else:
            subplot_titles.append("")
        if 2 * i < len(totals_plot_data.sub_titles):
            subplot_titles.append(totals_plot_data.sub_titles[2 * i + 1])
            subplot_titles.append(totals_plot_data.sub_titles[2 * i])
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


def same_apart_from_time(s1, s2) -> bool:
    return np.sum(np.abs(s1["dummy"] - s2["dummy"])) < 0.5


def preprocess_for_ts_plot(
    sf: SliceFinder, average_name: Optional[str] = None
) -> PlotData:
    if average_name is None:
        average_name = "Averages"

    df = pd.DataFrame(
        {
            "totals": sf.actual_totals,
            "weights": sf.weights,
            "Regr totals": sf.predicted_totals,
            "time": sf.time,
        }
    )
    df["reg_time_profile"] = 0.0

    # do a pass over the segments, sorting them into time-only and rest
    global_reg = 0.0
    global_time_profile_names = []
    nonflat_segments = []

    adj_avg = sf.y_adj.sum() / sf.weights.sum()

    rel_adj = sf.y_adj - adj_avg * sf.weights

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
        str(s["segment"]) for s in nonflat_segments
    ]
    sub_titles = [
        [f"{sf.total_name} for {s} ", f"{average_name} for {s}"] for s in seg_names
    ]
    sub_titles = sum(sub_titles, start=[])

    plot_data = PlotData(
        df, nonflat_segments, global_time_label, sf.total_name, average_name, sub_titles
    )
    return plot_data


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
                marker=dict(color="orange"),
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
                    line=dict(color="blue"),
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
                    line=dict(color="teal"),
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
                    marker=dict(color="purple"),
                    showlegend=showlegend and col == col_nums[0],
                ),
                row=row_num,
                col=col,
            )
