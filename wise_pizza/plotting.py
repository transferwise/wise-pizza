from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.io import to_image
from plotly.subplots import make_subplots

from wise_pizza.slicer import SliceFinder

pio.templates.default = "plotly_white"

import numpy as np
import pandas as pd
from IPython.display import Image


def plot_split_segments(
    sf_size: SliceFinder,
    sf_avg: SliceFinder,
    plot_is_static: bool = False,
    width: int = 2000,
    height: int = 500,
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
            "Normalised difference in sizes",
            "Impact of average chgs on overall total",
            "Normalised difference in averages",
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
    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)

        # Display the static image in the Jupyter notebook
        return Image(
            image_bytes,
            height=height + len(size_data.index) * 30,
            width=width + len(size_data.index) * 30,
        )
    else:
        fig.show()


def plot_segments(
    sf: SliceFinder,
    plot_is_static: bool = False,
    width: int = 2000,
    height: int = 500,
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
            "Simple segment averages",
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

    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)

        # Display the static image in the Jupyter notebook
        return Image(
            image_bytes,
            height=height + len(sf.segment_labels) * 30,
            width=width + len(sf.segment_labels) * 30,
        )
    else:
        fig.show()


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
    sf: SliceFinder, plot_is_static: bool = False, width: int = 1000, height: int = 1000
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

    trace2 = go.Bar(
        x=data["naive_avg"],
        y=data.index,
        orientation="h",
        name="Diff in averages",
        marker_color="#ff685f",
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_yaxes=False,
        subplot_titles=["", "Normalised difference in averages"],
    )

    fig.add_trace(trace1, 1, 1)
    fig.add_trace(trace2, 2, 1)
    fig["layout"]["yaxis2"].update(autorange="reversed")

    fig.update_layout(
        title="Segments contributing most to the change",
        #         showlegend = True,
        **waterfall_layout_args(sf, width, height)
    )

    if plot_is_static:
        # Convert the figure to a static image
        image_bytes = to_image(fig, format="png", scale=2)

        # Display the static image in the Jupyter notebook
        return Image(image_bytes, width=width, height=height)
    else:
        fig.show()
