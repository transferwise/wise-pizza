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
    regression: pd.Series
    bars: pd.Series
    subtitle: str


def plot_time_from_tree(
    sf: SliceFinderPlottingInterface,
    width: int = 1000,
    height: int = 1000,
    average_name: Optional[str] = None,
    plot_is_static: bool = False,
    return_fig: bool = False,
):
    plot_data = preprocess_for_ts_plot(sf, average_name)
    num_rows = len(plot_data)
    num_cols = len(plot_data[0])
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[data.subtitle for row in plot_data for data in row],
        specs=[[{"secondary_y": False}] * num_cols] * num_rows,
    )

    for i, row in enumerate(plot_data):
        for j, data in enumerate(row):
            simple_ts_plot(
                fig,
                data.regression.index,
                data.bars,
                data.regression,
                row_num=i + 1,
                col_num=j + 1,
                show_legend=False,
            )

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


def preprocess_for_ts_plot(
    sf: SliceFinderPlottingInterface,
    average_name: str,
) -> List[List[PlotData]]:
    out = []
    for row, s in enumerate(sf.segments):
        print(row, s)
        this_df = pd.DataFrame(
            {
                "time": sf.time,
                "totals": sf.totals * s["dummy"],
                "weights": sf.weights * s["dummy"],
                "pred_totals": sf.avg_prediction * sf.weights * s["dummy"],
            }
        )
        if sf.weight_total_prediction is not None:
            this_df["w_pred_totals"] = sf.weight_total_prediction * s["dummy"]

        time_df = this_df.groupby("time", as_index=False).sum()

        segment_name = (
            str(s["segment"]).replace(",", "<br>").replace(";", ",").replace("'", "")
        )
        data1 = PlotData(
            regression=time_df["pred_totals"] / time_df["weights"],
            bars=time_df["totals"] / time_df["weights"],
            subtitle=f"{average_name} for <br> {segment_name}",
        )

        if sf.weight_total_prediction is None:
            data2 = PlotData(
                regression=time_df["pred_totals"],
                bars=time_df["totals"],
                subtitle=f"{sf.total_name} for <br> {segment_name}",
            )
            out.append([data1, data2])
        else:
            data2 = PlotData(
                # Use predictions for both avg and weights if available
                regression=time_df["w_pred_totals"]
                * time_df["pred_totals"]
                / time_df["weights"],
                bars=time_df["totals"],
                subtitle=f"{sf.total_name} for <br> {segment_name}",
            )
            data3 = PlotData(
                regression=time_df["w_pred_totals"],
                bars=time_df["weights"],
                subtitle=f"{sf.size_name} for <br> {segment_name}",
            )
            out.append([data3, data1, data2])

    return out


def simple_ts_plot(
    fig,
    time,
    bars,
    line,
    row_num: int,
    col_num: int,
    show_legend: bool = False,
):

    fig.add_trace(
        go.Bar(
            x=time,
            y=bars,
            name=f"Actuals",
            marker=dict(color="#9fe870"),
            showlegend=show_legend,
        ),
        row=row_num,
        col=col_num,
    )
    if line is not None:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=line,
                mode="lines",
                name=f"Regression",
                line=dict(color="#485cc7"),
                showlegend=show_legend,
            ),
            row=row_num,
            col=col_num,
        )
    fig.update_layout(
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True)
    )