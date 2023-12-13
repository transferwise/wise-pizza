"""An example of Streamlit leveraging Wise pizza."""
# Streamlit part was initially developed by https://github.com/agusfigueroa-htg

import streamlit as st
import os, sys
import pandas as pd
from wise_pizza.explain import (
    explain_levels,
    explain_changes_in_totals,
    explain_changes_in_average,
)


# False if you want nice interactive plots
# True if you want static plots (Doesn't work on all platforms yet)
plot_is_static = False


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Wise Pizza", page_icon=":pizza:")

st.title("Wise Pizza powered by Streamlit")
st.text(
    "Only categorical columns are accepted, bucket the numeric ones if you wanna use those"
)

# upload the file from the computer
def load_data_upload():
    uploaded_file = st.file_uploader("Choose a file")
    if not uploaded_file:
        st.warning("Please input a dataset.")
        st.stop()
    st.success("Dataset inputted.")
    data = pd.read_csv(uploaded_file)
    return data


on = st.toggle("Use sample data from Github")
url_data = r"https://raw.githubusercontent.com/transferwise/wise-pizza/main/data/synth_data.csv"

# select the datasource, either local or from github
if on:
    st.write(f"Downloading data from {url_data}!")
    df = pd.read_csv(url_data)
else:
    df = load_data_upload()

# show dataset preview
st.text("Table preview")
st.table(df.head(10))

# ask the user for relevant dimensions
dims = st.multiselect(
    "Select the dimensions you want to include in the analysis",
    df.select_dtypes(exclude=["number"]).columns.tolist(),
)

# ask the user via streamlit to select if they want to run a comparison between subgroups or not
flag_comparison = st.toggle(
    "I want to run a comparison between two subgroups in my data"
)

# return columns that are candidate for comparison
def flag_columns(df):
    binary_columns = df.columns[df.nunique() == 2].tolist()
    if not binary_columns:
        st.warning(
            "No column in the dataset is binary, no comparison can be carried out"
        )
        st.stop()
    return binary_columns


if flag_comparison:
    # calculate binary columns
    binary_columns = flag_columns(df)

    # ask the user via streamlit for the flag column
    flag_column = st.selectbox(
        "What is the flag column of your dataset that defines the two subgroups?",
        binary_columns,
        index=None,
        placeholder="Select flag column column",
    )
    st.write("You selected:", flag_column)

    # wait until flag column is added
    if not flag_column:
        st.stop()

    # calculate unique flags in the dataset
    flags = sorted(df[flag_column].unique())

    # allow users to define what's "old" and "new" in the comparison
    flags_option = st.selectbox(
        "Which value in your flag column belongs to group A?", (flags)
    )


# ask the user via streamlit for the target column
totals = st.selectbox(
    "Name of column that contains totals per segment (e.g. GMV/revenue)",
    # display only numerical columns
    df.select_dtypes(include=["number"]).columns.tolist(),
    index=None,
    placeholder="Select target column",
)
st.write("You selected:", totals)

# ask the user via streamlit for the size column
size = st.selectbox(
    "Name of column containing segment size (e.g. number of users/number of transactions)",
    # display only numerical columns
    df.select_dtypes(include=["number"]).columns.tolist(),
    index=None,
    placeholder="Select volume column",
)
st.write("You selected:", size)

if not totals or not size or not dims:
    st.warning("Please input all fields above.")
    st.stop()

if not flag_comparison:
    st.subheader("Finding the juiciest slices", divider="rainbow")
    st.text("Find segments whose average is most different from the global one")
    ## finding juiciest slices
    solver = st.selectbox("Select a solver:", ("lasso", "lp"))
    min_segments = st.number_input(
        "Min segments: Minimum number of segments to find",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="min_segments",
    )
    max_segments = st.number_input(
        "Max segments: Maximum number of segments to find, defaults to min_segments",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="max_segments",
    )
    min_depth = st.number_input(
        "Min depth: Minimum number of dimension to constrain in segment definition",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="min_depth",
    )
    max_depth = st.number_input(
        "Max depth: Maximum number of dimension to constrain in segment definition",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        key="max_depth",
    )
    cluster_values = st.selectbox(
        "Cluster values: In addition to single-value slices, consider slices that consist of a group of segments from the same dimension with similar naive averages",
        (True, False),
    )

    width = st.number_input(
        "Width of the plot",
        min_value=100,
        max_value=10000,
        value=500,
        step=1,
        key="width",
    )
    height = st.number_input(
        "Height of the plot",
        min_value=100,
        max_value=10000,
        value=500,
        step=1,
        key="height",
    )
    apply_button = st.button("Apply")
    if apply_button:
        sf = explain_levels(
            df=df,
            dims=dims,
            total_name=totals,
            size_name=size,
            min_depth=min_depth,
            max_depth=max_depth,
            min_segments=min_segments,
            max_segments=max_segments,
            solver=solver,
            cluster_values=cluster_values,
        )
        plot_sf = sf.plot(width=width, height=height, return_fig=True)
        if not cluster_values:
            plot_sf_0 = plot_sf
            st.plotly_chart(plot_sf_0, use_container_width=True)
        else:
            plot_sf_0 = plot_sf[0]
            st.plotly_chart(plot_sf_0, use_container_width=True)
            plot_sf_1 = plot_sf[1]
            st.plotly_chart(plot_sf_1, use_container_width=True)

if flag_comparison:
    st.subheader("Analysing differences", divider="rainbow")
    st.text(
        "This section does compare the two groups defined by the flag. Old total is the group A you selected in the dropdown"
    )
    st.text(
        "Explain changes in totals: Let's look for segments that experience the largest change in the totals from previous period."
    )
    st.text(
        "Explain changes in average: Sometimes, rather than explaining the change in totals from one period to the next, one wishes to explain a change in averages"
    )

    data = df[df[flag_column] != flags_option]  # take the group to compare to
    pre_data = df[df[flag_column] == flags_option]  # take the group to be compared

    comparison_dimensions = list(filter(lambda x: x != flag_column, dims))

    explain = st.selectbox(
        "Select a function:",
        ("explain changes in totals", "explain changes in average"),
    )
    solver = st.selectbox("Select a solver:", ("lasso", "lp"))
    min_segments = st.number_input(
        "Min segments: Minimum number of segments to find",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="min_segments",
    )
    max_segments = st.number_input(
        "Max segments: Maximum number of segments to find, defaults to min_segments",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="max_segments",
    )
    min_depth = st.number_input(
        "Min depth: Minimum number of dimension to constrain in segment definition",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="min_depth",
    )
    max_depth = st.number_input(
        "Max depth: Maximum number of dimension to constrain in segment definition",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        key="max_depth",
    )
    cluster_values = st.selectbox(
        "Cluster values: In addition to single-value slices, consider slices that consist of a group of segments from the same dimension with similar naive averages",
        (True, False),
    )
    how = st.selectbox(
        "Select a method:", ("totals", "split_fits", "extra_dim", "force_dim")
    )
    st.text(
        '"totals": to only decompose segment totals (ignoring size vs average contribution)'
    )
    st.text(
        '"split_fits": to separately decompose contribution of size changes and average changes'
    )
    st.text(
        '"extra_dim": to treat size vs average change contribution as an additional dimension'
    )
    st.text(
        '"force_dim": like extra_dim, but each segment must contain a Change_from constraint'
    )
    width = st.number_input(
        "Width of the plot",
        min_value=100,
        max_value=10000,
        value=500,
        step=1,
        key="width",
    )
    height = st.number_input(
        "Height of the plot",
        min_value=100,
        max_value=10000,
        value=500,
        step=1,
        key="height",
    )
    changes_apply_button = st.button("Get results")
    ## running explain calculations
    if changes_apply_button:
        if explain == "explain changes in totals":
            sf1 = explain_changes_in_totals(
                df1=pre_data,
                df2=data,
                dims=comparison_dimensions,
                total_name=totals,
                size_name=size,
                min_depth=min_depth,
                max_depth=max_depth,
                min_segments=min_segments,
                max_segments=max_segments,
                solver=solver,
                cluster_values=cluster_values,
                how=how,
            )
        else:
            sf1 = explain_changes_in_average(
                df1=pre_data,
                df2=data,
                dims=comparison_dimensions,
                total_name=totals,
                size_name=size,
                min_depth=min_depth,
                max_depth=max_depth,
                min_segments=min_segments,
                max_segments=max_segments,
                solver=solver,
                cluster_values=cluster_values,
                how=how,
            )
        plot_sf1 = sf1.plot(width=width, height=height, return_fig=True)
        if not cluster_values:
            plot_sf_0 = plot_sf1
            st.plotly_chart(plot_sf_0, use_container_width=True)
        else:
            plot_sf_0 = plot_sf1[0]
            st.plotly_chart(plot_sf_0, use_container_width=True)
            plot_sf_1 = plot_sf1[1]
            st.plotly_chart(plot_sf_1, use_container_width=True)
