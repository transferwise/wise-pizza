"""An example of Streamlit leveraging Wise pizza."""

import streamlit as st
import os, sys
import pandas as pd
from wise_pizza.explain import explain_levels, explain_changes_in_totals, explain_changes_in_average


# False if you want nice interactive plots
# True if you want static plots (Doesn't work on all platforms yet)
plot_is_static = False


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Wise Pizza", page_icon=":pizza:")

st.title('Wise Pizza powered by Streamlit')
st.text('Only categorical columns are accepted, bucket the numeric ones if you wanna use those')

# upload the file from the computer
def load_data_upload():
    uploaded_file = st.file_uploader("Choose a file")
    if not uploaded_file:
        st.warning('Please input a dataset.')
        st.stop()
    st.success('Dataset inputted.')
    data = pd.read_csv(uploaded_file)
    return data

on = st.toggle('Use sample data from Github')
url_data = (r'https://raw.githubusercontent.com/transferwise/wise-pizza/main/data/synth_data.csv')

# select the datasource, either local or from github
if on:
    st.write(f'Downloading data from {url_data}!')
    df = pd.read_csv(url_data)
else:
    df=load_data_upload()

# show dataset preview
st.text('Table preview')
st.table(df.head(10))

# ask the user for relevant dimensions
dims = st.multiselect(
    "Select the dimensions you want to include in the analysis",
    df.select_dtypes(exclude=['number']).columns.tolist()
    )

# ask the user via streamlit to select if they want to run a comparison between subgroups or not
flag_comparison = st.toggle('I want to run a comparison between two subgroups in my data')

# return columns that are candidate for comparison
def flag_columns(df):
    binary_columns = df.columns[df.nunique() == 2].tolist()
    if not binary_columns:
        st.warning('No column in the dataset is binary, no comparison can be carried out')
        st.stop()
    return binary_columns

if flag_comparison:
    #calculate binary columns
    binary_columns=flag_columns(df)

    # ask the user via streamlit for the flag column
    flag_column = st.selectbox(
    "What is the flag column of your dataset that defines the two subgroups?",
    binary_columns,
    index=None,
    placeholder="Select flag column column",
    )
    st.write('You selected:', flag_column)

    # wait until flag column is added
    if not flag_column:
            st.stop()

    # calculate unique flags in the dataset
    flags = sorted(df[flag_column].unique())  

    # allow users to define what's "old" and "new" in the comparison
    flags_option = st.selectbox(
        'Which value in your flag column belongs to group A?',
        (flags))


# ask the user via streamlit for the target column
totals = st.selectbox(
   "Name of column that contains totals per segment (e.g. GMV/revenue)",
   # display only numerical columns
   df.select_dtypes(include=['number']).columns.tolist(),
   index=None,
   placeholder="Select target column",
)
st.write('You selected:', totals)

# ask the user via streamlit for the size column
size = st.selectbox(
   "Name of column containing segment size (e.g. number of users/number of transactions)",
   # display only numerical columns
   df.select_dtypes(include=['number']).columns.tolist(),
   index=None,
   placeholder="Select volume column",
)
st.write('You selected:', size)

st.subheader('Finding the juiciest slices', divider='rainbow')
st.text('This section does not compare groups, but rather checks which features have the most impact in the target column you selected.')

if not totals or not size or not dims:
        st.warning('Please input all fields above.')
        st.stop()

## finding juiciest slices
sf = explain_levels(
    df=df,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    solver="lasso",
    return_fig=True
)

# storing the plot in a variable
plot_sf=sf.plot(width=500, height=500)
# exposing the plot via streamlit
st.plotly_chart(plot_sf, use_container_width=True)

if flag_comparison:
    st.subheader('Analysing differences', divider='rainbow')
    st.text('This section does compare the two groups defined by the flag. Old total is the group A you selected in the dropdown')

    # creating the dataframes for the comparison calculations
    data = df[df[flag_column] != flags_option]  # take the group to compare to
    pre_data = df[df[flag_column] == flags_option]  # take the group to be compared

    # define the relevant dimensions for the comparison feature
    # for this, the flag column is to be excluded
    comparison_dimensions = list(filter(lambda x: x != flag_column, dims))

    ## running explain calculations
    sf1 = explain_changes_in_totals(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=2,
        min_segments=20,
        how="totals",
        solver="lasso",
        return_fig=True
    )
    # specifying a two column layout on streamlit
    col1, col2 = st.columns(2)
    # storing the plots in variables
    # exposing the plots via streamlit
    with col1:
        plot_sf1=sf1.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
        st.plotly_chart(plot_sf1, use_container_width=True)

    with col2:
        plot_sf2=sf1.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
        st.plotly_chart(plot_sf2, use_container_width=True)

    st.subheader('Decomposing differences', divider='rainbow')
    st.text('`split_fits` to separately decompose contribution of size changes and average changes')

    ## running explain calculations
    sf2 = explain_changes_in_totals(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=1,
        min_segments=10,
        how="split_fits",
        solver="lasso",
        return_fig=True
    )
    # storing the plot in a variable
    # exposing the plot via streamlit
    plot_sf=sf2.plot(width=500, height=500)
    st.plotly_chart(plot_sf, use_container_width=True)

    st.text('`extra_dim` to treat size vs average change contribution as an additional dimension')

    ## running explain calculations
    sf3 = explain_changes_in_totals(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=2,
        min_segments=20,
        how="extra_dim",
        solver="lasso",
        return_fig=True
    )
    # specifying a two column layout on streamlit
    col1, col2 = st.columns(2)

    # storing the plots in variables
    # exposing the plots via streamlit
    with col1:
        plot_sf1=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
        st.plotly_chart(plot_sf1, use_container_width=True)

    with col2:
        plot_sf2=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
        st.plotly_chart(plot_sf2, use_container_width=True)

    st.text('`force_dim` like extra_dim, but each segment must contain a Change_from constraint')

    ## running explain calculations
    sf3 = explain_changes_in_totals(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=2,
        min_segments=15,
        how="force_dim",
        solver="lasso",
        return_fig=True
    )
    # specifying a two column layout on streamlit
    col1, col2 = st.columns(2)

    # storing the plots in variables
    # exposing the plots via streamlit
    with col1:
        plot_sf1=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
        st.plotly_chart(plot_sf1, use_container_width=True)

    with col2:
        plot_sf2=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
        st.plotly_chart(plot_sf2, use_container_width=True)

    st.subheader('Explaining changes in average', divider='rainbow')

    ## running explain calculations
    sf4 = explain_changes_in_average(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=2,
        min_segments=20,
        how="totals",
        solver="lasso",
        return_fig=True
    )
    # specifying a two column layout on streamlit
    col1, col2 = st.columns(2)

    # storing the plots in variables
    # exposing the plots via streamlit
    with col1:
        plot_sf1=sf4.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
        st.plotly_chart(plot_sf1, use_container_width=True)

    with col2:
        plot_sf2=sf4.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
        st.plotly_chart(plot_sf2, use_container_width=True)

    ## running explain calculations
    sf6 = explain_changes_in_average(
        df1=pre_data,
        df2=data,
        dims=comparison_dimensions,
        total_name=totals,
        size_name=size,
        max_depth=2,
        min_segments=20,
        how="split_fits",
        solver="lasso",
        return_fig=True
    )

    # storing the plot in a variable
    plot_sf=sf6.plot(width=500, height=500)
    # exposing the plot via streamlit
    st.plotly_chart(plot_sf, use_container_width=True)