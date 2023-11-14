"""An example of Streamlit leveraging Wise pizza."""

import altair as alt
import pydeck as pdk
import streamlit as st

import os, sys
import datetime
import random
from typing import List
import copy
import gzip

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

from io import StringIO

import warnings
warnings.filterwarnings("ignore")

root_path = os.path.realpath('../..')
print(root_path)

# this assumes that all of the following files are checked in the same directory
sys.path.append(os.path.join(root_path,"wise-pizza"))

# create data-related directories
data_dir = os.path.realpath(os.path.join(root_path, 'wise-pizza/data'))
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
print(data_dir)

from wise_pizza import explain_levels, explain_changes_in_totals, explain_changes_in_average
# False if you want nice interactive plots
# True if you want static plots (Doesn't work on all platforms yet)
plot_is_static = False


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Wise Pizza", page_icon=":pizza:")

st.title('Wise Pizza powered by Streamlit')
st.text('Only categorical columns are accepted, bucket the numeric ones if you wanna use those')

def load_data_upload():
    uploaded_file = st.file_uploader("Choose a file")
    data = pd.read_csv(uploaded_file)
    return data

def load_data_snowflake(input_query, conn):
    cur = conn.cursor()
    cur.execute(input_query)
    sql_df = cur.fetch_pandas_all()
    return sql_df

on = st.toggle('Use sample data from Github')
url_data = (r'https://raw.githubusercontent.com/transferwise/wise-pizza/main/data/synth_data.csv')

if on:
    st.write(f'Downloading data from {url_data}!')
    df = pd.read_csv(url_data)
else:
    df=load_data_upload()

st.text('Table preview')
st.table(df.head(10))

totals = st.selectbox(
   "What is the target column that you want to analyse? e.g. GMV/revenue",
   df.columns,
   index=None,
   placeholder="Select target column",
)
st.write('You selected:', totals)


size = st.selectbox(
   "What is the volume column of your dataset? e.g. number of users/transactions",
   df.columns,
   index=None,
   placeholder="Select volume column",
)
st.write('You selected:', size)


flag_column = st.selectbox(
   "What is the flag column of your dataset you wanan split it by? Ensure this column is binary",
   df.columns,
   index=None,
   placeholder="Select time column",
)
st.write('You selected:', flag_column)

flags = sorted(df[flag_column].unique())  # unique flags in the dataset

if len(flags)>2:
    st.error('Your flag is not binary', icon="🚨")

flags_option = st.selectbox(
    'Which one in your data belongs to group A?',
    (flags))

candidates_excluded_columns = [element for element in df.columns if element not in [totals,size,flag_column]]

excluded_columns = st.multiselect(
    'Please select all columns that you want to exclude from the analysis',
    candidates_excluded_columns)

non_dimensional_columns = excluded_columns + [totals,size,flag_column]

dims = [element for element in df.columns if element not in non_dimensional_columns]

data = df[df[flag_column] != flags_option]  # take the group to compare to
pre_data = df[df[flag_column] == flags_option]  # take the group to be compared

st.table(df[dims].head(10))

st.subheader('Finding the juiciest slices', divider='rainbow')
st.text('This section does not compare groups, but rather checks which features have the most impact in the target column you selected.')

##Finding juiciest slices
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

plot_sf=sf.plot(width=500, height=500)
st.plotly_chart(plot_sf, use_container_width=True)

st.subheader('Analysing differences', divider='rainbow')
st.text('This section does compare the two groups defined by the flag. Old total is the group A you selected in the dropdown')

##explaining changes overall
sf1 = explain_changes_in_totals(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="totals",
    solver="lasso",
    return_fig=True
)
col1, col2 = st.columns(2)

with col1:
    plot_sf1=sf1.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
    st.plotly_chart(plot_sf1, use_container_width=True)

with col2:
    plot_sf2=sf1.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
    st.plotly_chart(plot_sf2, use_container_width=True)

st.subheader('Decomposing differences', divider='rainbow')
st.text('`split_fits` to separately decompose contribution of size changes and average changes')

sf2 = explain_changes_in_totals(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=1,
    min_segments=10,
    how="split_fits",
    solver="lasso",
    return_fig=True
)
plot_sf=sf2.plot(width=500, height=500)
st.plotly_chart(plot_sf, use_container_width=True)

st.text('`extra_dim` to treat size vs average change contribution as an additional dimension')
sf3 = explain_changes_in_totals(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="extra_dim",
    solver="lasso",
    return_fig=True
)

col1, col2 = st.columns(2)

with col1:
    plot_sf1=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
    st.plotly_chart(plot_sf1, use_container_width=True)

with col2:
    plot_sf2=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
    st.plotly_chart(plot_sf2, use_container_width=True)

st.text('`force_dim` like extra_dim, but each segment must contain a Change_from constraint')
sf3 = explain_changes_in_totals(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=15,
    how="force_dim",
    solver="lasso",
    return_fig=True
)
col1, col2 = st.columns(2)

with col1:
    plot_sf1=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
    st.plotly_chart(plot_sf1, use_container_width=True)

with col2:
    plot_sf2=sf3.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
    st.plotly_chart(plot_sf2, use_container_width=True)

st.subheader('Explaining changes in average', divider='rainbow')

sf4 = explain_changes_in_average(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="totals",
    solver="lasso",
    return_fig=True
)

col1, col2 = st.columns(2)

with col1:
    plot_sf1=sf4.plot(width=500, height=500, plot_is_static=plot_is_static)[0]
    st.plotly_chart(plot_sf1, use_container_width=True)

with col2:
    plot_sf2=sf4.plot(width=500, height=500, plot_is_static=plot_is_static)[1]
    st.plotly_chart(plot_sf2, use_container_width=True)

sf6 = explain_changes_in_average(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="split_fits",
    solver="lasso",
    return_fig=True
)
plot_sf=sf6.plot(width=500, height=500)
st.plotly_chart(plot_sf, use_container_width=True)