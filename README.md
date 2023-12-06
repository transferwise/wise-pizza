<img src="https://github.com/transferwise/wise-pizza/blob/main/docs/Wise_Logo.png?raw=True" width=10% height=10%>

# Wise Pizza: A library for automated figuring out most unusual segments

**WisePizza** is a library to find and visualise the most interesting slices in multidimensional data based on Lasso and LP solvers, which provides different functions to find segments whose average is most different from the global one or find segments most useful in explaining the difference between two datasets.

## The approach
WisePizza assumes you have a dataset with a number of discrete *dimensions* (could be currency, region, etc). For each
combination of dimensions, the dataset must have a *total* value (total of the metric over that segment, for example
the total volume in that region and currency), and an optional *size* value (set to 1 if not specified), this could for
example be the total number of customers for that region and currency. The *average* value of the outcome for the segment
is defined as total divided by size, in this example it would be the average volume per customer. 

`explain_levels` takes such a dataset and looks for a small number of 'simple' segments (each only constraining a 
small number of dimensions) that between them 
explain most of the variation in the averages; you could also think of them as the segments whose size-weighted
deviation from the overall dataset average is the largest. This trades off unusual averages (which will naturally 
occur more for smaller segments) against segment size.

Yet another way of looking at it is that we look for segments which, if their average was reset to the overall dataset
average, would move overall total the most.

`explain_changes_in_totals` and `explain_changes_in_average` take two datasets of the kind described above, with the same 
column names, and apply the same kind of logic to find the segments that contribute most to the difference (in total 
or average, respectively) between the two datasets, optionally splitting that into contributions from 
changes in segment size and changes in segment total.

Sometimes, rather than explaining the change in totals from one period to the next, one wishes to explain a change in averages. The analytics of this are a little different - for example, while (as long as all weights and totals are positive) increasing a segment size (other things remaining equal) always increases the overall total, it can increase or decrease the pverall average, depending on whether the average value of that segment is below or above the overall average.

<summary><strong><em>Table of Contents</em></strong></summary>

- [What can this do for you?](#what-can-this-do-for-you)
    - [Find interesting slices](#better-information-about-segments-and-subsegments-in-your-data)
    - [Comparison between two datasets](#understanding-differences-in-two-time-periods-or-two-dataframes)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Streamlit app](#streamlit-app)
    - [Docker container](#docker-container)
- [For Developers](#for-developers)
 - [Tests](#testing)

 ## What can this do for you?

The automated search for interesting segments can give you the following:

### 1. Better information about segments and subsegments in your data 
By using **WisePizza** and defining initial segments, you can find a segment which maximizes a specific outcome, such as adoption rates.


### 2. Understanding differences in two time periods or two dataframes
If you have two time periods or two datasets, you can find segments that experience the largest change in the totals from previous period/dataset.


## Installation
You can always get the newest *wise_pizza* release using ``pip``:
https://pypi.org/project/wise-pizza/
```
pip install wise-pizza
```

From the command line (another way):
```
pip install git+https://github.com/transferwise/wise-pizza.git
```

From Jupyter notebook (another way):
```
!pip install git+https://github.com/transferwise/wise-pizza.git
```

Or you can clone and run from source, in which case you should `pip -r requirements.txt` before running.

## Quick Start
The wisepizza package can be used for finding segments with unusual average:

```Python
sf = explain_levels(
    df=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    solver="lasso"
)
```

![plot](https://github.com/transferwise/wise-pizza/blob/main/docs/explain_levels.png?raw=True)

Or for finding changes between two datasets in totals:

```Python
sf1 = explain_changes_in_totals(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="totals",
    solver="lasso"
)
```

![plot](https://github.com/transferwise/wise-pizza/blob/main/docs/explain_changes_in_totals.png?raw=True)

Or for finding changes between two datasets in average:

```Python
sf1 = explain_changes_in_average(
    df1=pre_data,
    df2=data,
    dims=dims,
    total_name=totals,
    size_name=size,
    max_depth=2,
    min_segments=20,
    how="totals",
    solver="lasso"
)
```

![plot](https://github.com/transferwise/wise-pizza/blob/main/docs/explain_changes_in_average(totals).png?raw=True)

***In addition to single-value slices, consider slices that consist of a
    group of segments from the same dimension with similar naive averages***
For that goal you can use cluster_values=True parameter.

![plot](https://github.com/transferwise/wise-pizza/blob/main/docs/cluster_values.png?raw=True)

And then you can visualize differences:

```Python
sf.plot()
```

And check segments:

```Python
sf.segments
```

if you use cluster values, you can also check relevant cluster names:
```Python
sf.relevant_cluster_names
```

Please see the full example [here](https://github.com/transferwise/wise-pizza/blob/main/notebooks/Finding%20interesting%20segments.ipynb)

## Streamlit app (beta)
In the root directory of this repository, there is a Streamlit app. This is an interface that allows you to upload your own files and run analyses, as you saw in the Jupyter Notebook provided as an example.

To run this, you need to:
1. Create a virtual environment (e.g. using pyenv)
2. Activate the virtual environment
3. Run `pip -r requirements.txt` before running, to install necessary dependencies
4. Run `streamlit run streamlit_app.py` to run Streamlit

### Docker container

We created a Docker container that makes it easier to deploy this solution elsewhere

You need to first: Create the Docker image
Create the Docker image

```Python
docker build -t streamlit .      
```
And then simply run the image

```Python
docker run -p 8501:8501 streamlit    
```

## For Developers




### Testing
We use [PyTest](https://docs.pytest.org/) for testing. If you want to contribute code, make sure that the tests in tests/ run without errors.

*Wise-pizza is open sourced and maintained by Wise Plc. Copyright 2023 Wise Plc.*
