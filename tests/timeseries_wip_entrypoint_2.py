import os, sys
import pandas as pd

root_path = os.path.realpath("../..")
print(root_path)

# this assumes that all of the following files are checked in the same directory
sys.path.append(os.path.join(root_path, "wise-pizza"))

# create data-related directories
data_dir = os.path.realpath(os.path.join(root_path, "wise-pizza/data"))
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
print(data_dir)

from wise_pizza import explain_timeseries

df = pd.read_csv(
    os.path.join(data_dir, "volume_data_new.csv")
)  # replace this variable with your data
dims = [
    "CUSTOMER_TYPE",
    "STRATEGIC_PRODUCT",
    "SOURCE_CURRENCY",
    "TARGET_CURRENCY",
    "PRODUCT_USE_CASE",
    "REGION",
    "TRANS_VOL_BUCKET",
]  # dimensions to find segments
totals = "VOLUME_GBP"  # value to analyze
size = "NUM_CUSTOMERS"  #'NUM_TRANSACTIONS'  # number of objects
time = "ACTION_YM"
sf = explain_timeseries(
    df=df,
    dims=dims,
    max_segments=7,
    max_depth=2,
    total_name=totals,
    size_name=size,
    time_name=time,
    verbose=False,
    solver="tree",
    fit_sizes=True,
)
sf.plot(plot_is_static=False, height=1500, width=1000, average_name="VPC")
print(sf.summary())
print("yay!")
