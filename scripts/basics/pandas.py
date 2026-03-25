# %% [markdown]
# # pandas - Python data analysis library
#
# Please read through the following sections for a very brief overview if you are unfamiliar with the Python package [_pandas_](https://pandas.pydata.org/); otherwise, you can skip this material. For an in-depth overview see the [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).
#
# ## Pandas Data Structure
#
# [pandas](https://pandas.pydata.org/) allows for and builds the backbone for storing and manipulating scRNA-seq datasets as it is tabular data.
#
# The schematic below illustrates the general structure of a pandas [DataFrame](https://pandas.pydata.org/docs/reference/frame.html) - the primarily data structure underlying single-cell data analysis in Python. A DataFrame is defined by the
#
# 1. _index_: an annotation for each row.
# 2. _columns_: an annotation for each column.
# 3. _data_: numeric or string elements associated with each index and column value pair.
#
# <img src="https://pynative.com/wp-content/uploads/2021/02/dataframe.png" alt="genescore" width="600">

# %% [markdown]
# ## Library imports

# %%
import pandas as pd

# %% [markdown]
# ## Pandas basics

# %% [markdown]
# Let's define a simple DataFrame, that collects information on five students.

# %%
df = pd.DataFrame(
    [
        ["Joe", 20, 85.1, "A", "swimming"],
        ["Nat", 21, 77.8, "B", "reading"],
        ["Harry", 19, 91.54, "A", "nusic"],
        ["Sam", 20, 88.87, "A", "painting"],
        ["Monica", 22, 60.55, "B", "dancing"],
    ],
    index=[f"student_{student_id}" for student_id in range(5)],
    columns=["name", "age", "marks", "grade", "hobby"],
)

# Alternative but equivalent definition
# df = pd.DataFrame(
#     {
#         "name": ["Joe", "Nat", "Harry", "Sam", "Monica"],
#         "age": [20, 21, 19, 20, 22],
#         "marks": [85.1, 77.8, 91.54, 88.87, 60.55],
#         "grade": ["A", "B", "A", "A", "B"],
#         "hobby": ["swimming", "reading", "music", "painting", "dancing"],
#     },
#     index=[f"student_{student_id}" for student_id in range(5)],
# )

df

# %% [markdown]
# ### Indexing
#
# A major utility of pandas is easily accessing data with indexing functions. There are two primary ways to access data in pandas:
#
# **iloc:** allows you to access elements or sets of elements in the data based on numeric indexing, similar to how you would index any other array-like object:
# * `df.iloc[0, 1]`: accesses the element of the first row and second column
# * `df.iloc[[0, 1, 2],[0, 1, 2]]`: accesses the 3x3 sub-matrix spanning the first three rows and columns, respectively.
#
# **loc:** is very similar to iloc, but indexes based on the row or column names in the index and column header:
# * `df.loc["student_3", :]` accesses the row labeled `"student_3"`
# * `df.loc["student_3", "name"]`: accesses the entry of the `"name"` column of row `"student_4"`; i.e. equivalent to `df.iloc[3, 0]` without the need to know where in the DataFrame the information for the fourth student is stored.

# %%
# Accessing the second element in the first row
df.iloc[0, 1]

# %%
# Accessing the first three rows and columns
df.iloc[[0, 1, 2], [0, 1, 2]]

# %%
# Accessing the entries of the last row
df.iloc[-1, :]

# %%
# Accessing the data associtated with the id `"student_3"`
df.loc["student_3", :]

# %%
# Accessing the name associtated with the id `"student_3"`
df.loc["student_3", "name"]

# %%
# Accessing the name associtated with the id `"student_3"` via iloc
df.iloc[3, 0]

# %% [markdown]
# ### Commonly used methods
#
# Analyzing tabular data such as single-cell data, there are commonly used functions:
#
# * `sum`, `mean`, `std`, etc. compute the corresponding summary statistics (sum, average, standard deviation) of numerical rows or columns easily - on the whole DataFrame or a subset defined with iloc/loc. You can also specify the *axis* parameter to either compute over rows (axis=1) or columns (axis=0). See the examples below.
#
# * `groupby` aggregates the data based on column subsets to compute summary statistics within a category of interest, for example.
#
# * `sort_values` sorts the DataFrame based on a specified column - in ascending order if the column is numeric, and alphabetically if the column is of data type string.

# %%
# Summing all ages and marks, respectively
df.loc[:, ["age", "marks"]].sum(axis=0)

# %%
# Summing the ages and marks for each student
df.loc[:, ["age", "marks"]].sum(axis=1)

# %%
# Computing the mean age per grade
df.loc[:, ["age", "grade"]].groupby("grade").mean()

# %%
# Sort DataFrame by age
df.sort_values("age")

# %% [markdown]
# ### Reading and writing
#
# You can read and write DataFrames from and to disk using various formats. To retain data types and allow reading subsets of the data without having to load the entire dataset, use the parquet file format.
# 1. `pandas.read_csv()`: Reads data stored in any delimited file. By default, it assumes your data are comma-delimited, but you can specify any other delimiter with the 'sep' parameter. Other important parameters to know are 'header' (is there a dedicated header row in your data giving column names; if not, set 'header=None') and 'index_col' (is there a dedicated index column in your data giving row names; if so, set index_col=0).
# 2. `DataFrame.to_csv()`: Same idea, just for writing! The key parameters are 'header' (set False if you don't want to write out the dataframe header) and 'index' (same idea, for the index column).
# 3. `pandas.read_parquet()`, `DataFrame.to_parquet()`: Same concept as the corresponding CSV-methods, but the files are not in human readable format but retain data types. This should be the default method to read and write your data frames!
