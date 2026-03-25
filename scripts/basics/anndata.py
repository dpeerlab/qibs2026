# %% [markdown]
# # anndata - Annotated data
#
# Please read through the following sections for a very brief overview if you are unfamiliar with the Python package [_anndata_](https://anndata.readthedocs.io/en/latest/). For an in-depth overview head over to the anndata documentation and tutorials.
#
# ## AnnData data structure
#
# [_anndata_](https://anndata.readthedocs.io/en/latest/) allows for and builds the backbone for storing and manipulating scRNA-seq datasets as it is tabular data in a single object. Consider the following use case: you have the count matrix from a scRNA-seq experiment and associated metadata for the cells and genes. Using pandas, for example, you would work with three variables: one for the counts, one for the cell metadata, and one for the gene metadata. So when you filter out cells, you will have to update both the counts and cell metadata variables - with anndata, your variables are contained in a single variable - the AnnData object -, aligned along shared axes, and each subsetting updates all relevant entries automatically. Another major advantage of anndata is its support of sparse data structures.
#
# The schematic below illustrates the general structure of a an AnnData object:
#
# 1. _X_ contains your cell by gene matrix
# 2. _layers_ contains additional matrices of dimension cell by gene such as a transformation of _X_
# 3. _obs_ is a pandas DataFrame collecting all cell-specific data
# 4. _var_ is a pandas DataFrame collecting all gene-specific data
# 5. _obsm_ and _varm_ contain matrices whose rows are aligned with the rows and colums of _X_, respectively, but the number of columns is not pre-defined
# 6. _obsp_ and _varp_ contain square matrices aligned with the rows and colums of _X_, respectively
# 7. _uns_ is a dictionary that can hold any information
#
# <img src="https://raw.githubusercontent.com/scverse/anndata/main/docs/_static/img/anndata_schema.svg" alt="genescore" width="600">

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import anndata as ad

# %% [markdown]
# ## AnnData basics

# %% [markdown]
# Let's define a simple AnnData object, that collects a count matrix and some metadata and perform some basic manipulations.

# %%
n_obs = 10  # Number of cells
n_vars = 15  # Number of genes
adata = ad.AnnData(csr_matrix(np.random.poisson(1, size=(n_obs, n_vars)), dtype=np.float32))

# Define cell metadata
adata.obs = pd.DataFrame(
    {
        "patient_id": np.random.choice(["patient_1", "patient_2", "patient_3"], size=n_obs, replace=True),
        "site": np.random.choice(["MSK", "WCM"], size=n_obs, replace=True),
    },
    index=[f"obs_{obs_id}" for obs_id in range(n_obs)],
)
# Set data type of obs columns
adata.obs["patient_id"] = adata.obs["patient_id"].astype("category")

# Set names for variables (columns; genes)
adata.var_names = [f"var_{var_id}" for var_id in range(n_vars)]

adata

# %%
# Access cell names
adata.obs_names

# %%
# Access number of cells
adata.n_obs

# %%
# Access count data stored in X
adata.X

# %%
# Subset to patients 1 and 2
obs_mask = adata.obs["patient_id"].isin(["patient_1", "patient_2"])
adata = adata[obs_mask, :].copy()

# %%
# Verify subsetting
adata.obs["patient_id"].cat.categories

# %%
# Subset to first five observations and seven features
adata[:5, :7]

# %%
# Add data to layers
adata.layers["counts"] = adata.X.copy()
adata.layers["counts_dense"] = adata.X.toarray().copy()
adata

# %%
# Add gene metadata
adata.var["highly_variable"] = np.random.choice([True, False], size=adata.n_vars, replace=True)
adata
