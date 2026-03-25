# %% [markdown]
# # scRNA-seq data analysis
#
# Welcome to QIBS lab 1! In this lab, we will implement a standard scRNA-seq analysis pipeline, including
#
# 1. Data loading
# 2. Basic quality control
# 3. Data normalization and transformation
# 4. Feature selection
# 5. Dimensionality reduction
# 6. Cell typing
#
# Our dataset for today is a PBMC scRNA-seq dataset from the 10X genomics platform: the data has already been processed from raw sequencing reads to filtered count matrices with the [10X CellRanger pipeline](https://www.10xgenomics.com/support/software/cell-ranger/latest) - additional data cleaning will be required before delving into analysis, though.
#
# In a real-life analysis, you will likely use [_anndata_](https://anndata.readthedocs.io/en/stable/) and [_scanpy_](https://scanpy.readthedocs.io/en/stable/) to perform the outlined steps in Python. However, here, we will also implement each step using common Python packages for scientific computing like _numpy_, _pandas_, or _scipy_. To specify which kind of package and corresponding data to work with, we use the following color-codings for the exercises:
#
# <div style="padding: 10px; border-radius: 1px; width: 60%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px; width: 20%; display: inline-block; text-align: center; vertical-align: middle">
#     <b>Not Scanpy-based</b>
#   </div>
#   <div style="width: 5%; display: inline-block"></div>
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px; width: 20%; display: inline-block; text-align: center; vertical-align: middle">
#     <b>Scanpy-based</b>
#   </div>
#   <div style="width: 5%; display: inline-block"></div>
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px; width: 20%; display: inline-block; text-align: center; vertical-align: middle">
#     <b>Both</b>
#   </div>
# </div>
#
# Additionally, questions are color-coded green. Use the file _worksheet_1.xlsx_ to provide your answers and send them to weilerp@mskcc.org; name the file *worksheet_1_LASTNAME.xlsx*. **The deadline for submitting your answers is April 2 at 12pm ET**.
#
# <div style="padding: 10px; border-radius: 1px; width: 60%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px; width: 20%; display: inline-block; text-align: center; vertical-align: middle">
#     <b>Question</b>
#   </div>
# </div>
#
# ## Setup instructions
#
# Copy the data relevant for this lab into the repositories data directory.
#
# ```bash
# cp -r /athena/qibs_class/scratch/pfw4001/data/lab_1 /athena/qibs_class/scratch/CWID/qibs2026/data/
# ```
#
# Let's begin!

# %% [markdown]
# ## Library imports

# %%
import scanpy as sc

from qibs2026 import DATA_DIR  # noqa

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
sc.set_figure_params(frameon=False, transparent=True)

# %% [markdown]
# ## Constants

# %%
DATASET_ID = "lab_1"

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## 1. Data loading
#
# The output of CellRanger is a directory containing a number of files with different levels of processing. In most cases, we will work with the filtered - based on basic cell filtering - outputs in the sub-directory _filtered_feature_bc_matrix_ (see [here](https://www.10xgenomics.com/support/software/cell-ranger/latest/algorithms-overview/cr-gex-algorithm) for more details on CellRanger's filtering).
#
# The _filtered_feature_bc_matrix/_ sub-directory contains three files:
# 1. _features.tsv.gz_: a tab-delimited file with annotations for each gene in the data.
# 2. _barcodes.tsv.gz_: the cell barcodes.
# 3. _matrix.mtx.gz_: the data matrix, where columns (rows) correspond to the cells (genes) aligned with the other two files.

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 1.1</b>
#   </div>
#   <ul>
#     <li>Use <a href="https://pandas.pydata.org/">pandas</a> to read the cell information and save it in the variable <code>obs_data</code>, with column names <code>"cell_barcode"</code>; use the barcode column as index.</li>
#     <li>Use <a href="https://pandas.pydata.org/">pandas</a> to read the gene information and save it in the variable <code>var_data</code>, with column names <code>"ensembl_id"</code>, <code>"hgnc_id"</code>, and <code>"type"</code>.</li>
#     <li>Use <a href="https://scipy.org/">SciPy</a> to read the count data (varibale name <code>counts</code>) with cells as rows and genes as columns.</li>
#   </ul>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 1.2</b>
#   </div>
#   <p>Use <a href="https://scanpy.readthedocs.io/en/stable/">Scanpy</a> to read the count data and save it as a variable <code>adata</code>.</p>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# ## 2. Data overview

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 2.1</b>
#   </div>
#   <ol type="a">
#     <li>How many cells does the data include?</li>
#     <li>What format is the count data saved as and why?</li>
#   </ol>
# </div>

# %%
# Implement your solution here (using `counts`)

# %%
# Implement your solution here (using `adata`)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 2.1</b>
#   </div>
#   <ul>
#     <li>Make sure the counts are saved as a CSR matrix.</li>
#     <li>Confirm that the count data indeed consists of counts.</li>
#   </ul>
# </div>

# %%
# Implement your solution here (using `counts`)

# %%
# Implement your solution here (using `adata`)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 2.2</b>
#   </div>
#   Extract duplicate gene names from <code>var_data</code>.
# </div>

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 2.2</b>
#   </div>
#   <p>How many gene names are duplicated?</p>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 2.3</b>
#   </div>
#   Does the AnnData object contain the same number of duplicated genes? If not, why?</li>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 2.4</b>
#   </div>
#   Update the gene names in <code>var_data</code> to match <code>adata.var_names</code> and set the <code>hgnc_id</code> column as the index.
# </div>

# %%
# Implement your solution here

# %% [markdown]
# ## 3. Quality control
#
# As a first step, we will apply a first crude filtering to retain
# * cells expressing sufficiently many genes, and
# * genes expressed in sufficiently many cells.
#
# Cells expressing only few genes suggest these cells are impaired or not properly sequenced, and, vice-versa, if only few cells express a given gene, we cannot ensure statistical power for that gene.

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.1</b>
#   </div>
#   Remove cells expressing less than 100 transcripts.
# </div>

# %%
# Implement your solution here (cell filter)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.1</b>
#   </div>
#   <ol type="a">
#     <li>How many genes are not expressed?</li>
#     <li>How many genes are expressed but in less than 10 cells?</li>
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.2</b>
#   </div>
#   Remove genes expressed in less than 10 cells.
# </div>

# %%
# Implement your solution here (gene filter)

# %% [markdown]
# ### Cell-specific quality control metrics
#
# Following this first filtering, we can now compute quality control (QC) metrics to filter cells further - in a data-driven and statistically-grounded manner.
#
# As a first step, we compute the number of transcripts captured across all genes for each cell - the library size - and the number of total unique genes per cell. Cells expressing many genes and exhibiting a large library size will reveal outlier cells; such measurements can represent doublets, i.e., different cells sequenced as a single cell, for example.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.3</b>
#   </div>
#
#   <ul>
#     <li>Compute the library size, i.e., the , and report its median.</li>
#     <li>Compute the number of total unique genes captured in each cell, and report its median.</li>
#     <li>Compute the log1p-transformation of each statistics.</li>
#     <li>Add each statistic as a column to the <code>obs_data</code> DataFrame (names <code>"total_counts"</code>, <code>"log1p_total_counts"</code>, <code>"n_genes_by_counts"</code>, <code>"log1p_n_genes_by_counts"</code>).</li>
#   </ul>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.2</b>
#   </div>
#   <p>Why is a canonical log-transformation inadmissible for single-cell data?</p>
# </div>

# %%
# Implement your solution here (library size)

# %%
# Implement your solution here (number of genes per cell)

# %% [markdown]
# Additional important QC metrics are the total fraction of the transcripts that are ribosomal RNAs or that attributed to the mitochondrial genome: mitochondrial fraction has been show to increase in dead or dying cells [(Azizi et al. 2018)](https://pubmed.ncbi.nlm.nih.gov/29961579/); high fractions of ribosomal RNA entail less coverage of the for us relevant mRNA.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.4</b>
#   </div>
#
#   <ul>
#     <li>For each cell, compute the total and log1p-transformed total counts and percentage of counts originating from mitochondrial and ribosomal genes, respectively.</li>
#     <li>Add each statistic as a column to the <code>obs_data</code> DataFrame (names <code>"total_counts_"</code>, <code>"log1p_total_counts_"</code>, <code>"pct_counts_"</code> with suffixes <code>"mito"</code> and <code>"ribo"</code>).</li>
#   </ul>
# </div>

# %%
# Implement your solution here (mitochondrial genes)

# %%
# Implement your solution here (ribosomal genes)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.3</b>
#   </div>
#
#   <ol type="a">
#     <li>How many mitochondiral genes does the data include?</li>
#     <li>What is the highest mitochondrial percentage (rounded to two decimals)? Report the corresponding cell barcode.</li>
#     <li>How many ribosomal genes does the data include?</li>
#     <li>What is the highest ribosomal percentage (rounded to two decimals)? Report the corresponding cell barcode.</li>
#   </ol>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# ### Gene-specific quality control metrics
#
# Equivalent to cell-specific QC metrics, we can compute gene-specific metrics.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.5</b>
#   </div>
#   <ul>
#     <li>Add the number of cells expressing each gene as a column <code>n_cells_by_counts</code> to <code>var_data</code>.</li>
#     <li>Add the mean gene expression and its log1p transformation as columns <code>mean_counts</code> and <code>log1p_mean_counts</code> to <code>var_data</code>.</li>
#     <li>Add the percentage of cells not expressing each gene as a column <code>pct_dropout_by_counts</code> to <code>var_data</code>.</li>
#     <li>Add the total gene expression and its log1p transformation as columns <code>total_counts</code> and <code>log1p_total_counts</code> to <code>var_data</code></li>
#   </ul>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.4</b>
#   </div>
#
#   Which gene has the highest mean expression?
# </div>

# %%
# Implement your solution here (n_cells_by_counts)

# %%
# Implement your solution here (mean GEX)

# %%
# Implement your solution here (dropout percentage)

# %%
# Implement your solution here (total counts)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.6</b>
#   </div>
#   <ol>
#     <li>Visualize the distribution of the log1p-transformed library sizes and number of genes per cell, and the percentage of mitochondrial and ribosomal counts.</li>
#     <li>Plot the mitochondiral count percentage against the library size.</li>
#     <li>Plot the number of genes per cell against the library size.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (single-QC plots)

# %%
# Implement your solution here (QC metrics plotted against each other)

# %% [markdown]
# We can use the plots visualizing the relationship between mitochondrial count percentage and number of genes per cell versus the total counts per cell, respectively, to identify outlier cells. Specifically, we are looking for cell with
# 1. high percentage of mitochondrial genes and low total counts &rarr; likely dying cells
# 2. high number of genes expressed and large library size &rarr; potentially doublets
#
# Remember that high/low are always compared to the rest of the data! Our thresholds for filtering also depends on the system we are studying as we naturally expect stressed cells in a disease of inflammatory context, for example.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.7</b>
#   </div>
#   Remove cells expressing more than 60,000 transcripts or where mitochondrial genes account for at least 20 percent of the transcripts.
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.5</b>
#   </div>
#
#   What is the dimension of our dataset after filtering?
# </div>

# %%
# Implement your solution here

# %% [markdown]
# As a final step, we will perform the same analysis using Scanpy directly.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.8</b>
#   </div>
#   Repeat the quality control steps using Scanpy, i.e.,
#   <ol type="a">
#     <li>Remove cells expressing less than 100 transcripts.</li>
#     <li>Remove genes expressed in less than 10 cells.</li>
#     <li>Update the AnnData object with the cell- and gene-specific metrics defined above but no additional metrics.</li>
#     <li>Remove cells expressing more than 60,000 transcripts and where mitochondrial genes account for at least 20 percent of the transcripts.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (Scanpy-based workflow)

# %% [markdown]
# ## 4. Data preprocessing

# %% [markdown]
# ### Count normalization and transformation
#
# Following filtering cells and genes and equivalent to standard data analysis pipeline, it is now time to normalize our observations. Normalization ensures that all cells are comparable relative to each other - an aspect not automatically guaranteed since the capture rate can vary dramatically across cells and, thus, heavily confounding the count values in scRNA-seq experiments. For example, does a value of '2' transcripts mean the same thing in a cell which had 10,000 transcripts, vs. one that is 1,000 transcripts? Probably not!

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.1</b>
#   </div>
#   <ol type="a">
#     <li>Normalize the cell size to the median library size, and save the resulting array as <code>normalized_counts</code> in CSR format.</li>
#     <li>Log1p-transform the normalized counts, and save the resulting array as <code>log1p_normalized_counts</code>.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (normalization)

# %%
# Implement your solution here (log1p-transformation)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.2</b>
#   </div>
#   <ol type="a">
#     <li>Save the whole counts as a layer of <code>adata</code>, named <code>"counts"</code>.</li>
#     <li>Normalize the data to the median library size.</li>
#     <li>Log1p-transform the normalized data.</li>
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 4.1</b>
#   </div>
#
#   How can you keep track of whether or not your data is log1p-transformed in an AnnData-Scanpy-based workflow?
# </div>

# %%
# Implement your solution here

# %% [markdown]
# We will not z-score our data here, but it is an optional step to be aware of: in some cases, we want to scale each gene to zero mean and a standard deviation of 1. Such a transformation can also dramatically change the structure of your data.
#
# The guiding principle here: there are many subjective choices in single-cell analysis, each of which can have major consequences on your embeddings, visualizations, differential expression, etc. It is a good idea to play around with your data with multiple iterations of processing before finalizing an analysis. Always think about your specific biological system and question, and how you might justify each choice you make.

# %% [markdown]
# ### Feature selection
#
# As a next step, we will identify genes we even want to consider or focuse on in our analysis (beyond those we filtered earlier for technical reasons). Intuitvely, we want to focus on genes varying across our observations as they contain non-redundant information; similarly, housekeeping genes are ubiquitously expressed and do not carry much information on biological heterogeneity. Highly-variable gene (HVG) selection accomplishes this goal by first defining a measure of how variable each gene is across cells in our dataset; thresholding defines HVGs; subsequent downstream analyses will operate on these HVGs to generate embeddings and visualizations of cells.
#
# **Important:** unlike poorly-captured genes we filtered earlier, we do not actually want *remove* lowly-variable genes from your dataset entirely as some genes may be differential across conditions, cell types, etc. even when they are not statistically highly-variable. No method for detection of highly-variable genes is perfect, so the goal is to use HVGs for generating your embeddings/visualizations, but retaining all genes for future analysis.
#
# At first thought, you might consider simply computing the empirical variance of each gene across cells as a measure of variability. However, the variance strongly correlates with mean expression, i.e., filtering based on variance is nearly equivalent to picking the most highly expressed genes overall! In reality, sometimes genes with a low mean expression are going to be the most important, such as those expressed highly in a rare population of cells.
#
# To consider the scaling of the variance with the mean, the method developed by [Smillie et al. method](https://pubmed.ncbi.nlm.nih.gov/31348891/) attempts to find genes whose variance is higher than expected, given the mean expression of that gene. This is accomplished by first fitting a simple model where *dispersion* (the variance standardized by the mean) is a function of the mean, and the *residuals* (true dispersion - predicted dispersion) are a measure of variability. We will implement this method here, but Scanpy provides more robust and sophisticated HVG identification routines.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.3</b>
#   </div>
#   <ol type="a">
#     <li>For each gene, compute the mean and standard deviation of its log1p-normalized expression.</li>
#     <li>Compute the dispersion. To avoid dividing by zero, set zero mean values to 1 x 10-12</li>
#     <li>Using statsmodels, fit a locally weighted scatterplot smoothing function to the log-dispersion and log-mean.</li>
#     <li>Compute the residuals under the fitted model, and plot both the data, colored based on these residuals, and fitted curve.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (mean, standard deviation, and dispersion)

# %%
# Implement your solution here (model fit and residual computation)

# %%
# Implement your solution here (plotting)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.4</b>
#   </div>
#   Extract the 1000 most highly variable genes.
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 4.2</b>
#   </div>
#
#   What are the 5 most highly variable genes?
# </div>

# %%
# Implement your solution here (plotting)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.5</b>
#   </div>
#   Identify and annotate the 2000 most highly variable genes. Only provide the minimal set of function arguments and use default parameters otherwise.
#   </ol>
# </div>

# %%
# Implement your solution here

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.6</b>
#   </div>
#   Subset the data to the 2000 HVGs identified with Scanpy.
# </div>

# %%
# Implement your solution here

# %% [markdown]
# ## Dimensionality reduction
#
# With our data completely cleaned and a set of useful features selected, we will finally begin with embedding the cells in lower-dimensional space. Dimensionality reduction is often the first step to understanding your data. While we can glean very little from the way cells distribute in the 2000-dimensional space spanned along our highly-variable genes, the major axes of variation in this space can often be very interpretable. On top of that, dimensionality reduction can serve to de-noise our dataset. Most downstream analyses, such as visualization with UMAP or clustering cells, are often performed directly on reduced dimensions, rather than the full-dimensional per-gene data.
#
# In this lab, we will use the most standard approach for dimensionality reduction: Principal Components Analysis (PCA), a linear decomposition method which places cells on axes upon which the variance of the data is maximized.
#
# An extremely important step for downstream analysis is selecting which principal components to carry forward for visualization, graph building, and so forth. In PCA, the components are *ordered*, with the 'most important' components coming first. This ordering is based on a metric for the total amount of variance in the data explained by each component. If we were to compute all possible components (namely the total number of genes), we would explain all the variance, but generally we do not need to (or want to!) as later components tend to explain more of the *noise* and less of the *signal* in the data. We can choose a good number of PCs by plotting the variance ratio against the number of PCs, and selecting a threshold after which the variance added with additional PCs is negligible.
#
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 5.1</b>
#   </div>
#   <ol type="a">
#     <li>Compute the first 50 principal components with sci-kit learn based on the log1p-normalized data.</li>
#     <li>Plot the explained variance ratio agains the number of principal components.</li>
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 5.1</b>
#   </div>
#
#   What is a reasonable number of principal components to choose?
# </div>

# %%
# Implement your solution here (PC computation)

# %%
# Implement your solution here (plotting)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 5.2</b>
#   </div>
#   Select the 30 PCs associated with the highest explained variance ratio.
# </div>

# %%
# Implement your solution here (plotting)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 5.3</b>
#   </div>
#   Compute the PCA embedding with 30 PCs.
# </div>

# %%
# Implement your solution here (plotting)

# %% [markdown]
# ### UMAP
#
# Although PCA reduced our data representation from a 2000-dimensional space into a 30-dimensional space, visualizing data requires a 2D or 3D representation. An ideal compression will retain the global topology and relative distance between clusters - the objective of [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) (UMAP; see [here](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) for an intuitive and illustrative explanation of the algorithm).
#
# **Three important aspects to keep in mind:**
#
# 1. Never rely on 2D/3D representations for quantification! Always validate observations of findinings in higher-dimensional spaces such as the PCA representation or the original GEX space.
# 2. UMAP embeddings are rotation invariant &rarr; we can always rotate our embedding space to our preferred orientation
# 3. UMAP preserves the global topology and relative cluster distances &rarr; **ALWAYS** plot UMAP embeddings in square format with the aspect set to "auto" (default in Scanpy)
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 5.4</b>
#   </div>
#   <ol type="a">
#     <li>Use the <i>umap</i> package to compute the 2D UMAP coordinates, based on a 30-nearest neighbor graph and an effective minimum distance between embedded points of 0.5.</li>
#     <li>Add the coordinates as columns <code>"umap_1"</code> and <code>"umap_2"</code> to the appropriate DataFrame.</li>
#     <li>Visualize the UMAP embedding.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (plotting) (umap coordinates)

# %%
# Implement your solution here (plotting)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 5.5</b>
#   </div>
#   <ol type="a">
#     <li>Compute a k-nearest neighbor graph with k=30 (UMAP performs this step internally).</li>
#     <li>Compute the umap embedding with an effective minimum distance between embedded points of 0.5.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (neighbor graph and umap embedding)

# %%
# Implement your solution here (plotting)

# %% [markdown]
# ## Cell typing
#
# Havin visualized our data, we observe different clusters which naturally leads us to the question: how can we systematically cluster cells based on their GEX profile, and what do these clusters encode biologically?
#
# Here, we will use the Leiden algorithm to cluster our data: in a first step, the algorithm defines an initial, trivial partioning of a given kNN graph where each node is its own cluster. Following, the algorithm iteratively merges clusters to define partitions by taking into account the number of links between cells in a cluster compared to the overall expected number of links in the dataset.
#
# <img src="https://www.sc-best-practices.org/_images/clustering.jpeg" alt="genescore" width="75%">
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 6.1</b>
#   </div>
#   <ol type="a">
#     <li>Extract the weighted adjacency matrix from <code>adata</code>.</li>
#     <li>Using the Python package *igraph* define the corresponding undirected and weighted graph object named <code>graph</code>.</li>
#     <li>Compute the leiden clustering of <code>graph</code> based on the modularity objective and at resolution 1.</li>
#     <li>Add the leiden cluster assignment as a colum <code>"leiden"</code> to the appropriate DataFrame.</li>
#     <li>Plot the UMAP embedding colored by leiden cluster; use <code>vega_20_scanpy</code> for coloring.</li>
#   </ol>
# </div>

# %%
# Implement your solution here (igraph graph definition)

# %%
# Implement your solution here (leiden clustering)
from scanpy._utils.random import set_igraph_random_state  # isort: skip # noqa

# Use the context `with set_igraph_random_state(0):`

# %%
# Implement your solution here (plotting)
from scanpy.plotting.palettes import vega_20_scanpy  # isort: skip # noqa

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 6.2</b>
#   </div>
#   Compute the Leiden clustering as before and visualize it in UMAP space.
# </div>

# %%
# Implement your solution here (leiden clustering)

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 6.3</b>
#   </div>
#   To ensure reproducibility, update the leiden cluster annotation with <code>DATA_DIR / DATASET_ID / "results" / "leiden.parquet"</code>
# </div>

# %%
# Implement your solution here (leiden clustering)

# %% [markdown]
# Finally, to associate the identified clusters with biology, we identify genes differentially expressed in each cluster compared to the rest of the data, ultimatly allowing us to assign cell type labels to these clusters, for example, by studying the expression of known markers in across the clusters.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(86, 104, 134); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 6.4</b>
#   </div>
#   <ol type="a">
#     <li>Use a t-test to identify differentially expressed genes in leiden cluster 1 compared to all other cells. Assume that the variance between the two groups differs.</li>
#     <li>Adjust the p-values using Benjamini-Hochberg.</li>
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 6.1</b>
#   </div>
#   <ol>
#     <li>How many genes are significantly enriched in leiden cluster 1 at a false discover rate of 0.05?</li>
#     <li>Which are the three most differentially enriched genes?</li>
#   </ol>
# </div>

# %%
# Implement your solution here

# %%
# Implement your solution here

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(134, 86, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 6.5</b>
#   </div>
#   Compute DEGs for all groups and print the results (gene names, scores, log fold changes, p values, and adjusted p values) for group 1 as a pandas DataFrame.
# </div>

# %%
# Implement your solution here

# %% [markdown]
# `scanpy.pl.rank_genes_groups_dotplot` enables studying the expression of a set of given genes in the indentified Leiden cluster systematically: for each gene, the dot size indicates how many cells in a given group express a given gene, and the color of the dot encodes the group-specific mean expression. Here, we can see that cells in cluster 1 are likely Eryhtrocytes, and cells in cluster 7 T cells, for example.

# %%
# Uncomment to visualize the GEX using the dotplot
# sc.pl.rank_genes_groups_dotplot(
#     adata,
#     var_names={
#         "T": ["CD4", "CD8A", "CD3D"],
#         "B": ["CD79A"],
#         "Neutrophil": ["FCGR3B"],
#         "Monocyte": ["CD14"],
#         "Erythrocyte": ["HBA1", "HBA2", "HBB"],
#     },
#     standard_scale="var",
# )
