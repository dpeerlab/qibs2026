# %% [markdown]
# # Single-cell perturbation data
#
# In this session, we will discuss analysing single-cell perturbation data by analysing a single-cell CRISPR screen dataset, collected from the K562 lymphoblast cell line and published by [Norman et al. in 2019](https://www.science.org/doi/10.1126/science.aax4438#supplementary-materials). The study presents a library of guide RNAs designed for use with CRISPRa system - a system which enacts up-regulation of the targeted genes - in contrast to standard CRISPR-Cas9 - a system to selectively mutate targeted genes. That said, analysis of this data will entail determining which programs are induced or repressed in response to over-expression of the perturbed gene.
#
# This lab focuses on taking the data from its raw format through normalization and transformation specifically designed for Perturb-seq datasets to ultimately identify which genes each perturbation induces or represses. Furthermore, this experiment was designed to assess the combinatorial impact of two simultaneous perturbations. For example, if genes A and B are in different pathways which converge on the same targets, their joint over-expression may combine synergistically to drive target expression into overdrive; if they are instead on the same pathway, only the activation of the "more downstream" of the two might be important to drive the effect. To this end, we will develop a (simple) statistical model to delineate combinatorial impactss such as synerigistic and non-linear on gene expression.
#
# Let's started!
#
# ## Setup instructions
#
# You can either use the `DATA_DIR` variable to import the data from *data/lab_2/processed/*, or copy the data relevant for this lab into your repositories data directory with
#
# ```bash
# cp -r /athena/qibs_class/scratch/pfw4001/data/lab_2 /athena/qibs_class/scratch/CWID/qibs2026/data/
# ```
#
# and import from there.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

import anndata as ad
import scanpy as sc

from qibs2026 import DATA_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
sc.set_figure_params(frameon=False, transparent=True)

# %% [markdown]
# ## Constants

# %%
DATASET_ID = "lab_2"

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## 1. Data loading
#
# The data we are using today already includes metadata and QC metrics. The most relevant columns in `adata.obs` are
#
# 1. `UMI_count`: the library size.
# 2. `gemgroup`: the 10X lane of each cell, i.e., it is a technical parameter across which quality metrics (such as library size) may vary. As such, we may, for instance, observe systematic differences in the capture rate between different lanes, leading to "batch effects."
# 3. `guide_<GENE>`: a binary indicator for whether or not a guide for gene *GENE* was detected in that cell. By design of the experiment, for each cell, we expect only at most two genes to have a positive value for these columns.
# 4. `perturbation_name`: an identifier for single or combined perturbations, derived from the `guide_<GENE>` columns where combined perturbations follow the format `<GENE_A>+<GENE_B>`; otherwise the value is equal to the single-gene perturbation; the annotation for cells with no perturbations is `control`.
# 5. `gene_program`: the gene program each guide or combination of guides was intended to target based on the paper. If no known annotation is provided, the the corresponding value is `Unknown`.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 1.1</b>
#   </div>
#   Use <i>anndata</i> to load the H5AD file <i>adata_annotated.h5ad</i>.
# </div>

# %%
# Implement your solution here
adata = ad.io.read_h5ad(DATA_DIR / DATASET_ID / "processed" / "adata_annotated.h5ad")
adata

# %% [markdown]
# ## 2. Data overview
#
# As a start, let's quickly check the information already present in the data to know what we are dealing with - a common and important step of any data anlysis pipeline.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 2.1</b>
#   </div>
#   Verify that there are indeed at most two guides per cell.
# </div>

# %%
# Implement your solution here
guide_rna_cols = adata.obs.columns[adata.obs.columns.str.contains(r"^guide_[A-Z]")]
assert adata.obs.loc[:, guide_rna_cols].sum(axis=1).max() <= 2

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 2.1</b>
#   </div>
#   <ol type="a">
#     <li>How many genes are annotated as highly variable?</li>
#     <blockquote>5000</blockquote>
#     <li>How many perturbations are there in total?</li>
#     <blockquote>236</blockquote>
#     <li>How many cells included a FOXA1 guide?</li>
#     <blockquote>2182</blockquote>
#     <li>Which gene is perturbed most frequently and how often? I.e., the gene corresponding to the guide captured most often.</li>
#     <blockquote>KLF1 (6131)</blockquote>
#   </ol>
# </div>

# %%
# Implement your solution here
print(f"Number of HVGs: {adata.var['highly_variable'].sum()}")

print(f"Number of total perturbations: {adata.obs['perturbation_name'].cat.categories.difference(['control']).size}")

print(f"Number cells including a FOXA1 guide: {adata.obs['guide_FOXA1'].sum()}")

n_perturbations = adata.obs.loc[:, guide_rna_cols].sum(axis=0).sort_values(ascending=False)
print(f"Most frequently perturbed gene: {n_perturbations.index[0].removeprefix('guide_')} ({n_perturbations.iloc[0]})")

# %% [markdown]
# The data includes pre-computed UMAP embeddings which we can use to visualize cells including a specific guide or the gene programs assigned, for example.

# %%
# Cells with a FOXA1 guide
fig = sc.pl.embedding(
    adata,
    basis="umap",
    color="guide_FOXA1",
    cmap="binary",
    vmin=-0.1,  # to ensure 0 does not correspond to white
    size=25,
    title="Cells with FOXA1 guide",
    return_fig=True,
)
fig.set_size_inches(6, 6)

# %%
# Cells colored by the gene program targeted by their corresponding guide
fig = sc.pl.embedding(adata, basis="umap", color="gene_program", title="Targeted gene program", size=5, return_fig=True)
fig.set_size_inches(6, 6)

# %% [markdown]
# ## 3. Data preprocessing
#
# Perturb-seq experiments are a special scRNA-seq experiment to understanding the impact of perturbations. As such, data preprocessing is similar to our previous scRNA-seq data analysis - library size is a technical artifact across cells which must be accounted for, for example - but also entails additional steps to boost the signal in our data. Remember: we are trying to assess potentially minute impacts on gene expression data which is already rife with noise and heavily under-sampled. Specifically, we are interested in cell states upon perturbation and will, thus, place each in the context of our control cells.
#
# ### Data normalization
#
# #### Library size normalization
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.1</b>
#   </div>
#   <ol type="a">
#     <li>Normalize the library size of each cell to 10,000.</li>
#     <li>Subset to genes with mean normalized GEX larger than 0.5.</li>
#   </ol>
# </div>

# %%
# Implement your solution here
sc.pp.normalize_total(adata=adata, target_sum=1e4)

var_mask = adata.X.mean(axis=0).A1 >= 0.5
print(f"Removing {(~var_mask).sum()} genes with mean normalized GEX below 0.5.")
adata = adata[:, var_mask].copy()
adata

# %% [markdown]
# #### Z-scoring
#
# To assess the impact of perturbations, we compare the GEX observed in perturbed cells to those observed in control cells, i.e., a baseline. Thus, we z-score each gene w.r.t. GEX in control cells to (1) study GEX in terms of deviations relative to control cells and (2) place all genes on a similar scale. Here, to account for batch effects, we z-score each 10X lane separately (column _"gemgroup"_ in `adata.obs`). These transformed values no longer represent GEX itself but rather induction (positive z-score) and repression (negative z-score) upon perturbation.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.2</b>
#   </div>
#   Define the layer <i>znormed_wrt_control</i> as the 10X-lane-specific z-normalized data, i.e., for each lane
#   <ol type="a">
#     <li>Subset the normalized data to the given lane.</li>
#     <li>Compute the mean and standard deviation of control cells in this subset.</li>
#     <li>Set the standard deviation to at least 1e-6, to prevent division by zero and ensure numerical stability.</li>
#     <li>Z-score the subset.</li>
#   </ol>
# </div>
#
# **Self-check:** The overall mean z-normalized expression is -0.005573.

# %%
# Implement your solution here
adata.layers["znormed_wrt_control"] = np.empty(adata.shape)

for gemgroup in adata.obs["gemgroup"].cat.categories:
    gemgroup_mask = adata.obs["gemgroup"] == gemgroup
    gemgroup_gex = adata[gemgroup_mask, :].X.toarray()

    control_gemgroup_mask = (adata.obs["perturbation_name"] == "control")[gemgroup_mask]
    control_mean = gemgroup_gex[control_gemgroup_mask, :].mean(axis=0)
    control_std = np.maximum(gemgroup_gex[control_gemgroup_mask, :].std(axis=0), 1e-6)

    adata.layers["znormed_wrt_control"][gemgroup_mask, :] = (gemgroup_gex - control_mean) / control_std

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.1</b>
#   </div>
#   Which gene has the largest across-cell-average deviation? Report both the gene and the numerical value, rounded to four decimals.
#   <blockquote>ALAS2 (0.6438)</blockquote>
# </div>

# %%
# Implement your solution here
z_means = np.abs(adata.layers["znormed_wrt_control"].mean(axis=0))
z_mean_argmax = np.argmax(z_means)
print(
    f"Largest across-cell-average deviation observered in {adata.var_names[z_mean_argmax]} ({z_means[z_mean_argmax]:.4f})"
)

# %% [markdown]
# ### Pseudobulking
#
# We generally care about **average** perturbation effects. Thus, to boost our signal and mitigate the sparsity of single-cell data, we aggregate cells with the same perturbation. Specifically, since most analysis is in practice done at mean level, we just group cells with the same perturbation together, thereby defining a so-called *pseudobulk* representation of our data. Pseudobulking loses the per cell information, but retains the per gene resolution; i.e, we can save the corresponding data in the `adata.varm` slot.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 3.3</b>
#   </div>
#   <ol type="a">
#     <li>Define a pandas DataFrame with the z-normalized data as values, the cell barcodes as index, and gene names as columns.</li>
#     <li>Add the perturbation information as a column.</li>
#     <li>Use pandas <code>groupby</code> function to compute the mean z-scores for each perturbation.</li>
#     <li>Save the pseudobolk data as an entry <i>zscore_pseudobulk</i> in <code>adata.varm</code>.</li>
#   </ol>
# </div>

# %%
# Implement your solution here
pseudobulk = adata.to_df(layer="znormed_wrt_control")
pseudobulk["perturbation_name"] = adata.obs["perturbation_name"]

pseudobulk = pseudobulk.groupby("perturbation_name", observed=False).mean().T
pseudobulk.columns = pseudobulk.columns.astype(str)
adata.varm["zscore_pseudobulk"] = pseudobulk.loc[adata.var_names, :].copy()

del pseudobulk

# %% [markdown]
# Given the pseudobulked z-scores, we can embed the perturbations in a UMAP embedding similar to canonical scRNA-seq data. However, now, each dot corresponds to a perturbation, not a cell.

# %%
umap_model = UMAP(n_components=2)
data = umap_model.fit_transform(adata.varm["zscore_pseudobulk"].T)
data = pd.DataFrame(data, columns=["umap_1", "umap_2"], index=adata.varm["zscore_pseudobulk"].columns)
data = data.merge(
    adata.obs[["perturbation_name", "gene_program"]].drop_duplicates().set_index("perturbation_name"),
    left_index=True,
    right_index=True,
)

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(data=data, x="umap_1", y="umap_2", hue="gene_program", palette="colorblind", s=100, alpha=0.75, ax=ax)
ax.axis("off")
ax.legend(title="Targeted gene program", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.show()

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 3.2</b>
#   </div>
#   <ol type="a">
#     <li>On average, which gene is most induced?</li>
#     <blockquote>ALAS2</blockquote>
#     <li>On average, which gene is most repressed?</li>
#     <blockquote>RANBP1</blockquote>
#     <li>On average, which gene is most impacted, i.e., induced or repressed?</li>
#     <blockquote>HBZ</blockquote>
#     <li>Which perturbation induces TAL1 the most?</li>
#     <blockquote>FEV+ISL2</blockquote>
#     <li>Which perturbation represses BRD4 the most?</li>
#     <blockquote>SAMD1+TGFBR2</blockquote>
#     <li>Which perturbation represses CEBPB the most?</li>
#     <blockquote>CBFA2T3+PRDM1</blockquote>
#   </ol>
# </div>

# %%
# Implement your solution here
mean_pseudobulk_zscore = adata.varm["zscore_pseudobulk"].mean(axis=1)
print(f"Most induced gene: {mean_pseudobulk_zscore.idxmax()}")
print(f"Most repressed gene: {mean_pseudobulk_zscore.idxmin()}")
print(f"Most impacted gene: {adata.varm['zscore_pseudobulk'].abs().mean(axis=1).idxmax()}")

print(f"{adata.varm['zscore_pseudobulk'].loc['TAL1', :].idxmax()} induces TAL1 the most.")
print(f"{adata.varm['zscore_pseudobulk'].loc['BRD4', :].idxmin()} represses BRD4 the most.")
print(f"{adata.varm['zscore_pseudobulk'].loc['CEBPB', :].idxmin()} represses CEBPB the most.")

# %% [markdown]
# ## 4. Modeling the impacts of genetic interactions
#
# Having processed our data to express relative perturbation-induced changes, we can now focus on the actual biological question at hand: **how do pairs of genes interact functionally to manipulate a downstream target program**? Answering this quesion computationally amounts to comparing the effects of single-gene activation to joint (combined) activation of a gene pair.
#
# The basic premise is as follows: we will fit a linear model that predicts the effects $\Delta$ of joint activation of a pair of genes (A+B) as a linear combination of individual genes in that pair (A and B separately), i.e.,
#
# $$\Delta_{A+B} = \alpha \Delta_{A} + \beta \Delta_{B},$$
#
# with regression parameters $\alpha$ and $\beta$.
#
# Based on these fits, we then employ the following logic:
#
# 1. *Good prediction with (approximately) equal values for $\alpha$ and $\beta$* suggests perturbations impact gene expression additively in a synergystic effect.
# 2. *Good prediction with different values for $\alpha$ and $\beta$* suggests perturbations impact gene expression additively but the effect of the perturbed genes differs; e.g., if $\alpha$ is much larger than $\beta$, then gene A dominates in the sense that its impacts overwhelm the impacts of the other perturbation.
# 3. *Poor prediction* suggests that the genes combine in a non-linear fashion such as when over-expressing both genes is necessary to induce the effect, and each individual gene perturbation alone does not mimic the combined effect in any sense.
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.1</b>
#   </div>
#   Plot the pseudobulked z-scores of the <i>CBL+CNN1</i> perturbation against the ones of each individual perturbation, respectively. Based on this plot, Which of our three scenarios do you expect to hold?
# </div>

# %%
# Implement your solution here
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))
sns.scatterplot(adata.varm["zscore_pseudobulk"], x="CBL", y="CBL+CNN1", color="black", s=5, edgecolor=None, ax=ax[0])
ax[0].set(xlabel="Effect of CBL alone on GEX", ylabel="Combined effect of CBL and CNN1 on GEX")
sns.scatterplot(adata.varm["zscore_pseudobulk"], x="CNN1", y="CBL+CNN1", color="black", s=5, edgecolor=None, ax=ax[1])
ax[1].set(xlabel="Effect of CNN1 alone on GEX")
plt.show()


# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.2</b>
#   </div>
#   Define a function <code>linear_interaction_model</code> that takes the pseudobulked data and a double perturbation as input (keywords <code>pseudobulk</code> and <code>perturbation</code>) as well as an optional argument to return the model prediction (keyword <code>return_pred</code> with default set to <code>False</code>), fits the above defined linear model, and returns the fitted coefficients, their signed log-fold change, and the models coefficient of determination (R2; R-squared) value. Proceed as follows:
#   <ol type="a">
#     <li>Extract the individual gene names from the <code>perturbation</code> variable.</li>
#     <li>Define the data matrix for the linear model; i.e., extract the pseudobulk data corresponding to the individual genes of the double perturbation.</li>
#     <li>Define the response variable for the linear model; i.e., extract the pseudobulk data corresponding to the double perturbation.</li>
#     <li>Use <a href="https://scikit-learn.org/stable/">scikit-learn's</a> LinearRegression class to fit the linear model.</li>
#     <li>Extract the fitted coefficients and the R2 value.</li>
#     <li>Compute the signed log-foldchange of the coefficients, i.e., the log-foldchange of their absolute values multiplied by the product of their signs.</li>
#     <li>Return a pandas DataFrame indexed by the double perturbation with columns <i>coef_1</i>, <i>coef_2</i>, <i>lfc</i>, <i>r2</i>.</li>
#   </ol>
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 4.1</b>
#   </div>
#   Why do we compute a signed LFC?
#   <blockquote>The signs of the regression parameters may differ and, thus, lead to negative ratios for which the log is not defined.</blockquote>
# </div>
#
# **Self-check:** For the *CBL+CNN1* perturbation, the fitted coefficients are 1.1156 and 1.2811, with an R-squared value of 0.7395.


# %%
# Implement your solution here
def linear_interaction_model(
    pseudobulk: pd.DataFrame, perturbation: str, return_pred: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Fit and evaluate linear model.

    Parameters
    ----------
    pseudobulk
        Pseudobulked z-scored, indexed by gene name and with perturbations as column names.
    perturbation
        The name of the double perturbation.
    return_pred
        Return the model prediction.

    Returns
    -------
    If `return_pred==False`, a pandas DataFrame with columns
    * `coef_1`: coefficient for first gene in double perturbation
    * `coef_2`: coefficient for second gene in double perturbation
    * `lfc`: signed log-foldchange of coefficients
    * `r2`: r-squared of model fit
    otherwise the DataFrame and the model prediction as a pandas Series, index by gene names.
    """
    # Recover the individual genes of the perturbation
    gene_1, gene_2 = perturbation.split("+")

    # Define the data matrix and response variable
    X = pseudobulk[[gene_1, gene_2]].values
    y = pseudobulk[perturbation].values

    # Define and fit the linear regression
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Extract the data of the fit
    coef_1, coef_2 = model.coef_
    lfc = np.sign(coef_1) * np.sign(coef_2) * np.log(np.abs(coef_1) / np.abs(coef_2))
    r_squared = model.score(X, y)
    lm_result = pd.DataFrame(
        [[coef_1, coef_2, lfc, r_squared]], index=[perturbation], columns=["coef_1", "coef_2", "lfc", "r2"]
    )

    if not return_pred:
        return lm_result

    # Compute prediction
    lm_prediction = pd.Series(model.predict(X), index=pseudobulk.index)
    return lm_result, lm_prediction


# %%
lm_res = linear_interaction_model(pseudobulk=adata.varm["zscore_pseudobulk"], perturbation="CBL+CNN1")
lm_res


# %% [markdown]
# We want a possibility to visually assess how individual single perturbation compare to their corresponding double perturbation and the fitted model.
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.3</b>
#   </div>
#   Define a function <code>plot_linear_interaction_model</code> that takes the pseudobulked data, the result and prediction of the linear model fit and the number of genes to visualize as input (keywords <code>pseudobulk</code>, <code>lm_res</code>, <code>lm_pred</code>, and <code>n_genes</code>) and visualizes the pseudobulked deviations from baseline for the double perturbation, its individual genes, and the predicted deviation for a double perturbation based on the linear model. Proceed as follows:
#   <ol type="a">
#     <li>Extract the pseudobulked z-scores for the double perturbation and individual genes to define a DataFrame indexed by the first and second gene, the double perturbation itself, and the prediction of the linear model.</li>
#     <li>Subset this DataFrame to the <code>n_genes</code> most variable genes across the three perturbations.</li>
#     <li>Visualize the data frame with seaborn's <code>clustermap</code> function (keywords <code>cmap="coolwarm"</code>, <code>center=0</code>, <code>row_cluster=False</code>, <code>robust=True</code>).</li>
#   </ol>
# </div>


# %%
# Implement your solution here
def plot_linear_interaction_model(
    pseudobulk: pd.DataFrame, lm_res: pd.DataFrame, lm_pred: pd.Series, n_genes: int
) -> None:
    """Plot the clustermap of the z-scored pseudobulks for a double perturbation, its definint genes, and the linear model fit.

    Parameters
    ----------
    pseudobulk
        Pseudobulked z-scored, indexed by gene name and with perturbations as column names.
    lm_res
        Result of linear model fit.
    lm_pred
        Prediction of linear model fit.
    n_genes
        Number of highly variable genes to show.

    Returns
    -------
    Nothing, only plots the data.
    """
    # Extract the double pertrubation and its genes
    perturbation = lm_res.index[0]
    gene_1, gene_2 = perturbation.split("+")

    # Define data frame of pseudobulk z-scores
    data = pseudobulk[[gene_1, gene_2, perturbation]].T.copy()
    lm_pred = lm_pred.to_frame().T.copy()
    lm_pred.index = pd.Index([f"Linear model (lfc={lm_res.loc[perturbation, 'lfc']:.4f})"])
    data = pd.concat([data, lm_pred])

    # Subset to n_genes highly variable genes
    top_genes = data.var(axis=0).nlargest(n_genes).index
    data = data[top_genes]

    # Plot data
    sns.clustermap(data, cmap="coolwarm", center=0, row_cluster=False, robust=True, figsize=(12, 4))


# %% [markdown]
# ### Synergystic effect
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.4</b>
#   </div>
#   Use the previously defined functions to assess the <i>CBL+CNN1</i> perturbation (use <code>n_gene=100</code>) - an example of a synergystic effect.
# </div>

# %%
# Implement your solution here
lm_res, lm_pred = linear_interaction_model(
    pseudobulk=adata.varm["zscore_pseudobulk"], perturbation="CBL+CNN1", return_pred=True
)
plot_linear_interaction_model(pseudobulk=adata.varm["zscore_pseudobulk"], lm_res=lm_res, lm_pred=lm_pred, n_genes=100)
plt.show()

# %% [markdown]
# ### Double pertrubation with a dominant gene
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.5</b>
#   </div>
#   Use the previously defined functions to assess the <i>CEBPA+KLF1</i> perturbation (use <code>n_gene=100</code>) - an example for an effect with a dominant gene.
# </div>
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 4.2</b>
#   </div>
#   Which gene is the dominant one in the <i>CEBPA+KLF1</i> perturbation?
#   <blockquote>CEBPA</blockquote>
# </div>

# %%
# Implement your solution here
lm_res, lm_pred = linear_interaction_model(
    pseudobulk=adata.varm["zscore_pseudobulk"], perturbation="CEBPA+KLF1", return_pred=True
)
plot_linear_interaction_model(pseudobulk=adata.varm["zscore_pseudobulk"], lm_res=lm_res, lm_pred=lm_pred, n_genes=100)
plt.show()

# %% [markdown]
# ### Non-linear effect
#
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.5</b>
#   </div>
#   Use the previously defined functions to assess the <i>PLK4+STIL</i> perturbation (use <code>n_gene=100</code>) - an example for a non-linear effect.
# </div>

# %%
# Implement your solution here
lm_res, lm_pred = linear_interaction_model(
    pseudobulk=adata.varm["zscore_pseudobulk"], perturbation="PLK4+STIL", return_pred=True
)
plot_linear_interaction_model(pseudobulk=adata.varm["zscore_pseudobulk"], lm_res=lm_res, lm_pred=lm_pred, n_genes=100)
plt.show()

# %% [markdown]
# ### Systematic assessment of all double perturbations

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(83, 83, 83); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Exercise 4.5</b>
#   </div>
#   Use the <code>linear_interaction_model</code> function to collect the linear model results for all double perturbations in a single pandas DataFrame, indexed by the double perturbation.
# </div>

# %%
# Implement your solution here
double_perturbations = adata.varm["zscore_pseudobulk"].columns
double_perturbations = double_perturbations[double_perturbations.str.contains("+", regex=False)]

lm_res = pd.concat(
    [
        linear_interaction_model(pseudobulk=adata.varm["zscore_pseudobulk"], perturbation=perturbation)
        for perturbation in double_perturbations
    ]
)
lm_res.head()

# %%
fig, ax = plt.subplots(figsize=(12, 12))

cmap = sns.color_palette("viridis", as_cmap=True)
vmin, vmax = lm_res["r2"].min(), lm_res["r2"].max()
norm = Normalize(vmin=vmin, vmax=vmax)

sns.scatterplot(
    data=lm_res, x="coef_1", y="coef_2", hue="r2", palette=cmap, norm=norm, legend=False, s=100, alpha=0.8, ax=ax
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="R-squared")

max_val = lm_res[["coef_1", "coef_2"]].max().max() + 0.5
min_val = lm_res[["coef_1", "coef_2"]].min().min() - 0.5
ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="coef1 = coef2")

ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

for _, row in lm_res.iterrows():
    ax.annotate(row.name, (row["coef_1"], row["coef_2"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

ax.set(aspect="equal")
plt.show()

# %% [markdown]
# <div style="padding: 10px; border-radius: 1px; width: 98%">
#   <div style="background-color: rgb(114, 134, 86); color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: -10px -10px 10px -10px">
#     <b>Question 4.3</b>
#   </div>
#   <ol type="a">
#     <li>Which double perturbation is best explained by the linear model?</li>
#     <blockquote>TBX2+TBX3</blockquote>
#     <li>Which double perturbation appears to be most non-linear?</li>
#     <blockquote>BAK1+BCL2L11</blockquote>
#     <li>Which double perturbation is the most synergystic, amongst those with R-squared > 0.6?</li>
#     <blockquote>FOXA1+KLF1</blockquote>
#     <li>Which gene is the most dominant, amongst double perturbations with R-squared > 0.6?</li>
#     <blockquote>PRTG</blockquote>
#   </ol>
# </div>

# %%
# Implement your solution here
print(f"Double perturbation explained best by linear model: {lm_res['r2'].idxmax()} (R2={lm_res['r2'].max()})")
print(f"Double perturbation explained the least by linear model: {lm_res['r2'].idxmin()} (R2={lm_res['r2'].min()})")

row_mask = lm_res["r2"] > 0.6
print(f"Most synergystic double perturbation with R2>0.6: {lm_res.loc[row_mask, 'lfc'].abs().idxmin()}")

perturbation = lm_res.loc[lm_res["r2"] > 0.6, "lfc"].abs().idxmax()
if lm_res.loc[perturbation, "lfc"] > 0:
    most_dominant_gene = perturbation.split("+")[0]
else:
    most_dominant_gene = perturbation.split("+")[1]
print(f"Most dominant gene amongst double perturbation with R2>0.6: {most_dominant_gene}")
