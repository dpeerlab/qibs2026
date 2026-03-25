# QIBS 2026 - Module 6 workshop

Welcome to the QIBS 2026 module 6 labs! In our four sessions together, you will gain hands-on experience in analyzing
single-cell sequencing data. The dates for each lab are

-   **Lab 1**: Friday, March 25, 2026, 12-2pm ET
-   **Lab 2**: Friday, April 3, 2026, 12-2pm ET
-   **Lab 3**: Thursday, April 23, 2026, 9-11am ET
-   **Lab 4**: Friday, April 24, 2026, 12-2pm ET

In this README, I will provide the necessary details to complete each lab, so please **always pull the latest version of
the repository at the beginning of our sessions!**

## Setup

I am assuming the local SSH config includes aliases to ssh into the Cayuga login node and compute nodes, i.e.,

```
Host cayuga-login
  Hostname cayuga-login1.cac.cornell.edu
  IdentityFile path/to/cayuga/ssh/key
  User CWID
  RequestTTY force
  ServerAliveInterval 60
  TCPKeepAlive yes

# Update HostName with assigned hostname after job submission
Host cayuga-compute
  HostName c0010
  ProxyJump cayuga-login
  IdentityFile path/to/cayuga/ssh/key
  User CWID
```

where _CWID_ is your user id; please adjust any following commands if your setup differs.

### General

Please setup this repository as follows

```bash
ssh cayuga-login
cd /athena/qibs_class/scratch/CWID/optional/additional/path
git clone https://github.com/dpeerlab/qibs2026.git
```

All provided commands assume that the qibs2026 repository is at the base of your scratch directory; i.e., make sure the
adjust the commands if your setup differs.

### Job submission

```bash
# ssh into cayuga login node
ssh cayuga-login
# start a screen session
screen -S qibs2026_module_6
# start an interactive session
srun -n1 --pty --partition=qibs_class --mem=16G --time=02:00:00 bash -i
# hostname to update HostName in ssh config
hostname
```

Following, you can connect to the Cayuga compute node via the IDE of your choice, preferably [VS Code](https://code.visualstudio.com/)
for our sessions.

### Jupyter notebook setup

On the compute node, you can activate the conda environment that I am providing with

```bash
conda activate /athena/cayuga_0083/scratch/pfw4001/envs/qibs2026-py312
```

When you use the environment for the first time, you may have to run

```bash
conda activate /athena/cayuga_0083/scratch/pfw4001/envs/qibs2026-py312
python -m ipykernel install --user --name qibs2026-py312
```

in order to use the kernel for your Jupyter notebooks.

### QIBS2026 repository details

For our purporse, the important parts of this repository are

-   `data/`: this directory will contain the data we are using
-   `notebooks/`: you will find the notebooks for each session here

In combination with the provided conda environment, the Python package corresponding to this repo _qibs2026_ provides
easy access the data directory in Python from anywhere via the `DATA_DIR` variable

```python
from qibs2026 import DATA_DIR
```

## Lab instructions

### Lab 1

In our first lab, we will implement a standard scRNA-seq analysis pipeline. The corresponding Jupyter notebook includes
questions for which you will provide the answers to using the worksheet you have received.

-   **Data**: copy the data for this lab into the data directory

```bash
cp -r /athena/qibs_class/scratch/pfw4001/data/lab_1 /athena/qibs_class/scratch/CWID/qibs2026/data/
```

-   **Notebook**: `notebooks/lab_1/sc_basics.ipynb`
-   **Deadline for worksheet**: April 2 at 12pm ET

Please email the worksheet to weilerp@mskcc.org, using the file format `worksheet_1_LASTNAME.xlsx`

## Installation

There is no need to install anything since we will use an existing conda environment. However, these are the commands I
used to set it up

```bash
conda create -n qibs-py312 python=3.12 --yes && conda activate qibs-py312
pip install -e ".[dev,jupyter]"
pre-commit install

python -m ipykernel install --user --name qibs-py312 --display-name "qibs-py312"
```
