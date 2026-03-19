# QIBS 2026 - Module 6 workshop

## Installation

```bash
conda create -n qibs-py312 python=3.12 --yes && conda activate qibs-py312
pip install -e ".[dev,jupyter]"
pre-commit install

python -m ipykernel install --user --name qibs-py312 --display-name "qibs-py312"
```
