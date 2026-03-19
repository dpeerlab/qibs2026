# QUIBS 2026 - Module 6 workshop

## Installation

```bash
conda create -n quibs-py312 python=3.12 --yes && conda activate quibs-py312
pip install -e ".[dev,jupyter]"
pre-commit install

python -m ipykernel install --user --name quibs-py312 --display-name "quibs-py312"
```
