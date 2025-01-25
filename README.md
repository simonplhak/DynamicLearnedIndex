# Dynamic Learned Index

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/Coda-Research-Group/DynamicLearnedIndex/actions/workflows/ci.yml/badge.svg)](https://github.com/Coda-Research-Group/DynamicLearnedIndex/actions/workflows/ci.yml)

## Installation

This project uses [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage the environment.

```shell
# Setup the environment and install the dependencies
conda env create --file environment.yml
conda activate DynamicLearnedIndex
pip install -e .

# Development only
pre-commit install # Install the pre-commit hooks to check for code style problems
```

## Running Experiments

```shell
# Download the datasets
cd data
wget 'https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
wget 'http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
cd ..

# Run the experiments
python3 experiments/run.py --compaction-strategy leveling
```

## Notes

Implementation details:
- The LAION dataset is converted from `float16` to `float32` to take advantage of the `float32` arithmetic capabilities of existing CPUs, otherwise a significant performance loss will be observed.

Development tips:
- During development, you can use the following command `pre-commit run --all-files` to run the pre-commit hooks. It will run the `ruff` linter and formatter to check for code style problems.

## TODO: Folder Structure

- `data`: Contains the datasets used in the local experiments.

## Resources

- https://github.com/psu-db/dynamic-extension
- https://cstheory.stackexchange.com/questions/7642/i-dreamt-of-a-data-structure-does-it-exist
- https://cstheory.stackexchange.com/questions/17328/optimal-insertion-times-in-insertion-only-data-structures-beyond-bentley-saxe
