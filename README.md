# Dynamic Learned Index implementation in Rust

## Dev

### Setup

#### Python

Setup python environment

```shell
cd py_dynamic_learned_index
conda env create
conda activate DynamicLearnedIndexRust
uv install
```

Build package

```shell
cd py_dynamic_learned_index
conda activate DynamicLearnedIndexRust
maturin develop
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```