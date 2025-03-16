# Dynamic Learned Index implementation in Rust

## Dev

### Run

```shell
cargo run -p cli_dynamic_learned_index
```

### Setup

#### Libtorch

Crate depends on `tch-rs` dependency that serves as a wrappet to torch c++ implementation. 
Follow the intallation from [tch-rs homepage](https://github.com/LaurentMazare/tch-rs).
If you use python environment from `py_dynamic_learned_index` directory, the libtorch is installed in the virtual environment.
To link it with `tch-rs` you need to set `LIBTORCH_USE_PYTORCH=1` and `LD_LIBRARY_PATH=py_dynamic_learned_index/.venv/lib/python3.12/site-packages/torch/lib` (this path might be different if you decided to use different libtorch installation).
Repository provides a default setting for VScode in `.vscode/settings.json`, so hopefully if you use VScode, the environment will work.

When using libtorch from pip installation, you need to call cargo build within the environment where the torch package is installed.

#### Python

Setup python environment

```shell
cd py_dynamic_learned_index  # environment must be created here to safisfy maturin build system
uv venv
source .venv/bin/activate
uv sync
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```

Update dynamic_learned_index dependency

```shell
cd py_dynamic_learned_index
maturin develop
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```