# Dynamic Learned Index implementation in Rust

## Dev

### Run

```shell
cargo run -p cli_dynamic_learned_index
```

### Setup

#### SIMD

Thi library uses SIMD instructions for performance. It needs to know at a compile time the number of bits in SIMD register that CPU supports. To specify the number of bits go to [`dynamic_learned_index/src/constants.rs`](dynamic_learned_index/src/constants.rs) and change `SIMD_REGISTER_SIZE` constant (do not change any other constants). To find out how many bits in SIMD register your CPU support visit manufacturer webpage (in Linux you can find your CPU model via command `cat /proc/cpuinfo | grep -i 'model name'`). 

#### Libtorch

Crate depends on `tch-rs` dependency that serves as a wrapper for `libtorch` c++ implementation. 
Follow the installation instructions for `libtorch` from [tch-rs homepage](https://github.com/LaurentMazare/tch-rs).

When using libtorch from pip installation, you need to call cargo build within the environment where the torch package is installed.

#### Python

Setup python environment

```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cd py_dynamic_learned_index
maturin develop
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```

Update dynamic_learned_index dependency

```shell
cd py_dynamic_learned_index
maturin develop
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```