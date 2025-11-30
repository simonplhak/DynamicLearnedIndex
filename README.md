# Dynamic Learned Index implementation in Rust

![High level overview of the index](images/highlevel.excalidraw.png "High level overview of the index")

## Dev

You can find some useful commands in vscode tasks.

The whole project is written in Rust and uses `cargo` as a build system. The project is divided into several crates:
- `dynamic_learned_index`: main library crate that implements the dynamic learned index
- `cli_dynamic_learned_index`: CLI crate that provides a command line interface to run experiments and build the index
- `py_dynamic_learned_index`: Python crate that provides a Python interface to the dynamic learned index

### Run

```shell
cargo run -p cli_dynamic_learned_index
```

### Profiling

Add these lines to [`Cargo.toml`](Cargo.toml) to enable profiling:

```toml
[build]
rustflags = ["-C", "force-frame-pointers=yes"]
```

Then you can run the profiler with the following command:

```
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --bin=cli_dynamic_learned_index -- experiment 20250713.profiler  data/k300/ --skip-validation --start-from-one --force -n 10 --limit 50000
```

## Build

Thi library uses SIMD instructions for performance. It needs to know at a compile time the number of bits in SIMD register that CPU supports. To specify the number of bits go to [`dynamic_learned_index/src/constants.rs`](dynamic_learned_index/src/constants.rs) and change `SIMD_REGISTER_SIZE` constant (do not change any other constants). To find out how many bits in SIMD register your CPU support visit manufacturer webpage (in Linux you can find your CPU model via command `cat /proc/cpuinfo | grep -i 'model name'`). 


### Linking with Python

To link the Rust library with Python, we use `maturin` to build a Python package. This allows us to use the Rust code as a Python module.

Setup python environment

```shell
cd py_dynamic_learned_index
uv sync
maturin develop --release
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```

Update dynamic_learned_index dependency

```shell
cd py_dynamic_learned_index
maturin develop --release
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```


## Running experiments via CLI

### Dataset

```yaml
dataset:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
queries:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
ground_truth:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
```

`<dataset_name>`: name of the underlining dataset under h5

`<type>`: type of the dataset, can be `h5`

`<path>`: path to the dataset file relative from dataset config directory.

Default values can be found via `cli_dynamic_learned_index defaults dataset` command.

### Index

Example config:

```yaml
compaction_strategy:
  type: bentley_saxe  # curently only possible value
  rebuild_strategy: no_rebuild # currently only possible value
levels:
  0:
    model:
      layers:
      - type: linear
        value: 256
      - type: relu
      ...  # add more layers if needed
      # be aware that always index adds an output layer
      train_params:
        threshold_samples: 1000
        batch_size: 8
        epochs: 3
        max_iters: 10
      retrain_params:
        threshold_samples: 1000
        batch_size: 8
        epochs: 3
        max_iters: 10
    bucket_size: 5000
  5:
    ...
buffer_size: 5000
input_shape: 768
arity: 3
device: cpu  # only cpu supported currently
distance_fn: dot  # can be dot or l2
delete_method: oid_to_bucket  # only one method currently
```

`compaction_strategy`: compaction strategy, can be `bentley_saxe`

`levels`: levels of the index, specification for each level are taken from the previous level until the level index matches the level in the config. The first level is always 0.

`layers`: layers of the model, can be `linear`, `relu`

`train_params`: training parameters for the model

Default values can be found via `cli_dynamic_learned_index defaults dataset` command.

### Running experiments

To run experiments via CLI, you can use the `cli_dynamic_learned_index` binary. Example:

```shell
./target/release/cli_dynamic_learned_index experiment test_build data/k300 --force
```


## Python API

Example can be found in [`py_dynamic_learned_index/example.py`](py_dynamic_learned_index/example.py) directory.


## Docker

To run the project in a Docker container.

```shell
docker build -t dli-cli --exclude py_dynamic_learned_index .
docker run -it --rm -v ${PWD}/experiments_data:/app/experiments_data -v ${PWD}/data:/app/data -v ${PWD}/configs:/app/configs dli-cli
```


## Benchmarks

To run benchmarks, use the following command:

```shell
cargo bench -p dynamic_learned_index
```

In basic case there is just candle model benchmark. To run tch model benchmark, enable `tch` feature:

```shell
cargo bench -p dynamic_learned_index --features tch
```

### Libtorch installation

Crate depends on `tch-rs` dependency that serves as a wrapper for `libtorch` c++ implementation. 
Follow the installation instructions for `libtorch` from [tch-rs homepage](https://github.com/LaurentMazare/tch-rs).

When using libtorch from pip installation, you need to call cargo build within the environment where the torch package is installed.

```shell
# build is stored in `./target/release`
# entrypoint is in `./target/release/cli_dynamic_learned_index` binary
cargo build --release
```