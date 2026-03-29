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

## Build

If you want to build the `cli_dynamic_learned_index` workspace, you need to have hdf5 library installed on your system. You can install it via your package manager.

### Feature flags

**dynamic_learned_index** crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `tch`: enables the use of `tch-rs` library for model training
  - You need to have `libtorch` installed on your system to use this feature. Follow the installation instructions for `libtorch` from [tch-rs homepage](https://github.com/LaurentMazare/tch-rs).
- `candle`: enabled by default, enables the use of `candle` library for model training
- `mkl`: enables the use of `mkl` library for linear algebra operations, can be used with `candle` feature to speed up training on CPU

**cli_dynamic_learned_index** crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `kmeans`: just for running benchmarks for kmeans implementation, not used in the main codebase
- `kentro`: just for running benchmarks for kentro implementation, not used in the main codebase

**py_dynamic_learned_index* crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `mkl`: enables the use of `mkl` library for linear algebra operations, can be used with `candle` feature to speed up training on CPU
- `tch`: enables the use of `tch-rs` library for model training, can be used with `candle` feature to speed up training on CPU
- `candle`: enables the use of `candle` library for model training, can be used with `mkl` or `tch` features to speed up training on CPU

### Linking with Python

To link the Rust library with Python, we use `maturin` to build a Python package. This allows us to use the Rust code as a Python module.

Setup python environment

```shell
cd py_dynamic_learned_index
uv sync
uv run maturin develop --release
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```

Then you can install the package via pip:

```shell
pip install ./target/wheels/py_dynamic_learned_index*.whl
```

Building the wheel in release mode:

```shell
cd py_dynamic_learned_index
uv sync
uv run maturin build --release
uv pip install ./target/wheels/py_dynamic_learned_index*.whl
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
  bucket_size: 5000
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

## Rust API

The `dynamic_learned_index` crate provides a public API for building and querying indices programmatically from Rust code.

```rust
use dynamic_learned_index::{Index, IndexBuilder, IndexConfig, SearchParams, SearchStrategy};

// Build an index with f32 float type
let mut index = IndexBuilder::<f32>::default()
    .input_shape(768)
    .buffer_size(5000)
    .bucket_size(5000)
    .arity(3)
    .build()?;

// Insert data
index.insert(vec![1.0, 2.0, 3.0], 0)?;

// Search for k nearest neighbors
let query = vec![1.0, 2.0, 3.0];
let results = index.search(&query, 10)?;
index.delete(0)?;
```

## Python API

Example can be found in [`py_dynamic_learned_index/example.py`](py_dynamic_learned_index/example.py) directory.

## Docker

To run the project in a Docker container.

```shell
docker build -t dli-cli --exclude py_dynamic_learned_index .
docker run -it --rm -v ${PWD}/experiments_data:/app/experiments_data -v ${PWD}/data:/app/data -v ${PWD}/configs:/app/configs dli-cli
```

## Profiling

Recommended approach to optimize code is to go through cycle of measuring, analyzing and optimizing.

### Measurment

The measurment focuses on measuring search time. There are two measurements:

- for 3 individual queries (good for debugging)
- for the whole query set (good for overall performance measurment)

```shell
# ensure that you have built the latest release version
cargo build --release

# we need to build the index and serialize it
./target/release/cli_dynamic_learned_index experiment profiler  data/k300/ \
  -i configs/example.yaml \
  --skip-validation \
  --start-from-one \
  --force \
  --skip-search \
  --output-dir index_dump

# run benchmark and save as a baseline
cargo bench -p cli_dynamic_learned_index --bench index_benchmarks -- --warm-up-time 5 --measurement-time 15 --save-baseline base

# run benchmark and compare with the baseline
cargo bench -p cli_dynamic_learned_index --bench index_benchmarks -- --warm-up-time 5 --measurement-time 15 --baseline base 
```

Results are in `target/criterion/report/index.html` file.

```shell
# open the report
open target/criterion/report/index.html
```

### Analyzing

```shell
# ensure that you have built the latest release version
cargo build --release

# we need to build the index and serialize it
./target/release/cli_dynamic_learned_index experiment profiler  data/k300/ \
  -i configs/example.yaml \
  --skip-validation \
  --start-from-one \
  --log-output stdout \
  --force \
  --skip-search \
  --output-dir index_dump

# run profiler
# profiler will load the index from `index_dump` directory
# starts benchmark
# searches 100 queries and generate flamegraph
# ends benchmark
cargo run -p cli_dynamic_learned_index --example profiler
```

The flamegraph is stored in `flamegraph.svg` file. You can open it in a web browser.

```shell
# open the flamegraph 
google-chrome flamegraph.svg
```

### Benchmarks for individual implementations

To run benchmarks, use the following command:

```shell
cargo bench -p cli_dynamic_learned_index
```

In basic case there is just candle model benchmark. To run tch model benchmark, enable `tch` feature:

```shell
# model benchmarks are in dynamic_learned_index due to the visibility
cargo bench -p dynamic_learned_index --features tch
```
