# py-dynamic-learned-index

Python bindings for **Dynamic Learned Index** - a high-performance approximate nearest neighbor search library that uses learned neural network models to efficiently index and query unstructured data.

## Overview

The Dynamic Learned Index uses a multi-level hierarchical structure where each level contains a neural network model trained to predict the most likely bucket containing the requested data. This learned approach significantly improves search performance compared to traditional indexing methods.

## Installation

Install from PyPI:

```bash
pip install py-dynamic-learned-index
```

## Quick Start

```python
from py_dynamic_learned_index import Index

# Create an index
index = Index(
    input_shape=768,           # Dimension of your vectors
    buffer_size=5000,          # Write buffer size
    bucket_size=5000,          # Max records per bucket
    arity=3,                   # Fanout for compaction
    device="cpu"               # Use "cpu" or "cuda"
)

# Insert vectors with IDs
vector = [0.1, 0.2, 0.3, ...]  # 768-dimensional vector
index.insert(vector, id=1)

# Search for k nearest neighbors
query = [0.15, 0.25, 0.35, ...]
results = index.search(query, k=10)  # Returns top-10 nearest neighbors

# Delete an entry
index.delete(id=1)
```

## Configuration

The index supports various configuration options to optimize for your use case:

- **input_shape**: Vector dimension (must match your data)
- **buffer_size**: Size of write buffer before flushing to index
- **bucket_size**: Maximum records per bucket before splitting
- **arity**: Fanout for tree compaction
- **device**: Compute device ("cpu" or "cuda")
- **distance_fn**: Distance metric ("dot" for cosine similarity, "l2" for euclidean)

For more details on configuration, see the [main project README](https://github.com/simonplhak/DynamicLearnedIndex).

## Features

- **Learned Indexing**: Uses neural networks to guide search, not just data distribution
- **Dynamic**: Supports online insertions and deletions
- **Multi-level**: Hierarchical structure for better scalability
- **Fast**: Optimized Rust implementation with Python bindings
- **Flexible**: Configurable models, distance functions, and compaction strategies

## Examples

Full example with data loading and querying:

```python
import numpy as np
from py_dynamic_learned_index import Index

# Create index for 768-dimensional vectors (e.g., embeddings)
index = Index(input_shape=768, buffer_size=5000, bucket_size=5000)

# Generate sample data
n_vectors = 100000
vectors = np.random.randn(n_vectors, 768).astype(np.float32)

# Insert vectors
for i, vector in enumerate(vectors):
    index.insert(vector, id=i)

# Search
query = vectors[0]  # Use first vector as query
neighbors = index.search(query, k=10)
print(f"Top 10 neighbors: {neighbors}")
```

For more examples, check the [example.py](example.py) file in the repository.

## Performance

The Dynamic Learned Index is optimized for:
- High-dimensional vector search (e.g., embeddings from language/vision models)
- Large-scale datasets (millions of vectors)
- Both batch and online query scenarios

Performance depends on your data distribution, vector dimensionality, and hardware configuration.

## Heavy Users: Building from Source

### Simple Development Build

For a quick development build:

```bash
git clone https://github.com/plhis/DynamicLearnedIndex.git
cd DynamicLearnedIndex/py_dynamic_learned_index
pip install maturin
maturin develop --release
```

### Sdist Installation with Custom Features

For advanced users building from source distribution with specific features:

```bash
# Set up environment
export CARGO_TARGET_DIR=$SCRATCHDIR/cargo-target
mkdir -p $CARGO_TARGET_DIR

# Install Rust (if not already installed)
if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Install and set Rust nightly
rustup install nightly
rustup default nightly

# Install build dependencies
pip install maturin meson-python meson ninja cython

# Set compiler flags (optional)
export CC=gcc
export CXX=g++
export UV_LINK_MODE=copy

# Build with custom features (e.g., measure_time, mix)
export MATURIN_BUILD_ARGS="--features measure_time,mix"

# Install from source distribution without binary wheels
pip install --force-reinstall --no-binary :all: py-dynamic-learned-index --no-build-isolation

# Verify installation
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"
```

**Note**: Set `MATURIN_BUILD_ARGS` with the features you need (e.g., `measure_time`, `mix`, `tch`, `candle`, `mkl`).

## Requirements

- Python >= 3.9
- For GPU support (cuda feature): NVIDIA GPU and CUDA toolkit

## Documentation

For detailed configuration and usage documentation, see the [main project repository](https://github.com/simonplhak/DynamicLearnedIndex).

## License

Licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later).

See the [LICENSE](../LICENSE) file in the main repository for full license text.

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.
