# Start from the official PyPA Manylinux 2014 base
FROM quay.io/pypa/manylinux_2_28_x86_64

# Install Rust with the specific nightly toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-12-11
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy rust-toolchain.toml for cargo to automatically select the right toolchain
COPY rust-toolchain.toml /

# Install HDF5 and build dependencies using manylinux's package manager (dnf)
RUN dnf install -y \
    pkg-config \
    hdf5-devel \
    gcc \
    gcc-c++ \
    cmake \
    wget \
    unzip \
    ca-certificates \
    && dnf clean all

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set up the Python environment using Manylinux's built-in Python 3.12
# (Located at /opt/python/cp312-cp312/bin/python)
RUN uv venv /venv --python /opt/python/cp312-cp312/bin/python
ENV PATH="/venv/bin:${PATH}"

# Pre-install PyTorch and Maturin so they are cached in the image
RUN uv pip install torch==2.10.0 maturin

# Set the magic environment variable permanently for this container
ENV LIBTORCH_USE_PYTORCH=1

# Copy Cargo workspace files first for dependency caching
COPY Cargo.toml /
COPY Cargo.lock /
COPY cli_dynamic_learned_index/ /cli_dynamic_learned_index/
COPY dynamic_learned_index/ /dynamic_learned_index/
COPY measure_time_macro/ /measure_time_macro/
COPY py_dynamic_learned_index/ /py_dynamic_learned_index/
COPY service/ /service/

ENV SIMD_REGISTRY_BITS=256
ENV BUCKET_SCALING_FACTOR=1.0

# Build dependencies to cache them (this layer will be cached unless Cargo files change)
RUN cargo build --release

# Set the working directory to where we will mount the code
WORKDIR /io