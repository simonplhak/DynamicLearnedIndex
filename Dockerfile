# FROM ubuntu:22.04 AS builder
FROM rust:latest AS builder
LABEL stage=builder-cpu

# todo remove
RUN cargo

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    python3 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip \
    && unzip libtorch.zip -d libtorch \
    && rm libtorch.zip
ENV LIBTORCH=/opt/libtorch/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/libtorch/lib

ENV SIMD_REGISTRY_BITS=256
ENV BUCKET_SCALING_FACTOR=1.0

WORKDIR /app


# COPY Cargo.* cli_dynamic_learned_index/Cargo.* dynamic_learned_index/Cargo.* rust-toolchain.toml ./
# RUN mkdir -p dynamic_learned_index/src && echo "fn main() {println!(\"Hello, world!\");}" > dynamic_learned_index/src/lib.rs
# RUN mkdir -p cli_dynamic_learned_index/src && echo "fn main() {println!(\"Hello, world!\");}" > cli_dynamic_learned_index/src/main.rs

# RUN cargo build --release

COPY . .

# RUN python3 -m venv env \
#     && . env/bin/activate  \
#     && pip install -r requirements.txt


# RUN . env/bin/activate \
#     && cd py_dynamic_learned_index \
#     && maturin develop

RUN cargo build --release --workspace --exclude py_dynamic_learned_index

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    python3 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/libtorch /opt/libtorch
ENV LIBTORCH=/opt/libtorch/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/libtorch/lib

COPY --from=builder /app/target/release/cli_dynamic_learned_index /app


ENTRYPOINT ["/app/cli_dynamic_learned_index"]
