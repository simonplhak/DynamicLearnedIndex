FROM rust:latest AS builder
LABEL stage=builder-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY Cargo.* rust-toolchain.toml ./
COPY ./cli_dynamic_learned_index/ ./cli_dynamic_learned_index/
COPY ./dynamic_learned_index/ ./dynamic_learned_index/
COPY ./py_dynamic_learned_index/ ./py_dynamic_learned_index/
COPY ./measure_time_macro/ ./measure_time_macro/


RUN cargo build \
    --release \
    --workspace \
    --exclude py_dynamic_learned_index \
    --no-default-features \
    --features candle


ENTRYPOINT ["/app/target/release/cli_dynamic_learned_index"]
