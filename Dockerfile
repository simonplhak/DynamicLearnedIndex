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
    && rm -rf /var/lib/apt/lists/*



ENV SIMD_REGISTRY_BITS=256
ENV BUCKET_SCALING_FACTOR=1.0

WORKDIR /app


COPY Cargo.* rust-toolchain.toml ./
COPY cli_dynamic_learned_index/Cargo.* cli_dynamic_learned_index/
COPY dynamic_learned_index/Cargo.*  dynamic_learned_index/
COPY measure_time_macro/Cargo.* measure_time_macro/
COPY py_dynamic_learned_index/Cargo.* py_dynamic_learned_index/
RUN mkdir -p dynamic_learned_index/src && echo "fn main() {println!(\"Hello, world!\");}" > dynamic_learned_index/src/lib.rs
RUN mkdir -p cli_dynamic_learned_index/src && echo "fn main() {println!(\"Hello, world!\");}" > cli_dynamic_learned_index/src/main.rs
RUN mkdir -p measure_time_macro/src && echo "fn main() {println!(\"Hello, world!\");}" > measure_time_macro/src/lib.rs
RUN mkdir -p py_dynamic_learned_index/src && echo "fn main() {println!(\"Hello, world!\");}" > py_dynamic_learned_index/src/lib.rs

RUN cargo build --release --workspace --exclude py_dynamic_learned_index
RUN cargo clean -p cli_dynamic_learned_index -p dynamic_learned_index -p measure_time_macro -r

RUN rm dynamic_learned_index/src/lib.rs cli_dynamic_learned_index/src/main.rs measure_time_macro/src/lib.rs

COPY ./cli_dynamic_learned_index/src ./cli_dynamic_learned_index/src
COPY ./dynamic_learned_index/src ./dynamic_learned_index/src
COPY ./measure_time_macro/src ./measure_time_macro/src

RUN cargo build --release --workspace --exclude py_dynamic_learned_index

FROM rust:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libssl3 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Copy HDF5 libraries from builder to ensure version compatibility
COPY --from=builder /usr/lib/x86_64-linux-gnu/libhdf5*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsz.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libaec.so* /usr/lib/x86_64-linux-gnu/


COPY --from=builder /app/target/release/cli_dynamic_learned_index /app


ENTRYPOINT ["/app/cli_dynamic_learned_index"]
