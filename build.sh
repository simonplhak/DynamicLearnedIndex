#!/bin/bash
# Stop on any error
set -e

# Host user IDs for fixing permissions later
HOST_UID=$(id -u)
HOST_GID=$(id -g)

echo "📦 Ensuring the builder image is up to date..."
# Docker will use its cache for this unless you change the Dockerfile
docker build -t tch-manylinux-builder -f Dockerfile.builder .

echo "⚙️ Compiling the wheel..."
# Run the container, mount the code, and compile
docker run --rm -v "$(pwd):/io" tch-manylinux-builder bash -c "
    # We are already inside the activated virtual environment with torch installed
    cd /io/py_dynamic_learned_index
    export LIBTORCH_USE_PYTORCH=1
    export RAYON_NUM_THREADS=8
    # Build the wheel with maturin, skipping auditwheel since we're in a compatible environment
    maturin build --release \
        --compatibility manylinux_2_28 \
        --features measure_time,tch \
        --skip-auditwheel \
        --out /io/py_dynamic_learned_index/wheels
    
    echo '🔐 Fixing file permissions...'
    # 1. Fix the wheels directory we just created
    chown -R ${HOST_UID}:${HOST_GID} /io/py_dynamic_learned_index/wheels
    
    # 2. Safely fix Cargo's target directory, whether it's in the workspace root or the subfolder
    if [ -d '/io/target' ]; then
        chown -R ${HOST_UID}:${HOST_GID} /io/target
    fi
    if [ -d '/io/py_dynamic_learned_index/target' ]; then
        chown -R ${HOST_UID}:${HOST_GID} /io/py_dynamic_learned_index/target
    fi
"

echo "✅ Build complete! Your universal wheel is waiting in:"
ls -lh target/wheels/