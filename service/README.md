# Dynamic Learned Index Service

A web service that exposes the Dynamic Learned Index through a REST API using Axum and Tokio.

## Building

From the workspace root:

```bash
cargo build --release -p service
```

Or run directly:

```bash
cargo run -p service
```

## Running

```bash
cargo run -p service
```

The service will start on `http://127.0.0.1:3000`

## API Endpoints

### Health Check

```bash
GET /health
```

Returns: `OK`

### Search

Search for similar vectors in the index.

```bash
POST /search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10
}
```

**Parameters:**

- `vector`: Array of floats representing the query vector (required)
- `k`: Number of results to return (optional, default: 10)

**Response:**

```json
{
  "results": [1, 5, 3, 7, ...]
}
```

### Insert

Insert a vector into the index.

```bash
POST /insert
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "id": 1
}
```

**Parameters:**

- `vector`: Array of floats representing the vector (required)
- `id`: Unique identifier for the vector (required)

**Response:**

```json
{
  "success": true,
  "message": "Vector with id 1 inserted successfully"
}
```

## Configuration

The service loads configuration from a `.env` file and supports two configuration methods:

### Method 1: YAML Configuration File (Recommended)

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set the path to your configuration file:
   ```env
   INDEX_CONFIG_PATH=./configs/example.yaml
   ```

3. The service will load the index configuration from the specified YAML file.

Example configuration files are provided:
- `configs/example.yaml` - Detailed configuration with comments
- `configs/minimal.yaml` - Minimal configuration

### Method 2: Default Configuration

If `INDEX_CONFIG_PATH` is not set in `.env`, the service will use a hardcoded default configuration:
- **Vector dimension (input_shape)**: 128
- **Buffer size**: 1000
- **Arity**: 2
- **Device**: CPU
- **Distance function**: L2
- **Delete method**: OidToBucket
- **Compaction strategy**: BentleySaxe with NoRebuild

### YAML Configuration File Structure

```yaml
# Compaction strategy: determines how levels are managed during insertions
compaction_strategy: bentley_saxe:no_rebuild  # Options: bentley_saxe:no_rebuild, bentley_saxe:basic_rebuild, bentley_saxe:greedy_rebuild

# Level index configuration
levels:
  model:
    layers:
      - size: 128
      - size: 64
    train_params:
      threshold_samples: 1000      # Samples before model retraining
      batch_size: 256               # Training batch size
      epochs: 3                      # Number of training epochs
      max_iters: 10                 # Max clustering iterations
      retrain_strategy: no_retrain   # Options: no_retrain, from_scratch
  bucket_size: 1000

# Size of the buffer for new insertions
buffer_size: 1000

# Vector dimension (must match inserted vectors)
input_shape: 128

# Tree arity (children per node)
arity: 2

# Device: cpu or gpu
device:
  cpu

# Distance metric: l2 or dot
distance_fn: l2

# Delete method (currently only oid_to_bucket)
delete_method: oid_to_bucket
```

## Dependencies

- **axum**: A modular web framework for building APIs
- **tokio**: Async runtime with full features
- **serde/serde_json**: Serialization/deserialization
- **serde_yaml**: YAML configuration parsing
- **dotenvy**: .env file loading
- **dynamic_learned_index**: The core learned index implementation
- **tower**: Middleware and service abstractions
- **tracing**: Structured logging

## Example Usage

### Insert vectors:

```bash
curl -X POST http://127.0.0.1:3000/insert \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, ...], "id": 1}'
```

### Search:

```bash
curl -X POST http://127.0.0.1:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, ...], "k": 5}'
```

### Health check:

```bash
curl http://127.0.0.1:3000/health
```

## Notes

- The index is thread-safe and can handle concurrent requests
- Each request locks the index for the duration of the operation
- For production use, consider implementing more sophisticated concurrency patterns or read-write locks for better performance on read-heavy workloads
- The vector dimension must match the `input_shape` configured in the index
- Configuration files use YAML format and must be valid according to the dynamic_learned_index IndexConfig structure
