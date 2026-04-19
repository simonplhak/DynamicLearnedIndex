import io
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from py_dynamic_learned_index import (
    DynamicLearnedIndex,
    DynamicLearnedIndexBuilder,
    log_init,
)

# import py_dynamic_learned_index as rust

# log_init = rust.log_init

# DTYPE can be "f16" or "f32", default "f16", override with env DLI_DTYPE
DTYPE = os.getenv("DLI_DTYPE", "f16")

# Initialize Rust logger to write to a buffer
log_buffer = io.StringIO()
log_init(log_buffer, "debug")

builder = DynamicLearnedIndexBuilder(dtype=DTYPE)
input_shape = 768
builder = (
    builder.buffer_size(100)  # size of the buffer
    .bucket_size(100)  # size of the bucket
    .input_shape(input_shape)  # the shape of the input vector
    .distance_fn("dot")  # options: dot|l2
    .arity(3)  # arity of the tree structure created by the index
    .compaction_strategy(
        "bentley_saxe:basic_rebuild"
    )  # type of levelling used to construct new levels of tree;
    .delete_method("oid_to_bucket")
    # MODEL SPECIFICATIONS
    .add_linear_layer(256)
    .add_relu_layer()
    .train_batch_size(8)
    .train_epochs(3)
    .train_threshold_samples(100)
    .quantize(True)
    .retrain_strategy("from_scratch")  # possible values: no_retrain, from_scratch
)
# options for compaction strategy:
#   compaction strategy types: bentley_saxe
#   rebuild strategies: no_rebuild, basic_rebuild

# optional: make model device use gpu
# builder = builder.device(gpu:0)

# TBD
# builder.levels(...)
# builder.bucket_size(10)  # size of the individial buckets

index: DynamicLearnedIndex = builder.build()

# OR load from yaml
# index: DynamicLearnedIndex = DynamicLearnedIndexBuilder.from_yaml(
#     "../configs/example.yaml"
# ).build()


if DTYPE == "f16":
    queries = [np.random.rand(input_shape).astype(np.float16).view(np.uint16) for _ in range(1000)]
else:
    queries = [np.random.rand(input_shape).astype(np.float32) for _ in range(1000)]
for i, query in enumerate(queries):
    index.insert(query, i)

k = 3
n_candidates = 300
search_strategy = "model"  # options: knn, model
print("INITIAL SEARCH IN INDEX")
for i in range(0, len(queries) - 1, 50):
    res = index.search(
        queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy
    )
    print(f'For query "{i}" index found: {res}')
print()

# index statistics
print("INDEX STATISTICS")
print(
    f"n_buckets={index.n_buckets()}\n"
    f"n_levels={index.n_levels()}\n"
    f"occupied={index.occupied()}\n"
    f"empty_buckets={index.n_empty_buckets()}\n"
    f"buffer_occupied={index.buffer_occupied()}"
)
for level_idx in range(index.n_levels()):
    print(
        f"  LEVEL: {level_idx}\n"
        f"      level_occupied={index.level_occupied(level_idx)}\n"
        f"      level_n_buckets={index.level_n_buckets(level_idx)}\n"
        f"      level_total_size={index.level_total_size(level_idx)}\n"
        f"      level_n_empty_buckets={index.level_n_empty_buckets(level_idx)}"
    )
    for bucket_idx in range(index.level_n_buckets(level_idx)):
        print(
            f"      BUCKET: {bucket_idx}"
            f"          bucket_occupied={index.bucket_occupied(level_idx, bucket_idx)}"
        )
print()

# delete
id_to_delete = 0
deleted_query, deleted_id = index.delete(id_to_delete)

print(f"""delete:
      deleted_query is same as inserted query:
      {(deleted_query == queries[id_to_delete]).all()},
      {id_to_delete=},
      {deleted_id=}""")
print()


# delte non-existing id
id_to_delete = len(queries)
res = index.delete(id_to_delete)
print(f"deleting non-existing id results in: {res}")
print()

# serialize index to disk

working_dir = Path("index_dump")
if working_dir.exists():  # remove dir if exists
    print("Removing existing directory for index dump")
    shutil.rmtree(working_dir)

# working dir must not exist before dumping index
index.dump(str(working_dir))
print(f"Index dumped into {str(working_dir)}")
print()

# load index from disk

loaded_index = DynamicLearnedIndexBuilder.from_disk(str(working_dir))
assert index.n_buckets() == loaded_index.n_buckets()
assert index.n_levels() == loaded_index.n_levels()
assert index.occupied() == loaded_index.occupied()
assert index.n_empty_buckets() == loaded_index.n_empty_buckets()

print("INDEX STATISTICS FOR LOADED INDEX")
print(
    f"n_buckets={index.n_buckets()}\n"
    f"n_levels={index.n_levels()}\n"
    f"occupied={index.occupied()}\n"
    f"empty_buckets={index.n_empty_buckets()}\n"
    f"buffer_occupied={index.buffer_occupied()}"
)
for level_idx in range(index.n_levels()):
    print(
        f"  LEVEL: {level_idx}\n"
        f"      level_occupied={index.level_occupied(level_idx)}\n"
        f"      level_n_buckets={index.level_n_buckets(level_idx)}\n"
        f"      level_total_size={index.level_total_size(level_idx)}\n"
        f"      level_n_empty_buckets={index.level_n_empty_buckets(level_idx)}"
    )
    for bucket_idx in range(index.level_n_buckets(level_idx)):
        print(
            f"      BUCKET: {bucket_idx}"
            f"          bucket_occupied={index.bucket_occupied(level_idx, bucket_idx)}"
        )
print()

print("SEARCH IN LOADED INDEX")
for i in range(0, len(queries) - 1, 50):
    res = index.search(
        queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy
    )
    loaded_res = loaded_index.search(
        queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy
    )
    assert (res == loaded_res).all()
    print(f'For query "{i}" loaded index found: {loaded_res}')
print()

print("MEMORY USAGE")


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"


memory_usage = index.memory_usage()

print(
    f"memory usage of the index: {memory_usage} bytes -> {format_bytes(memory_usage)}"
)

# uncomment if needed
# print("CAPTURED LOGS")
# import json

# for log in log_buffer.getvalue().split("\n")[:-1]:
#     log = json.loads(log)
#     print(log)
# log_buffer.close()
