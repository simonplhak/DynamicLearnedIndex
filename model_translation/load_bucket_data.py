import numpy as np
import json

bin_filename = "buckets.bin"
meta_filename = "buckets_meta.json"
FLOAT_SIZE = 4  # f32 is 4 bytes

with open(meta_filename, 'r') as f:
    buckets_meta = json.load(f)

print(f"Found metadata for {len(buckets_meta)} buckets.")

all_data = np.memmap(bin_filename, dtype=np.float32, mode='r')

buckets = []
for i, meta in enumerate(buckets_meta):
    byte_offset = meta["offset"]
    count = meta["count"]
    start_index = byte_offset // FLOAT_SIZE
    end_index = start_index + count
    bucket_data = np.array(all_data[start_index : end_index])
    buckets.append(bucket_data)

print(f"Loaded buckets: {buckets}")