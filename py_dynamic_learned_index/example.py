import numpy as np

from py_dynamic_learned_index import DynamicLearnedIndexBuilder, DynamicLearnedIndex

builder = DynamicLearnedIndexBuilder()
input_shape = 768
builder = (
    builder
        .buffer_size(10)  # size of the buffer
        .input_shape(input_shape)  # the shape of the input vector
        .distance_fn('dot')  # options: dot|l2
        .arity(3)  # arity of the tree structure created by the index
        .levelling('bentley_saxe')  # type of levelling used to construct new levels of tree; options: bentley_saxe
)

# optional: make model device use gpu
# builder = builder.device(gpu:0)

# TBD
# builder.levels(...)
# builder.bucket_size(10)  # size of the individial buckets

index: DynamicLearnedIndex = builder.build()

# OR load from yaml
# index: DynamicLearnedIndex = DynamicLearnedIndexBuilder.from_yaml("../configs/example.yaml").build()

print(index)


queries = [np.random.rand(input_shape).astype(np.float32) for _ in range(1000)]
for i, query in enumerate(queries):
    index.insert(query, i)

k = 3
nprobe = 1
search_strategy = 'model'  # options: knn, model
for i in range(0, len(queries) - 1, 50):
    res = index.search(queries[i], k, nprobe=nprobe, search_strategy=search_strategy)
    print(f'For query "{i}" index found: {res}')

