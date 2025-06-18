import numpy as np

from py_dynamic_learned_index import DynamicLearnedIndexBuilder, DynamicLearnedIndex

builder = DynamicLearnedIndexBuilder()
input_shape = 768
builder = builder.buffer_size(10).input_shape(input_shape)
# TBD
# builder.levelling(...)
# builder.levels(...)
# builder.arity(...)
# builder.device(...)

index: DynamicLearnedIndex = builder.build()

# OR load from yaml
# index: DynamicLearnedIndex = DynamicLearnedIndexBuilder.from_yaml("../configs/example.yaml").build()

print(index)


queries = [np.random.rand(input_shape).astype(np.float32) for _ in range(1000)]
for i, query in enumerate(queries):
    index.insert(query, i)

k = 3
for i in range(0, len(queries) - 1, 50):
    res = index.search(queries[i], k)
    print(f'For query "{i}" index found: {res}')

