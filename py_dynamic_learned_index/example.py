from pathlib import Path
import shutil
import numpy as np

from py_dynamic_learned_index import DynamicLearnedIndexBuilder, DynamicLearnedIndex

builder = DynamicLearnedIndexBuilder()
input_shape = 768
builder = (
    builder
        .buffer_size(100)  # size of the buffer
        .bucket_size(100)  # size of the bucket
        .input_shape(input_shape)  # the shape of the input vector
        .distance_fn('dot')  # options: dot|l2
        .arity(3)  # arity of the tree structure created by the index
        .compaction_strategy('bentley_saxe:basic_rebuild')  # type of levelling used to construct new levels of tree;
        .delete_method('oid_to_bucket')
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
# index: DynamicLearnedIndex = DynamicLearnedIndexBuilder.from_yaml("../configs/example.yaml").build()

print(index)


queries = [np.random.rand(input_shape).astype(np.float32) for _ in range(1000)]
for i, query in enumerate(queries):
    index.insert(query, i)

k = 3
n_candidates = 300
search_strategy = 'model'  # options: knn, model
for i in range(0, len(queries) - 1, 50):
    res = index.search(queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy)
    print(f'For query "{i}" index found: {res}')


# index statistics
print(f'n_buckets={index.n_buckets()}; n_levels={index.n_levels()}; occupied={index.occupied()}, empty_buckets={index.n_empty_buckets()}')

# delete
id_to_delete = 0
deleted_query, deleted_id = index.delete(id_to_delete)
print(f"""delete: 
      deleted_query is same as inserted query: {(deleted_query == queries[id_to_delete]).all()}, 
      {id_to_delete=}, 
      {deleted_id=}""")


# delte non-existing id
id_to_delete = len(queries)
res = index.delete(id_to_delete)
print(f'deleting non-existing id results in: {res}')


# verbose search that returns additional statistics
res, statistics = index.verbose_search(queries[0], k, n_candidates=n_candidates, search_strategy=search_strategy)
print(f"""Verbose search: 
      {res=}, 
      {statistics=}, 
      {statistics.total_visited_buckets=}, 
      {statistics.total_visited_records=}""")

# verbose delete
id_to_delete = 2
(deleted_query, deleted_id), deleted_stats = index.verbose_delete(id_to_delete)
print(f"""delete: 
      deleted_query is same as inserted query: {(deleted_query == queries[id_to_delete]).all()}, 
      {id_to_delete=}, 
      {deleted_id=}, 
      {deleted_stats=}, 
      {deleted_stats.affected_level=}""")


# serialize index to disk

working_dir = Path('index_dump')
if working_dir.exists():  # remove dir if exists
    print('Removing existing directory for index dump')
    shutil.rmtree(working_dir)

# working dir must not exist before dumping index
index.dump(str(working_dir))



# load index from disk

loaded_index = DynamicLearnedIndexBuilder.from_disk(str(working_dir)).build()
assert index.n_buckets() == loaded_index.n_buckets()
assert index.n_levels() == loaded_index.n_levels()
assert index.occupied() == loaded_index.occupied()
assert index.n_empty_buckets() == loaded_index.n_empty_buckets()
print(f'n_buckets={loaded_index.n_buckets()}; n_levels={loaded_index.n_levels()}; occupied={loaded_index.occupied()}, empty_buckets={loaded_index.n_empty_buckets()}')

for i in range(0, len(queries) - 1, 50):
    res = index.search(queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy)
    loaded_res = loaded_index.search(queries[i], k, n_candidates=n_candidates, search_strategy=search_strategy)
    assert (res == loaded_res).all()
    print(f'For query "{i}" loaded index found: {loaded_res}')