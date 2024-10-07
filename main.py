from __future__ import annotations

import h5py
import torch
from faiss import METRIC_L2

from bliss import BLISSIndex
from dummy import DummyIndex
from framework import Framework
from lmi import LMIIndex

SEED = 42
torch.manual_seed(SEED)

ARITY = 3
BUCKET_SIZE = 20_000
DIMENSIONALITY = 768
BUCKET_SHAPE = (BUCKET_SIZE, DIMENSIONALITY)
METRIC = METRIC_L2
KEEP_MAX = False  # Related to METRIC
DATASET_SIZE = 300_000
# N_QUERIES = 100

# Load the dataset
X = torch.from_numpy(h5py.File('laion2B-en-clip768v2-n=300K.h5', 'r')['emb'][:DATASET_SIZE]).to(torch.float32)  # type: ignore
Q = torch.from_numpy(h5py.File('public-queries-2024-laion2B-en-clip768v2-n=10k.h5', 'r')['emb'][:]).to(  # type: ignore
    torch.float32,
)
GT = torch.from_numpy(
    h5py.File('gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5', 'r')['knns'][:],  # type: ignore
)
assert X.shape[1] == DIMENSIONALITY

# Create the framework
# framework = Framework(
#     BLISSIndex,
#     # DummyIndex,
#     # LMIIndex,
#     ARITY,
#     BUCKET_SHAPE,
#     METRIC,
#     KEEP_MAX,
# )

# # Insert the dataset one object at a time
# for i in range(len(X)):
#     framework.insert(X[i], i)

#     if (i + 1) % (len(X) // 10) == 0:
#         print(f'Inserted {i+1} objects')

#     assert framework.get_n_objects() == i + 1, f'Wrong number of objects: {framework.get_n_objects()} != {i + 1}'

# torch.save(framework, 'framework.pt')


framework = torch.load('framework.pt')

framework.print_stats()

# Search
k = 10

import faiss

# D, I = faiss.knn(Q, X, k, metric=METRIC)

# print(Q.shape)
# print(GT.shape)

# print(torch.from_numpy((I + 1)[:, :k]))
# print(GT[:, :k])
# _, I = faiss.knn(Q, X, k, metric=METRIC)

recall = 0
for i in range(len(Q)):
    _, I = framework.search(Q[i], k)
    # _, I = faiss.knn(Q[i : i + 1], X, k, metric=METRIC)
    # I[i]
    # I = I[0]
    # print(torch.from_numpy(I + 1), GT[i, :k])
    # print((I + 1).tolist())
    recall += len(set((I[0] + 1).tolist()).intersection(set(GT[i, :k].tolist()))) / k
    # recall += (torch.from_numpy(I + 1) == GT[i, :k]).sum().item() / len(GT)
    # print(recall)
    # exit(0)
print(recall / len(Q))  # 0.058560000000003706

# recall = (torch.from_numpy((I + 1)[:, :k]) == GT[:, :k]).sum() / len(GT)
# print(recall)
# print(framework.levels[0].buckets[0].ids)
# print(framework.levels[0].buckets[1].ids)
# print(framework.levels[0].buckets[2].ids)

# index.get_n_objects()
# index.top_level_bucket.get_n_objects()
# index.levels[0].get_n_objects()
# index.levels[0].buckets[0].get_n_objects()
# index.levels[0].buckets[1].get_n_objects()
# index.levels[0].buckets[2].get_n_objects()
# index.levels[1].get_n_objects()
# for i in range(ARITY**2):
#     print(i, index.levels[1].buckets[i].get_n_objects())


# # Obtain a query from the user -- here we sample a random query from the dataset
# query = X[torch.randint(0, n, (1,))]
# # Number of neighbors to look for
# k = 10
# # Search for the k nearest neighbors
# nearest_neighbors = lmi.search(query, k)

# # Evaluate the accuracy of the LMI's result

# # Calculate the ground truth for the query over the whole dataset
# ground_truth = torch.argsort(torch.cdist(query, X)).reshape(-1)[:k]

# # Calculate the recall -- closer to 1 is better
# recall = (
#     len(set(nearest_neighbors.tolist()).intersection(set(ground_truth.tolist())))
#     / k
# )
# print(f"Recall: {recall}")

############################

# import h5py

# DATASET_SIZE = 1_000
# X = torch.from_numpy(h5py.File('laion2B-en-clip768v2-n=100K.h5', 'r')['emb'][:DATASET_SIZE]).to(torch.float32)  # type: ignore

# bliss = BLISSIndex(20, faiss.METRIC_L2, (200, 768))
# b = Bucket((DATASET_SIZE, 768), faiss.METRIC_L2)
# b.insert(X, np.arange(DATASET_SIZE))
# bliss.train([b])

# for i, b in bliss.buckets.items():
#     print(i, b.get_n_objects())
