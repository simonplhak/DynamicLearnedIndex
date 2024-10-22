from __future__ import annotations

import h5py
import torch
from faiss import METRIC_L2
from loguru import logger
from tqdm import tqdm

from bentley_saxe import BentleySaxe
from bliss import BLISSIndex
from dummy import DummyIndex
from leveling import Leveling
from lmi import LMIIndex
from utils import measure_runtime

SEED = 42
torch.manual_seed(SEED)

ARITY = 3
BUCKET_SIZE = 200
DIMENSIONALITY = 768
BUCKET_SHAPE = (BUCKET_SIZE, DIMENSIONALITY)
METRIC = METRIC_L2
KEEP_MAX = False  # Related to METRIC
DATASET_SIZE = 10_000
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
# framework = BentleySaxe(
framework = Leveling(
    BLISSIndex,
    # DummyIndex,
    # LMIIndex,
    ARITY,
    BUCKET_SHAPE,
    METRIC,
    KEEP_MAX,
)

# Insert the dataset one object at a time
for i in range(len(X)):
    framework.insert(X[i], i)

    if (i + 1) % (len(X) // 10) == 0:
        logger.info(f'Inserted {i+1} objects')

    assert framework.get_n_objects() == i + 1, f'Wrong number of objects: {framework.get_n_objects()} != {i + 1}'

# torch.save(framework, 'framework.pt')
# framework = torch.load('framework.pt')
framework.print_stats()


@measure_runtime
def perform_search(k: int, nprobe: int) -> float:
    recall = 0
    for i in tqdm(range(len(Q))):
        _, I = framework.search(Q[i], k, nprobe)
        # _, I = faiss.knn(Q[i : i + 1], X, k, metric=METRIC)
        # I[i]
        # I = I[0]
        # logger.info(torch.from_numpy(I + 1), GT[i, :k])
        # logger.info((I + 1).tolist())
        recall += len(set((I[0] + 1).tolist()).intersection(set(GT[i, :k].tolist()))) / k
        # recall += (torch.from_numpy(I + 1) == GT[i, :k]).sum().item() / len(GT)
        # logger.info(recall)
        # exit(0)
    return recall / len(Q)  # 0.058560000000003706


# Search
k = 10
logger.info(f'{k=}')
for nprobe in range(1, 20 + 1):
    logger.info(f'{nprobe=}')
    logger.info(perform_search(k, nprobe))

# import faiss

# D, I = faiss.knn(Q, X, k, metric=METRIC)

# logger.info(Q.shape)
# logger.info(GT.shape)

# logger.info(torch.from_numpy((I + 1)[:, :k]))
# logger.info(GT[:, :k])
# _, I = faiss.knn(Q, X, k, metric=METRIC)


# recall = (torch.from_numpy((I + 1)[:, :k]) == GT[:, :k]).sum() / len(GT)
# logger.info(recall)
# logger.info(framework.levels[0].buckets[0].ids)
# logger.info(framework.levels[0].buckets[1].ids)
# logger.info(framework.levels[0].buckets[2].ids)

# index.get_n_objects()
# index.top_level_bucket.get_n_objects()
# index.levels[0].get_n_objects()
# index.levels[0].buckets[0].get_n_objects()
# index.levels[0].buckets[1].get_n_objects()
# index.levels[0].buckets[2].get_n_objects()
# index.levels[1].get_n_objects()
# for i in range(ARITY**2):
#     logger.info(i, index.levels[1].buckets[i].get_n_objects())


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
# logger.info(f"Recall: {recall}")

############################

# import h5py

# DATASET_SIZE = 1_000
# X = torch.from_numpy(h5py.File('laion2B-en-clip768v2-n=100K.h5', 'r')['emb'][:DATASET_SIZE]).to(torch.float32)  # type: ignore

# bliss = BLISSIndex(20, faiss.METRIC_L2, (200, 768))
# b = Bucket((DATASET_SIZE, 768), faiss.METRIC_L2)
# b.insert(X, np.arange(DATASET_SIZE))
# bliss.train([b])

# for i, b in bliss.buckets.items():
#     logger.info(i, b.get_n_objects())
