from __future__ import annotations

import time
from dataclasses import dataclass

import h5py
import torch
from faiss import METRIC_INNER_PRODUCT
from loguru import logger
from tqdm import tqdm

from configuration import DistanceConfig, ExperimentConfig, FrameworkConfig, SearchConfig
from leveling import Leveling
from lmi import LMIIndex
from utils import measure_runtime

SEED = 42
torch.manual_seed(SEED)


experiment_config = ExperimentConfig(
    FrameworkConfig(
        LMIIndex,
        arity=3,
        bucket_shape=(200, 768),
        distance=DistanceConfig(
            METRIC_INNER_PRODUCT,
            keep_max=True,
        ),
        sample_percentage=0.1,
    ),
    [SearchConfig(k=10, nprobe=nprobe) for nprobe in [1, 2]],
)

logger.info(experiment_config)

# Load the dataset
DATASET_SIZE = 10_000
X = torch.from_numpy(h5py.File('laion2B-en-clip768v2-n=300K.h5', 'r')['emb'][:DATASET_SIZE]).to(torch.float32)  # type: ignore
Q = torch.from_numpy(h5py.File('public-queries-2024-laion2B-en-clip768v2-n=10k.h5', 'r')['emb'][:]).to(  # type: ignore
    torch.float32,
)
GT = torch.from_numpy(
    h5py.File('gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5', 'r')['knns'][:],  # type: ignore
)

# Create the framework
framework = Leveling(experiment_config.framework_config)


# Insert the dataset one object at a time
@measure_runtime
def insert_objects(X: torch.Tensor) -> None:
    for i in range(len(X)):
        framework.insert(X[i], i)

        if (i + 1) % (len(X) // 10) == 0:
            logger.info(f'Inserted {i+1} objects')

        assert framework.get_n_objects() == i + 1, f'Wrong number of objects: {framework.get_n_objects()} != {i + 1}'

    logger.info(f'Inserted {len(X)} objects')


insert_objects(X)

# torch.save(framework, 'framework.pt')
# framework = torch.load('framework.pt')
framework.print_stats()


@dataclass
class SearchResult:
    database_size: int
    n_queries: int
    recall_per_query: list[float]
    n_candidates_per_query: list[int]
    total_search_time: float

    def add(self, recall: float, n_candidates: int) -> None:
        self.recall_per_query.append(recall)
        self.n_candidates_per_query.append(n_candidates)

    def avg_recall(self) -> float:
        return sum(self.recall_per_query) / self.n_queries * 100

    def avg_n_candidates(self) -> float:
        return sum(self.n_candidates_per_query) / self.n_queries

    def candidates_percentage(self) -> float:
        return self.avg_n_candidates() / self.database_size * 100

    def avg_time_per_query_in_ms(self) -> float:
        return self.total_search_time / self.n_queries * 1_000

    def log_stats(self) -> None:
        avg_recall = self.avg_recall()
        avg_n_candidates = self.avg_n_candidates()
        candidates_percentage = self.candidates_percentage()
        avg_time_per_query_in_ms = self.avg_time_per_query_in_ms()

        logger.info(
            f'{avg_recall:.2f}%, '
            f'{avg_n_candidates:.2f} candidates ({candidates_percentage:.1f}%), '
            f'{avg_time_per_query_in_ms:.2f}ms per query',
        )


@measure_runtime
def perform_search(db_size: int, config: SearchConfig) -> SearchResult:
    recall_per_query = []
    n_candidates_per_query = []

    s = time.time()
    for i in tqdm(range(len(Q))):
        _, I, n_query_candidates = framework.search(Q[i], config.k, config.nprobe)
        recall = len(set((I[0] + 1).tolist()).intersection(set(GT[i, : config.k].tolist()))) / config.k

        recall_per_query.append(recall)
        n_candidates_per_query.append(n_query_candidates)
    e = time.time() - s

    return SearchResult(db_size, len(Q), recall_per_query, n_candidates_per_query, e)


# Search
for config in experiment_config.search_configs:
    logger.info(f'{config=}')
    result = perform_search(len(X), config)
    result.log_stats()
    # TODO: store result in a file

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
