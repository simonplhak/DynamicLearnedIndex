from __future__ import annotations

import socket
import time

import h5py
import torch
from faiss import METRIC_INNER_PRODUCT
from loguru import logger
from tqdm import tqdm

from configuration import DistanceConfig, ExperimentConfig, FrameworkConfig, SamplingConfig, SearchConfig
from leveling import Leveling
from lmi import LMIIndex
from search_result import SearchResult
from utils import measure_runtime

SEED = 42
torch.manual_seed(SEED)


experiment_config = ExperimentConfig(
    FrameworkConfig(
        LMIIndex,
        arity=3,
        bucket_shape=(200 if socket.gethostname() == 'Pro.local' else 1_000, 768),
        distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
        sampling=SamplingConfig(percentage=0.1, threshold=100_000),
    ),
    [SearchConfig(k=10, nprobe=nprobe) for nprobe in [1, 2]],
)

logger.info(experiment_config)

# Load the dataset
DATASET_SIZE = 10_000 if socket.gethostname() == 'Pro.local' else 300_000
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
stats = framework.collect_stats()
logger.info(stats)


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
