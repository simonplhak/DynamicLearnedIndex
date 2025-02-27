from __future__ import annotations

import time
from pathlib import Path

from cli import parse_arguments
from datasets import load_data
from execution_environment import detect_environment
from hnswlib import Index
from loguru import logger
from torch import Tensor, arange, empty, float32, manual_seed

from dli.utils import (
    measure_memory_usage,
    measure_runtime,
    obtain_commit_hash,
    obtain_dirty_state,
    obtain_metacentrum_experiment_id,
    time_fmt,
)

SEED = 42
manual_seed(SEED)

EXPERIMENTAL_RESULTS_DIR = Path('experiments/results')

commit_hash, dirty_state = obtain_commit_hash(), obtain_dirty_state()
experiment_id = (
    f'{obtain_metacentrum_experiment_id()}-{time.strftime("%Y%m%d-%H%M%S")}'
    f'-{commit_hash}{"-dirty" if dirty_state else ""}'
)

logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'experiment.log', backtrace=True, diagnose=True)
logger.add(EXPERIMENTAL_RESULTS_DIR / experiment_id / 'serialized.log', backtrace=True, diagnose=True, serialize=True)

args = parse_arguments()
experiment_config = detect_environment().create_config(args, commit_hash, dirty_state)

X, Q, GT = load_data(experiment_config.dataset_config, experiment_config.dataset_path_prefix)
logger.info(f'Loaded dataset of {len(X)} objects of size {(X.element_size() * X.nelement()) / 1024**2:.0f} MB')

X = X.to(float32)


@measure_memory_usage
@measure_runtime
def create_index_bulk(X: Tensor, ef_construction: int, M: int) -> Index:
    index = Index(space='ip', dim=X.shape[1])  # possible options are l2, cosine or ip
    index.init_index(max_elements=len(X), ef_construction=ef_construction, M=M)
    index.add_items(X, arange(len(X)))
    return index


def search_index(index: Index, ef_search: int, Q: Tensor, k: int) -> Tensor:
    index.set_ef(ef_search)  # ef should always be > k
    I, _ = index.knn_query(Q, k=k)
    return I


M = 8  # number of connections each vertex will have
ef_construction = 100  # depth of layers explored during index construction

k = 30

logger.info(f'Adding {len(X)} objects to index')
start = time.time()
index = create_index_bulk(X, ef_construction, M)  # ! Requires max_elements to be set before adding elements!
logger.info(f'Index construction time: {time_fmt(time.time() - start)}')
logger.info(f'Insert throughput: {int(len(X) / (time.time() - start))} IPS')

logger.info(f'Searching for {len(Q)} queries')
logger.info('ef_search,search_time,search_throughput,recall')

for ef_search in [40, 50, 60, 70, 80, 90, 100]:
    assert ef_search > k  # ef should always be > k

    start = time.time()
    I = search_index(index, ef_search, Q, k)

    recalls = empty(len(Q), dtype=float32)
    for i in range(len(Q)):
        recalls[i] = len(set((I[i, :] + 1).tolist()).intersection(set(GT[i, :k].tolist()))) / k

    logger.info(f'{ef_search},{time_fmt(time.time() - start)},{int(len(Q) / (time.time() - start))},{recalls.mean()}')
